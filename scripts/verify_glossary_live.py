#!/usr/bin/env python3
"""End-to-end check of the glossary against a live gateway and a real vLLM.

Everything in the test suite runs against stubs: the engine is faked, and no
test has ever put a glossary term through the actual model. This script is the
missing half. It drives the real deployment and asserts on what comes back.

The term and alias below are not arbitrary. They are the pair measured against
the served checkpoint while the feature was being designed (see
docs/2026-08-21_glossary_memory_design.md, "Measured on the served checkpoint"):
TranslateGemma renders "multi-query attention" as توجه چندگانه, and the termbase
is supposed to rewrite that to توجه چندپرسشی. So a mismatch here is a real
regression, not a badly chosen fixture.

Run it on the host where the gateway runs, after enabling the feature:

    TG_ADMIN_API_KEY=... python3 scripts/verify_glossary_live.py \
        --base-url http://localhost:8000

It creates its own entries and domains, exercises them, and removes them
again, so it is safe to run against a deployment that already has a termbase
-- with one caveat: if an entry for the same source term already exists, the
create returns 409 and the script stops rather than touching data it did not
create.

Checks 9 and 10 cover the two modes that alter behaviour beyond post-hoc
replacement. Neither asserts that the model obeys, because neither mechanism
can compel it: exact mode can only ask the decoder to carry a placeholder
through, and a forbidden ban can only push it off a wording, never onto the
right one. What they assert is that the gateway is honest about which
happened -- a placeholder never reaches a caller, and a forbidden rendering is
never returned unflagged.

Stdlib only, so it runs anywhere that can reach the gateway.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

SOURCE_TERM = "multi-query attention"
CANONICAL = "توجه چندپرسشی"
ALIAS = "توجه چندگانه"
SENTENCE = "The model relies on multi-query attention."
# Created and removed by check 8; named so it cannot collide with a real one.
DOMAIN = "verify-script-temporary"
# A second, *enabled* domain, so check 8 can prove the available list filters
# by enabled rather than merely being empty.
DOMAIN_ENABLED = "verify-script-enabled"

# exact mode: an invented product name, so the model has no learned rendering
# for it and any faithful output must be the sentinel carried through. Azure's
# own dynamic-dictionary documentation uses the same shape of example.
EXACT_TERM = "Wordomatic"
EXACT_SENTENCE = "We deployed Wordomatic on the cluster last week."

PASS = "PASS"
FAIL = "FAIL"


class Gateway:
    def __init__(self, base_url: str, admin_key: str | None):
        self.base_url = base_url.rstrip("/")
        self.admin_key = admin_key

    def request(self, method: str, path: str, payload=None, admin=False):
        """Return (status, parsed_body). Never raises for an HTTP error status."""
        data = json.dumps(payload).encode() if payload is not None else None
        headers = {"Content-Type": "application/json"} if data else {}
        if admin and self.admin_key:
            headers["X-Admin-Key"] = self.admin_key
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return response.status, json.loads(response.read() or b"null")
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", "replace")
            try:
                return error.code, json.loads(body)
            except json.JSONDecodeError:
                return error.code, {"raw": body[:400]}
        except urllib.error.URLError as error:
            # An unreachable gateway is a setup problem, not a failed check --
            # say so plainly instead of dumping a traceback.
            print(f"\nCannot reach {self.base_url}: {error.reason}")
            print("Is the gateway running, and is --base-url right?")
            raise SystemExit(2) from None


class Checks:
    def __init__(self):
        self.failures = 0

    def check(self, label: str, ok: bool, detail: str = "") -> bool:
        print(f"  [{PASS if ok else FAIL}] {label}")
        if detail:
            print(f"         {detail}")
        if not ok:
            self.failures += 1
        return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--admin-key",
        default=os.environ.get("TG_ADMIN_API_KEY"),
        help="Defaults to $TG_ADMIN_API_KEY.",
    )
    parser.add_argument("--text", default=SENTENCE)
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Leave the entry behind instead of deleting it.",
    )
    args = parser.parse_args()

    if not args.admin_key:
        print("No admin key. Set TG_ADMIN_API_KEY or pass --admin-key.")
        return 2

    gateway = Gateway(args.base_url, args.admin_key)
    checks = Checks()
    entry_id = None
    domain_created = False
    exact_entry_id = None

    print(f"\ngateway: {gateway.base_url}\n")

    # --- 0. Is the running container the code you think it is? -----------
    # A passing run against a stale image proves nothing about a change you
    # just made. The admin PATCH routes exist only in the newer code, so the
    # served OpenAPI schema is a reliable, unauthenticated tell.
    print("0. Deployed code")
    status, schema = gateway.request("GET", "/openapi.json")
    paths = schema.get("paths", {}) if status == 200 else {}
    has_patch = "patch" in paths.get("/admin/glossary/entries/{entry_id}", {})
    # exact mode and the forbidden re-decode ship together with the patch
    # routes, so one probe covers the build these checks assume.
    if not checks.check(
        "the admin PATCH routes are served",
        has_patch,
        ""
        if has_patch
        else "This container predates them. Rebuild and restart:\n"
        "         docker compose up -d --build translategemma-api",
    ):
        return 1

    # --- 1. Is the feature actually on? ----------------------------------
    print("1. Feature is enabled")
    status, info = gateway.request("GET", "/model-info")
    if not checks.check(
        "/model-info reachable", status == 200, f"status {status}"
    ):
        return 1
    enabled = bool(info.get("glossary_enabled"))
    checks.check(
        "glossary_enabled is true",
        enabled,
        "" if enabled else "Set TG_GLOSSARY_ENABLED=true and restart the gateway.",
    )
    if not enabled:
        return 1
    print(f"         serving {info.get('served_system')} via {info.get('upstream')}")

    try:
        # --- 2. Admin surface --------------------------------------------
        print("\n2. Admin surface")
        status, _ = gateway.request("GET", "/admin/glossary/entries")
        checks.check(
            "unauthenticated request is refused",
            status == 401,
            f"status {status} (expected 401)",
        )

        status, created = gateway.request(
            "POST",
            "/admin/glossary/entries",
            {
                "src_lang": info.get("default_source_lang", "en"),
                "tgt_lang": info.get("default_target_lang", "fa"),
                "source_term": SOURCE_TERM,
                "target_term": CANONICAL,
                "aliases": [ALIAS],
            },
            admin=True,
        )
        if status == 409:
            print(
                f"  [{FAIL}] an entry for {SOURCE_TERM!r} already exists.\n"
                "         Refusing to modify data this script did not create.\n"
                "         Delete it first, or run against a clean deployment."
            )
            return 1
        if not checks.check("entry created", status == 201, f"status {status}"):
            return 1
        entry_id = created.get("id")

        # --- 3. Dry run, before any model call ---------------------------
        print("\n3. Dry run (no model call)")
        status, dry = gateway.request(
            "POST",
            "/admin/glossary/dry-run",
            {
                "text": args.text,
                "src_lang": info.get("default_source_lang", "en"),
                "tgt_lang": info.get("default_target_lang", "fa"),
            },
            admin=True,
        )
        matches = dry.get("matches", []) if status == 200 else []
        checks.check(
            "the term is matched in the source",
            status == 200 and any(m["source_term"] == SOURCE_TERM for m in matches),
            f"status {status}, matches={[m.get('matched_text') for m in matches]}",
        )

        # --- 4. The real thing -------------------------------------------
        print("\n4. Translation with the termbase applied")
        status, result = gateway.request("POST", "/translate", {"text": args.text})
        if not checks.check("translate succeeded", status == 200, f"status {status}"):
            print(f"         {result}")
            return 1

        translation = result.get("translation", "")
        raw = result.get("raw_translation")
        report = result.get("glossary") or {}
        applied = report.get("applied", [])
        misses = report.get("misses", [])

        print(f"         raw:         {raw}")
        print(f"         translation: {translation}")

        checks.check(
            "glossary_version is reported",
            result.get("glossary_version") is not None,
            f"version={result.get('glossary_version')}",
        )
        checks.check(
            "the canonical term is in the translation",
            CANONICAL in translation,
            f"expected {CANONICAL!r}",
        )
        checks.check(
            "raw_translation is present and differs from translation",
            raw is not None and raw != translation,
            "If these are equal the model already produced the canonical form; "
            "that is a pass for the model and a no-op for the termbase.",
        )
        checks.check(
            "exactly one term reported applied",
            len(applied) == 1 and applied[0]["source_term"] == SOURCE_TERM,
            f"applied={applied}, misses={misses}",
        )

        # --- 5. The per-request off switch -------------------------------
        print("\n5. terminology_mode=off bypasses the termbase")
        status, off = gateway.request(
            "POST", "/translate", {"text": args.text, "terminology_mode": "off"}
        )
        checks.check(
            "no glossary keys when mode is off",
            status == 200 and "glossary" not in off and "glossary_version" not in off,
            f"status {status}, keys={sorted(off)}",
        )

        # --- 6. An unknown domain must be refused, not ignored ------------
        print("\n6. An unknown domain is refused")
        status, _ = gateway.request(
            "POST", "/translate", {"text": args.text, "domain": "definitely-not-a-domain"}
        )
        checks.check(
            "unknown domain returns 404",
            status == 404,
            f"status {status} (a 200 here would mean silent fallback)",
        )

        # --- 7. Correcting an entry keeps its id -------------------------
        print("\n7. PATCH corrects in place")
        status, patched = gateway.request(
            "PATCH",
            f"/admin/glossary/entries/{entry_id}",
            {"aliases": [ALIAS, "توجه چند پرسشی"]},
            admin=True,
        )
        checks.check(
            "patch preserves the entry id",
            status == 200 and patched.get("id") == entry_id,
            f"status {status}, id={patched.get('id')} (was {entry_id})",
        )
        checks.check(
            "the new alias is persisted",
            "توجه چند پرسشی" in (patched.get("aliases") or []),
            f"aliases={patched.get('aliases')}",
        )

        status, disabled = gateway.request(
            "PATCH", f"/admin/glossary/entries/{entry_id}", {"enabled": False}, admin=True
        )
        status, dry = gateway.request(
            "POST",
            "/admin/glossary/dry-run",
            {
                "text": args.text,
                "src_lang": info.get("default_source_lang", "en"),
                "tgt_lang": info.get("default_target_lang", "fa"),
            },
            admin=True,
        )
        checks.check(
            "disabling an entry takes effect without a restart",
            status == 200 and dry.get("matches") == [],
            f"matches={dry.get('matches')}",
        )
        gateway.request(
            "PATCH", f"/admin/glossary/entries/{entry_id}", {"enabled": True}, admin=True
        )

        # --- 8. A disabled domain says so --------------------------------
        print("\n8. A disabled domain is reported as disabled")
        for name in (DOMAIN, DOMAIN_ENABLED):
            gateway.request(
                "POST",
                "/admin/glossary/domains",
                {
                    "name": name,
                    "src_lang": info.get("default_source_lang", "en"),
                    "tgt_lang": info.get("default_target_lang", "fa"),
                },
                admin=True,
            )
        domain_created = True
        gateway.request(
            "PATCH", f"/admin/glossary/domains/{DOMAIN}", {"enabled": False}, admin=True
        )
        status, refused = gateway.request(
            "POST", "/translate", {"text": args.text, "domain": DOMAIN}
        )
        detail = str(refused.get("detail", ""))
        checks.check(
            "a disabled domain is refused as disabled, not as missing",
            status == 404 and "disabled" in detail.lower(),
            f"status {status}, detail={detail!r}",
        )
        # Asking about a domain that does not exist, so the error carries the
        # available list. Both halves matter: the disabled one must be absent
        # AND the enabled one present -- without the second assertion this
        # passes on any deployment whose list is simply empty, which is what
        # happened the first time this check ran.
        status, unknown = gateway.request(
            "POST", "/translate", {"text": args.text, "domain": "definitely-not-a-domain"}
        )
        offered = str(unknown.get("detail", "")).split("Available:")[-1]
        checks.check(
            "the available list names enabled domains",
            DOMAIN_ENABLED in offered,
            f"offered={offered.strip()!r}",
        )
        checks.check(
            "the available list omits disabled domains",
            DOMAIN not in offered,
            f"offered={offered.strip()!r}",
        )

        # --- 9. exact mode: the model never sees the term ----------------
        print("\n9. exact mode keeps a term verbatim")
        status, exact_created = gateway.request(
            "POST",
            "/admin/glossary/entries",
            {
                "src_lang": info.get("default_source_lang", "en"),
                "tgt_lang": info.get("default_target_lang", "fa"),
                "source_term": EXACT_TERM,
                "target_term": EXACT_TERM,
                "target_mode": "exact",
            },
            admin=True,
        )
        if checks.check(
            "an exact entry is accepted", status == 201, f"status {status}"
        ):
            exact_entry_id = exact_created.get("id")
            status, result = gateway.request(
                "POST", "/translate", {"text": EXACT_SENTENCE}
            )
            translation = result.get("translation", "")
            report = result.get("glossary") or {}
            applied = report.get("applied", [])
            misses = report.get("misses", [])
            print(f"         {translation}")
            checks.check(
                "the term survives character-for-character",
                status == 200 and EXACT_TERM in translation,
                f"status {status}, expected {EXACT_TERM!r} in the output",
            )
            checks.check(
                "no placeholder leaked to the caller",
                "__TG_TERM" not in translation,
                f"translation={translation!r}",
            )
            reported_exact = [a for a in applied if a.get("mode") == "exact"]
            lost = [m for m in misses if m.get("reason") == "sentinel_lost"]
            checks.check(
                "the outcome is reported honestly",
                bool(reported_exact) != bool(lost),
                f"applied(exact)={reported_exact}, misses={misses}",
            )
            if lost:
                print(
                    "         NOTE: the model did not carry the placeholder through.\n"
                    "         The gateway fell back correctly, but exact mode is not\n"
                    "         reliable on this checkpoint -- run the sentinel probe at\n"
                    "         scale before depending on it."
                )

        # --- 10. a forbidden rendering is removed or reported -------------
        print("\n10. a forbidden rendering never passes silently")
        gateway.request(
            "PATCH",
            f"/admin/glossary/entries/{entry_id}",
            {"aliases": [], "forbidden": [ALIAS]},
            admin=True,
        )
        status, result = gateway.request("POST", "/translate", {"text": args.text})
        translation = result.get("translation", "")
        violations = (result.get("glossary") or {}).get("violations", [])
        print(f"         {translation}")
        # The contract is not "the model obeys" -- a ban can only push it off a
        # wording, never onto the right one. What must hold is that a forbidden
        # rendering is never returned without being flagged.
        checks.check(
            "a forbidden rendering is either gone or reported",
            status == 200 and (ALIAS not in translation or bool(violations)),
            f"status {status}, violations={violations}",
        )
        if ALIAS in translation:
            print("         NOTE: the re-decode did not shift the model; reported instead.")
        else:
            print("         The re-decode moved the model off the forbidden rendering.")

    finally:
        if exact_entry_id is not None:
            gateway.request(
                "DELETE", f"/admin/glossary/entries/{exact_entry_id}", admin=True
            )
            print(f"cleanup: deleted exact entry {exact_entry_id}")
        if domain_created:
            for name in (DOMAIN, DOMAIN_ENABLED):
                gateway.request("DELETE", f"/admin/glossary/domains/{name}", admin=True)
            print(f"cleanup: deleted domains {DOMAIN!r}, {DOMAIN_ENABLED!r}")
        if entry_id is not None and not args.keep:
            status, _ = gateway.request(
                "DELETE", f"/admin/glossary/entries/{entry_id}", admin=True
            )
            print(f"\ncleanup: deleted entry {entry_id} (status {status})")
        elif entry_id is not None:
            print(f"\ncleanup: kept entry {entry_id} as requested")

    print()
    if checks.failures:
        print(f"{checks.failures} check(s) FAILED.")
        return 1
    print("All checks passed. The glossary works end to end against the real model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
