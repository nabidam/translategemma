#!/usr/bin/env python3
"""Probe what negative logit_bias actually does on the served vLLM.

Answers the questions docs/2026-08-21_glossary_memory_design.md leaves open for
phase 2.5 (negative-constraint re-decode) and that reading vllm/v0.13.0 source
cannot settle:

1. Does logit_bias measurably suppress a token under temperature=0, or is it
   accepted and ignored?
2. How strong does the bias need to be? -100 is the OpenAI convention, but the
   value that actually flips a greedy argmax is a property of this checkpoint's
   logit scale, not of the API.
3. What does the model emit instead? A banned rendering that is replaced by a
   worse one is not a win.
4. Is the bias request-scoped or prompt-scoped when one request carries several
   prompts? Source says request-scoped; this confirms it against the running
   server rather than trusting the reading.
5. Does /v1/completions reject bad_words, as the v0.13.0 schema implies?

Stdlib only, so it runs inside the vLLM container or anywhere that can reach the
server. No transformers: token ids come from vLLM's own /tokenize endpoint, so
the ids are by construction the ones the server uses.

Usage:
    python3 scripts/probe_logit_bias.py --base-url http://localhost:8000
    python3 scripts/probe_logit_bias.py --base-url ... --text "Your source text."
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

# Bias strengths to sweep, weakest first. -100 is the OpenAI convention and the
# value most callers reach for; the sweep exists because the value that flips a
# greedy argmax depends on this checkpoint's logit scale.
BIAS_SWEEP = (-1.0, -5.0, -20.0, -100.0)

DEFAULT_TEXT = "The model relies on multi-query attention to process the genome sequence."


def post(base_url: str, path: str, payload: dict, api_key: str | None) -> dict:
    """POST JSON and return the parsed body, raising with the server's text on error."""
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"{path} -> HTTP {error.code}: {body}") from error


def get(base_url: str, path: str, api_key: str | None) -> dict:
    request = urllib.request.Request(f"{base_url.rstrip('/')}{path}", method="GET")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def resolve_model(base_url: str, api_key: str | None) -> str:
    data = get(base_url, "/v1/models", api_key)
    models = [entry["id"] for entry in data.get("data", [])]
    if not models:
        raise RuntimeError("/v1/models returned no models.")
    if len(models) > 1:
        print(f"  note: several models served, using the first of {models}")
    return models[0]


def tokenize(base_url: str, model: str, text: str, api_key: str | None) -> list[int]:
    """Token ids straight from the server, so they match what it decodes with."""
    payload = {"model": model, "prompt": text, "add_special_tokens": True}
    return post(base_url, "/tokenize", payload, api_key)["tokens"]


def complete(
    base_url: str,
    model: str,
    prompt_ids: list[list[int]],
    max_tokens: int,
    api_key: str | None,
    logit_bias: dict[str, float] | None = None,
    logprobs: int | None = None,
) -> list[dict]:
    """Greedy completion. Returns choices ordered by their index field."""
    payload = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "skip_special_tokens": True,
    }
    if logit_bias:
        payload["logit_bias"] = logit_bias
    if logprobs is not None:
        payload["logprobs"] = logprobs
    # The logprobs block carries token STRINGS, not ids. Ids come from this
    # vLLM-specific flag (CompletionRequest.return_token_ids, present in v0.13.0),
    # which populates choice["token_ids"]. Ids are what logit_bias is keyed by,
    # so asking for strings and tokenizing them back would risk a mismatch.
    payload["return_token_ids"] = True
    choices = post(base_url, "/v1/completions", payload, api_key)["choices"]
    return sorted(choices, key=lambda choice: int(choice["index"]))


def first_generated_token_ids(choice: dict, count: int) -> list[int]:
    """Ids of the first `count` generated tokens."""
    token_ids = choice.get("token_ids")
    if token_ids:
        return list(token_ids[:count])
    return []


def banner(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="e.g. http://localhost:8000")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--ban-first-n",
        type=int,
        default=1,
        help="How many of the baseline's leading generated tokens to ban.",
    )
    args = parser.parse_args()

    model = resolve_model(args.base_url, args.api_key)
    print(f"model: {model}")
    prompt_ids = tokenize(args.base_url, model, args.text, args.api_key)
    print(f"prompt: {len(prompt_ids)} tokens")

    # --- 1. Baseline -----------------------------------------------------
    banner("1. Baseline (no bias)")
    baseline = complete(
        args.base_url, model, [prompt_ids], args.max_tokens, args.api_key, logprobs=1
    )[0]
    baseline_text = baseline.get("text", "")
    print(f"text: {baseline_text!r}")

    banned = first_generated_token_ids(baseline, args.ban_first_n)
    if not banned:
        print(
            "\nFAIL: the response carried no token_ids, so the probe cannot pick a token\n"
            "to ban. This vLLM may predate CompletionRequest.return_token_ids."
        )
        return 1
    print(f"banning first {len(banned)} generated token id(s): {banned}")

    # --- 2. Bias strength sweep ------------------------------------------
    banner("2. Does negative bias suppress, and at what strength?")
    suppressing_bias = None
    for bias in BIAS_SWEEP:
        biased = complete(
            args.base_url,
            model,
            [prompt_ids],
            args.max_tokens,
            args.api_key,
            logit_bias={str(token_id): bias for token_id in banned},
            logprobs=1,
        )[0]
        text = biased.get("text", "")
        emitted = first_generated_token_ids(biased, len(banned))
        avoided = not set(banned) & set(emitted)
        print(f"\nbias={bias:>7}: avoided={avoided}  first_ids={emitted}")
        print(f"  text: {text!r}")
        if avoided and suppressing_bias is None:
            suppressing_bias = bias

    if suppressing_bias is None:
        print(
            "\nRESULT: no bias in the sweep changed the leading token.\n"
            "logit_bias is accepted but not effective here -- phase 2.5 is not viable\n"
            "on this deployment and the design should fall back to reporting the miss."
        )
    else:
        print(f"\nRESULT: bias {suppressing_bias} was the weakest that suppressed.")
        print("Judge the substituted text above: a worse rendering is not an improvement.")

    # --- 3. Request-scoped or prompt-scoped? ------------------------------
    banner("3. Scope check: two prompts, one bias dict")
    both = complete(
        args.base_url,
        model,
        [prompt_ids, prompt_ids],
        args.max_tokens,
        args.api_key,
        logit_bias={str(token_id): -100.0 for token_id in banned},
        logprobs=1,
    )
    affected = [
        not set(banned) & set(first_generated_token_ids(choice, len(banned)))
        for choice in both
    ]
    print(f"per-prompt avoided: {affected}")
    if all(affected):
        print(
            "RESULT: request-scoped, as the v0.13.0 schema implies. A batched\n"
            "/completions call cannot carry per-segment bans, so phase 2.5 costs one\n"
            "extra single-prompt request per violating segment."
        )
    else:
        print("RESULT: unexpected -- bias did not apply uniformly. Investigate before planning.")

    # --- 4. bad_words: applied, or accepted and ignored? ------------------
    banner("4. Is bad_words APPLIED on /v1/completions?")

    # Control first. vLLM tolerates unknown request fields rather than
    # rejecting them, so "the server returned 200" proves nothing on its own --
    # an ignored parameter and a supported one look identical from the status
    # code. A field the server cannot possibly know establishes the baseline for
    # what "ignored" looks like.
    control_payload = {
        "model": model,
        "prompt": [prompt_ids],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "zzz_not_a_real_parameter": True,
    }
    try:
        post(args.base_url, "/v1/completions", control_payload, args.api_key)
        tolerates_unknown = True
    except RuntimeError:
        tolerates_unknown = False
    print(f"control: server {'IGNORES' if tolerates_unknown else 'REJECTS'} unknown fields")

    # Now the functional test: ban the text the baseline actually produced and
    # see whether the output moves.
    baseline_head = baseline_text[:20].rstrip()
    if not baseline_head:
        print("SKIP: baseline produced no text to ban.")
        return 0
    try:
        banned_run = post(
            args.base_url,
            "/v1/completions",
            {
                "model": model,
                "prompt": [prompt_ids],
                "max_tokens": args.max_tokens,
                "temperature": 0.0,
                "bad_words": [baseline_head],
            },
            args.api_key,
        )["choices"][0]
    except RuntimeError as error:
        print(f"RESULT: rejected outright.\n  {error}")
        return 0

    changed = banned_run.get("text", "") != baseline_text
    print(f"banned {baseline_head!r} -> output changed: {changed}")
    print(f"  text: {banned_run.get('text', '')!r}")
    if changed:
        print(
            "\nRESULT: bad_words is APPLIED here. Prefer it over logit_bias -- it bans a\n"
            "token SEQUENCE, which is what a multi-token Farsi term needs, instead of\n"
            "penalising individual tokens wherever they appear."
        )
    elif tolerates_unknown:
        print(
            "\nRESULT: accepted but INERT -- the same behaviour as the fake field above.\n"
            "This is the dangerous case: HTTP 200, no effect, no warning. Negative\n"
            "constraints must then use per-token logit_bias, and banning a term's first\n"
            "token also penalises it inside unrelated words in the same segment."
        )
    else:
        print("\nRESULT: accepted and had no effect, though unknown fields are rejected.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
