"""Admin routes for the termbase.

Mounted only when the glossary is enabled, so on a default deployment these
paths are 404 rather than 401 -- there is nothing there to authorize against.

The static key is the whole authorization boundary for this gateway: there is
no reverse-proxy auth in front of it. CORS is not authorization and plays no
part here.
"""

import secrets

from fastapi import APIRouter, Body, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field

from .service import GlossaryService, UnknownDomainError

# Entries matching one of these exactly are refused. Google ignores such
# entries silently; refusing loudly is better, because an entry that is
# accepted and then never fires looks like a broken feature. Exact match only:
# "the" is refused, "the cloud" is not.
STOPWORDS = frozenset(
    """
    a about an and are as at be by com for from how i in is it of on or that
    the this to was what when where who will with www edu
    """.split()
)

# Widened when sentinel protection ships (phase 2 of the design document).
SUPPORTED_TARGET_MODES = frozenset({"preferred"})


class DomainIn(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    src_lang: str
    tgt_lang: str
    description: str | None = None


class EntryIn(BaseModel):
    src_lang: str
    tgt_lang: str
    source_term: str = Field(min_length=1, max_length=512)
    target_term: str = Field(min_length=1, max_length=512)
    domain: str | None = None
    target_mode: str = "preferred"
    aliases: list[str] = []
    forbidden: list[str] = []
    case_sensitive: bool = False
    whole_word: bool = True
    priority: int = 0
    notes: str | None = None
    created_by: str | None = None


class DryRunIn(BaseModel):
    text: str
    src_lang: str
    tgt_lang: str
    domain: str | None = None


def build_router(service: GlossaryService, api_key: str) -> APIRouter:
    async def require_key(x_admin_key: str | None = Header(default=None)) -> None:
        # compare_digest so a wrong key costs the same time as a right one.
        # It also requires ASCII-only strings: headers are latin-1 decoded, so
        # a byte above 127 in the header reaches us as a non-ASCII str and
        # would otherwise raise TypeError here, turning the gateway's only
        # authorization boundary into an unauthenticated 500 on malformed
        # input rather than the 401 an unauthenticated caller must get.
        valid = (
            x_admin_key is not None
            and x_admin_key.isascii()
            and secrets.compare_digest(x_admin_key, api_key)
        )
        if not valid:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing admin key.")

    router = APIRouter(
        prefix="/admin/glossary", tags=["glossary-admin"], dependencies=[Depends(require_key)]
    )

    @router.get("/domains")
    async def list_domains():
        return [
            {
                "name": domain.name,
                "src_lang": domain.src_lang,
                "tgt_lang": domain.tgt_lang,
                "description": domain.description,
                "enabled": domain.enabled,
            }
            for domain in await service.store.list_domains()
        ]

    @router.post("/domains", status_code=status.HTTP_201_CREATED)
    async def create_domain(payload: DomainIn):
        try:
            domain = await service.store.create_domain(
                payload.name, payload.src_lang, payload.tgt_lang, payload.description
            )
        except ValueError as error:
            raise HTTPException(status.HTTP_409_CONFLICT, str(error)) from error
        await service.reload()
        return {"name": domain.name}

    @router.delete("/domains/{name}")
    async def delete_domain(name: str):
        if not await service.store.delete_domain(name):
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No such domain: {name!r}")
        await service.reload()
        return {"deleted": name}

    @router.get("/entries")
    async def list_entries(domain: str | None = None):
        try:
            entries = await service.store.list_entries(domain)
        except ValueError as error:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error
        return [
            {
                "id": entry.id,
                "source_term": entry.source_term,
                "target_term": entry.target_term,
                "target_mode": entry.target_mode,
                "aliases": entry.aliases,
                "forbidden": entry.forbidden,
                "enabled": entry.enabled,
            }
            for entry in entries
        ]

    @router.post("/entries", status_code=status.HTTP_201_CREATED)
    async def create_entry(payload: EntryIn):
        if payload.target_mode not in SUPPORTED_TARGET_MODES:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                f"target_mode {payload.target_mode!r} is not supported yet; "
                f"use one of {sorted(SUPPORTED_TARGET_MODES)}.",
            )
        if payload.source_term.strip().lower() in STOPWORDS:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                f"{payload.source_term!r} is a stopword and would never be applied. "
                "Multi-word phrases containing one are fine.",
            )
        try:
            entry = await service.store.create_entry(
                domain_name=payload.domain,
                src_lang=payload.src_lang,
                tgt_lang=payload.tgt_lang,
                source_term=payload.source_term,
                target_term=payload.target_term,
                target_mode=payload.target_mode,
                aliases=payload.aliases,
                forbidden=payload.forbidden,
                case_sensitive=payload.case_sensitive,
                whole_word=payload.whole_word,
                priority=payload.priority,
                notes=payload.notes,
                created_by=payload.created_by,
            )
        except ValueError as error:
            # Not a substring match on the message: create_entry's duplicate
            # message embeds the caller-supplied source_term verbatim, so an
            # admin naming their term e.g. "No such domain" would otherwise
            # be told their entry's domain is missing instead of that their
            # entry is a duplicate. Ask the store what is actually true
            # instead. A request with no domain at all can never be a domain
            # error -- only a named, missing domain can be.
            domain_missing = (
                payload.domain is not None
                and await service.store.get_domain(payload.domain) is None
            )
            code = status.HTTP_404_NOT_FOUND if domain_missing else status.HTTP_409_CONFLICT
            raise HTTPException(code, str(error)) from error
        await service.reload()
        return {"id": entry.id}

    @router.delete("/entries/{entry_id}")
    async def delete_entry(entry_id: int):
        if not await service.store.delete_entry(entry_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No entry {entry_id}.")
        await service.reload()
        return {"deleted": entry_id}

    @router.post("/reload")
    async def reload():
        return {"version": await service.reload()}

    @router.post("/dry-run")
    async def dry_run(payload: DryRunIn = Body(...)):
        """Show what would fire on this text. No model call, no side effects."""
        try:
            index = await service.resolve(payload.src_lang, payload.tgt_lang, payload.domain)
        except UnknownDomainError as error:
            # An endpoint whose whole purpose is catching mistakes before they
            # reach traffic must not answer a typo'd domain with a 500.
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                f"Unknown glossary domain {error.name!r}. Available: {error.available}",
            ) from error
        spans = service.plan(index, payload.text)
        return {
            "version": index.version,
            "matches": [
                {
                    "source_term": span.term.source_term,
                    "target_term": span.term.target_term,
                    "mode": span.term.target_mode,
                    "start": span.start,
                    "end": span.end,
                    "matched_text": payload.text[span.start : span.end],
                }
                for span in spans
            ],
        }

    return router
