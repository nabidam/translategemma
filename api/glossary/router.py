"""Admin routes for the termbase.

Mounted only when the glossary is enabled, so on a default deployment these
paths are 404 rather than 401 -- there is nothing there to authorize against.

The static key is the whole authorization boundary for this gateway: there is
no reverse-proxy auth in front of it. CORS is not authorization and plays no
part here.
"""

import secrets
from collections.abc import Callable

from fastapi import APIRouter, Body, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from .normalize import normalize_lang_code
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

    # Lowercased and shape-checked at write time: load_terms compares
    # language codes with `==` (see glossary/store.py), so an entry or domain
    # stored as "EN" can never match a request carrying "en" -- a typo here
    # is a feature that looks broken forever, not a loud failure.
    @field_validator("src_lang", "tgt_lang")
    @classmethod
    def _normalize_lang(cls, value: str) -> str:
        return normalize_lang_code(value)


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

    @field_validator("src_lang", "tgt_lang")
    @classmethod
    def _normalize_lang(cls, value: str) -> str:
        return normalize_lang_code(value)


class DomainPatch(BaseModel):
    """Patchable fields of a domain. Omitted fields are left alone.

    The name is not patchable: it is the selector callers send, so renaming
    would silently break every request naming it.
    """

    description: str | None = None
    enabled: bool | None = None


class EntryPatch(BaseModel):
    """Patchable fields of an entry. Omitted fields are left alone.

    Identity -- source term, language pair, domain, case sensitivity -- is not
    patchable: those form the unique key, and changing one makes it a different
    entry. Preserving the id is the point, since that is what an administrator
    tracks in the misses report while curating aliases.
    """

    target_term: str | None = Field(default=None, min_length=1, max_length=512)
    aliases: list[str] | None = None
    forbidden: list[str] | None = None
    priority: int | None = None
    whole_word: bool | None = None
    enabled: bool | None = None
    notes: str | None = None


class DryRunIn(BaseModel):
    text: str
    src_lang: str
    tgt_lang: str
    domain: str | None = None


def build_router(
    get_service: Callable[[], "GlossaryService | None"], api_key: str
) -> APIRouter:
    """Build the admin router.

    Takes a getter rather than a service instance so the router can be mounted
    when the app is constructed, while the service it talks to is created later
    by the lifespan. Mounting at construction is what keeps route registration
    idempotent: mounting as a side effect of a re-entrant lifespan appends a
    second copy of every route each time it runs.
    """

    def _service() -> GlossaryService:
        service = get_service()
        if service is None:
            # Reachable only if the app was built with the glossary enabled and
            # the lifespan has not finished (or has already torn down).
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE, "Glossary is not ready."
            )
        return service

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
        service = _service()
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
        service = _service()
        try:
            domain = await service.store.create_domain(
                payload.name, payload.src_lang, payload.tgt_lang, payload.description
            )
        except ValueError as error:
            raise HTTPException(status.HTTP_409_CONFLICT, str(error)) from error
        await service.reload()
        return {"name": domain.name}

    @router.patch("/domains/{name}")
    async def update_domain(name: str, payload: DomainPatch):
        service = _service()
        fields = payload.model_dump(exclude_unset=True, exclude_none=True)
        if not fields:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY, "No patchable fields supplied."
            )
        domain = await service.store.update_domain(name, **fields)
        if domain is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No such domain: {name!r}")
        await service.reload()
        return {"name": domain.name, "enabled": domain.enabled,
                "description": domain.description}

    @router.delete("/domains/{name}")
    async def delete_domain(name: str):
        service = _service()
        if not await service.store.delete_domain(name):
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No such domain: {name!r}")
        await service.reload()
        return {"deleted": name}

    @router.get("/entries")
    async def list_entries(domain: str | None = None):
        service = _service()
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
        service = _service()
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
        if payload.domain is not None:
            # store.load_terms filters entries by the ENTRY's own language
            # pair and never consults the domain row's src_lang/tgt_lang (see
            # glossary/store.py) -- so those two columns are enforced here,
            # at write time, rather than left as unread, purely advisory
            # metadata that could quietly drift from what an entry actually
            # declares.
            domain = await service.store.get_domain(payload.domain)
            if domain is not None and (
                domain.src_lang != payload.src_lang or domain.tgt_lang != payload.tgt_lang
            ):
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_ENTITY,
                    f"Domain {payload.domain!r} is {domain.src_lang}->{domain.tgt_lang}; "
                    f"this entry is {payload.src_lang}->{payload.tgt_lang}. An entry's "
                    "language pair must match the domain it is created in.",
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

    @router.patch("/entries/{entry_id}")
    async def update_entry(entry_id: int, payload: EntryPatch):
        service = _service()
        fields = payload.model_dump(exclude_unset=True, exclude_none=True)
        if not fields:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY, "No patchable fields supplied."
            )
        entry = await service.store.update_entry(entry_id, **fields)
        if entry is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No entry {entry_id}.")
        await service.reload()
        return {
            "id": entry.id,
            "source_term": entry.source_term,
            "target_term": entry.target_term,
            "aliases": entry.aliases,
            "forbidden": entry.forbidden,
            "priority": entry.priority,
            "whole_word": entry.whole_word,
            "enabled": entry.enabled,
        }

    @router.delete("/entries/{entry_id}")
    async def delete_entry(entry_id: int):
        service = _service()
        if not await service.store.delete_entry(entry_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No entry {entry_id}.")
        await service.reload()
        return {"deleted": entry_id}

    @router.post("/reload")
    async def reload():
        service = _service()
        return {"version": await service.reload()}

    @router.post("/dry-run")
    async def dry_run(payload: DryRunIn = Body(...)):
        """Show what would fire on this text. No model call, no side effects."""
        service = _service()
        try:
            index = await service.resolve(payload.src_lang, payload.tgt_lang, payload.domain)
        except UnknownDomainError as error:
            # An endpoint whose whole purpose is catching mistakes before they
            # reach traffic must not answer a typo'd domain with a 500.
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                f"{error.reason} Available: {error.available}",
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
