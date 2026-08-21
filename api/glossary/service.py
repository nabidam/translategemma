"""The glossary's public face: one object the request path talks to.

Snapshots are built per (src_lang, tgt_lang, domain) and cached. They are
immutable, and a reload replaces the cache dictionary wholesale rather than
mutating it, so a request that already holds a snapshot finishes against the
termbase it started with and readers need no lock.
"""

from collections.abc import Sequence

from .apply import Report, apply_preferred
from .matcher import Index, Span, build_index, find_spans
from .store import GlossaryStore


class UnknownDomainError(Exception):
    """A request named a domain that does not exist.

    Carries the valid names so the API can say what the caller could have
    meant. A typo silently answered from the global termbase is the failure
    this whole layer exists to prevent.
    """

    def __init__(self, name: str, available: list[str]):
        super().__init__(f"No such glossary domain: {name!r}. Available: {available}")
        self.name = name
        self.available = available


class GlossaryService:
    def __init__(
        self,
        database_url: str,
        default_domain: str | None = None,
        unknown_domain: str = "reject",
    ):
        self.store = GlossaryStore(database_url)
        self._default_domain = default_domain
        self._unknown_domain = unknown_domain
        self._snapshots: dict[tuple[str, str, str | None], Index] = {}
        self._version = 1

    async def start(self) -> None:
        await self.store.create_all()
        self._version = await self.store.version()

    async def aclose(self) -> None:
        await self.store.aclose()

    @property
    def version(self) -> int:
        return self._version

    async def reload(self) -> int:
        """Drop every cached snapshot. The next request rebuilds what it needs."""
        self._snapshots = {}
        self._version = await self.store.version()
        return self._version

    async def resolve(self, src_lang: str, tgt_lang: str, domain: str | None) -> Index:
        """The snapshot for this request, building and caching it on first use."""
        name = domain if domain is not None else self._default_domain
        key = (src_lang, tgt_lang, name)
        cached = self._snapshots.get(key)
        if cached is not None:
            return cached

        try:
            terms, version = await self.store.load_terms(src_lang, tgt_lang, name)
        except ValueError:
            if self._unknown_domain == "fallback":
                terms, version = await self.store.load_terms(src_lang, tgt_lang, None)
            else:
                available = [row.name for row in await self.store.list_domains()]
                raise UnknownDomainError(name or "", available) from None

        index = build_index(terms, version)
        # Rebind rather than mutate: a concurrent reader either sees the old
        # dictionary or the new one, never a half-updated one.
        self._snapshots = {**self._snapshots, key: index}
        self._version = version
        return index

    def plan(self, index: Index, text: str) -> list[Span]:
        return find_spans(index, text)

    def apply(self, translation: str, spans: Sequence[Span]) -> tuple[str, Report]:
        return apply_preferred(translation, spans)
