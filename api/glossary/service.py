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
    # (src_lang, tgt_lang, domain) is caller-controlled free text on an
    # unauthenticated endpoint, so the snapshot cache is bounded rather than
    # left to grow with every novel combination a caller sends. One gateway
    # in front of a handful of language pairs and domains needs far fewer
    # than this; it exists to cap the cost of an adversarial or buggy caller,
    # not to size a real workload.
    _MAX_SNAPSHOTS = 32

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
        if self._default_domain is not None:
            domain = await self.store.get_domain(self._default_domain)
            if domain is None or not domain.enabled:
                available = [row.name for row in await self.store.list_domains() if row.enabled]
                raise ValueError(
                    f"TG_GLOSSARY_DEFAULT_DOMAIN={self._default_domain!r} does not name an "
                    f"enabled glossary domain. Available: {available}. Every caller that omits "
                    "`domain` resolves through this setting, so a typo here breaks the normal "
                    "path for every ordinary translation, not just requests that name a domain."
                )
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
        # Lowercased, not the strict shape check admin writes get: a
        # translation request's source_lang/target_lang have already passed
        # through the rest of the gateway by the time they reach here, and
        # this call must not be the first place a caller sees a 422 for a
        # code that every other endpoint already accepted. Entries are
        # written lowercase (see glossary/router.py), so this is what makes
        # "EN" in a request still match an entry stored under "en".
        src_lang = src_lang.strip().lower()
        tgt_lang = tgt_lang.strip().lower()
        name = domain if domain is not None else self._default_domain
        key = (src_lang, tgt_lang, name)
        cached = self._snapshots.get(key)
        if cached is not None:
            # Touch: move `key` to the most-recently-used position (dicts
            # preserve insertion order) so it survives eviction while it is
            # still being used. Rebuilt as a new dict rather than reordered
            # in place -- same rebind-not-mutate discipline as a write below,
            # so a concurrent reader iterating the old dict never observes a
            # half-updated one, and a caller already holding this Index is
            # unaffected either way: eviction only ever drops it from the
            # cache, never mutates the Index itself.
            self._snapshots = {
                **{k: v for k, v in self._snapshots.items() if k != key},
                key: cached,
            }
            return cached

        try:
            terms, version = await self.store.load_terms(src_lang, tgt_lang, name)
        except ValueError:
            if self._unknown_domain == "fallback":
                terms, version = await self.store.load_terms(src_lang, tgt_lang, None)
            else:
                available = [
                    row.name for row in await self.store.list_domains() if row.enabled
                ]
                raise UnknownDomainError(name or "", available) from None

        index = build_index(terms, version)
        # Rebind rather than mutate: a concurrent reader either sees the old
        # dictionary or the new one, never a half-updated one.
        updated = {**self._snapshots, key: index}
        if len(updated) > self._MAX_SNAPSHOTS:
            # Evict the single least-recently-used entry (dict order: oldest
            # first, since every hit above re-inserts at the end). A caller
            # already holding the evicted Index keeps using it -- eviction
            # only removes it from this cache, it never touches the object.
            oldest_key = next(iter(updated))
            updated = {k: v for k, v in updated.items() if k != oldest_key}
        self._snapshots = updated
        # `self._version` is intentionally NOT set here. Two concurrent
        # resolve() calls for different language pairs can race, and
        # whichever reads the older revision last would otherwise move the
        # counter backwards. The revision is a single global counter and
        # every admin write path calls reload(), which reads it directly
        # from the store -- so resolve() has nothing correct to add here.
        return index

    def plan(self, index: Index, text: str) -> list[Span]:
        return find_spans(index, text)

    def apply(self, translation: str, spans: Sequence[Span]) -> tuple[str, Report]:
        return apply_preferred(translation, spans)
