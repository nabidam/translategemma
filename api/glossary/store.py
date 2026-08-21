"""Async persistence for the termbase.

Nothing here runs during a translation: `load_terms` is called when a snapshot
is built, and the snapshot answers requests. Querying the database per term per
request is the thing this design exists to avoid.
"""

from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from .matcher import Term
from .models import Base, Domain, Entry, Revision


class GlossaryStore:
    def __init__(self, database_url: str):
        self._engine = create_async_engine(database_url, future=True)
        self._session = async_sessionmaker(self._engine, expire_on_commit=False)

    async def create_all(self) -> None:
        async with self._engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with self._session() as session:
            existing = await session.scalar(select(Revision).limit(1))
            if existing is None:
                session.add(Revision(id=1, version=1))
                await session.commit()

    async def aclose(self) -> None:
        await self._engine.dispose()

    # ---------------------------------------------------------------- version

    async def _bump(self, session: AsyncSession) -> None:
        await session.execute(update(Revision).values(version=Revision.version + 1))

    async def version(self) -> int:
        async with self._session() as session:
            return await session.scalar(select(Revision.version).limit(1)) or 1

    # ---------------------------------------------------------------- domains

    async def list_domains(self) -> list[Domain]:
        async with self._session() as session:
            result = await session.scalars(select(Domain).order_by(Domain.name))
            return list(result)

    async def get_domain(self, name: str) -> Domain | None:
        async with self._session() as session:
            return await session.scalar(select(Domain).where(Domain.name == name))

    async def create_domain(
        self, name: str, src_lang: str, tgt_lang: str, description: str | None = None
    ) -> Domain:
        async with self._session() as session:
            domain = Domain(
                name=name, src_lang=src_lang, tgt_lang=tgt_lang, description=description
            )
            session.add(domain)
            try:
                await self._bump(session)
                await session.commit()
            except IntegrityError as error:
                await session.rollback()
                raise ValueError(f"Domain {name!r} already exists.") from error
            return domain

    async def delete_domain(self, name: str) -> bool:
        async with self._session() as session:
            result = await session.execute(delete(Domain).where(Domain.name == name))
            await self._bump(session)
            await session.commit()
            return result.rowcount > 0

    # ---------------------------------------------------------------- entries

    async def list_entries(self, domain_name: str | None = None) -> list[Entry]:
        async with self._session() as session:
            statement = select(Entry).order_by(Entry.source_term)
            if domain_name is not None:
                domain = await session.scalar(select(Domain).where(Domain.name == domain_name))
                if domain is None:
                    raise ValueError(f"No such domain: {domain_name!r}")
                statement = statement.where(Entry.domain_id == domain.id)
            result = await session.scalars(statement)
            return list(result)

    async def create_entry(self, domain_name: str | None = None, **fields) -> Entry:
        async with self._session() as session:
            domain_id = None
            if domain_name is not None:
                domain = await session.scalar(select(Domain).where(Domain.name == domain_name))
                if domain is None:
                    raise ValueError(f"No such domain: {domain_name!r}")
                domain_id = domain.id

            # The unique constraint alone cannot catch this: SQL treats every
            # NULL domain_id as distinct from every other, so two global
            # entries for the same term would otherwise pass silently.
            domain_filter = (
                Entry.domain_id.is_(None) if domain_id is None else Entry.domain_id == domain_id
            )
            conflict = await session.scalar(
                select(Entry).where(
                    domain_filter,
                    Entry.src_lang == fields.get("src_lang"),
                    Entry.tgt_lang == fields.get("tgt_lang"),
                    Entry.source_term == fields.get("source_term"),
                    Entry.case_sensitive == fields.get("case_sensitive", False),
                )
            )
            if conflict is not None:
                raise ValueError(
                    f"An entry for {fields.get('source_term')!r} already exists in this scope."
                )

            entry = Entry(domain_id=domain_id, **fields)
            session.add(entry)
            try:
                await self._bump(session)
                await session.commit()
            except IntegrityError as error:
                await session.rollback()
                raise ValueError(
                    f"An entry for {fields.get('source_term')!r} already exists in this scope."
                ) from error
            return entry

    async def delete_entry(self, entry_id: int) -> bool:
        async with self._session() as session:
            result = await session.execute(delete(Entry).where(Entry.id == entry_id))
            await self._bump(session)
            await session.commit()
            return result.rowcount > 0

    # ----------------------------------------------------------------- terms

    async def load_terms(
        self, src_lang: str, tgt_lang: str, domain_name: str | None
    ) -> tuple[list[Term], int]:
        """Global entries, with a domain's entries layered over them.

        A domain entry shadows a global entry with the same source term: the
        more specific termbase wins, and the global one is not also applied.
        """
        async with self._session() as session:
            version = await session.scalar(select(Revision.version).limit(1)) or 1

            domain_id = None
            if domain_name is not None:
                domain = await session.scalar(
                    select(Domain).where(Domain.name == domain_name, Domain.enabled.is_(True))
                )
                if domain is None:
                    raise ValueError(f"No such domain: {domain_name!r}")
                domain_id = domain.id

            statement = select(Entry).where(
                Entry.src_lang == src_lang,
                Entry.tgt_lang == tgt_lang,
                Entry.enabled.is_(True),
            )
            rows = list(await session.scalars(statement))

        by_source: dict[str, Entry] = {}
        for entry in rows:
            if entry.domain_id is not None and entry.domain_id != domain_id:
                continue
            key = entry.source_term.lower()
            existing = by_source.get(key)
            # A domain row replaces a global row for the same term.
            if existing is None or (existing.domain_id is None and entry.domain_id is not None):
                by_source[key] = entry

        terms = [
            Term(
                entry_id=entry.id,
                source_term=entry.source_term,
                target_term=entry.target_term,
                target_mode=entry.target_mode,
                aliases=tuple(entry.aliases or ()),
                forbidden=tuple(entry.forbidden or ()),
                case_sensitive=entry.case_sensitive,
                whole_word=entry.whole_word,
                priority=entry.priority,
            )
            for entry in by_source.values()
        ]
        return terms, version
