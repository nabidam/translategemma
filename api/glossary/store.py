"""Async persistence for the termbase.

Nothing here runs during a translation: `load_terms` is called when a snapshot
is built, and the snapshot answers requests. Querying the database per term per
request is the thing this design exists to avoid.
"""

from pathlib import Path

from sqlalchemy import delete, event, select, update
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from .matcher import Term
from .models import Base, Domain, Entry, Revision


class GlossaryStore:
    def __init__(self, database_url: str):
        self._database_url = database_url
        self._engine = create_async_engine(database_url, future=True)
        self._session = async_sessionmaker(self._engine, expire_on_commit=False)

        # SQLite disables foreign-key enforcement per connection unless told
        # otherwise, and there's no ORM-level cascade defined to substitute --
        # without this, Entry.domain_id's ondelete="CASCADE" is inert and
        # delete_domain orphans its entries. Guarded to SQLite only: this
        # store is written against a repository interface precisely so
        # PostgreSQL can replace it, and that dialect must not receive a
        # SQLite pragma.
        if self._engine.sync_engine.dialect.name == "sqlite":

            @event.listens_for(self._engine.sync_engine, "connect")
            def _enable_sqlite_fk(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

    async def create_all(self) -> None:
        self._ensure_sqlite_directory()
        async with self._engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with self._session() as session:
            existing = await session.scalar(select(Revision).limit(1))
            if existing is None:
                session.add(Revision(id=1, version=1))
                await session.commit()

    def _ensure_sqlite_directory(self) -> None:
        """SQLite opens a file but will not create the directory it lives in.

        The default TG_GLOSSARY_DB_URL (./data/glossary.db) and the compose
        deployment's mounted volume (/data/glossary.db) both name a directory
        nothing else in this process creates, so without this an operator who
        only sets TG_GLOSSARY_ENABLED sees an opaque "unable to open database
        file" instead of a working default. Guarded to SQLite, like the FK
        pragma above: a file path is meaningless for another dialect.
        """
        if self._engine.sync_engine.dialect.name != "sqlite":
            return
        database = make_url(self._database_url).database
        if not database or database == ":memory:":
            return
        Path(database).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

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
            deleted = result.rowcount > 0
            if deleted:
                # A miss shouldn't invalidate every cached snapshot for nothing.
                await self._bump(session)
            await session.commit()
            return deleted

    async def update_domain(self, name: str, **fields) -> Domain | None:
        """Patch a domain in place. Returns None when there is no such domain.

        Only `description` and `enabled` are patchable. The name is the selector
        callers send and part of the identity a snapshot is keyed by; renaming
        would silently break every caller naming it, so a rename is a delete
        plus a create, deliberately.
        """
        async with self._session() as session:
            domain = await session.scalar(select(Domain).where(Domain.name == name))
            if domain is None:
                return None
            for key, value in fields.items():
                setattr(domain, key, value)
            await self._bump(session)
            await session.commit()
            return domain


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
            # entries for the same term would otherwise pass silently. This
            # check is still racy for the domain_id IS NULL scope specifically
            # -- the DB constraint can never back it up there -- which is
            # acceptable only because the deployment is one gateway with one
            # administrator (see requirements.txt).
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
            deleted = result.rowcount > 0
            if deleted:
                # A miss shouldn't invalidate every cached snapshot for nothing.
                await self._bump(session)
            await session.commit()
            return deleted

    async def update_entry(self, entry_id: int, **fields) -> Entry | None:
        """Patch an entry in place. Returns None when there is no such entry.

        Identity is not patchable: `source_term`, the language pair, the domain
        and `case_sensitive` together form the unique key, and changing any of
        them makes it a different entry. Preserving the id is the whole point --
        it is what an administrator is tracking in the misses report while
        curating aliases, and delete-plus-create loses it.
        """
        async with self._session() as session:
            entry = await session.scalar(select(Entry).where(Entry.id == entry_id))
            if entry is None:
                return None
            for key, value in fields.items():
                setattr(entry, key, value)
            await self._bump(session)
            await session.commit()
            return entry


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
                # Looked up without the enabled filter so a disabled domain can
                # be reported as disabled rather than as nonexistent. They are
                # equally unusable, but an administrator who just turned one off
                # should not be told it does not exist.
                domain = await session.scalar(
                    select(Domain).where(Domain.name == domain_name)
                )
                if domain is not None and not domain.enabled:
                    raise ValueError(f"Domain {domain_name!r} is disabled.")
                if domain is None:
                    raise ValueError(f"No such domain: {domain_name!r}")
                domain_id = domain.id

            statement = (
                select(Entry)
                .where(
                    Entry.src_lang == src_lang,
                    Entry.tgt_lang == tgt_lang,
                    Entry.enabled.is_(True),
                )
                # Deterministic order: if two rows still collide on the merge
                # key below, the winner must not depend on SQLite's whim.
                .order_by(Entry.id)
            )
            rows = list(await session.scalars(statement))

        by_source: dict[tuple[str, bool], Entry] = {}
        for entry in rows:
            if entry.domain_id is not None and entry.domain_id != domain_id:
                continue
            # The unique constraint deliberately lets a case-sensitive and a
            # case-insensitive row for the same spelling coexist in one scope
            # (e.g. "Genome" case-sensitive alongside "genome" general) -- the
            # merge key must include case_sensitive or one silently vanishes.
            # A case-sensitive entry keeps its exact spelling in the key:
            # lowercasing it would merge "Genome" and "GENOME", which is
            # exactly what case-sensitivity exists to keep apart.
            spelling = entry.source_term if entry.case_sensitive else entry.source_term.lower()
            key = (spelling, entry.case_sensitive)
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
