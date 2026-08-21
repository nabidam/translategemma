"""Termbase tables.

A domain is a row rather than a free string on the entry: it gives a request's
`domain` a stable identity, an existence check that turns a caller's typo into
a 404 instead of a silently general translation, and somewhere to hang a
per-termbase version.

Global entries are `domain_id IS NULL` rather than a reserved row, so "the
global layer always applies" is a property of the query and cannot be switched
off or deleted by an admin action.
"""

from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Domain(Base):
    __tablename__ = "glossary_domain"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    src_lang: Mapped[str] = mapped_column(String(16))
    tgt_lang: Mapped[str] = mapped_column(String(16))
    description: Mapped[str | None] = mapped_column(String(512), default=None)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )


class Entry(Base):
    __tablename__ = "glossary_entry"
    __table_args__ = (
        UniqueConstraint(
            "domain_id",
            "src_lang",
            "tgt_lang",
            "source_term",
            "case_sensitive",
            name="uq_glossary_entry_scope",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    domain_id: Mapped[int | None] = mapped_column(
        ForeignKey("glossary_domain.id", ondelete="CASCADE"), default=None, index=True
    )
    src_lang: Mapped[str] = mapped_column(String(16), index=True)
    tgt_lang: Mapped[str] = mapped_column(String(16), index=True)
    source_term: Mapped[str] = mapped_column(String(512))
    target_term: Mapped[str] = mapped_column(String(512))
    # Only 'preferred' is accepted until sentinel protection ships. The column
    # exists now so that phase is additive rather than a migration.
    target_mode: Mapped[str] = mapped_column(String(16), default="preferred")
    aliases: Mapped[list] = mapped_column(JSON, default=list)
    forbidden: Mapped[list] = mapped_column(JSON, default=list)
    case_sensitive: Mapped[bool] = mapped_column(Boolean, default=False)
    whole_word: Mapped[bool] = mapped_column(Boolean, default=True)
    priority: Mapped[int] = mapped_column(Integer, default=0)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    notes: Mapped[str | None] = mapped_column(String(1024), default=None)
    created_by: Mapped[str | None] = mapped_column(String(128), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )


class Revision(Base):
    """A single monotonic counter, bumped on every write.

    A request stamps its response with the value it read, so a served
    translation can be attributed to the exact termbase that produced it.
    """

    __tablename__ = "glossary_revision"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
