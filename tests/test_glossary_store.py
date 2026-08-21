"""Persistence, scoping and the layering of a domain over the global termbase."""

import pytest
from sqlalchemy import func, select

from glossary.models import Entry
from glossary.store import GlossaryStore


@pytest.fixture
async def store():
    store = GlossaryStore("sqlite+aiosqlite:///:memory:")
    await store.create_all()
    yield store
    await store.aclose()


async def test_load_terms_returns_global_entries_when_no_domain_is_given(store):
    await store.create_entry(
        domain_name=None, src_lang="en", tgt_lang="fa", source_term="genome", target_term="X"
    )
    terms, version = await store.load_terms("en", "fa", None)
    assert [term.source_term for term in terms] == ["genome"]
    assert version >= 1


async def test_a_domain_layers_on_top_of_global(store):
    await store.create_domain("medical", "en", "fa", "Clinical terms")
    await store.create_entry(
        domain_name=None, src_lang="en", tgt_lang="fa", source_term="genome", target_term="G"
    )
    await store.create_entry(
        domain_name="medical", src_lang="en", tgt_lang="fa", source_term="lesion", target_term="L"
    )
    terms, _ = await store.load_terms("en", "fa", "medical")
    assert sorted(term.source_term for term in terms) == ["genome", "lesion"]


async def test_a_domain_entry_shadows_a_global_entry_for_the_same_term(store):
    await store.create_domain("medical", "en", "fa", None)
    await store.create_entry(
        domain_name=None, src_lang="en", tgt_lang="fa", source_term="culture", target_term="GLOBAL"
    )
    await store.create_entry(
        domain_name="medical", src_lang="en", tgt_lang="fa", source_term="culture", target_term="DOMAIN"
    )
    terms, _ = await store.load_terms("en", "fa", "medical")
    assert len(terms) == 1
    assert terms[0].target_term == "DOMAIN"


async def test_entries_of_another_language_pair_are_not_loaded(store):
    await store.create_entry(
        domain_name=None, src_lang="de", tgt_lang="fr", source_term="genom", target_term="X"
    )
    terms, _ = await store.load_terms("en", "fa", None)
    assert terms == []


async def test_disabled_entries_are_not_loaded(store):
    await store.create_entry(
        domain_name=None, src_lang="en", tgt_lang="fa", source_term="genome",
        target_term="X", enabled=False,
    )
    terms, _ = await store.load_terms("en", "fa", None)
    assert terms == []


async def test_duplicate_source_term_in_the_same_scope_is_rejected(store):
    await store.create_entry(
        domain_name=None, src_lang="en", tgt_lang="fa", source_term="genome", target_term="X"
    )
    with pytest.raises(ValueError, match="already exists"):
        await store.create_entry(
            domain_name=None, src_lang="en", tgt_lang="fa", source_term="genome", target_term="Y"
        )


async def test_creating_an_entry_in_an_unknown_domain_is_rejected(store):
    with pytest.raises(ValueError, match="No such domain"):
        await store.create_entry(
            domain_name="nope", src_lang="en", tgt_lang="fa", source_term="x", target_term="y"
        )


async def test_version_increases_on_every_write(store):
    _, first = await store.load_terms("en", "fa", None)
    await store.create_entry(
        domain_name=None, src_lang="en", tgt_lang="fa", source_term="genome", target_term="X"
    )
    _, second = await store.load_terms("en", "fa", None)
    assert second > first


async def test_deleting_a_domain_removes_its_entries(store):
    await store.create_domain("medical", "en", "fa", None)
    await store.create_entry(
        domain_name="medical", src_lang="en", tgt_lang="fa", source_term="lesion", target_term="L"
    )
    assert await store.delete_domain("medical") is True
    async with store._session() as session:
        count = await session.scalar(select(func.count()).select_from(Entry))
    assert count == 0


async def test_deleting_a_nonexistent_domain_returns_false(store):
    assert await store.delete_domain("nope") is False


async def test_deleting_an_entry_removes_it(store):
    entry = await store.create_entry(
        domain_name=None, src_lang="en", tgt_lang="fa", source_term="genome", target_term="X"
    )
    assert await store.delete_entry(entry.id) is True
    async with store._session() as session:
        count = await session.scalar(select(func.count()).select_from(Entry))
    assert count == 0


async def test_deleting_a_nonexistent_entry_returns_false(store):
    assert await store.delete_entry(999) is False


async def test_deleting_a_nonexistent_domain_does_not_bump_the_version(store):
    _, before = await store.load_terms("en", "fa", None)
    await store.delete_domain("nope")
    _, after = await store.load_terms("en", "fa", None)
    assert after == before


async def test_deleting_a_nonexistent_entry_does_not_bump_the_version(store):
    _, before = await store.load_terms("en", "fa", None)
    await store.delete_entry(999)
    _, after = await store.load_terms("en", "fa", None)
    assert after == before


async def test_entries_differing_only_by_case_sensitivity_both_survive(store):
    await store.create_entry(
        domain_name=None, src_lang="en", tgt_lang="fa", source_term="Genome",
        target_term="CS", case_sensitive=True,
    )
    await store.create_entry(
        domain_name=None, src_lang="en", tgt_lang="fa", source_term="Genome",
        target_term="CI", case_sensitive=False,
    )
    terms, _ = await store.load_terms("en", "fa", None)
    assert sorted(term.target_term for term in terms) == ["CI", "CS"]


async def test_domain_shadowing_still_holds_with_case_sensitivity_in_the_key(store):
    await store.create_domain("medical", "en", "fa", None)
    await store.create_entry(
        domain_name=None, src_lang="en", tgt_lang="fa", source_term="culture", target_term="GLOBAL"
    )
    await store.create_entry(
        domain_name="medical", src_lang="en", tgt_lang="fa", source_term="culture", target_term="DOMAIN"
    )
    terms, _ = await store.load_terms("en", "fa", "medical")
    assert len(terms) == 1
    assert terms[0].target_term == "DOMAIN"
