"""Persistence, scoping and the layering of a domain over the global termbase."""

import pytest

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
