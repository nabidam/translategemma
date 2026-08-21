"""Snapshot caching, domain resolution, and the unknown-domain policy."""

import pytest

from glossary.service import GlossaryService, UnknownDomainError


def build(**overrides):
    options = {
        "database_url": "sqlite+aiosqlite:///:memory:",
        "default_domain": None,
        "unknown_domain": "reject",
    }
    options.update(overrides)
    return GlossaryService(**options)


@pytest.fixture
async def service():
    service = build()
    await service.start()
    yield service
    await service.aclose()


async def test_omitting_a_domain_resolves_the_global_layer(service):
    await service.store.create_entry(
        domain_name=None, src_lang="en", tgt_lang="fa", source_term="genome", target_term="X"
    )
    await service.reload()
    index = await service.resolve("en", "fa", None)
    assert [term.source_term for term in index.terms] == ["genome"]


async def test_an_unknown_domain_is_rejected_and_lists_what_exists(service):
    await service.store.create_domain("medical", "en", "fa", None)
    with pytest.raises(UnknownDomainError) as error:
        await service.resolve("en", "fa", "medcial")
    assert error.value.available == ["medical"]


async def test_fallback_policy_uses_the_global_layer_for_an_unknown_domain():
    service = build(unknown_domain="fallback")
    await service.start()
    try:
        index = await service.resolve("en", "fa", "nope")
        assert index.terms == ()
    finally:
        await service.aclose()


async def test_the_default_domain_applies_when_the_request_omits_one():
    service = build(default_domain="medical")
    await service.start()
    try:
        await service.store.create_domain("medical", "en", "fa", None)
        await service.store.create_entry(
            domain_name="medical", src_lang="en", tgt_lang="fa",
            source_term="lesion", target_term="L",
        )
        await service.reload()
        index = await service.resolve("en", "fa", None)
        assert [term.source_term for term in index.terms] == ["lesion"]
    finally:
        await service.aclose()


async def test_snapshots_are_cached_and_reload_replaces_them(service):
    first = await service.resolve("en", "fa", None)
    assert await service.resolve("en", "fa", None) is first
    await service.store.create_entry(
        domain_name=None, src_lang="en", tgt_lang="fa", source_term="genome", target_term="X"
    )
    await service.reload()
    assert await service.resolve("en", "fa", None) is not first


async def test_plan_and_apply_round_trip(service):
    await service.store.create_entry(
        domain_name=None, src_lang="en", tgt_lang="fa",
        source_term="genome", target_term="ژنوم",
        aliases=["گنوم"],
    )
    await service.reload()
    index = await service.resolve("en", "fa", None)
    spans = service.plan(index, "The genome sequence.")
    assert len(spans) == 1
    result, report = service.apply("توالی گنوم.", spans)
    assert "ژنوم" in result
    assert report.applied[0].count == 1
