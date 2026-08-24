"""HTTP-layer coverage for the unknown-domain policy.

`_resolve_index` converting `UnknownDomainError` into an HTTP 404 is a binding
rule of the design ("an unknown domain is a 404, not a silent fallback"), but
before this file it was only verified one layer down, at
`GlossaryService.resolve` (see `test_glossary_service.py`). Nothing exercised
the conversion into the actual HTTP response the `/translate` endpoint
returns, which is what a real caller sees. This drives the real FastAPI route
through `httpx.AsyncClient` + `ASGITransport`, with a stub engine (no vLLM)
and a real `GlossaryService` on an in-memory sqlite database, so the endpoint
code under test -- `_resolve_index`, `get_glossary`, the route body -- runs
unmodified.
"""

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.exc import OperationalError

import main as main_module
from config import System
from glossary.service import GlossaryService
from translator import TranslationEngine


class FakeSettings:
    """Only what TranslationEngine touches; no vLLM, no tokenizer."""

    max_concurrent_requests = 4
    batch_size = 8
    split_sentences = False
    served_system = System.ADAPTER


class StubEngine(TranslationEngine):
    """Answers every request with a fixed string; no vLLM involved."""

    def __init__(self, output="stub"):
        super().__init__(FakeSettings())
        self._output = output

    @property
    def is_loaded(self):
        return True

    async def _generate(self, segments, system, source_lang, target_lang, max_new_tokens):
        return [self._output] * len(segments)


@pytest.fixture
async def client():
    """The real app, wired to a stub engine and a real, empty glossary."""
    # An app of this test's own, so nothing it does can leak into another.
    app = main_module.create_app()
    glossary = GlossaryService(
        database_url="sqlite+aiosqlite:///:memory:",
        default_domain=None,
        unknown_domain="reject",
    )
    await glossary.start()
    await glossary.store.create_domain("medical", "en", "fa", None)

    app.state.engine = StubEngine()
    app.state.glossary = glossary
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as http_client:
            http_client.glossary = glossary
            # The app this client is bound to, so a test can swap the stub
            # engine without reaching for a module-level singleton.
            http_client.app = app
            yield http_client
    finally:
        await glossary.aclose()
        app.state.engine = None
        app.state.glossary = None


async def test_an_unknown_domain_on_translate_is_a_404_naming_what_exists(client):
    response = await client.post(
        "/translate",
        json={"text": "The genome.", "source_lang": "en", "target_lang": "fa", "domain": "medcial"},
    )
    assert response.status_code == 404
    body = response.json()
    assert "medcial" in body["detail"]
    assert "medical" in body["detail"]


async def test_a_known_domain_on_translate_succeeds(client):
    response = await client.post(
        "/translate",
        json={"text": "The genome.", "source_lang": "en", "target_lang": "fa", "domain": "medical"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["translation"] == "stub"
    # The glossary ran (empty termbase, so a no-op report) -- proof the 404
    # path above is a real branch and not just "the glossary never engages".
    assert body["glossary"]["status"] == "no_match"


async def test_an_unknown_domain_on_translate_batch_is_also_a_404(client):
    response = await client.post(
        "/translate/batch",
        json={"texts": ["The genome."], "source_lang": "en", "target_lang": "fa", "domain": "medcial"},
    )
    assert response.status_code == 404
    assert "medical" in response.json()["detail"]


async def test_translate_batch_populates_raw_translations_when_the_glossary_runs(client):
    response = await client.post(
        "/translate/batch",
        json={"texts": ["The genome.", "Another."], "source_lang": "en", "target_lang": "fa", "domain": "medical"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["raw_translations"] == ["stub", "stub"]
    assert body["glossary_version"] is not None
    assert len(body["glossary"]) == 2


async def test_an_invalid_terminology_mode_is_a_422(client):
    response = await client.post(
        "/translate",
        json={"text": "The genome.", "domain": "medical", "terminology_mode": "OFF"},
    )
    assert response.status_code == 422


async def test_terminology_mode_off_disables_matching_on_a_populated_termbase(client):
    # Before this test nothing in the suite ever drove the `mode == "off"`
    # branch in main._resolve_index with an actual term to *not* apply --
    # test_request_fields_are_accepted_when_the_feature_is_off in
    # test_glossary_disabled.py covers the field's acceptance with the
    # feature off entirely, not this branch with it on.
    await client.glossary.store.create_entry(
        domain_name=None, src_lang="en", tgt_lang="fa",
        source_term="genome", target_term="ژنوم", aliases=["گنوم"],
    )
    await client.glossary.reload()
    client.app.state.engine = StubEngine(output="گنوم است.")
    try:
        response = await client.post(
            "/translate",
            json={"text": "The genome.", "source_lang": "en", "target_lang": "fa",
                  "terminology_mode": "off"},
        )
    finally:
        client.app.state.engine = StubEngine()
    assert response.status_code == 200
    body = response.json()
    assert body["translation"] == "گنوم است."
    for key in ("raw_translation", "glossary_version", "glossary"):
        assert key not in body


async def test_a_glossary_store_failure_degrades_to_plain_translation(client, monkeypatch):
    # An unmounted volume or an unreadable database file must not take
    # translation down while the model is perfectly healthy -- the glossary's
    # whole premise is that it is always one restart away from being
    # harmless. UnknownDomainError (a caller error) must still 404, which
    # test_an_unknown_domain_on_translate_is_a_404_naming_what_exists above
    # covers; this is the store-level failure, which must degrade instead.
    async def _boom(*args, **kwargs):
        raise OperationalError("SELECT 1", {}, Exception("disk I/O error"))

    monkeypatch.setattr(client.glossary.store, "load_terms", _boom)
    response = await client.post(
        "/translate",
        json={"text": "The genome.", "source_lang": "en", "target_lang": "fa"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["translation"] == "stub"
    for key in ("raw_translation", "glossary_version", "glossary"):
        assert key not in body


async def test_end_to_end_translate_rewrites_output_with_a_populated_termbase(client):
    # The full chain -- store -> service -> index -> translator ->
    # serializer -> JSON -- with a term that actually rewrites. Every other
    # HTTP test in this file exercises an empty termbase, so the report is
    # always `no_match` and the rewrite path itself is never driven end to
    # end through the real app.
    await client.glossary.store.create_entry(
        domain_name=None, src_lang="en", tgt_lang="fa",
        source_term="genome", target_term="ژنوم", aliases=["گنوم"],
    )
    await client.glossary.reload()
    client.app.state.engine = StubEngine(output="گنوم است.")
    try:
        response = await client.post(
            "/translate",
            json={"text": "The genome.", "source_lang": "en", "target_lang": "fa"},
        )
    finally:
        client.app.state.engine = StubEngine()
    assert response.status_code == 200
    body = response.json()
    assert body["translation"] != body["raw_translation"]
    assert body["raw_translation"] == "گنوم است."
    assert "ژنوم" in body["translation"]
    assert body["glossary"]["applied"]
    assert body["glossary"]["applied"][0]["source_term"] == "genome"
