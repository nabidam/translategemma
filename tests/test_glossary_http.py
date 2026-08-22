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

    def __init__(self):
        super().__init__(FakeSettings())

    @property
    def is_loaded(self):
        return True

    async def _generate(self, segments, system, source_lang, target_lang, max_new_tokens):
        return ["stub"] * len(segments)


@pytest.fixture
async def client():
    """The real app, wired to a stub engine and a real, empty glossary."""
    app = main_module.app
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
