"""Admin CRUD, its authorization boundary, and the dry-run endpoint."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

import main as main_module
from config import get_settings
from glossary.router import build_router
from glossary.service import GlossaryService
from translator import TranslationEngine

KEY = "test-key"
HEADERS = {"X-Admin-Key": KEY}


@pytest.fixture
async def client():
    service = GlossaryService(database_url="sqlite+aiosqlite:///:memory:")
    await service.start()
    app = FastAPI()
    app.include_router(build_router(service, KEY))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        client.service = service
        yield client
    await service.aclose()


async def test_requests_without_the_key_are_rejected(client):
    response = await client.get("/admin/glossary/entries")
    assert response.status_code == 401


async def test_requests_with_a_wrong_key_are_rejected(client):
    response = await client.get("/admin/glossary/entries", headers={"X-Admin-Key": "wrong"})
    assert response.status_code == 401


async def test_a_non_ascii_key_is_rejected_rather_than_crashing(client):
    # HTTP headers are latin-1 decoded, so a byte above 127 reaches the
    # dependency as a non-ASCII str. secrets.compare_digest raises TypeError
    # on that input; the auth boundary must answer 401, not 500. httpx itself
    # requires header *values* to be ASCII (or bytes) client-side, so the
    # non-ASCII byte is sent as a raw bytes value -- Starlette still decodes
    # incoming header bytes as latin-1 into a non-ASCII str on the server side,
    # which is what actually reaches the dependency under test.
    response = await client.get(
        "/admin/glossary/entries", headers={"X-Admin-Key": "wrong-\xff".encode("latin-1")}
    )
    assert response.status_code == 401


async def test_create_and_list_an_entry(client):
    created = await client.post(
        "/admin/glossary/entries",
        headers=HEADERS,
        json={
            "src_lang": "en", "tgt_lang": "fa",
            "source_term": "genome", "target_term": "ژنوم",
        },
    )
    assert created.status_code == 201
    listed = await client.get("/admin/glossary/entries", headers=HEADERS)
    assert [item["source_term"] for item in listed.json()] == ["genome"]


async def test_a_stopword_entry_is_refused(client):
    response = await client.post(
        "/admin/glossary/entries",
        headers=HEADERS,
        json={"src_lang": "en", "tgt_lang": "fa", "source_term": "the", "target_term": "X"},
    )
    assert response.status_code == 422
    assert "stopword" in response.text


async def test_a_multi_word_phrase_containing_a_stopword_is_allowed(client):
    response = await client.post(
        "/admin/glossary/entries",
        headers=HEADERS,
        json={
            "src_lang": "en", "tgt_lang": "fa",
            "source_term": "the cloud", "target_term": "X",
        },
    )
    assert response.status_code == 201


async def test_exact_mode_is_refused_until_phase_two(client):
    response = await client.post(
        "/admin/glossary/entries",
        headers=HEADERS,
        json={
            "src_lang": "en", "tgt_lang": "fa", "source_term": "wordomatic",
            "target_term": "wordomatic", "target_mode": "exact",
        },
    )
    assert response.status_code == 422


async def test_duplicate_entries_conflict(client):
    payload = {
        "src_lang": "en", "tgt_lang": "fa", "source_term": "genome", "target_term": "X",
    }
    assert (await client.post("/admin/glossary/entries", headers=HEADERS, json=payload)).status_code == 201
    duplicate = await client.post("/admin/glossary/entries", headers=HEADERS, json=payload)
    assert duplicate.status_code == 409


async def test_a_duplicate_source_term_matching_the_domain_error_text_is_still_a_conflict(client):
    # store.create_entry's duplicate message embeds the caller-supplied
    # source_term verbatim: "An entry for 'No such domain' already exists in
    # this scope." Branching on substrings of that message would misread this
    # as an unknown-domain 404 instead of the duplicate it actually is.
    payload = {
        "src_lang": "en", "tgt_lang": "fa",
        "source_term": "No such domain", "target_term": "X",
    }
    first = await client.post("/admin/glossary/entries", headers=HEADERS, json=payload)
    assert first.status_code == 201
    duplicate = await client.post("/admin/glossary/entries", headers=HEADERS, json=payload)
    assert duplicate.status_code == 409


async def test_dry_run_reports_matches_without_translating(client):
    await client.post(
        "/admin/glossary/entries",
        headers=HEADERS,
        json={
            "src_lang": "en", "tgt_lang": "fa",
            "source_term": "genome", "target_term": "ژنوم",
        },
    )
    response = await client.post(
        "/admin/glossary/dry-run",
        headers=HEADERS,
        json={"text": "The genome sequence.", "src_lang": "en", "tgt_lang": "fa"},
    )
    assert response.status_code == 200
    matches = response.json()["matches"]
    assert matches[0]["source_term"] == "genome"
    assert matches[0]["start"] == 4


async def test_creating_a_domain_and_listing_it(client):
    created = await client.post(
        "/admin/glossary/domains",
        headers=HEADERS,
        json={"name": "medical", "src_lang": "en", "tgt_lang": "fa"},
    )
    assert created.status_code == 201
    listed = await client.get("/admin/glossary/domains", headers=HEADERS)
    assert [item["name"] for item in listed.json()] == ["medical"]


# ---------------------------------------------------------------------------
# The tests above build a standalone FastAPI() and never touch main.lifespan,
# so they cannot catch the router failing to be mounted in the real app (a
# dropped import, a mis-guarded `if`, a typo'd settings attribute). This test
# drives main.app through its actual startup path -- the only place that
# wiring is exercised at all.
# ---------------------------------------------------------------------------

LIFESPAN_KEY = "lifespan-test-key"


class _NoLoadTranslationEngine(TranslationEngine):
    """A TranslationEngine that skips the real load().

    lifespan() constructs `TranslationEngine(settings)` by name and runs
    `engine.load()` in a worker thread; the real implementation pulls a live
    tokenizer via load_processor(...), which this test must not depend on
    (network/local checkpoint). Swapping the class main.py references lets
    lifespan's own code -- engine construction, glossary startup, and the
    admin router mount -- run completely unmodified while only the expensive
    tokenizer load is stubbed out. Follows the same "subclass TranslationEngine,
    stub what touches the outside world" approach as StubEngine in
    tests/test_glossary_http.py.
    """

    def load(self):
        pass

    @property
    def is_loaded(self) -> bool:
        return True


def test_the_admin_router_is_actually_mounted_by_the_real_lifespan(monkeypatch):
    """Drives main.app's real startup, not a hand-built FastAPI()."""
    monkeypatch.setenv("TG_GLOSSARY_ENABLED", "true")
    monkeypatch.setenv("TG_ADMIN_API_KEY", LIFESPAN_KEY)
    monkeypatch.setenv("TG_GLOSSARY_DB_URL", "sqlite+aiosqlite:///:memory:")
    get_settings.cache_clear()
    monkeypatch.setattr(main_module, "TranslationEngine", _NoLoadTranslationEngine)
    try:
        with TestClient(main_module.app) as test_client:
            unauthorized = test_client.get("/admin/glossary/domains")
            assert unauthorized.status_code == 401

            authorized = test_client.get(
                "/admin/glossary/domains", headers={"X-Admin-Key": LIFESPAN_KEY}
            )
            assert authorized.status_code == 200
    finally:
        get_settings.cache_clear()
