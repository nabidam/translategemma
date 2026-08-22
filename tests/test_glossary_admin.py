"""Admin CRUD, its authorization boundary, and the dry-run endpoint."""

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from glossary.router import build_router
from glossary.service import GlossaryService

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
