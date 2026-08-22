"""With TG_GLOSSARY_ENABLED=false, the feature must be absent, not merely bypassed."""

import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import main as main_module
from config import get_settings
from glossary.matcher import build_index
from translator import TranslationEngine

SYSTEM = "adapter"


class FakeSettings:
    """Only what TranslationEngine touches; no vLLM, no tokenizer."""

    max_concurrent_requests = 4
    batch_size = 8
    split_sentences = False
    served_system = SYSTEM


class RecordingEngine(TranslationEngine):
    """A TranslationEngine whose upstream is a recorded list of canned outputs."""

    def __init__(self, outputs):
        super().__init__(FakeSettings())
        self._outputs = outputs
        self.sent: list[list[str]] = []

    @property
    def is_loaded(self):
        return True

    async def _generate(self, segments, system, source_lang, target_lang, max_new_tokens):
        self.sent.append(list(segments))
        return list(self._outputs)


class _StubTranslationEngine(TranslationEngine):
    """Skips the real load() (no tokenizer, no vLLM) and answers with a fixed string.

    Lets a request drive the real main.app -- construction, lifespan, routing,
    response building -- without a live vLLM or a mounted checkpoint. Follows
    the same "subclass TranslationEngine, stub what touches the outside world"
    approach as _NoLoadTranslationEngine in tests/test_glossary_admin.py and
    StubEngine in tests/test_glossary_http.py.
    """

    def load(self):
        pass

    @property
    def is_loaded(self) -> bool:
        return True

    async def _generate(self, segments, system, source_lang, target_lang, max_new_tokens):
        return ["stub"] * len(segments)


def _reload_app(monkeypatch):
    """Re-execute main.py to get a fresh FastAPI() app, wired to the stub engine.

    main.app is a module-level singleton, imported once for the whole pytest
    session. FastAPI never undoes an app.include_router() call, so a prior
    test in this same process that ran the real lifespan with the glossary
    enabled (e.g. the lifespan test in tests/test_glossary_admin.py) leaves
    the admin router mounted on that shared object permanently -- reusing it
    here would make "the router is absent" pass or fail depending on test
    order rather than on the behaviour under test. A real restart, which is
    what disabling the glossary via an env var actually relies on, gets a
    brand-new process and therefore a brand-new FastAPI() object; reloading
    main.py mirrors that precisely instead of trusting the possibly
    already-mutated shared singleton.
    """
    importlib.reload(main_module)
    monkeypatch.setattr(main_module, "TranslationEngine", _StubTranslationEngine)
    return main_module.app


async def test_segments_sent_upstream_are_identical_with_and_without_a_glossary():
    # The guarantee the kill switch rests on: turning the feature off cannot
    # change a single token the model sees.
    texts = ["The genome sequence.", "A second sentence."]

    without = RecordingEngine(["الف", "ب"])
    await without.translate(texts, SYSTEM, "en", "fa", 128, False, glossary_index=None)

    with_empty = RecordingEngine(["الف", "ب"])
    await with_empty.translate(
        texts, SYSTEM, "en", "fa", 128, False, glossary_index=build_index([], version=1)
    )

    assert without.sent == with_empty.sent == [texts]


def test_no_database_file_is_created_when_disabled(tmp_path, monkeypatch):
    """No database file materialises anywhere under a tmp_path root when the
    real app serves a real request with the feature off.

    The brief's version of this test only asserted that a default Settings()
    has glossary_enabled is False and that an *empty* temp dir contains no
    .db file -- true before anything ever runs, and would stay true even if a
    bug quietly opened a database file somewhere the test never looked. This
    version chdirs into tmp_path (so a relative TG_GLOSSARY_DB_URL like the
    default `./data/glossary.db` would land inside it), drives an actual
    /translate request through the real main.app, and then searches the whole
    tree.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TG_GLOSSARY_ENABLED", raising=False)
    get_settings.cache_clear()
    try:
        app = _reload_app(monkeypatch)
        with TestClient(app) as client:
            response = client.post(
                "/translate",
                json={"text": "hello", "domain": "medical", "terminology_mode": "enforce"},
            )
            assert response.status_code == 200
    finally:
        get_settings.cache_clear()
    assert list(Path(tmp_path).rglob("*.db")) == []


def test_admin_routes_are_absent_not_unauthorized(monkeypatch):
    # 404 rather than 401: with the feature off there is nothing to authorize.
    #
    # The brief's version built a bare FastAPI() and asserted 404 against it,
    # which is nearly worthless: a bare app trivially 404s everything, so it
    # would pass even if the production wiring in main.lifespan were broken
    # (e.g. the `if settings.glossary_enabled:` guard flipped, or the router
    # mounted unconditionally). This drives the real main.app through its
    # actual startup path with the feature off, mirroring how
    # test_the_admin_router_is_actually_mounted_by_the_real_lifespan in
    # tests/test_glossary_admin.py proves the mounted case.
    monkeypatch.delenv("TG_GLOSSARY_ENABLED", raising=False)
    get_settings.cache_clear()
    try:
        app = _reload_app(monkeypatch)
        with TestClient(app) as client:
            response = client.get("/admin/glossary/entries")
            assert response.status_code == 404
    finally:
        get_settings.cache_clear()


def test_response_body_carries_no_glossary_keys_when_disabled(monkeypatch):
    # Checks the serialized JSON body from a real request, not the model
    # definition -- response_model_exclude_none only hides the glossary
    # fields if the handler actually leaves them None, which is a runtime
    # behaviour and not something a schema inspection would catch.
    monkeypatch.delenv("TG_GLOSSARY_ENABLED", raising=False)
    get_settings.cache_clear()
    try:
        app = _reload_app(monkeypatch)
        with TestClient(app) as client:
            response = client.post(
                "/translate",
                json={"text": "hello", "domain": "medical", "terminology_mode": "enforce"},
            )
    finally:
        get_settings.cache_clear()
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"translation", "system", "source_lang", "target_lang"}
    for key in ("raw_translation", "glossary_version", "glossary"):
        assert key not in body


@pytest.mark.parametrize("field", ["domain", "terminology_mode"])
def test_request_fields_are_accepted_when_the_feature_is_off(field):
    # A caller that learned to send these must not start failing when an
    # operator turns the glossary off mid-incident.
    from schemas import Prompt

    prompt = Prompt(text="hello", **{field: "anything"})
    assert getattr(prompt, field) == "anything"


def test_disabling_the_glossary_does_not_destroy_its_data(tmp_path, monkeypatch):
    """Enable, write an entry, disable, re-enable: the entry must still be there.

    Turning the feature off must mean "stop opening the database", never
    "the database no longer matters". This cycles a fresh main.app (one per
    phase, via _reload_app -- see its docstring for why) through
    enable -> disable -> enable against the same on-disk sqlite file, the way
    an operator actually flips the switch across restarts.
    """
    db_path = tmp_path / "glossary.db"
    db_url = f"sqlite+aiosqlite:///{db_path}"
    key = "kill-switch-key"

    def configure(enabled: bool):
        monkeypatch.setenv("TG_GLOSSARY_ENABLED", "true" if enabled else "false")
        if enabled:
            monkeypatch.setenv("TG_ADMIN_API_KEY", key)
        else:
            monkeypatch.delenv("TG_ADMIN_API_KEY", raising=False)
        monkeypatch.setenv("TG_GLOSSARY_DB_URL", db_url)
        get_settings.cache_clear()
        return _reload_app(monkeypatch)

    try:
        # Enable, write an entry.
        app = configure(True)
        with TestClient(app) as client:
            created = client.post(
                "/admin/glossary/entries",
                headers={"X-Admin-Key": key},
                json={
                    "src_lang": "en",
                    "tgt_lang": "fa",
                    "source_term": "genome",
                    "target_term": "ژنوم",
                },
            )
            assert created.status_code == 201
        assert db_path.exists()

        # Disable: the admin surface disappears, the file must not be touched.
        app = configure(False)
        with TestClient(app) as client:
            assert client.get("/admin/glossary/entries").status_code == 404
        assert db_path.exists()

        # Re-enable: the entry written before the outage is still there.
        app = configure(True)
        with TestClient(app) as client:
            listed = client.get("/admin/glossary/entries", headers={"X-Admin-Key": key})
            assert [item["source_term"] for item in listed.json()] == ["genome"]
    finally:
        get_settings.cache_clear()
        importlib.reload(main_module)
