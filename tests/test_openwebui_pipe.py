"""Exercise the OpenWebUI translation pipe (scripts/openwebui_pipe.py) — the
deterministic pseudo-model front end generated from scripts/translation_core.py.

v0.11.3 specifics these tests pin down (verified against the v0.11.3 source,
backend/open_webui/functions.py::generate_function_chat_completion):
- The pipe is called as ``pipe(body, __user__, __chat_id__, __message_id__,
  __files__, __event_emitter__, ...)``; only parameters present in the
  method signature are passed.
- ``__files__`` is ``metadata['files']`` — the same item list the tool's
  ``__files__`` receives (frontend file dicts), so resolution and file
  reading behave exactly as in the tool.
- With ``stream=True`` (the chat default) a returned sync generator is
  consumed line by line; each plain-string line is wrapped by the framework
  into an OpenAI delta chunk. The pipe must therefore yield plain text.
- A plain string return is wrapped into a single chunk (stream) or a full
  completion (non-stream) by the framework.
- Pipe replies are plain assistant messages (no tool-call marker), so the
  core's history fallback must run with include_assistant_history=False or
  "translate this" re-translates the previous pipe reply.
"""

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "openwebui_pipe.py"


# ---------------------------------------------------------------------------
# Hermetic stubs (same as the tool suite): requests + pydantic come from the
# OpenWebUI container in production.
# ---------------------------------------------------------------------------


def _install_stubs(monkeypatch):
    class _FieldInfo:
        def __init__(self, default=None, **kwargs):
            self.default = default

    def Field(default=None, **kwargs):
        return _FieldInfo(default)

    class BaseModel:
        def __init__(self, **kwargs):
            for name, value in vars(type(self)).items():
                if name.startswith("_") or not isinstance(value, _FieldInfo):
                    continue
                setattr(self, name, kwargs.get(name, value.default))

    pydantic_stub = types.ModuleType("pydantic")
    pydantic_stub.BaseModel = BaseModel
    pydantic_stub.Field = Field
    monkeypatch.setitem(sys.modules, "pydantic", pydantic_stub)

    requests_stub = types.ModuleType("requests")
    requests_stub.post = None  # tests replace it
    monkeypatch.setitem(sys.modules, "requests", requests_stub)


def _load_pipe(monkeypatch):
    _install_stubs(monkeypatch)
    spec = importlib.util.spec_from_file_location("owui_pipe_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_capturing_backend(module, calls, translation="TRANSLATION_OUTPUT"):
    class FakeResp:
        status_code = 200

        def json(self):
            return {"translation": translation}

    def fake_post(url, json=None, timeout=None):
        calls.append(dict(json))
        return FakeResp()

    module.requests.post = fake_post


def _chats_stub(monkeypatch, captured):
    class FakeChats:
        @staticmethod
        async def add_message_files_by_id_and_message_id(chat_id, message_id, files):
            captured.append((chat_id, message_id, files))
            return files

    chats_mod = types.ModuleType("open_webui.models.chats")
    chats_mod.Chats = FakeChats
    models_mod = types.ModuleType("open_webui.models")
    open_webui_mod = types.ModuleType("open_webui")
    monkeypatch.setitem(sys.modules, "open_webui", open_webui_mod)
    monkeypatch.setitem(sys.modules, "open_webui.models", models_mod)
    monkeypatch.setitem(sys.modules, "open_webui.models.chats", chats_mod)


def _run_pipe(module, valves=None, body=None, **kwargs):
    pipe = module.Pipe()
    if valves:
        for key, value in valves.items():
            setattr(pipe.valves, key, value)
    if body is None:
        body = {"messages": [], "stream": False}
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(pipe.pipe(body=body, **kwargs))
    finally:
        loop.close()


def user_msg(text):
    return {"role": "user", "content": [{"type": "text", "text": text}]}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReplyShape:
    def test_message_text_non_stream_returns_translation(self, monkeypatch):
        module = _load_pipe(monkeypatch)
        calls = []
        _make_capturing_backend(module, calls)
        sentence = "The quick brown fox jumps over the lazy dog."
        body = {"messages": [user_msg(sentence)], "stream": False}
        out = _run_pipe(module, body=body)
        assert out == "TRANSLATION_OUTPUT"
        # Short single line: direct mode.
        assert calls[0]["text"] == sentence
        assert calls[0]["split_sentences"] is False

    def test_stream_returns_text_generator(self, monkeypatch):
        module = _load_pipe(monkeypatch)
        calls = []
        long_doc = "\n\n".join(f"Paragraph {i} of the document. " * 40 for i in range(3))
        _make_capturing_backend(module, calls, translation=long_doc)
        body = {"messages": [user_msg(long_doc)], "stream": True}
        out = _run_pipe(module, body=body)
        assert not isinstance(out, str)
        chunks = list(out)
        assert all(isinstance(c, str) and c for c in chunks)
        assert "".join(chunks) == long_doc

    def test_stream_chunks_bounded(self, monkeypatch):
        module = _load_pipe(monkeypatch)
        _make_capturing_backend(
            module, [], translation="word " * 2000
        )  # one 10k-char line, no paragraph breaks
        body = {"messages": [user_msg("some input text")], "stream": True}
        chunks = list(_run_pipe(module, body=body))
        assert "".join(chunks) == "word " * 2000
        assert all(len(c) <= 1500 for c in chunks)


class TestResolution:
    def test_system_prompt_never_translated(self, monkeypatch):
        module = _load_pipe(monkeypatch)
        calls = []
        _make_capturing_backend(module, calls)
        messages = [
            {"role": "system", "content": "You are a helpful assistant with secrets."},
            user_msg("translate this"),
        ]
        out = _run_pipe(module, body={"messages": messages, "stream": False})
        assert "Nothing to translate" in out
        assert calls == []

    def test_nothing_to_translate_error(self, monkeypatch):
        module = _load_pipe(monkeypatch)
        calls = []
        _make_capturing_backend(module, calls)
        out = _run_pipe(module, body={"messages": [user_msg("translate this")], "stream": False})
        assert "Nothing to translate" in out
        assert "defaults: en" in out
        assert calls == []

    def test_unreadable_file_error(self, monkeypatch):
        module = _load_pipe(monkeypatch)
        calls = []
        _make_capturing_backend(module, calls)
        captured = []
        _chats_stub(monkeypatch, captured)
        out = _run_pipe(
            module,
            body={"messages": [user_msg("translate this")], "stream": False},
            __files__=[{"type": "file", "id": "f1", "name": "doc.md"}],
        )
        assert "The attached file (doc.md) has no readable text yet" in out
        assert "Using Entire Document" in out
        assert calls == []
        assert captured == []

    def test_attached_file_item_resolved(self, monkeypatch):
        module = _load_pipe(monkeypatch)
        calls = []
        _make_capturing_backend(module, calls)
        files = [
            {
                "type": "file",
                "id": "f1",
                "name": "doc.md",
                "file": {"data": {"content": "First paragraph of the document."}},
            }
        ]
        out = _run_pipe(
            module,
            body={"messages": [user_msg("translate this")], "stream": False},
            __files__=files,
        )
        assert out == "TRANSLATION_OUTPUT"
        assert calls[0]["text"] == "First paragraph of the document."

    def test_previous_pipe_reply_not_retranslated(self, monkeypatch):
        # Pipe-mode regression: the previous reply is plain assistant text
        # with no tool-call marker, so the tool-mode relay detection cannot
        # see it. include_assistant_history=False is what keeps "translate
        # this" latching onto the original user sentence instead.
        module = _load_pipe(monkeypatch)
        calls = []
        _make_capturing_backend(module, calls)
        messages = [
            user_msg("The quick brown fox jumps over the lazy dog."),
            {"role": "assistant", "content": "سگ قهوه‌ای تیز از روی لاک‌پشت تنبل می‌پرد."},
            user_msg("translate this"),
        ]
        _run_pipe(module, body={"messages": messages, "stream": False})
        assert calls[0]["text"] == "The quick brown fox jumps over the lazy dog."

    def test_translate_command_overrides_languages(self, monkeypatch):
        module = _load_pipe(monkeypatch)
        calls = []
        _make_capturing_backend(module, calls)
        messages = [user_msg("/translate en de some words to translate")]
        _run_pipe(module, body={"messages": messages, "stream": False})
        assert calls[0]["source_lang"] == "en"
        assert calls[0]["target_lang"] == "de"
        assert calls[0]["text"] == "some words to translate"

    def test_user_valves_dict_shape(self, monkeypatch):
        module = _load_pipe(monkeypatch)
        calls = []
        _make_capturing_backend(module, calls)
        messages = [user_msg("The quick brown fox jumps over the lazy dog.")]
        _run_pipe(
            module,
            body={"messages": messages, "stream": False},
            __user__={"id": "u1", "valves": {"target_lang": "ru"}},
        )
        assert calls[0]["target_lang"] == "ru"


class TestDelivery:
    def test_success_attached_to_message(self, monkeypatch):
        module = _load_pipe(monkeypatch)
        calls = []
        _make_capturing_backend(module, calls)
        captured = []
        _chats_stub(monkeypatch, captured)
        _run_pipe(
            module,
            body={
                "messages": [user_msg("The quick brown fox jumps over the lazy dog.")],
                "stream": False,
            },
            __chat_id__="chat-1",
            __message_id__="msg-1",
            __user__={"id": "u1"},
        )
        assert len(captured) == 1
        chat_id, message_id, files = captured[0]
        assert (chat_id, message_id) == ("chat-1", "msg-1")
        entry = files[0]
        assert entry["name"] == "translation-fa.md"
        assert entry["type"] == "file"

    def test_attach_file_valve_off_skips_attachment(self, monkeypatch):
        module = _load_pipe(monkeypatch)
        calls = []
        _make_capturing_backend(module, calls)
        captured = []
        _chats_stub(monkeypatch, captured)
        out = _run_pipe(
            module,
            valves={"ATTACH_FILE": False},
            body={
                "messages": [user_msg("The quick brown fox jumps over the lazy dog.")],
                "stream": False,
            },
            __chat_id__="chat-1",
            __message_id__="msg-1",
            __user__={"id": "u1"},
        )
        assert out == "TRANSLATION_OUTPUT"
        assert captured == []

    def test_gateway_error_not_attached(self, monkeypatch):
        module = _load_pipe(monkeypatch)
        captured = []
        _chats_stub(monkeypatch, captured)

        class FakeResp:
            status_code = 502
            text = "upstream is down"

        module.requests.post = lambda url, json=None, timeout=None: FakeResp()
        out = _run_pipe(
            module,
            body={
                "messages": [user_msg("The quick brown fox jumps over the lazy dog.")],
                "stream": False,
            },
            __chat_id__="chat-1",
            __message_id__="msg-1",
            __user__={"id": "u1"},
        )
        assert out.startswith("Gateway Error (502):")
        assert captured == []

    def test_connection_error_not_attached(self, monkeypatch):
        # Regression: connection failures arrive as "Translation
        # connection error:" — a prefix the old error check missed, so a
        # file containing the error text got attached.
        module = _load_pipe(monkeypatch)
        captured = []
        _chats_stub(monkeypatch, captured)

        def boom(url, json=None, timeout=None):
            raise ConnectionError("connection refused")

        module.requests.post = boom
        out = _run_pipe(
            module,
            body={
                "messages": [user_msg("The quick brown fox jumps over the lazy dog.")],
                "stream": False,
            },
            __chat_id__="chat-1",
            __message_id__="msg-1",
            __user__={"id": "u1"},
        )
        assert out == "Translation connection error: connection refused"
        assert captured == []

    def test_empty_translation_not_attached(self, monkeypatch):
        # A 200 with an empty body is a failure too: no file, a visible
        # error instead of an empty reply.
        module = _load_pipe(monkeypatch)
        captured = []
        _chats_stub(monkeypatch, captured)

        class FakeResp:
            status_code = 200

            def json(self):
                return {"translation": ""}

        module.requests.post = lambda url, json=None, timeout=None: FakeResp()
        out = _run_pipe(
            module,
            body={
                "messages": [user_msg("The quick brown fox jumps over the lazy dog.")],
                "stream": False,
            },
            __chat_id__="chat-1",
            __message_id__="msg-1",
            __user__={"id": "u1"},
        )
        assert out == "The translator returned no text (empty response)."
        assert captured == []

    def test_status_events_emitted(self, monkeypatch):
        module = _load_pipe(monkeypatch)
        calls = []
        _make_capturing_backend(module, calls)
        events = []

        async def emitter(event):
            events.append(event)

        _run_pipe(
            module,
            body={
                "messages": [user_msg("The quick brown fox jumps over the lazy dog.")],
                "stream": False,
            },
            __event_emitter__=emitter,
        )
        # Two status chips plus the file-attachment event (ATTACH_FILE on).
        assert [e["type"] for e in events] == ["status", "status", "chat:message:files"]
        assert events[0]["data"]["done"] is False
        assert "Translating 9 words [en" in events[0]["data"]["description"]
        assert events[1]["data"]["done"] is True
        assert events[2]["data"]["files"][0]["name"] == "translation-fa.md"

    def test_stored_as_real_downloadable_file(self, monkeypatch):
        module = _load_pipe(monkeypatch)
        calls = []
        _make_capturing_backend(module, calls)

        class FakeFiles:
            @staticmethod
            async def insert_new_file(user_id, form):
                class Record:
                    id = form.id

                    def model_dump(self):
                        return {
                            "id": form.id,
                            "filename": form.filename,
                            "data": form.data,
                        }

                return Record()

        class FakeFileForm:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class FakeStorage:
            @staticmethod
            def upload_file(contents, filename, tags):
                return contents.read(), f"files/{filename}"

        files_mod = types.ModuleType("open_webui.models.files")
        files_mod.Files = FakeFiles
        files_mod.FileForm = FakeFileForm
        storage_mod = types.ModuleType("open_webui.storage.provider")
        storage_mod.Storage = FakeStorage
        models_mod = types.ModuleType("open_webui.models")
        storage_pkg = types.ModuleType("open_webui.storage")
        open_webui_mod = types.ModuleType("open_webui")
        monkeypatch.setitem(sys.modules, "open_webui", open_webui_mod)
        monkeypatch.setitem(sys.modules, "open_webui.models", models_mod)
        monkeypatch.setitem(sys.modules, "open_webui.models.files", files_mod)
        monkeypatch.setitem(sys.modules, "open_webui.storage", storage_pkg)
        monkeypatch.setitem(sys.modules, "open_webui.storage.provider", storage_mod)

        captured = []

        class FakeChats:
            @staticmethod
            async def add_message_files_by_id_and_message_id(chat_id, message_id, files):
                captured.append(files)
                return files

        chats_mod = types.ModuleType("open_webui.models.chats")
        chats_mod.Chats = FakeChats
        monkeypatch.setitem(sys.modules, "open_webui.models.chats", chats_mod)

        _run_pipe(
            module,
            body={
                "messages": [user_msg("The quick brown fox jumps over the lazy dog.")],
                "stream": False,
            },
            __chat_id__="chat-1",
            __message_id__="msg-1",
            __user__={"id": "u1"},
        )
        entry = captured[0][0]
        assert entry["id"] == entry["url"]  # file id: /files/{id}/content
        assert entry["file"]["data"]["content"] == "TRANSLATION_OUTPUT"
