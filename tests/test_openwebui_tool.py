"""Exercise the OpenWebUI translate tool (scripts/openwebui_tool.py) against the
message shapes OpenWebUI v0.11.3 actually produces.

v0.11.3 specifics these tests pin down (verified against the v0.11.3 source):
- With tools enabled, user message content is a LIST of content parts, not a
  string (middleware.add_file_context).
- Attached files arrive as ``<attached_files><file id="..."/></attached_files>``
  metadata tags; full-context uploads carry the document inside a RAG
  ``<context><source ...>BODY</source></context>`` block.
- At tool-call time the LAST message is the assistant tool_call, so the current
  user message is not ``messages[-1]``.
- ``__files__`` is scoped to the current message.

The tool must therefore self-resolve the text to translate instead of relying
on the base model to copy document text into the tool argument (which is where
truncation and multi-source confusion happened).
"""

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "openwebui_tool.py"

# ---------------------------------------------------------------------------
# Hermetic stubs: the tool declares `requirements: requests, pydantic` (provided
# by the OpenWebUI container). Stub both so this suite runs in any environment.
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


def _load_tool(monkeypatch):
    _install_stubs(monkeypatch)
    spec = importlib.util.spec_from_file_location("owui_tool_under_test", MODULE_PATH)
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


def _run(module, valves=None, **kwargs):
    tool = module.Tools()
    if valves:
        for key, value in valves.items():
            setattr(tool.valves, key, value)
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(tool.translate(**kwargs))
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# v0.11.3 message builders
# ---------------------------------------------------------------------------

RAG_TEMPLATE = (
    "### Task:\n"
    "Respond to the user query using the provided context, incorporating inline "
    "citations in the format [id] only when the <source> tag includes an id.\n\n"
    "<context>\n{{CONTEXT}}\n</context>\n\n"
)


def rag_user_msg(body, query, as_list=True, name="doc.md"):
    context = f'<source id="1" name="{name}" resource-type="file" resource-id="f123">{body}</source>\n'
    text = RAG_TEMPLATE.replace("{{CONTEXT}}", context) + query
    if as_list:
        return {"role": "user", "content": [{"type": "text", "text": text}]}
    return {"role": "user", "content": text}


def attached_files_tag(file_id, name, content_type="text/markdown"):
    return (
        f'<attached_files>\n'
        f'<file type="file" id="{file_id}" url="{file_id}" '
        f'content_type="{content_type}" name="{name}"/>\n'
        f"</attached_files>\n\n"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFileContentResolution:
    def test_full_context_pdf_translates_body(self, monkeypatch):
        module = _load_tool(monkeypatch)
        calls = []
        _make_capturing_backend(module, calls)
        body = "First paragraph of the document.\nSecond paragraph."
        messages = [
            rag_user_msg(body, "translate this", as_list=True),
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "function": {"name": "translate", "arguments": "{}"}}]},
        ]
        out = _run(
            module,
            text="this",
            __messages__=messages,
            __files__=[{"type": "file", "id": "f123", "name": "doc.pdf"}],
        )
        assert out == "TRANSLATION_OUTPUT"
        assert "First paragraph of the document." in calls[0]["text"]
        assert "### Task:" not in calls[0]["text"]
        assert "<source" not in calls[0]["text"]
        assert calls[0]["split_sentences"] is True

    def test_markdown_list_content_extracted(self, monkeypatch):
        # Content arrives as a list of parts; the pre-fix tool only handled str.
        module = _load_tool(monkeypatch)
        calls = []
        _make_capturing_backend(module, calls)
        body = "# Title\n\nMarkdown body line one.\nMarkdown body line two."
        messages = [rag_user_msg(body, "translate this", as_list=True)]
        _run(module, text="", __messages__=messages, __files__=[])
        assert "Markdown body line two." in calls[0]["text"]

    def test_string_content_context(self, monkeypatch):
        module = _load_tool(monkeypatch)
        calls = []
        _make_capturing_backend(module, calls)
        messages = [rag_user_msg("STRING CONTENT DOC BODY here.", "translate this", as_list=False)]
        _run(module, text="this", __messages__=messages, __files__=[])
        assert calls[0]["text"] == "STRING CONTENT DOC BODY here."

    def test_tag_only_message_uses_files_embedded_content(self, monkeypatch):
        module = _load_tool(monkeypatch)
        calls = []
        _make_capturing_backend(module, calls)
        messages = [
            {"role": "user", "content": [{"type": "text", "text": attached_files_tag("f77", "report.pdf", "application/pdf")}]},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c", "function": {"name": "translate", "arguments": "{}"}}]},
        ]
        files = [{"type": "file", "id": "f77", "name": "report.pdf", "file": {"data": {"content": "PDF BODY FROM FILES"}}}]
        _run(module, text="", __messages__=messages, __files__=files)
        assert calls[0]["text"] == "PDF BODY FROM FILES"

    def test_http_url_file_ignored(self, monkeypatch):
        module = _load_tool(monkeypatch)
        calls = []
        _make_capturing_backend(module, calls)
        messages = [
            {"role": "user", "content": "translate this"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c", "function": {"name": "translate", "arguments": "{}"}}]},
        ]
        files = [{"type": "file", "id": "http://external/x.pdf", "url": "http://external/x.pdf", "name": "x.pdf"}]
        out = _run(module, text="this", __messages__=messages, __files__=files)
        assert out.startswith("Error:")

    def test_image_file_skipped(self, monkeypatch):
        module = _load_tool(monkeypatch)
        calls = []
        _make_capturing_backend(module, calls)
        messages = [rag_user_msg("IMG DOC CONTEXT", "translate this")]
        files = [{"type": "image", "id": "img1", "name": "a.png", "file": {"data": {"content": "NOT A DOC"}}}]
        _run(module, text="this", __messages__=messages, __files__=files)
        assert calls[0]["text"] == "IMG DOC CONTEXT"

    def test_multiple_sources_in_one_context_block(self, monkeypatch):
        module = _load_tool(monkeypatch)
        calls = []
        _make_capturing_backend(module, calls)
        context = '<source id="1" name="a.md" resource-id="a">FILE A BODY</source>\n<source id="2" name="b.md" resource-id="b">FILE B BODY</source>\n'
        messages = [{"role": "user", "content": [{"type": "text", "text": RAG_TEMPLATE.replace("{{CONTEXT}}", context) + "\ntranslate both"}]}]
        _run(module, text="this", __messages__=messages, __files__=[])
        assert "FILE A BODY" in calls[0]["text"]
        assert "FILE B BODY" in calls[0]["text"]


class TestLongDocuments:
    def test_full_long_text_sent_without_truncation(self, monkeypatch):
        module = _load_tool(monkeypatch)
        calls = []
        _make_capturing_backend(module, calls)
        long_body = ("The quick brown fox jumps over the lazy dog. " * 500).strip()
        assert len(long_body) > 10_000  # beyond view_file's default cap
        messages = [rag_user_msg(long_body, "translate this", as_list=True)]
        out = _run(module, text="", __messages__=messages, __files__=[])
        assert calls[0]["text"] == long_body
        assert out == "TRANSLATION_OUTPUT"


class TestMultiSource:
    def test_second_uploaded_file_not_first(self, monkeypatch):
        module = _load_tool(monkeypatch)
        calls = []
        _make_capturing_backend(module, calls)
        body_a = "SOURCE A CONTENT about apples."
        body_b = "SOURCE B CONTENT about oranges."
        messages = [
            rag_user_msg(body_a, "translate this", as_list=True),
            {"role": "assistant", "content": None, "tool_calls": [{"id": "cA", "function": {"name": "translate", "arguments": "{}"}}]},
            {"role": "tool", "content": "TRANSLATION OF A", "tool_call_id": "cA"},
            {"role": "assistant", "content": "TRANSLATION OF A"},
            rag_user_msg(body_b, "translate this", as_list=True),
            {"role": "assistant", "content": None, "tool_calls": [{"id": "cB", "function": {"name": "translate", "arguments": "{}"}}]},
        ]
        _run(module, text="this", __messages__=messages, __files__=[])
        assert calls[0]["text"] == body_b

    def test_second_pasted_source_not_first(self, monkeypatch):
        module = _load_tool(monkeypatch)
        calls = []
        _make_capturing_backend(module, calls)
        messages = [
            {"role": "user", "content": "translate this\n\nPASTED SOURCE ONE with some words."},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "function": {"name": "translate", "arguments": "{}"}}]},
            {"role": "tool", "content": "TRANSLATION ONE", "tool_call_id": "c1"},
            {"role": "assistant", "content": "TRANSLATION ONE"},
            {"role": "user", "content": "translate this\n\nPASTED SOURCE TWO with other words."},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c2", "function": {"name": "translate", "arguments": "{}"}}]},
        ]
        _run(module, text="this", __messages__=messages, __files__=[])
        assert calls[0]["text"] == "PASTED SOURCE TWO with other words."

    def test_second_file_via_embedded_files(self, monkeypatch):
        module = _load_tool(monkeypatch)
        calls = []
        _make_capturing_backend(module, calls)
        body_a = "SOURCE A CONTENT about apples."
        body_b = "SOURCE B CONTENT about oranges."
        messages = [
            rag_user_msg(body_a, "translate this", as_list=True),
            {"role": "assistant", "content": None, "tool_calls": [{"id": "cA", "function": {"name": "translate", "arguments": "{}"}}]},
            {"role": "tool", "content": "TRANSLATION OF A", "tool_call_id": "cA"},
            {"role": "assistant", "content": "TRANSLATION OF A"},
            {"role": "user", "content": [{"type": "text", "text": attached_files_tag("fB", "b.md") + "translate this"}]},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "cB", "function": {"name": "translate", "arguments": "{}"}}]},
        ]
        files_b = [{"type": "file", "id": "fB", "name": "b.md", "file": {"data": {"content": body_b}}}]
        _run(module, text="this", __messages__=messages, __files__=files_b)
        assert calls[0]["text"] == body_b

    def test_retranslate_picks_source_not_previous_translation(self, monkeypatch):
        module = _load_tool(monkeypatch)
        calls = []
        _make_capturing_backend(module, calls)
        messages = [
            {"role": "user", "content": "translate this\n\nRETRANSLATE SOURCE TEXT please."},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "function": {"name": "translate", "arguments": "{}"}}]},
            {"role": "tool", "content": "OLD TRANSLATION", "tool_call_id": "c1"},
            {"role": "assistant", "content": "OLD TRANSLATION"},
            {"role": "user", "content": "translate this to German"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c2", "function": {"name": "translate", "arguments": "{}"}}]},
        ]
        _run(module, text="this", __messages__=messages, __files__=[])
        assert calls[0]["text"] == "RETRANSLATE SOURCE TEXT please."


class TestHistoryReferences:
    def test_ocr_output_then_translate_this(self, monkeypatch):
        # The source is a previous ASSISTANT message (OCR extraction); a plain
        # assistant message that is not a translate relay must remain eligible.
        module = _load_tool(monkeypatch)
        calls = []
        _make_capturing_backend(module, calls)
        ocr_text = "EXTRACTED OCR TEXT: Hello from the scanned page."
        messages = [
            {"role": "user", "content": attached_files_tag("f9", "scan.pdf", "application/pdf") + "extract the text from this pdf"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c9", "function": {"name": "view_file", "arguments": "{}"}}]},
            {"role": "tool", "content": "{...}", "tool_call_id": "c9"},
            {"role": "assistant", "content": ocr_text},
            {"role": "user", "content": "translate this"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c2", "function": {"name": "translate", "arguments": "{}"}}]},
        ]
        _run(module, text="this", __messages__=messages, __files__=[])
        assert calls[0]["text"] == ocr_text

    def test_translate_the_above_explanation(self, monkeypatch):
        module = _load_tool(monkeypatch)
        calls = []
        _make_capturing_backend(module, calls)
        explanation = "CRISPR is a gene-editing tool. It uses a guide RNA to cut DNA."
        messages = [
            {"role": "user", "content": "Explain CRISPR in two sentences."},
            {"role": "assistant", "content": explanation},
            {"role": "user", "content": "Translate the above into Persian."},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c", "function": {"name": "translate", "arguments": "{}"}}]},
        ]
        _run(module, text="the above", __messages__=messages, __files__=[])
        assert calls[0]["text"] == explanation

    def test_persian_referential(self, monkeypatch):
        module = _load_tool(monkeypatch)
        calls = []
        _make_capturing_backend(module, calls)
        messages = [
            {"role": "assistant", "content": "EN TEXT TO TRANSLATE FROM HISTORY."},
            {"role": "user", "content": "\u0645\u062a\u0646 \u0628\u0627\u0644\u0627 \u0631\u0627 \u062a\u0631\u062c\u0645\u0647 \u06a9\u0646"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c", "function": {"name": "translate", "arguments": "{}"}}]},
        ]
        _run(module, text="\u062a\u0631\u062c\u0645\u0647 \u06a9\u0646", __messages__=messages, __files__=[])
        assert calls[0]["text"] == "EN TEXT TO TRANSLATE FROM HISTORY."

    def test_nothing_found_returns_helpful_error(self, monkeypatch):
        module = _load_tool(monkeypatch)
        calls = []
        _make_capturing_backend(module, calls)
        out = _run(
            module,
            text="this",
            __messages__=[{"role": "user", "content": "translate this"}],
            __files__=[],
        )
        assert out.startswith("Error: No text or file content found")
        assert calls == []


class TestSystemPromptProtection:
    def test_system_prompt_never_used_as_source(self, monkeypatch):
        # Regression: a referential-only user turn ("translate this") plus an
        # attachment whose content is not yet available must NOT fall back to
        # the system prompt as the translation source. The old fallback loop
        # scanned every message and returned the system prompt, which then got
        # translated instead of the user's document.
        module = _load_tool(monkeypatch)
        calls = []
        _make_capturing_backend(module, calls)
        system_prompt = (
            "You are a translation engine powered by TranslateGemma 27B. "
            "You have exactly ONE job: translate. You never do anything else."
        )
        out = _run(
            module,
            text="this",
            __messages__=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "translate this"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "c1", "function": {"name": "translate", "arguments": "{}"}}],
                },
            ],
            __files__=[
                {"id": "f1", "name": "doc.md", "type": "file", "url": "f1", "file": {}},
            ],
        )
        assert out.startswith("Error:")
        assert "no readable text" in out
        assert system_prompt not in out
        assert calls == []  # nothing was sent to the gateway

    def test_system_prompt_skipped_but_real_history_used(self, monkeypatch):
        # The system prompt is skipped, but a genuine earlier assistant/user
        # source is still found by the fallback.
        module = _load_tool(monkeypatch)
        calls = []
        _make_capturing_backend(module, calls)
        source = "REAL EARLIER SOURCE TEXT that should be translated."
        messages = [
            {"role": "system", "content": "SYSTEM PROMPT — must never be translated."},
            {"role": "assistant", "content": source},
            {"role": "user", "content": "translate this"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "c", "function": {"name": "translate", "arguments": "{}"}}],
            },
        ]
        _run(module, text="this", __messages__=messages, __files__=[])
        assert calls[0]["text"] == source


class TestInlineAndCommands:
    def test_explicit_inline_text(self, monkeypatch):
        module = _load_tool(monkeypatch)
        calls = []
        _make_capturing_backend(module, calls)
        _run(module, text="Hello world, this is inline text to translate.", __messages__=[], __files__=[])
        assert calls[0]["text"] == "Hello world, this is inline text to translate."
        assert calls[0]["split_sentences"] is False

    def test_model_dumps_whole_message(self, monkeypatch):
        # Some base models pass the entire current message (RAG preamble and
        # all) as the text argument; mine the context block out of it.
        module = _load_tool(monkeypatch)
        calls = []
        _make_capturing_backend(module, calls)
        dumped = RAG_TEMPLATE.replace("{{CONTEXT}}", '<source id="1" name="d.md" resource-id="x">DUMPED DOC BODY</source>\n') + "\ntranslate this"
        _run(module, text=dumped, __messages__=[], __files__=[])
        assert calls[0]["text"] == "DUMPED DOC BODY"

    def test_translate_command_with_pair(self, monkeypatch):
        module = _load_tool(monkeypatch)
        calls = []
        _make_capturing_backend(module, calls)
        _run(module, text="/translate fr fa Bonjour le monde", __messages__=[], __files__=[])
        assert calls[0]["source_lang"] == "fr"
        assert calls[0]["target_lang"] == "fa"
        assert calls[0]["text"] == "Bonjour le monde"

    def test_translate_command_with_region_code(self, monkeypatch):
        module = _load_tool(monkeypatch)
        calls = []
        _make_capturing_backend(module, calls)
        _run(module, text="/translate en pt-BR Hello there friend", __messages__=[], __files__=[])
        assert calls[0]["target_lang"] == "pt-br"

    def test_simple_translate_command(self, monkeypatch):
        module = _load_tool(monkeypatch)
        calls = []
        _make_capturing_backend(module, calls)
        _run(module, text="/translate Good morning everyone", __messages__=[], __files__=[])
        assert calls[0]["text"] == "Good morning everyone"

    def test_modal_instruction_line_stripped_from_explicit_text(self, monkeypatch):
        # Model forwards the whole modal string; the blank-line-separated
        # instruction prefix must not reach the gateway.
        module = _load_tool(monkeypatch)
        calls = []
        _make_capturing_backend(module, calls)
        _run(
            module,
            text="Translate the following text to Persian:\n\nThe cell is the basic unit of life.",
            __messages__=[],
            __files__=[],
        )
        assert calls[0]["text"] == "The cell is the basic unit of life."

    def test_real_sentence_starting_with_translate_untouched(self, monkeypatch):
        module = _load_tool(monkeypatch)
        calls = []
        _make_capturing_backend(module, calls)
        body = "Translate this word carefully. It is a title."
        _run(module, text=body, __messages__=[], __files__=[])
        assert calls[0]["text"] == body

    def test_prompt_modal_instruction_stripped(self, monkeypatch):
        module = _load_tool(monkeypatch)
        calls = []
        _make_capturing_backend(module, calls)
        messages = [
            {"role": "user", "content": "Translate the following text to Persian:\n\nThe cell is the basic unit of life."},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c", "function": {"name": "translate", "arguments": "{}"}}]},
        ]
        _run(module, text="this", __messages__=messages, __files__=[])
        assert calls[0]["text"] == "The cell is the basic unit of life."


class TestLanguages:
    def test_user_valves_override(self, monkeypatch):
        module = _load_tool(monkeypatch)
        calls = []
        _make_capturing_backend(module, calls)
        _run(
            module,
            text="Some text here to translate ok",
            __messages__=[],
            __files__=[],
            __user__={"valves": {"source_lang": "de", "target_lang": "ru"}},
        )
        assert calls[0]["source_lang"] == "de"
        assert calls[0]["target_lang"] == "ru"

    def test_explicit_args_beat_valves(self, monkeypatch):
        module = _load_tool(monkeypatch)
        calls = []
        _make_capturing_backend(module, calls)
        _run(
            module,
            text="Some text here to translate ok",
            source_lang="en",
            target_lang="fa",
            __messages__=[],
            __files__=[],
            __user__={"valves": {"source_lang": "de", "target_lang": "ru"}},
        )
        assert calls[0]["source_lang"] == "en"
        assert calls[0]["target_lang"] == "fa"


class TestSanitization:
    """OCR markdown carries base64 image payloads and HTML; sending them to
    the MT model produces garbage and overflows the relay back through the
    base model. They must be stripped before the gateway call."""

    def test_base64_data_uri_stripped(self, monkeypatch):
        module = _load_tool(monkeypatch)
        calls = []
        _make_capturing_backend(module, calls)
        blob = "iVBORw0KGgoAAAANSUhEUg" * 50
        text = f"First paragraph about the method.\n\n![figure](data:image/png;base64,{blob})\n\nSecond paragraph about the results."
        _run(module, text=text, __messages__=[], __files__=[])
        sent = calls[0]["text"]
        assert "iVBORw0KGgo" not in sent
        assert "data:image" not in sent
        assert "First paragraph about the method." in sent
        assert "Second paragraph about the results." in sent

    def test_image_alt_text_kept(self, monkeypatch):
        module = _load_tool(monkeypatch)
        calls = []
        _make_capturing_backend(module, calls)
        blob = "AAAA" * 20
        _run(
            module,
            text=f"The plot below. ![Training loss curve](data:image/png;base64,{blob}) It declines steadily.",
            __messages__=[],
            __files__=[],
        )
        sent = calls[0]["text"]
        assert "Training loss curve" in sent
        assert "It declines steadily." in sent
        assert "base64" not in sent

    def test_html_comments_and_tags_stripped(self, monkeypatch):
        module = _load_tool(monkeypatch)
        calls = []
        _make_capturing_backend(module, calls)
        text = "<!-- OCR confidence 0.99 --><p>The <b>model</b> achieves state-of-the-art results.</p>"
        _run(module, text=text, __messages__=[], __files__=[])
        assert calls[0]["text"] == "The model achieves state-of-the-art results."

    def test_image_only_document_errors_without_gateway_call(self, monkeypatch):
        module = _load_tool(monkeypatch)
        calls = []
        _make_capturing_backend(module, calls)
        blob = "iVBORw0KGgo" * 100
        # Empty alt text -> nothing translatable remains after stripping.
        out = _run(
            module,
            text=f"![](data:image/png;base64,{blob})",
            __messages__=[],
            __files__=[],
        )
        assert out.startswith("Error: The content contains no translatable text")
        assert calls == []

    def test_image_with_alt_keeps_only_alt(self, monkeypatch):
        module = _load_tool(monkeypatch)
        calls = []
        _make_capturing_backend(module, calls)
        blob = "iVBORw0KGgo" * 100
        _run(
            module,
            text=f"![Figure 1: The plot](data:image/png;base64,{blob}) and the caption continues.",
            __messages__=[],
            __files__=[],
        )
        sent = calls[0]["text"]
        assert "Figure 1: The plot" in sent
        assert "and the caption continues." in sent
        assert "iVBORw0KGgo" not in sent


class TestLargeTranslationAttachment:
    """Large translations must not be relayed through the base model (it would
    re-emit the whole document, overflowing its context). They are delivered
    as a chat file attachment, the way OpenWebUI's own generate_image tool
    delivers images."""

    @staticmethod
    def _events():
        events = []

        async def emit(event):
            events.append(event)

        return events, emit

    def test_large_translation_attached_not_relayed(self, monkeypatch):
        module = _load_tool(monkeypatch)
        calls = []
        big = "پاراگراف ترجمه شده. " * 500  # ~9000 chars, above the 8000 default
        _make_capturing_backend(module, calls, translation=big)
        events, emit = self._events()
        out = _run(
            module,
            text="A source sentence that is long enough to split into parts for sure.",
            __messages__=[],
            __files__=[],
            __event_emitter__=emit,
        )
        # The model gets an acknowledgement, not the translation.
        assert big not in out
        assert "attached" in out.lower()
        assert "NEVER restate" in out
        file_event = [e for e in events if e.get("type") == "chat:message:files"]
        assert len(file_event) == 1
        entry = file_event[0]["data"]["files"][0]
        assert entry["content"] == big
        assert entry["type"] == "file"
        assert entry["name"].endswith(".translated.fa.md") or entry["name"].startswith("translation-")
        assert entry["url"].startswith("data:text/markdown")
        assert ";base64," in entry["url"]

    def test_small_translation_relayed_inline_and_attached(self, monkeypatch):
        # v2.4: small results are relayed inline (default behaviour) AND
        # attached as a downloadable file alongside the reply.
        module = _load_tool(monkeypatch)
        calls = []
        small = "This is a short translation."
        _make_capturing_backend(module, calls, translation=small)
        events, emit = self._events()
        out = _run(
            module,
            text="A source sentence that is long enough to split into parts for sure.",
            __messages__=[],
            __files__=[],
            __event_emitter__=emit,
        )
        # The model still gets the full text to relay verbatim.
        assert out == small
        file_event = [e for e in events if e.get("type") == "chat:message:files"]
        assert len(file_event) == 1
        entry = file_event[0]["data"]["files"][0]
        assert entry["content"] == small
        assert entry["type"] == "file"
        assert entry["url"].startswith("data:text/markdown")

    def test_small_translation_no_file_when_valve_off(self, monkeypatch):
        module = _load_tool(monkeypatch)
        calls = []
        small = "This is a short translation."
        _make_capturing_backend(module, calls, translation=small)
        events, emit = self._events()
        out = _run(
            module,
            valves={"ALWAYS_ATTACH_FILE": False},
            text="A source sentence that is long enough to split into parts for sure.",
            __messages__=[],
            __files__=[],
            __event_emitter__=emit,
        )
        assert out == small
        assert not [e for e in events if e.get("type") == "chat:message:files"]

    def test_small_translation_stored_as_real_downloadable_file(self, monkeypatch):
        # v2.4 with storage available: even a small translation becomes a real
        # OpenWebUI file, and the tool result stays the relay text.
        module = _load_tool(monkeypatch)
        calls = []
        small = "This is a short translation."
        _make_capturing_backend(module, calls, translation=small)

        uploaded = {}

        class FakeFileForm:
            def __init__(self, **kwargs):
                uploaded["form"] = kwargs

        class FakeRecord:
            @property
            def id(self):
                return uploaded["form"]["id"]

            def model_dump(self):
                return {
                    "id": uploaded["form"]["id"],
                    "filename": uploaded["form"]["filename"],
                    "data": uploaded["form"]["data"],
                    "meta": uploaded["form"]["meta"],
                }

        class FakeFiles:
            @staticmethod
            async def insert_new_file(user_id, form_data, db=None):
                uploaded["user_id"] = user_id
                return FakeRecord()

        class FakeStorage:
            @staticmethod
            def upload_file(fileobj, filename, tags):
                data = fileobj.read()
                uploaded["bytes"] = data
                uploaded["filename"] = filename
                return data, f"/tmp/uploads/{filename}"

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

        events, emit = self._events()
        out = _run(
            module,
            text="A source sentence that is long enough to split into parts for sure.",
            __messages__=[],
            __files__=[],
            __user__={"id": "user-1"},
            __event_emitter__=emit,
        )
        # Relay behaviour is untouched.
        assert out == small
        # Real file stored, entry references the file id (downloadable).
        assert uploaded["bytes"] == small.encode("utf-8")
        assert uploaded["form"]["data"]["content"] == small
        file_id = uploaded["form"]["id"]
        entry = [e for e in events if e.get("type") == "chat:message:files"][0][
            "data"
        ]["files"][0]
        assert entry["id"] == file_id
        assert entry["url"] == file_id
        assert not entry["url"].startswith("data:")

    def test_error_never_attached(self, monkeypatch):
        module = _load_tool(monkeypatch)
        upstream_error = "vLLM rejected the request (400): prompt too long. " * 300
        calls = []

        class _BoomResp:
            status_code = 502
            text = upstream_error

            def json(self):
                raise RuntimeError("no json on errors")

        def fake_post(url, json=None, timeout=None):
            calls.append(dict(json))
            return _BoomResp()

        module.requests.post = fake_post
        events, emit = self._events()
        out = _run(
            module,
            text="A source sentence that is long enough to split into parts for sure.",
            __messages__=[],
            __files__=[],
            __event_emitter__=emit,
        )
        assert out == f"Gateway Error (502): {upstream_error}"
        assert not [e for e in events if e.get("type") == "chat:message:files"]

    def test_attachment_persisted_to_chat_message(self, monkeypatch):
        module = _load_tool(monkeypatch)
        calls = []
        big = "long translation body " * 700
        _make_capturing_backend(module, calls, translation=big)

        persisted = {}

        class FakeChats:
            @staticmethod
            async def add_message_files_by_id_and_message_id(chat_id, message_id, files):
                persisted["chat_id"] = chat_id
                persisted["message_id"] = message_id
                persisted["files"] = files
                return files

        chats_mod = types.ModuleType("open_webui.models.chats")
        chats_mod.Chats = FakeChats
        models_mod = types.ModuleType("open_webui.models")
        open_webui_mod = types.ModuleType("open_webui")
        monkeypatch.setitem(sys.modules, "open_webui", open_webui_mod)
        monkeypatch.setitem(sys.modules, "open_webui.models", models_mod)
        monkeypatch.setitem(sys.modules, "open_webui.models.chats", chats_mod)

        events, emit = self._events()
        _run(
            module,
            text="translate this",
            __messages__=[
                {"role": "user", "content": "translate this"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "c", "function": {"name": "translate", "arguments": "{}"}}],
                },
            ],
            # Readable file: its name feeds the attachment filename.
            __files__=[
                {
                    "id": "f1",
                    "name": "paper.md",
                    "type": "file",
                    "url": "f1",
                    "file": {"data": {"content": "A source document with plenty of real words to translate properly."}},
                }
            ],
            __chat_id__="chat-123",
            __message_id__="msg-456",
            __event_emitter__=emit,
        )
        assert persisted["chat_id"] == "chat-123"
        assert persisted["message_id"] == "msg-456"
        assert persisted["files"][0]["name"] == "paper.translated.fa.md"
        assert persisted["files"][0]["content"] == big

    def test_attachment_stored_as_real_downloadable_file(self, monkeypatch):
        # With OpenWebUI storage available, the translation must become a real
        # file (Storage + Files DB) so the chat chip can download it via
        # /files/{id}/content — a bare data-URI entry has no download path.
        module = _load_tool(monkeypatch)
        calls = []
        big = "long translation body " * 700
        _make_capturing_backend(module, calls, translation=big)

        uploaded = {}

        class FakeFileForm:
            def __init__(self, **kwargs):
                uploaded["form"] = kwargs

        class FakeRecord:
            @property
            def id(self):
                return uploaded["form"]["id"]

            def model_dump(self):
                return {
                    "id": uploaded["form"]["id"],
                    "filename": uploaded["form"]["filename"],
                    "data": uploaded["form"]["data"],
                    "meta": uploaded["form"]["meta"],
                }

        class FakeFiles:
            @staticmethod
            async def insert_new_file(user_id, form_data, db=None):
                uploaded["user_id"] = user_id
                return FakeRecord()

        class FakeStorage:
            @staticmethod
            def upload_file(fileobj, filename, tags):
                data = fileobj.read()
                uploaded["bytes"] = data
                uploaded["filename"] = filename
                uploaded["tags"] = tags
                return data, f"/tmp/uploads/{filename}"

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

        events, emit = self._events()
        _run(
            module,
            text="A source sentence that is long enough to split into parts for sure.",
            __messages__=[],
            __files__=[],
            __user__={"id": "user-1"},
            __event_emitter__=emit,
        )
        # Stored in file storage with the real translation bytes.
        assert uploaded["bytes"] == big.encode("utf-8")
        assert uploaded["filename"].endswith("_translation-fa.md")
        file_id = uploaded["form"]["id"]
        assert uploaded["tags"]["OpenWebUI-File-Id"] == file_id
        assert uploaded["form"]["data"]["content"] == big
        assert uploaded["form"]["data"]["status"] == "completed"
        assert uploaded["form"]["meta"]["content_type"] == "text/markdown"
        assert uploaded["user_id"] == "user-1"
        # The chat entry references the file id (downloadable), not a data URI.
        file_event = [e for e in events if e.get("type") == "chat:message:files"]
        entry = file_event[0]["data"]["files"][0]
        assert entry["id"] == file_id
        assert entry["url"] == file_id
        assert not entry["url"].startswith("data:")
        assert entry["file"]["meta"]["content_type"] == "text/markdown"
