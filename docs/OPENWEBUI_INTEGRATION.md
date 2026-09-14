# TranslateGemma 27B OpenWebUI (v0.11.3) Integration Guide

This guide details how to integrate the fine-tuned and merged **TranslateGemma-27B** model (served with vLLM on an offline host) into **OpenWebUI v0.11.3**, with complete support for:
1. **Official multimodal template compliance** and stop-token preservation (`[1, 106]`).
2. **Context resolution** (e.g., *"translate this"*, *"translate the above message"*).
3. **Document & file translation** (handling full file uploads with structure-preserving chunked batching).
4. **Interactive `/translate` slash command** with default language fallback.

> **Verified against the OpenWebUI v0.11.3 source.** The tool in section 2 is
> version 2.4.0 and is the canonical copy kept in `scripts/openwebui_tool.py`
> (tests: `tests/test_openwebui_tool.py`). It is written for how v0.11.3
> *actually* delivers file content to tools (see section 5), which differs from
> older OpenWebUI versions:
> - With tools enabled, the user message `content` is a **list of content
>   parts**, not a string.
> - Uploaded files reach the message as `<attached_files><file id="..."/></attached_files>`
>   metadata tags; **full-context** uploads additionally carry the whole
>   document inside a RAG `<context><source ...>BODY</source></context>` block.
> - At tool-call time the last message is the assistant `tool_call`, not the
>   user message.
> - The built-in `view_file` helper caps reads at 10,000 chars by default —
>   the tool therefore reads file content itself, bypassing that cap.

---

## 1. Architecture Overview

TranslateGemma requires three mandatory inputs for any translation:
1. **`text`**: The input segment or document to translate.
2. **`source_lang`**: Source language ISO code (e.g., `en`, `fa`, `de`, `fr`, `ru`).
3. **`target_lang`**: Target language ISO code (e.g., `fa`, `en`, `de`, `fr`, `ru`).

Standard OpenWebUI chat endpoints pass only a flat string (`{"role": "user", "content": "..."}`). Connecting directly to raw vLLM fails for three reasons:
- **Official Schema Invariant**: TranslateGemma expects a multimodal-style content list:
  ```json
  [{"type": "text", "source_lang_code": "en", "target_lang_code": "fa", "text": "..."}]
  ```
- **SFT Jinja Prefix Indentation**: As documented in `docs/2026-08-10_adapter_degeneration_analysis.md`, the fine-tuned adapter requires Jinja indentation (`<start_of_turn>model\n\n        `). Calling `add_generation_prompt=True` drops this prefix and causes degenerative repetition.
- **Turn-Ending Stop Tokens**: `config.json` sets `eos_token_id: 1` (`<eos>`), but SFT teaches the model to end turns with `<end_of_turn>` (`106`). If `stop_token_ids: [1, 106]` is not explicitly supplied, generation loops indefinitely.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           OpenWebUI v0.11.3                             │
│                                                                         │
│  User uploads document or types: /translate [text]                      │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ OpenWebUI Tool: translategemma_tool (Workspace > Tools, v2.5.0)   │  │
│  │ - Self-resolves the text: RAG <context> block of the current      │  │
│  │   message, __files__ full content (in-process, no 10k cap),       │  │
│  │   <attached_files> ids, or conversation history — the base model  │  │
│  │   never has to copy document text into the tool argument          │  │
│  │ - Disambiguates multi-source chats: current message wins; a       │  │
│  │   previous translation is never re-selected as a source           │  │
│  │ - Strips non-translatable OCR artifacts (base64 images, HTML)     │  │
│  │ - Every result attaches to the chat as a file (downloadable)      │  │
│  │ - Structure-preserving split: split_sentences=True (>250 chars)   │  │
│  │ - Sends live progress status chips via __event_emitter__          │  │
│  └─────────────────────────────────┬─────────────────────────────────┘  │
└────────────────────────────────────┼────────────────────────────────────┘
                                     │ HTTP POST (Internal Docker Network)
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ translategemma-api Gateway (:8000/translate)                            │
│ - Validates inputs (source_lang, target_lang, text)                     │
│ - Structure-preserving chunking: blocks -> lines -> sentences           │
│ - Renders official SFT Jinja template via prompting.py                  │
│ - Resolves stop token IDs: [1, 106] (<end_of_turn>)                     │
│ - Sets greedy decoding: temperature=0.0, top_p=1.0, top_k=-1            │
│ - Encodes to token IDs and batches requests to vLLM /v1/completions     │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ Batched Token IDs + stop_token_ids
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ translategemma-vllm (:8000)                                             │
│ - vLLM engine serving merged translategemma-27b weights                 │
│ - Decodes tokens using continuous batching & PagedAttention             │
└─────────────────────────────────────────────────────────────────────────┘
```

*Optional deterministic mode: a **Pipe** pseudo-model (`Workspace > Functions`,
section 2b) runs the same flow with **no base model at all** — the translation
IS the assistant reply, so nothing is ever relayed or re-emitted.*

---

## 2. OpenWebUI Tool Implementation (`Workspace > Tools`)

In OpenWebUI v0.11.3, custom extensibility uses **Tools** (`class Tools:`).

This v2.5.0 tool implementation includes:
- **Self-served content resolution**: The tool reads `__messages__` and `__files__` itself and finds the text to translate. The base model only says *"translate"* — it does **not** retype document content into the tool argument. This removes the two failure modes of v1:
  - *Long documents got shortened*: the model could not reproduce 10k+ chars verbatim in a tool argument (and the built-in `view_file` caps at 10,000 chars by default). The tool now sends the complete text to the gateway.
  - *Mixed-up sources*: with two or more sources in one chat, v1's "last non-empty message" fallback could grab the previous translation or an old source. v2 resolves in a deterministic order (below) and never selects a prior `translate` result as a source.
- **v0.11.3 message-format aware**: handles list-of-parts message content, `<attached_files>`/`<file id=...>` tags, RAG `<context>`/`<source>` blocks, and the fact that the last message at call time is the assistant `tool_call`.
- **OCR artifact stripping (v2.2)**: before the gateway call, the resolved text is cleaned of non-translatable payloads — base64 image data-URIs, markdown images (alt text kept), HTML comments and tags (inner text kept). OCR-generated markdown embeds image data inline; "translating" it produced garbage and inflated the token count until the relay back through the base model overflowed its context window. A document that is images/binary only now returns an explicit "no translatable text" error.
- **A downloadable file for every translation (v2.2 attachment, storage in v2.3, always-on in v2.4)**: every successful translation is attached to the chat as a file via OpenWebUI's own `chat:message:files` mechanism (persisted to the message via `Chats.add_message_files_by_id_and_message_id`) and stored as a **real OpenWebUI file** (`Storage.upload_file` + `Files.insert_new_file`, no RAG processing), so the chat chip previews the markdown and its name is a working link to `/files/{id}/content` — i.e. the file can actually be **downloaded**; the file also appears in Workspace → Files. For translations longer than the `RELAY_MAX_CHARS` valve (default 8000 chars) the file is the *only* delivery path: the base model receives a one-line acknowledgement instruction instead of the document-sized text (it never reads or re-emits it). For shorter results the file is attached **alongside** the normal reply, which the model still relays verbatim (`ALWAYS_ATTACH_FILE` valve, default on, restores the attach-only-when-large behaviour when off). Without storage access the attachment degrades to a content-only (preview-only) entry.
- **Structure-preserving chunking**: For inputs exceeding 250 characters or containing newlines, it automatically sets `split_sentences: True`. The gateway splits the document into **blocks** (paragraphs, headings, list/table lines) and keeps the original separators as data: a block that fits the token budget translates as a whole unit (also the best unit for MT quality), an over-budget block descends to lines (structural blocks) or `pysbd` sentences with their original gaps (prose), and a unit still over budget is hard-sliced at token boundaries. Code fences (```/~~~) and `$$` math blocks pass through **verbatim** — they never reach the model. Rejoining is exact (`translated unit + original separator`), so the output keeps the input's markdown structure instead of collapsing to one line.
- **User & Global Valves**: Preserves default language fallbacks (`en` ➔ `fa`).
- **Shared deterministic core (v2.5.0)**: resolution, sanitization, the gateway call, and file delivery live in one canonical core (`scripts/translation_core.py`) shared with the section-2b pipe. Both deployables are **generated** by `scripts/build_deployables.py` (which also re-splices these doc blocks to stay byte-identical) — edit the core or a `*.front` file, never the generated `.py`.

**Resolution order** (first match wins):
1. `text` argument, when it is real content (not a reference like `"this"`). If the model instead dumps the whole current message, the `<context>` block inside it is mined.
2. The `<context>` block of the **current** user message (full-context uploads — PDF, markdown, all file types).
3. Content of files attached to the **current** message (`__files__`): embedded `file.data.content` first, then the file record's `data.content` read in-process (full text, no 10k viewer cap).
4. `<file id="..."/>` ids referenced in the current message, fetched in-process.
5. The current message's own text, minus the reference/instruction line (covers *"translate this: …"* and the `/translate` modal).
6. The most recent earlier `<context>` block in the conversation.
7. The last substantive message in history (user **or** assistant — so *"OCR extract → translate this"* works), skipping tool results and messages that merely relay a previous `translate` result.

Navigate in OpenWebUI to **Workspace → Tools → Add Tool (`+`)**, name it `translategemma_tool`, and paste (identical to `scripts/openwebui_tool.py`):

```python
"""
title: TranslateGemma Translation Tool (Context & File Aware)
author: TranslateGemma Team
description: High-accuracy translation using finetuned TranslateGemma 27B. Self-resolves the text to translate from uploaded documents, conversation context, and inline text — no manual copy/paste by the model. Never falls back to the system prompt. Strips non-translatable artifacts (base64 image blobs, HTML) before translating. Every successful translation is attached to the chat as a downloadable file (stored in OpenWebUI file storage); large translations are delivered as that file alone, with a short acknowledgement instead of making the base model re-emit them.
version: 2.5.0
license: MIT
requirements: requests, pydantic
"""
"""Shared core for the OpenWebUI translation front ends.

This module holds everything the OpenWebUI *tool* and the *pipe* have in
common: deterministic source resolution (files, context blocks, message
text, history — never the system prompt), OCR artifact sanitization, the
gateway/vLLM call, and the downloadable-file delivery (OpenWebUI file
storage + chat attachment). It is the canonical implementation; the two
deployables are single self-contained files (OpenWebUI execs them as
standalone modules, so they cannot import this file) and are generated by
scripts/build_deployables.py, which inlines this source into each front end.

Imports `requests` at module level (a declared requirement of both
deployables) and only touches `open_webui.*` lazily inside functions, so it
also runs in a plain test environment.
"""

import asyncio
import base64
import os
import re

import requests

__all__ = [
    "TEXT_LIKE_EXT",
    "TranslationClient",
    "attached_file_ids",
    "context_blocks",
    "current_remainder",
    "file_text",
    "file_text_by_id",
    "is_referential",
    "last_user_message",
    "msg_text",
    "parse_command",
    "resolve_languages",
    "resolve_source",
    "sanitize_document",
    "store_translation_file",
    "strip_wrappers",
    "translation_relay_indices",
    "upload_translation_file",
]

TEXT_LIKE_EXT = (
    ".md", ".markdown", ".txt", ".text", ".csv", ".tsv", ".json", ".html",
    ".htm", ".xml", ".rst", ".yaml", ".yml", ".ini", ".log",
)
_MAX_RAW_READ_BYTES = 20 * 1024 * 1024

# Non-translatable artifacts. OCR-generated markdown (pymupdf4llm, docling,
# tesseract post-processing) routinely embeds full base64 image payloads
# inline. Sent to the MT model they are "translated" into garbage, blow the
# token count up by orders of magnitude, and the inflated translation then
# overflows the base model's context on the way back. Stripped before the
# gateway call, in the tool that knows it is handling documents.
_DATA_URI_RE = re.compile(r"data:[A-Za-z0-9/+.=-]+;base64,[A-Za-z0-9+/=\s]+")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>]*>")
_BLANK_RUN_RE = re.compile(r"\n{3,}")
# Collapse the space runs left where tags/blobs were stripped, without
# touching line-start indentation (code blocks).
_SPACE_RUN_RE = re.compile(r"(?<=\S)[ \t]{2,}(?=\S)")

_REFERENTIAL = {
    "",
    "this",
    "that",
    "it",
    "these",
    "those",
    "above",
    "the above",
    "the above message",
    "the above text",
    "the text",
    "this text",
    "that text",
    "previous",
    "previous message",
    "this file",
    "that file",
    "the file",
    "attached file",
    "the document",
    "this document",
    "translate",
    "translate this",
    "translate that",
    "translate it",
    "translate the above",
    "translate the above message",
    "translate the above text",
    "translate the text",
    "translate this text",
    "translate previous",
    "translate previous message",
    "translate this file",
    "translate that file",
    "translate the file",
    "translate file",
    "translate the document",
    "translate this document",
    "ترجمه",
    "ترجمه کن",
    "این را ترجمه کن",
    "متن بالا را ترجمه کن",
    "بالا را ترجمه کن",
    "این متن را ترجمه کن",
    "این فایل را ترجمه کن",
    "فایل را ترجمه کن",
    "متن را ترجمه کن",
    "متن بالا",
}

_REFERENCE_PREFIXES = (
    "translate this",
    "translate that",
    "translate the above",
    "translate the text",
    "translate file",
    "translate the file",
    "translate the document",
    "translate previous",
    "translate attached",
)

_INSTRUCTION_LINE_RE = re.compile(
    r"^(please\s+)?(translate|convert|translate the following|traduis|übersetze|ترجمه)",
    re.IGNORECASE,
)


# ------------------------------------------------------------ message utils


def msg_text(message: dict) -> str:
    """Message content as text, for both plain strings and content-part lists."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return "\n".join(parts)
    return ""


def last_user_message(messages) -> tuple:
    """(index, text) of the most recent user message, or (-1, '')."""
    if not messages:
        return -1, ""
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, dict) and msg.get("role") == "user":
            return i, msg_text(msg)
    return -1, ""


def context_blocks(text: str) -> list:
    """All OpenWebUI RAG context blocks in `text`, inner <source> tags stripped."""
    blocks = []
    for m in re.finditer(r"<context>(.*?)</context>", text or "", re.DOTALL):
        inner = re.sub(r"</?source[^>]*>", "", m.group(1))
        inner = inner.strip()
        if inner:
            blocks.append(inner)
    return blocks


def strip_wrappers(text: str) -> str:
    """Remove file-tag wrappers an LLM may have copied verbatim into `text`."""
    text = re.sub(r"<attached_files>.*?</attached_files>", "", text or "", flags=re.DOTALL)
    text = re.sub(r"<knowledge>.*?</knowledge>", "", text, flags=re.DOTALL)
    return text.strip()


def attached_file_ids(text: str) -> list:
    ids = []
    for m in re.finditer(r"<file\b[^>]*?/?>", text or ""):
        idm = re.search(r'id="([^"]+)"', m.group(0))
        if idm:
            ids.append(idm.group(1))
    return ids


def is_referential(text: str) -> bool:
    clean = (text or "").strip().lower()
    if clean in _REFERENTIAL:
        return True
    # Prefix matches count only for short phrases: an explicitly passed
    # long argument is content, even one starting with "translate this".
    return any(clean.startswith(prefix) for prefix in _REFERENCE_PREFIXES) and len(clean) < 30


def parse_command(text: str) -> tuple:
    """Split a /translate command into (body, source_lang, target_lang)."""
    clean = (text or "").strip()
    m = re.match(
        r"^/translate\s+([a-zA-Z]{2,5}(?:-[a-zA-Z0-9]+)?)"
        r"(?:->|:|\s+)"
        r"([a-zA-Z]{2,5}(?:-[a-zA-Z0-9]+)?)\s+(.+)$",
        clean,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        return m.group(3).strip(), m.group(1).lower(), m.group(2).lower()
    m = re.match(r"^/translate\s+(.+)$", clean, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip(), None, None
    return clean, None, None


def current_remainder(current_text: str, reference: str) -> str:
    """Text carried by the current user message beyond the reference/instruction."""
    t = current_text or ""
    t = re.sub(r"<attached_files>.*?</attached_files>", "", t, flags=re.DOTALL)
    t = re.sub(r"<context>.*?</context>", "", t, flags=re.DOTALL)
    t = re.sub(r"<knowledge>.*?</knowledge>", "", t, flags=re.DOTALL)

    kept = []
    for line in t.splitlines():
        s = line.strip()
        if (
            not kept
            and s
            and len(s) < 120
            and _INSTRUCTION_LINE_RE.match(s)
        ):
            continue
        kept.append(line)
    t = "\n".join(kept).strip()

    ref = (reference or "").strip()
    if ref and t == ref:
        t = ""
    elif ref and ref.isascii() and len(t) > len(ref):
        # trailing "translate this" style addendum (ASCII refs only:
        # whitespace token boundaries do not exist in e.g. Persian)
        t = re.sub(re.escape(ref) + r"[\s,:：]*$", "", t, flags=re.IGNORECASE).strip()

    if not t or is_referential(t):
        return ""
    return t if len(t) >= 8 else ""


def translation_relay_indices(messages: list) -> set:
    """Indices of assistant messages that merely relay a `translate` tool result.

    Those are translations, not sources: falling back onto them would
    translate the previous translation.
    """
    translate_call_ids = set()
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            fn = (tc.get("function") or {}).get("name", "")
            if fn == "translate" and tc.get("id"):
                translate_call_ids.add(tc.get("id"))
    if not translate_call_ids:
        return set()
    relays = set()
    for i, msg in enumerate(messages):
        if isinstance(msg, dict) and msg.get("role") == "tool" and msg.get("tool_call_id") in translate_call_ids:
            for j in range(i + 1, len(messages)):
                if messages[j].get("role") == "assistant":
                    relays.add(j)
                    break
    return relays


# ------------------------------------------------------------------ resolve


def resolve_from_messages(
    clean: str,
    messages,
    current_idx: int,
    current_text: str,
    attached_text: str,
    include_assistant_history: bool = True,
) -> tuple:
    """Deterministic resolution of the text to translate.

    Returns (text, source) where source labels where the text came from.
    Order: current message context block -> attached-file text ->
    current message remainder -> earlier context blocks -> last
    substantive history message. Never returns the system prompt, tool
    results, or a prior `translate` relay as a source.

    `include_assistant_history=False` (the pipe's mode): the history
    fallback then only scans earlier USER messages. Pipe replies are plain
    assistant messages with no tool-call marker to identify them, so
    including assistant text would let "translate this" latch onto the
    previous translation and re-translate it.
    """
    if messages:
        ctx = context_blocks(current_text)
        if ctx:
            return ctx[-1], "document context"

    if attached_text.strip():
        return attached_text.strip(), "attached file"

    remainder = current_remainder(current_text, clean)
    if remainder:
        return remainder, "message text"

    if messages:
        limit = current_idx if current_idx >= 0 else len(messages)
        earlier = []
        for msg in messages[:limit]:
            if isinstance(msg, dict) and msg.get("role") == "user":
                earlier.extend(context_blocks(msg_text(msg)))
        if earlier:
            return earlier[-1], "earlier document"

        relays = translation_relay_indices(messages)
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if not isinstance(msg, dict):
                continue
            if i == current_idx or i in relays:
                continue
            role = msg.get("role")
            # System/developer prompts are instructions, not sources:
            # falling back onto one is how a chat ends up "translating"
            # the system prompt.
            if role in ("system", "developer", "function", "tool"):
                continue
            if role == "assistant" and (
                not include_assistant_history or msg.get("tool_calls")
            ):
                continue
            content = msg_text(msg).strip()
            content = re.sub(
                r"<attached_files>.*?</attached_files>", "", content, flags=re.DOTALL
            ).strip()
            content = re.sub(
                r"<context>.*?</context>", "", content, flags=re.DOTALL
            ).strip()
            if not content:
                continue
            if content.startswith("/translate"):
                continue
            # Referential-only messages ("translate this") carry no source;
            # messages that carry a source alongside the reference do.
            remainder = current_remainder(content, "")
            if not remainder:
                continue
            return remainder, "conversation history"
    return "", ""


async def resolve_source(text_arg: str, messages, files, include_assistant_history: bool = True) -> dict:
    """The full source-resolution pipeline shared by the tool and the pipe.

    `text_arg` is the tool's `text` argument, or "" for the pipe (which
    always resolves from the conversation itself). Returns a dict with
    'text', 'source', 'file_names', 'unreadable_files', 'cmd_src',
    'cmd_tgt'.
    """
    # 1. Attached files on the CURRENT message (full text, no viewer caps).
    file_parts = []
    file_names = []
    unreadable_files = []
    for item in files or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") is not None and item.get("type") not in ("file", "doc", "text", "note"):
            continue
        ft = await file_text(item)
        label = str(item.get("name") or item.get("id") or "attached file")
        if ft and ft.strip():
            file_parts.append(ft.strip())
            file_names.append(label)
        else:
            unreadable_files.append(label)
    attached_text = "\n\n".join(file_parts)

    # 2. Parse /translate command for language overrides.
    clean, cmd_src, cmd_tgt = parse_command(text_arg)

    # 3. Resolve the text. An explicit non-referential `text` wins, unless
    #    it is the model dumping the whole message (then mine the context).
    if not is_referential(clean) and clean:
        ctx = context_blocks(clean)
        clean = strip_wrappers(clean)
        resolved = ctx[-1] if ctx else clean
        source = "document context" if ctx else "provided text"
        if not ctx:
            # Prompt-modal templates wrap the text in a short instruction
            # line; strip it only when the body is blank-line-separated,
            # so a real sentence that happens to start with "translate"
            # is never touched.
            m = re.match(r"^([^\n]{0,119})\n\n(.+)$", resolved, re.DOTALL)
            if m and _INSTRUCTION_LINE_RE.match(m.group(1).strip()):
                resolved = m.group(2)
        if not resolved.strip() and attached_text.strip():
            resolved, source = attached_text, "attached file"
    else:
        current_idx, current_text = last_user_message(messages)
        # attached-file ids referenced in the current message
        if not attached_text.strip():
            ids = attached_file_ids(current_text)
            fetched = []
            for fid in ids:
                t = await file_text_by_id(fid)
                if t and t.strip():
                    fetched.append(t.strip())
                else:
                    unreadable_files.append(fid)
            if fetched:
                attached_text = "\n\n".join(fetched)
        resolved, source = resolve_from_messages(
            clean,
            messages,
            current_idx,
            current_text,
            attached_text,
            include_assistant_history=include_assistant_history,
        )

    return {
        "text": (resolved or "").strip(),
        "source": source,
        "file_names": file_names,
        "unreadable_files": unreadable_files,
        "cmd_src": cmd_src,
        "cmd_tgt": cmd_tgt,
    }


# --------------------------------------------------------------- sanitise


def sanitize_document(text: str) -> str:
    """Remove non-translatable artifacts from a document before translation.

    Order matters: data URIs first (they can contain anything, including
    angle brackets that would confuse the tag stripping), then markdown
    images (keeping the alt text, which *is* translatable), then HTML
    comments and tags (keeping inner text). Collapses blank runs. Returns
    the cleaned text; may be empty when the input was only images/binary.
    """
    if not text:
        return ""
    text = _DATA_URI_RE.sub(" ", text)
    text = _MD_IMAGE_RE.sub(lambda m: m.group(1).strip(), text)
    text = _HTML_COMMENT_RE.sub(" ", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _BLANK_RUN_RE.sub("\n\n", text)
    text = _SPACE_RUN_RE.sub(" ", text)
    return text.strip()


# ------------------------------------------------------------ file text


async def file_record(file_id: str):
    try:
        from open_webui.models.files import Files

        return await Files.get_file_by_id(str(file_id))
    except Exception:
        return None


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


async def raw_disk_text(record) -> str:
    """Naive text read for text-like files whose extraction is pending or failed.

    The raw upload bytes are on disk as soon as the upload completes, so
    this keeps .md/.txt usable even when the extraction pipeline has not
    written ``data.content`` yet (or failed on it). Binary formats are
    left alone: guessing at a PDF byte stream is worse than an honest error.
    """
    try:
        name = str(getattr(record, "filename", "") or "")
        meta = getattr(record, "meta", None)
        content_type = str(meta.get("content_type") or "") if isinstance(meta, dict) else ""
        text_like = name.lower().endswith(TEXT_LIKE_EXT) or "text/" in content_type
        if not text_like:
            return ""
        rel_path = getattr(record, "path", None)
        if not rel_path:
            return ""
        from open_webui.storage.provider import Storage

        abs_path = await asyncio.to_thread(Storage.get_file, rel_path)
        if not abs_path or not os.path.isfile(abs_path):
            return ""
        if os.path.getsize(abs_path) > _MAX_RAW_READ_BYTES:
            return ""
        raw = await asyncio.to_thread(_read_bytes, abs_path)
        for enc in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            return ""
        if "\x00" in text[:4096]:
            return ""
        return text.strip()
    except Exception:
        return ""


def record_content(record) -> str:
    if record is not None:
        data = getattr(record, "data", None)
        if isinstance(data, dict) and data.get("content"):
            return str(data["content"])
    return ""


async def file_text(item: dict) -> str:
    """Full extracted text of one attached-file item, no viewer char caps."""
    if not isinstance(item, dict):
        return ""
    file_obj = item.get("file")
    if isinstance(file_obj, dict):
        data = file_obj.get("data")
        if isinstance(data, dict) and data.get("content"):
            return str(data["content"])
    file_id = item.get("id") or item.get("url")
    if not file_id or str(file_id).startswith(("http://", "https://", "data:")):
        return ""
    record = await file_record(str(file_id))
    content = record_content(record)
    if content:
        return content
    return await raw_disk_text(record)


async def file_text_by_id(file_id: str) -> str:
    record = await file_record(file_id)
    content = record_content(record)
    if content:
        return content
    return await raw_disk_text(record)


# ------------------------------------------------------------- languages


def resolve_languages(source_lang, target_lang, user, default_src, default_tgt) -> tuple:
    """explicit args > user valves > system defaults, all lowercased.

    `user['valves']` arrives as a dict in tests and as an instance of the
    front end's UserValves class in production (OpenWebUI builds it with
    `module.UserValves(**stored)`), so both shapes are read defensively —
    pydantic v2 models are not mappings, and `**instance` would silently
    drop the user's valves.
    """
    src = (source_lang or "").strip().lower()
    tgt = (target_lang or "").strip().lower()

    user_valves = user.get("valves") if isinstance(user, dict) else None

    def _valve(name: str) -> str:
        if user_valves is None:
            return ""
        if isinstance(user_valves, dict):
            return str(user_valves.get(name) or "")
        return str(getattr(user_valves, name, "") or "")

    if not src:
        src = _valve("source_lang").strip().lower()
    if not tgt:
        tgt = _valve("target_lang").strip().lower()

    if not src:
        src = (default_src or "").strip().lower()
    if not tgt:
        tgt = (default_tgt or "").strip().lower()
    return src, tgt


# ----------------------------------------------------------------- client


class TranslationClient:
    """The gateway (or direct vLLM) call. Synchronous: the caller decides
    whether to run it in the event loop or a worker thread."""

    def __init__(
        self,
        backend_mode: str = "gateway",
        gateway_url: str = "http://translategemma-api:8000/translate",
        vllm_url: str = "http://translategemma-vllm:8000/v1/chat/completions",
        vllm_model_name: str = "model",
        max_new_tokens: int = 512,
        timeout_seconds: int = 300,
    ):
        self.backend_mode = backend_mode
        self.gateway_url = gateway_url
        self.vllm_url = vllm_url
        self.vllm_model_name = vllm_model_name
        self.max_new_tokens = max_new_tokens
        self.timeout_seconds = timeout_seconds

    def translate(self, text: str, src: str, tgt: str, split_sentences: bool) -> str:
        """One translation. Returns the text, or an 'Error'/'Gateway
        Error'/'vLLM Error' string on failure — the caller decides what to
        do with errors (never attach a file for them)."""
        if self.backend_mode == "gateway":
            payload = {
                "text": text,
                "source_lang": src,
                "target_lang": tgt,
                "max_new_tokens": self.max_new_tokens,
                "split_sentences": split_sentences,
            }
            resp = requests.post(
                self.gateway_url,
                json=payload,
                timeout=self.timeout_seconds,
            )
            if resp.status_code != 200:
                return f"Gateway Error ({resp.status_code}): {resp.text}"
            return resp.json().get("translation", "")
        payload = {
            "model": self.vllm_model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "source_lang_code": src,
                            "target_lang_code": tgt,
                            "text": text,
                        }
                    ],
                }
            ],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": self.max_new_tokens,
            "stop_token_ids": [1, 106],
        }
        resp = requests.post(
            self.vllm_url,
            json=payload,
            timeout=self.timeout_seconds,
        )
        if resp.status_code != 200:
            return f"vLLM Error ({resp.status_code}): {resp.text}"
        choices = resp.json().get("choices", [])
        return (
            choices[0].get("message", {}).get("content", "")
            if choices
            else "No translation returned."
        )


# ------------------------------------------------------------ attachments


def build_file_name(file_names: list, tgt: str) -> str:
    name = f"translation-{tgt}.md"
    if file_names:
        base = str(file_names[0]).rsplit(".", 1)[0]
        if base and not base.startswith(("http", "data:")):
            name = f"{base}.translated.{tgt}.md"
    return name


async def upload_translation_file(translation: str, name: str, user_id: str):
    """Store the translation in OpenWebUI's own file storage.

    Returns the FileModel record, or None when storage is unavailable
    (the caller then falls back to a content-only attachment). No RAG
    processing happens — the translation is never embedded into any
    knowledge collection, it is a plain downloadable file.
    """
    try:
        import io
        import uuid

        from open_webui.models.files import FileForm, Files
        from open_webui.storage.provider import Storage
    except Exception:
        return None
    try:
        file_id = str(uuid.uuid4())
        tags = {
            "OpenWebUI-User-Id": str(user_id),
            "OpenWebUI-File-Id": file_id,
        }
        contents, file_path = await asyncio.to_thread(
            Storage.upload_file,
            io.BytesIO(translation.encode("utf-8")),
            f"{file_id}_{name}",
            tags,
        )
        return await Files.insert_new_file(
            str(user_id),
            FileForm(
                id=file_id,
                filename=name,
                path=file_path,
                data={"content": translation, "status": "completed"},
                meta={
                    "name": name,
                    "content_type": "text/markdown",
                    "size": len(contents),
                    "data": {},
                },
            ),
        )
    except Exception:
        return None


async def store_translation_file(
    translation: str,
    file_names: list,
    tgt: str,
    chat_id,
    message_id,
    event_emitter,
    user,
) -> str:
    """Store the translation as a downloadable chat file. Returns the name.

    When possible the translation is stored as a real OpenWebUI file
    (file storage + DB record): the chat chip's modal then previews the
    markdown and the file name is a link to /files/{id}/content, which
    the browser downloads. Without storage access it degrades to a
    content-only entry (preview only). Persistence and the
    chat:message:files event are best effort — a failure here must not
    fail the translation itself.
    """
    name = build_file_name(file_names, tgt)

    user_id = str((user or {}).get("id") or "")
    record = await upload_translation_file(translation, name, user_id) if user_id else None

    file_entry = {
        "type": "file",
        "name": name,
        "content_type": "text/markdown",
        "size": len(translation.encode("utf-8")),
    }
    if record is not None:
        # url is the file id: FileItem/FileItemModal resolve
        # /files/{id}/content (Content-Disposition: attachment for
        # non-text/plain types -> the browser downloads the file).
        file_entry["id"] = record.id
        file_entry["url"] = record.id
        file_entry["file"] = record.model_dump()
    else:
        encoded = base64.b64encode(translation.encode("utf-8")).decode("ascii")
        file_entry["url"] = f"data:text/markdown;charset=utf-8;base64,{encoded}"
        file_entry["content"] = translation

    # Persist the attachment to the chat message so it survives reloads.
    # Best effort: a failure here must not fail the translation itself.
    chat_id = str(chat_id or "")
    if (
        chat_id
        and str(message_id)
        and not chat_id.startswith(("temp:", "channel:"))
    ):
        try:
            from open_webui.models.chats import Chats

            saved = await Chats.add_message_files_by_id_and_message_id(
                chat_id, str(message_id), [file_entry]
            )
            if saved:
                file_entry = saved[0]
        except Exception:
            pass

    if event_emitter:
        try:
            await event_emitter(
                {"type": "chat:message:files", "data": {"files": [file_entry]}}
            )
        except Exception:
            pass

    return name

from typing import Any, Callable, Optional
from pydantic import BaseModel, Field


class Tools:
    class Valves(BaseModel):
        BACKEND_MODE: str = Field(
            default="gateway",
            description="Backend mode: 'gateway' (translategemma-api gateway) or 'vllm'",
        )
        GATEWAY_URL: str = Field(
            default="http://translategemma-api:8000/translate",
            description="URL to the TranslateGemma FastAPI gateway (/translate)",
        )
        VLLM_URL: str = Field(
            default="http://translategemma-vllm:8000/v1/chat/completions",
            description="URL to vLLM chat completions if querying vLLM directly",
        )
        VLLM_MODEL_NAME: str = Field(
            default="model",
            description="Model identifier served by vLLM",
        )
        DEFAULT_SOURCE_LANG: str = Field(
            default="en",
            description="Default source language ISO code (e.g. en, fa, de, fr, ru)",
        )
        DEFAULT_TARGET_LANG: str = Field(
            default="fa",
            description="Default target language ISO code (e.g. fa, en, de, fr, ru)",
        )
        MAX_NEW_TOKENS: int = Field(
            default=512,
            description="Maximum new tokens per segment (one sentence; 512 is the gateway default and 4x less KV-cache pressure than 2048 on long documents)",
        )
        TIMEOUT_SECONDS: int = Field(
            default=300,
            description="Timeout in seconds for long documents",
        )
        RELAY_MAX_CHARS: int = Field(
            default=8000,
            description=(
                "Translations longer than this many characters are delivered as a "
                "chat file attachment with a short acknowledgement, instead of "
                "being re-emitted by the base model. The base model would "
                "otherwise have to copy the whole translation into its reply, "
                "which overflows its own context window and is slow. 0 disables "
                "the acknowledgement mode (always relay inline)."
            ),
        )
        ALWAYS_ATTACH_FILE: bool = Field(
            default=True,
            description=(
                "Always attach the translation as a downloadable chat file, even "
                "when the result is small enough to be relayed inline — the file "
                "is created alongside the normal reply. Set False to attach a "
                "file only for large results."
            ),
        )

    class UserValves(BaseModel):
        source_lang: str = Field(
            default="",
            description="Personal default source language code (leave blank for system default)",
        )
        target_lang: str = Field(
            default="",
            description="Personal default target language code (leave blank for system default)",
        )

    def __init__(self):
        self.valves = self.Valves()

    # ------------------------------------------------------------- translate

    async def translate(
        self,
        text: str = "",
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        split_sentences: Optional[bool] = None,
        __messages__: Optional[list] = None,
        __files__: Optional[list] = None,
        __user__: Optional[dict] = None,
        __chat_id__: Optional[str] = None,
        __message_id__: Optional[str] = None,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Translates text, documents, or conversation context using finetuned TranslateGemma 27B. ALWAYS translate the ENTIRE content — never a summary or partial excerpt, and never ask the user whether to translate all or part. A file/document attached with no instruction is a complete request to translate it in full (default target: Persian).

        :param text: Text to translate. For uploaded files and references like 'this' or 'the above', leave it empty or pass 'this' — the tool locates the document/message itself. Pass inline/pasted text here verbatim. `/translate [src] [tgt] ...` is also accepted.
        :param source_lang: Source language code (e.g. 'en', 'fa', 'de', 'fr', 'ru').
        :param target_lang: Target language code (e.g. 'fa', 'en', 'de', 'fr', 'ru').
        :param split_sentences: Whether to chunk long documents structure-preservingly (blocks/lines/sentences) for concurrent batching.
        :return: Translated text. Every successful translation is attached to the chat as a downloadable file (ALWAYS_ATTACH_FILE valve). Translations longer than the RELAY_MAX_CHARS valve are returned to the model as a short acknowledgement (the attached file carries the content) instead of the full text.
        """
        # 1-3. Deterministic source resolution (files, context blocks, message
        #      text, history) — never the system prompt, never a prior relay.
        info = await resolve_source(text, __messages__, __files__)
        file_names = info["file_names"]
        unreadable_files = info["unreadable_files"]
        if info["cmd_src"]:
            source_lang = info["cmd_src"]
        if info["cmd_tgt"]:
            target_lang = info["cmd_tgt"]

        resolved = info["text"]
        if not resolved:
            if unreadable_files:
                names = ", ".join(unreadable_files[:3])
                return (
                    f"Error: The attached file ({names}) has no readable text yet. "
                    "Its content extraction may still be running or failed — wait a "
                    "moment and send the message again. For files, use the "
                    "'Using Entire Document' upload mode (Settings → Interface → "
                    "File → Default Upload Mode) so the full text is available."
                )
            return (
                "Error: No text or file content found to translate. "
                "Upload a file (with full-context upload mode), paste the text, "
                "or point at a specific earlier message."
            )

        # OCR markdown and scraped pages carry non-translatable payloads
        # (base64 image data, HTML). Translating them produces garbage and
        # inflates the token count until the relay back through the base
        # model overflows its context window.
        resolved = sanitize_document(resolved)
        if not resolved:
            return (
                "Error: The content contains no translatable text (only images, "
                "binary data, or markup). Nothing was sent to the translator."
            )

        src, tgt = resolve_languages(
            source_lang,
            target_lang,
            __user__,
            self.valves.DEFAULT_SOURCE_LANG,
            self.valves.DEFAULT_TARGET_LANG,
        )

        if split_sentences is None:
            split_sentences = len(resolved) > 250 or "\n" in resolved

        word_count = len(resolved.split())
        mode_desc = "with structure-preserving splitting" if split_sentences else "direct"
        if info["source"] == "attached file" and file_names:
            source_desc = f"file: {', '.join(file_names[:2])}"
        else:
            source_desc = info["source"] or "text"

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Translating {word_count} words [{src} \u279c {tgt}] ({mode_desc}, from {source_desc})...",
                        "done": False,
                    },
                }
            )

        try:
            client = TranslationClient(
                backend_mode=self.valves.BACKEND_MODE,
                gateway_url=self.valves.GATEWAY_URL,
                vllm_url=self.valves.VLLM_URL,
                vllm_model_name=self.valves.VLLM_MODEL_NAME,
                max_new_tokens=self.valves.MAX_NEW_TOKENS,
                timeout_seconds=self.valves.TIMEOUT_SECONDS,
            )
            result = client.translate(resolved, src, tgt, split_sentences)

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Translation complete ({word_count} words).",
                            "done": True,
                        },
                    }
                )
            is_error = result.startswith(("Error:", "Gateway Error", "vLLM Error"))
            if result and not is_error:
                if (
                    self.valves.RELAY_MAX_CHARS > 0
                    and len(result) > self.valves.RELAY_MAX_CHARS
                ):
                    # The base model must not re-emit a document-sized result.
                    name = await store_translation_file(
                        result,
                        file_names,
                        tgt,
                        __chat_id__,
                        __message_id__,
                        __event_emitter__,
                        __user__,
                    )
                    return (
                        f"Translation complete: {word_count} words, {src} to {tgt}. The full "
                        f"translation is already attached to the chat as the file '{name}' and "
                        "is visible to the user. Your reply must be a single short sentence "
                        "acknowledging that the translation is ready and attached (for "
                        "example: 'Done - the full translation is attached.'). NEVER restate, "
                        "quote, summarize, or repeat any of the translated content."
                    )
                if self.valves.ALWAYS_ATTACH_FILE:
                    # Default behaviour: the model relays the full text verbatim,
                    # and the file is attached alongside the reply.
                    await store_translation_file(
                        result,
                        file_names,
                        tgt,
                        __chat_id__,
                        __message_id__,
                        __event_emitter__,
                        __user__,
                    )
            return result

        except Exception as error:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Translation failed: {str(error)}",
                            "done": True,
                        },
                    }
                )
            return f"Translation connection error: {str(error)}"
```

---

## 2b. Deterministic Translation Pipe (optional, `Workspace > Functions`)

The tool above works through the base model: the model decides to call it
and re-emits short results. That is the right shape for *conversational*
translation, but it keeps the base model in the loop — which is where every
relay overflow, chattiness, and input-side context error in section 8 came
from. The pipe removes that loop entirely.

A **pipe function** in OpenWebUI v0.11.3 appears in the model list as a
pseudo-model. When a user selects it, the pipe's `pipe()` handles the whole
request: it resolves the text (files, pasted text, references — the same
deterministic core as the tool), calls the TranslateGemma gateway directly,
and streams the full structured translation back as the assistant reply,
attaching it as a downloadable file. **No chat LLM is called at any point**,
so:

- the document never enters any LLM prompt (no input-side context overflow);
- nothing is ever relayed or re-emitted (no output-side overflow, no
  `max_tokens` concerns on a base model);
- there is no model to ask "full, part, or summary?" — a message to the pipe
  *is* a translation request.

**Install** (alongside the tool; both can coexist — pipe for documents,
tool+model for chat):

1. `Workspace > Functions > + New`, name it e.g. `translategemma_pipe`.
2. Type: it auto-detects the `Pipe` class. Paste the code below (identical
   to `scripts/openwebui_pipe.py`, generated by `scripts/build_deployables.py`).
3. Save. The pipe shows up in the model list; select it for document
   translation chats. `/translate [src] [tgt] ...` still overrides the
   languages; per-user language valves work as with the tool.

Valves: the tool's gateway/language/timeout valves (same defaults) plus
`ATTACH_FILE` (default on — the translation is always part of the reply;
the file is a convenience copy). The pipe has no `RELAY_MAX_CHARS`: there is
no base model to protect.

and paste (identical to `scripts/openwebui_pipe.py`):

```python
"""
title: TranslateGemma Translation Pipe (Deterministic)
author: TranslateGemma Team
description: Select this pipe as the model and every message becomes a translation request. Files, pasted text, and references like "translate this" are resolved deterministically, the document never touches a chat LLM (no context overflow, no chatty replies, nothing re-emitted), the full structured translation is streamed back as the reply, and it is attached as a downloadable file.
version: 1.0.0
license: MIT
requirements: requests, pydantic
"""
"""Shared core for the OpenWebUI translation front ends.

This module holds everything the OpenWebUI *tool* and the *pipe* have in
common: deterministic source resolution (files, context blocks, message
text, history — never the system prompt), OCR artifact sanitization, the
gateway/vLLM call, and the downloadable-file delivery (OpenWebUI file
storage + chat attachment). It is the canonical implementation; the two
deployables are single self-contained files (OpenWebUI execs them as
standalone modules, so they cannot import this file) and are generated by
scripts/build_deployables.py, which inlines this source into each front end.

Imports `requests` at module level (a declared requirement of both
deployables) and only touches `open_webui.*` lazily inside functions, so it
also runs in a plain test environment.
"""

import asyncio
import base64
import os
import re

import requests

__all__ = [
    "TEXT_LIKE_EXT",
    "TranslationClient",
    "attached_file_ids",
    "context_blocks",
    "current_remainder",
    "file_text",
    "file_text_by_id",
    "is_referential",
    "last_user_message",
    "msg_text",
    "parse_command",
    "resolve_languages",
    "resolve_source",
    "sanitize_document",
    "store_translation_file",
    "strip_wrappers",
    "translation_relay_indices",
    "upload_translation_file",
]

TEXT_LIKE_EXT = (
    ".md", ".markdown", ".txt", ".text", ".csv", ".tsv", ".json", ".html",
    ".htm", ".xml", ".rst", ".yaml", ".yml", ".ini", ".log",
)
_MAX_RAW_READ_BYTES = 20 * 1024 * 1024

# Non-translatable artifacts. OCR-generated markdown (pymupdf4llm, docling,
# tesseract post-processing) routinely embeds full base64 image payloads
# inline. Sent to the MT model they are "translated" into garbage, blow the
# token count up by orders of magnitude, and the inflated translation then
# overflows the base model's context on the way back. Stripped before the
# gateway call, in the tool that knows it is handling documents.
_DATA_URI_RE = re.compile(r"data:[A-Za-z0-9/+.=-]+;base64,[A-Za-z0-9+/=\s]+")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>]*>")
_BLANK_RUN_RE = re.compile(r"\n{3,}")
# Collapse the space runs left where tags/blobs were stripped, without
# touching line-start indentation (code blocks).
_SPACE_RUN_RE = re.compile(r"(?<=\S)[ \t]{2,}(?=\S)")

_REFERENTIAL = {
    "",
    "this",
    "that",
    "it",
    "these",
    "those",
    "above",
    "the above",
    "the above message",
    "the above text",
    "the text",
    "this text",
    "that text",
    "previous",
    "previous message",
    "this file",
    "that file",
    "the file",
    "attached file",
    "the document",
    "this document",
    "translate",
    "translate this",
    "translate that",
    "translate it",
    "translate the above",
    "translate the above message",
    "translate the above text",
    "translate the text",
    "translate this text",
    "translate previous",
    "translate previous message",
    "translate this file",
    "translate that file",
    "translate the file",
    "translate file",
    "translate the document",
    "translate this document",
    "ترجمه",
    "ترجمه کن",
    "این را ترجمه کن",
    "متن بالا را ترجمه کن",
    "بالا را ترجمه کن",
    "این متن را ترجمه کن",
    "این فایل را ترجمه کن",
    "فایل را ترجمه کن",
    "متن را ترجمه کن",
    "متن بالا",
}

_REFERENCE_PREFIXES = (
    "translate this",
    "translate that",
    "translate the above",
    "translate the text",
    "translate file",
    "translate the file",
    "translate the document",
    "translate previous",
    "translate attached",
)

_INSTRUCTION_LINE_RE = re.compile(
    r"^(please\s+)?(translate|convert|translate the following|traduis|übersetze|ترجمه)",
    re.IGNORECASE,
)


# ------------------------------------------------------------ message utils


def msg_text(message: dict) -> str:
    """Message content as text, for both plain strings and content-part lists."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return "\n".join(parts)
    return ""


def last_user_message(messages) -> tuple:
    """(index, text) of the most recent user message, or (-1, '')."""
    if not messages:
        return -1, ""
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, dict) and msg.get("role") == "user":
            return i, msg_text(msg)
    return -1, ""


def context_blocks(text: str) -> list:
    """All OpenWebUI RAG context blocks in `text`, inner <source> tags stripped."""
    blocks = []
    for m in re.finditer(r"<context>(.*?)</context>", text or "", re.DOTALL):
        inner = re.sub(r"</?source[^>]*>", "", m.group(1))
        inner = inner.strip()
        if inner:
            blocks.append(inner)
    return blocks


def strip_wrappers(text: str) -> str:
    """Remove file-tag wrappers an LLM may have copied verbatim into `text`."""
    text = re.sub(r"<attached_files>.*?</attached_files>", "", text or "", flags=re.DOTALL)
    text = re.sub(r"<knowledge>.*?</knowledge>", "", text, flags=re.DOTALL)
    return text.strip()


def attached_file_ids(text: str) -> list:
    ids = []
    for m in re.finditer(r"<file\b[^>]*?/?>", text or ""):
        idm = re.search(r'id="([^"]+)"', m.group(0))
        if idm:
            ids.append(idm.group(1))
    return ids


def is_referential(text: str) -> bool:
    clean = (text or "").strip().lower()
    if clean in _REFERENTIAL:
        return True
    # Prefix matches count only for short phrases: an explicitly passed
    # long argument is content, even one starting with "translate this".
    return any(clean.startswith(prefix) for prefix in _REFERENCE_PREFIXES) and len(clean) < 30


def parse_command(text: str) -> tuple:
    """Split a /translate command into (body, source_lang, target_lang)."""
    clean = (text or "").strip()
    m = re.match(
        r"^/translate\s+([a-zA-Z]{2,5}(?:-[a-zA-Z0-9]+)?)"
        r"(?:->|:|\s+)"
        r"([a-zA-Z]{2,5}(?:-[a-zA-Z0-9]+)?)\s+(.+)$",
        clean,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        return m.group(3).strip(), m.group(1).lower(), m.group(2).lower()
    m = re.match(r"^/translate\s+(.+)$", clean, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip(), None, None
    return clean, None, None


def current_remainder(current_text: str, reference: str) -> str:
    """Text carried by the current user message beyond the reference/instruction."""
    t = current_text or ""
    t = re.sub(r"<attached_files>.*?</attached_files>", "", t, flags=re.DOTALL)
    t = re.sub(r"<context>.*?</context>", "", t, flags=re.DOTALL)
    t = re.sub(r"<knowledge>.*?</knowledge>", "", t, flags=re.DOTALL)

    kept = []
    for line in t.splitlines():
        s = line.strip()
        if (
            not kept
            and s
            and len(s) < 120
            and _INSTRUCTION_LINE_RE.match(s)
        ):
            continue
        kept.append(line)
    t = "\n".join(kept).strip()

    ref = (reference or "").strip()
    if ref and t == ref:
        t = ""
    elif ref and ref.isascii() and len(t) > len(ref):
        # trailing "translate this" style addendum (ASCII refs only:
        # whitespace token boundaries do not exist in e.g. Persian)
        t = re.sub(re.escape(ref) + r"[\s,:：]*$", "", t, flags=re.IGNORECASE).strip()

    if not t or is_referential(t):
        return ""
    return t if len(t) >= 8 else ""


def translation_relay_indices(messages: list) -> set:
    """Indices of assistant messages that merely relay a `translate` tool result.

    Those are translations, not sources: falling back onto them would
    translate the previous translation.
    """
    translate_call_ids = set()
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            fn = (tc.get("function") or {}).get("name", "")
            if fn == "translate" and tc.get("id"):
                translate_call_ids.add(tc.get("id"))
    if not translate_call_ids:
        return set()
    relays = set()
    for i, msg in enumerate(messages):
        if isinstance(msg, dict) and msg.get("role") == "tool" and msg.get("tool_call_id") in translate_call_ids:
            for j in range(i + 1, len(messages)):
                if messages[j].get("role") == "assistant":
                    relays.add(j)
                    break
    return relays


# ------------------------------------------------------------------ resolve


def resolve_from_messages(
    clean: str,
    messages,
    current_idx: int,
    current_text: str,
    attached_text: str,
    include_assistant_history: bool = True,
) -> tuple:
    """Deterministic resolution of the text to translate.

    Returns (text, source) where source labels where the text came from.
    Order: current message context block -> attached-file text ->
    current message remainder -> earlier context blocks -> last
    substantive history message. Never returns the system prompt, tool
    results, or a prior `translate` relay as a source.

    `include_assistant_history=False` (the pipe's mode): the history
    fallback then only scans earlier USER messages. Pipe replies are plain
    assistant messages with no tool-call marker to identify them, so
    including assistant text would let "translate this" latch onto the
    previous translation and re-translate it.
    """
    if messages:
        ctx = context_blocks(current_text)
        if ctx:
            return ctx[-1], "document context"

    if attached_text.strip():
        return attached_text.strip(), "attached file"

    remainder = current_remainder(current_text, clean)
    if remainder:
        return remainder, "message text"

    if messages:
        limit = current_idx if current_idx >= 0 else len(messages)
        earlier = []
        for msg in messages[:limit]:
            if isinstance(msg, dict) and msg.get("role") == "user":
                earlier.extend(context_blocks(msg_text(msg)))
        if earlier:
            return earlier[-1], "earlier document"

        relays = translation_relay_indices(messages)
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if not isinstance(msg, dict):
                continue
            if i == current_idx or i in relays:
                continue
            role = msg.get("role")
            # System/developer prompts are instructions, not sources:
            # falling back onto one is how a chat ends up "translating"
            # the system prompt.
            if role in ("system", "developer", "function", "tool"):
                continue
            if role == "assistant" and (
                not include_assistant_history or msg.get("tool_calls")
            ):
                continue
            content = msg_text(msg).strip()
            content = re.sub(
                r"<attached_files>.*?</attached_files>", "", content, flags=re.DOTALL
            ).strip()
            content = re.sub(
                r"<context>.*?</context>", "", content, flags=re.DOTALL
            ).strip()
            if not content:
                continue
            if content.startswith("/translate"):
                continue
            # Referential-only messages ("translate this") carry no source;
            # messages that carry a source alongside the reference do.
            remainder = current_remainder(content, "")
            if not remainder:
                continue
            return remainder, "conversation history"
    return "", ""


async def resolve_source(text_arg: str, messages, files, include_assistant_history: bool = True) -> dict:
    """The full source-resolution pipeline shared by the tool and the pipe.

    `text_arg` is the tool's `text` argument, or "" for the pipe (which
    always resolves from the conversation itself). Returns a dict with
    'text', 'source', 'file_names', 'unreadable_files', 'cmd_src',
    'cmd_tgt'.
    """
    # 1. Attached files on the CURRENT message (full text, no viewer caps).
    file_parts = []
    file_names = []
    unreadable_files = []
    for item in files or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") is not None and item.get("type") not in ("file", "doc", "text", "note"):
            continue
        ft = await file_text(item)
        label = str(item.get("name") or item.get("id") or "attached file")
        if ft and ft.strip():
            file_parts.append(ft.strip())
            file_names.append(label)
        else:
            unreadable_files.append(label)
    attached_text = "\n\n".join(file_parts)

    # 2. Parse /translate command for language overrides.
    clean, cmd_src, cmd_tgt = parse_command(text_arg)

    # 3. Resolve the text. An explicit non-referential `text` wins, unless
    #    it is the model dumping the whole message (then mine the context).
    if not is_referential(clean) and clean:
        ctx = context_blocks(clean)
        clean = strip_wrappers(clean)
        resolved = ctx[-1] if ctx else clean
        source = "document context" if ctx else "provided text"
        if not ctx:
            # Prompt-modal templates wrap the text in a short instruction
            # line; strip it only when the body is blank-line-separated,
            # so a real sentence that happens to start with "translate"
            # is never touched.
            m = re.match(r"^([^\n]{0,119})\n\n(.+)$", resolved, re.DOTALL)
            if m and _INSTRUCTION_LINE_RE.match(m.group(1).strip()):
                resolved = m.group(2)
        if not resolved.strip() and attached_text.strip():
            resolved, source = attached_text, "attached file"
    else:
        current_idx, current_text = last_user_message(messages)
        # attached-file ids referenced in the current message
        if not attached_text.strip():
            ids = attached_file_ids(current_text)
            fetched = []
            for fid in ids:
                t = await file_text_by_id(fid)
                if t and t.strip():
                    fetched.append(t.strip())
                else:
                    unreadable_files.append(fid)
            if fetched:
                attached_text = "\n\n".join(fetched)
        resolved, source = resolve_from_messages(
            clean,
            messages,
            current_idx,
            current_text,
            attached_text,
            include_assistant_history=include_assistant_history,
        )

    return {
        "text": (resolved or "").strip(),
        "source": source,
        "file_names": file_names,
        "unreadable_files": unreadable_files,
        "cmd_src": cmd_src,
        "cmd_tgt": cmd_tgt,
    }


# --------------------------------------------------------------- sanitise


def sanitize_document(text: str) -> str:
    """Remove non-translatable artifacts from a document before translation.

    Order matters: data URIs first (they can contain anything, including
    angle brackets that would confuse the tag stripping), then markdown
    images (keeping the alt text, which *is* translatable), then HTML
    comments and tags (keeping inner text). Collapses blank runs. Returns
    the cleaned text; may be empty when the input was only images/binary.
    """
    if not text:
        return ""
    text = _DATA_URI_RE.sub(" ", text)
    text = _MD_IMAGE_RE.sub(lambda m: m.group(1).strip(), text)
    text = _HTML_COMMENT_RE.sub(" ", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _BLANK_RUN_RE.sub("\n\n", text)
    text = _SPACE_RUN_RE.sub(" ", text)
    return text.strip()


# ------------------------------------------------------------ file text


async def file_record(file_id: str):
    try:
        from open_webui.models.files import Files

        return await Files.get_file_by_id(str(file_id))
    except Exception:
        return None


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


async def raw_disk_text(record) -> str:
    """Naive text read for text-like files whose extraction is pending or failed.

    The raw upload bytes are on disk as soon as the upload completes, so
    this keeps .md/.txt usable even when the extraction pipeline has not
    written ``data.content`` yet (or failed on it). Binary formats are
    left alone: guessing at a PDF byte stream is worse than an honest error.
    """
    try:
        name = str(getattr(record, "filename", "") or "")
        meta = getattr(record, "meta", None)
        content_type = str(meta.get("content_type") or "") if isinstance(meta, dict) else ""
        text_like = name.lower().endswith(TEXT_LIKE_EXT) or "text/" in content_type
        if not text_like:
            return ""
        rel_path = getattr(record, "path", None)
        if not rel_path:
            return ""
        from open_webui.storage.provider import Storage

        abs_path = await asyncio.to_thread(Storage.get_file, rel_path)
        if not abs_path or not os.path.isfile(abs_path):
            return ""
        if os.path.getsize(abs_path) > _MAX_RAW_READ_BYTES:
            return ""
        raw = await asyncio.to_thread(_read_bytes, abs_path)
        for enc in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            return ""
        if "\x00" in text[:4096]:
            return ""
        return text.strip()
    except Exception:
        return ""


def record_content(record) -> str:
    if record is not None:
        data = getattr(record, "data", None)
        if isinstance(data, dict) and data.get("content"):
            return str(data["content"])
    return ""


async def file_text(item: dict) -> str:
    """Full extracted text of one attached-file item, no viewer char caps."""
    if not isinstance(item, dict):
        return ""
    file_obj = item.get("file")
    if isinstance(file_obj, dict):
        data = file_obj.get("data")
        if isinstance(data, dict) and data.get("content"):
            return str(data["content"])
    file_id = item.get("id") or item.get("url")
    if not file_id or str(file_id).startswith(("http://", "https://", "data:")):
        return ""
    record = await file_record(str(file_id))
    content = record_content(record)
    if content:
        return content
    return await raw_disk_text(record)


async def file_text_by_id(file_id: str) -> str:
    record = await file_record(file_id)
    content = record_content(record)
    if content:
        return content
    return await raw_disk_text(record)


# ------------------------------------------------------------- languages


def resolve_languages(source_lang, target_lang, user, default_src, default_tgt) -> tuple:
    """explicit args > user valves > system defaults, all lowercased.

    `user['valves']` arrives as a dict in tests and as an instance of the
    front end's UserValves class in production (OpenWebUI builds it with
    `module.UserValves(**stored)`), so both shapes are read defensively —
    pydantic v2 models are not mappings, and `**instance` would silently
    drop the user's valves.
    """
    src = (source_lang or "").strip().lower()
    tgt = (target_lang or "").strip().lower()

    user_valves = user.get("valves") if isinstance(user, dict) else None

    def _valve(name: str) -> str:
        if user_valves is None:
            return ""
        if isinstance(user_valves, dict):
            return str(user_valves.get(name) or "")
        return str(getattr(user_valves, name, "") or "")

    if not src:
        src = _valve("source_lang").strip().lower()
    if not tgt:
        tgt = _valve("target_lang").strip().lower()

    if not src:
        src = (default_src or "").strip().lower()
    if not tgt:
        tgt = (default_tgt or "").strip().lower()
    return src, tgt


# ----------------------------------------------------------------- client


class TranslationClient:
    """The gateway (or direct vLLM) call. Synchronous: the caller decides
    whether to run it in the event loop or a worker thread."""

    def __init__(
        self,
        backend_mode: str = "gateway",
        gateway_url: str = "http://translategemma-api:8000/translate",
        vllm_url: str = "http://translategemma-vllm:8000/v1/chat/completions",
        vllm_model_name: str = "model",
        max_new_tokens: int = 512,
        timeout_seconds: int = 300,
    ):
        self.backend_mode = backend_mode
        self.gateway_url = gateway_url
        self.vllm_url = vllm_url
        self.vllm_model_name = vllm_model_name
        self.max_new_tokens = max_new_tokens
        self.timeout_seconds = timeout_seconds

    def translate(self, text: str, src: str, tgt: str, split_sentences: bool) -> str:
        """One translation. Returns the text, or an 'Error'/'Gateway
        Error'/'vLLM Error' string on failure — the caller decides what to
        do with errors (never attach a file for them)."""
        if self.backend_mode == "gateway":
            payload = {
                "text": text,
                "source_lang": src,
                "target_lang": tgt,
                "max_new_tokens": self.max_new_tokens,
                "split_sentences": split_sentences,
            }
            resp = requests.post(
                self.gateway_url,
                json=payload,
                timeout=self.timeout_seconds,
            )
            if resp.status_code != 200:
                return f"Gateway Error ({resp.status_code}): {resp.text}"
            return resp.json().get("translation", "")
        payload = {
            "model": self.vllm_model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "source_lang_code": src,
                            "target_lang_code": tgt,
                            "text": text,
                        }
                    ],
                }
            ],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": self.max_new_tokens,
            "stop_token_ids": [1, 106],
        }
        resp = requests.post(
            self.vllm_url,
            json=payload,
            timeout=self.timeout_seconds,
        )
        if resp.status_code != 200:
            return f"vLLM Error ({resp.status_code}): {resp.text}"
        choices = resp.json().get("choices", [])
        return (
            choices[0].get("message", {}).get("content", "")
            if choices
            else "No translation returned."
        )


# ------------------------------------------------------------ attachments


def build_file_name(file_names: list, tgt: str) -> str:
    name = f"translation-{tgt}.md"
    if file_names:
        base = str(file_names[0]).rsplit(".", 1)[0]
        if base and not base.startswith(("http", "data:")):
            name = f"{base}.translated.{tgt}.md"
    return name


async def upload_translation_file(translation: str, name: str, user_id: str):
    """Store the translation in OpenWebUI's own file storage.

    Returns the FileModel record, or None when storage is unavailable
    (the caller then falls back to a content-only attachment). No RAG
    processing happens — the translation is never embedded into any
    knowledge collection, it is a plain downloadable file.
    """
    try:
        import io
        import uuid

        from open_webui.models.files import FileForm, Files
        from open_webui.storage.provider import Storage
    except Exception:
        return None
    try:
        file_id = str(uuid.uuid4())
        tags = {
            "OpenWebUI-User-Id": str(user_id),
            "OpenWebUI-File-Id": file_id,
        }
        contents, file_path = await asyncio.to_thread(
            Storage.upload_file,
            io.BytesIO(translation.encode("utf-8")),
            f"{file_id}_{name}",
            tags,
        )
        return await Files.insert_new_file(
            str(user_id),
            FileForm(
                id=file_id,
                filename=name,
                path=file_path,
                data={"content": translation, "status": "completed"},
                meta={
                    "name": name,
                    "content_type": "text/markdown",
                    "size": len(contents),
                    "data": {},
                },
            ),
        )
    except Exception:
        return None


async def store_translation_file(
    translation: str,
    file_names: list,
    tgt: str,
    chat_id,
    message_id,
    event_emitter,
    user,
) -> str:
    """Store the translation as a downloadable chat file. Returns the name.

    When possible the translation is stored as a real OpenWebUI file
    (file storage + DB record): the chat chip's modal then previews the
    markdown and the file name is a link to /files/{id}/content, which
    the browser downloads. Without storage access it degrades to a
    content-only entry (preview only). Persistence and the
    chat:message:files event are best effort — a failure here must not
    fail the translation itself.
    """
    name = build_file_name(file_names, tgt)

    user_id = str((user or {}).get("id") or "")
    record = await upload_translation_file(translation, name, user_id) if user_id else None

    file_entry = {
        "type": "file",
        "name": name,
        "content_type": "text/markdown",
        "size": len(translation.encode("utf-8")),
    }
    if record is not None:
        # url is the file id: FileItem/FileItemModal resolve
        # /files/{id}/content (Content-Disposition: attachment for
        # non-text/plain types -> the browser downloads the file).
        file_entry["id"] = record.id
        file_entry["url"] = record.id
        file_entry["file"] = record.model_dump()
    else:
        encoded = base64.b64encode(translation.encode("utf-8")).decode("ascii")
        file_entry["url"] = f"data:text/markdown;charset=utf-8;base64,{encoded}"
        file_entry["content"] = translation

    # Persist the attachment to the chat message so it survives reloads.
    # Best effort: a failure here must not fail the translation itself.
    chat_id = str(chat_id or "")
    if (
        chat_id
        and str(message_id)
        and not chat_id.startswith(("temp:", "channel:"))
    ):
        try:
            from open_webui.models.chats import Chats

            saved = await Chats.add_message_files_by_id_and_message_id(
                chat_id, str(message_id), [file_entry]
            )
            if saved:
                file_entry = saved[0]
        except Exception:
            pass

    if event_emitter:
        try:
            await event_emitter(
                {"type": "chat:message:files", "data": {"files": [file_entry]}}
            )
        except Exception:
            pass

    return name

import asyncio
from typing import Any, Callable, Optional
from pydantic import BaseModel, Field

# Reply chunks for streaming: paragraph-sized, so the chat renders
# progressively without fragmenting a line.
_CHUNK_CHARS = 1500


def _reply_chunks(text: str):
    """Yield streaming chunks that rejoin to `text` exactly.

    The framework emits each yielded line as its own delta and never joins
    them back together, so paragraph breaks must ride on the chunk — a
    splitter that drops the separator would flatten the whole document.
    """
    parts = (text or "").split("\n\n")
    for i, part in enumerate(parts):
        sep = "" if i == len(parts) - 1 else "\n\n"
        if part:
            for start in range(0, len(part), _CHUNK_CHARS):
                piece = part[start : start + _CHUNK_CHARS]
                yield piece + sep if start + _CHUNK_CHARS >= len(part) else piece
        elif sep:
            yield sep


class Pipe:
    class Valves(BaseModel):
        BACKEND_MODE: str = Field(
            default="gateway",
            description="Backend mode: 'gateway' (translategemma-api gateway) or 'vllm'",
        )
        GATEWAY_URL: str = Field(
            default="http://translategemma-api:8000/translate",
            description="URL to the TranslateGemma FastAPI gateway (/translate)",
        )
        VLLM_URL: str = Field(
            default="http://translategemma-vllm:8000/v1/chat/completions",
            description="URL to vLLM chat completions if querying vLLM directly",
        )
        VLLM_MODEL_NAME: str = Field(
            default="model",
            description="Model identifier served by vLLM",
        )
        DEFAULT_SOURCE_LANG: str = Field(
            default="en",
            description="Default source language ISO code (e.g. en, fa, de, fr, ru)",
        )
        DEFAULT_TARGET_LANG: str = Field(
            default="fa",
            description="Default target language ISO code (e.g. fa, en, de, fr, ru)",
        )
        MAX_NEW_TOKENS: int = Field(
            default=512,
            description="Maximum new tokens per segment (512 is the gateway default and 4x less KV-cache pressure than 2048 on long documents)",
        )
        TIMEOUT_SECONDS: int = Field(
            default=300,
            description="Timeout in seconds for long documents",
        )
        ATTACH_FILE: bool = Field(
            default=True,
            description=(
                "Attach the translation as a downloadable chat file (stored in "
                "OpenWebUI file storage). The full translation is always part "
                "of the reply itself; the file is a convenience copy."
            ),
        )

    class UserValves(BaseModel):
        source_lang: str = Field(
            default="",
            description="Personal default source language code (leave blank for system default)",
        )
        target_lang: str = Field(
            default="",
            description="Personal default target language code (leave blank for system default)",
        )

    def __init__(self):
        self.valves = self.Valves()

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __chat_id__: Optional[str] = None,
        __message_id__: Optional[str] = None,
        __files__: Optional[list] = None,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ):
        """Handle one chat completion. The body's messages are the
        conversation; the reply IS the translation. No base model is
        involved at any point."""
        body = body or {}
        messages = body.get("messages") or []
        stream = bool(body.get("stream", False))

        current_idx, current_text = last_user_message(messages)
        # /translate [src] [tgt] ... works on the pipe too (language override
        # + optional inline text).
        text_arg = current_text if current_text.strip().startswith("/translate") else ""

        info = await resolve_source(
            text_arg, messages, __files__, include_assistant_history=False
        )
        file_names = info["file_names"]
        resolved = sanitize_document(info["text"])

        if not resolved:
            if info["unreadable_files"]:
                names = ", ".join(info["unreadable_files"][:3])
                error = (
                    f"The attached file ({names}) has no readable text yet. "
                    "Its content extraction may still be running or failed — wait a "
                    "moment and send the message again. For files, use the "
                    "'Using Entire Document' upload mode (Settings → Interface → "
                    "File → Default Upload Mode) so the full text is available."
                )
            else:
                error = (
                    "Nothing to translate. Attach a document or paste the text — "
                    "every message sent to this pipe is a translation request. "
                    "Use '/translate [src] [tgt] ...' to override the languages "
                    f"(defaults: {self.valves.DEFAULT_SOURCE_LANG} → "
                    f"{self.valves.DEFAULT_TARGET_LANG})."
                )
            return error

        src, tgt = resolve_languages(
            info["cmd_src"] or None,
            info["cmd_tgt"] or None,
            __user__,
            self.valves.DEFAULT_SOURCE_LANG,
            self.valves.DEFAULT_TARGET_LANG,
        )
        split = len(resolved) > 250 or "\n" in resolved
        word_count = len(resolved.split())
        source_desc = (
            f"file: {', '.join(file_names[:2])}"
            if info["source"] == "attached file" and file_names
            else info["source"] or "text"
        )

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Translating {word_count} words [{src} \u279c {tgt}] (from {source_desc})...",
                        "done": False,
                    },
                }
            )

        client = TranslationClient(
            backend_mode=self.valves.BACKEND_MODE,
            gateway_url=self.valves.GATEWAY_URL,
            vllm_url=self.valves.VLLM_URL,
            vllm_model_name=self.valves.VLLM_MODEL_NAME,
            max_new_tokens=self.valves.MAX_NEW_TOKENS,
            timeout_seconds=self.valves.TIMEOUT_SECONDS,
        )
        try:
            # The gateway call blocks; keep the event loop free for the
            # progress events and other chats.
            result = await asyncio.to_thread(client.translate, resolved, src, tgt, split)
        except Exception as error:
            result = f"Translation connection error: {str(error)}"

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Translation complete ({word_count} words).",
                        "done": True,
                    },
                }
            )

        if result.startswith(("Error:", "Gateway Error", "vLLM Error")):
            return result

        if self.valves.ATTACH_FILE:
            await store_translation_file(
                result,
                file_names,
                tgt,
                __chat_id__,
                __message_id__,
                __event_emitter__,
                __user__,
            )

        if stream:
            return _reply_chunks(result)
        return result
```

---

## 3. Custom Model Setup (`Workspace > Models`)

To make your model handle context and documents seamlessly without requiring users to manually toggle tools every session:

1. Go to **Workspace → Models → Create a Model (`+`)**.
2. Set:
   - **Name**: `TranslateGemma Translator`
   - **Model ID**: `translategemma-translator`
   - **Base Model**: Select your preferred base chat model.
   - **Tools**: Check the box for **`TranslateGemma Translation Tool`**.
   - **System Prompt**:
      ```text
      You are a translation engine powered by TranslateGemma 27B. You have
      exactly ONE job: translate. You never do anything else.

      ABSOLUTE RULES (never violate these):
      1. You NEVER ask the user a question. You NEVER ask "what do you want me
         to do?", "should I translate the full text or a summary?", "full or
         part?", or any other clarifying question. You NEVER present options,
         choices, or menus.
      2. You NEVER translate a summary, an excerpt, or "some parts". You ALWAYS
         translate the ENTIRE content, every time, unless the user explicitly
         points at one specific sentence or section.
      3. You NEVER add greetings, explanations, planning, or commentary before
         or after the translation. Your whole reply is the tool's output.
      4. You NEVER translate anything yourself with your own weights.

      DEFAULT ACTION — when in doubt, just translate:
      - The default target language is Persian (fa). Translate to Persian
        whenever the user does not name a target language.
      - A file or document that appears in the conversation (as an attachment
        or inside a <context> block) is ALWAYS the text to translate. If the
        user sends only a file and no other words, that is a complete
        instruction: call translate() on the whole document now.

      HOW TO CALL translate:
      1. File/document attached (with or without other text): call translate()
         with `text` left EMPTY. The tool reads the whole document itself.
         NEVER copy or retype the document into `text`.
      2. "translate this / that / it / the above" (or Persian equivalent):
         call translate() with `text` empty or "this".
      3. The user pasted plain text (no file): pass that text verbatim in
         `text`. An inline `/translate ...` command: forward it unchanged.
      4. Languages: if the user names source/target (codes or names), pass
         them as `source_lang` / `target_lang`. Otherwise pass nothing; the
         tool applies its defaults.

      THEN:
      - If the tool returns translation text: return it COMPLETE and VERBATIM
        and stop.
      - If the tool says the translation is already attached to the chat as a
        file: reply with ONE short sentence acknowledging that the
        translation is ready and attached. NEVER restate, quote, or summarize
        the translated content — it is not in your context and you must not
        invent it.
      - If the tool returns an error, show that error and stop.
      ```
 3. Save the model.

   > **If the model still asks questions or offers "full vs. summary"** after
   > this prompt: the base model is too "chatty" for a deterministic tool
   > relay. Pick a base model that follows system prompts strictly (small
   > instruct models and very large creative models both tend to add
   > commentary). The system prompt above is written to suppress that, but
   > base-model compliance is the last mile.

---

## 4. Prompt Template Configuration (`Workspace > Prompts`)

OpenWebUI parses prompt variables using simple tokens `{{variable_name}}` without spaces.

### Option A: Quick Translation (Default EN ➔ FA, 1-Field Modal)
1. Go to **Workspace → Prompts → Add Prompt (`+`)**.
2. Configure:
   - **Title**: `Translate`
   - **Command**: `/translate`
   - **Content**:
     ```text
     Translate the following text to Persian:

     {{text}}
     ```
3. Save.

**Behavior**:
- Typing `/translate` opens a modal with **only one field: `text`** (rendered as a textarea).
- On submission, it translates using default languages (`en` ➔ `fa`).

---

### Option B: Custom Language Pair Translation
1. Go to **Workspace → Prompts → Add Prompt (`+`)**.
2. Configure:
   - **Title**: `Translate Custom Pair`
   - **Command**: `/translate-pair`
   - **Content**:
     ```text
     Translate the following text from {{source_lang}} to {{target_lang}}:

     {{text}}
     ```
3. Save.

**Behavior**:
- Opens a modal with 3 fields: `source_lang`, `target_lang`, and `text`.

---

### Option C: Direct In-Chat Execution (Zero Modal)
When chatting with `TranslateGemma Translator`:
- Paste any text or upload a document and type: `translate this`
- Type inline:
  - `/translate Cellular biology is the study of cell structure.`
  - `/translate fr fa Bonjour le monde`

---

## 5. Document & File Upload Best Practices (v0.11.3-verified)

**This is the single most important setting for translation.** It is the
root cause of "long text gets shortened" and "markdown says no content".

1. **Upload with FULL context, not focused retrieval.**
   In OpenWebUI v0.11.3 the **default** upload mode is *focused retrieval*:
   the file is chunked into a vector store and only the top-k chunks matching
   a generated query are injected into the prompt. For a document you want
   translated *in full*, that is wrong — you get a partial excerpt (shortened
   output) or, when no chunk clears the relevance threshold, **nothing**
   (the "no content in the file" symptom).

   Set uploads to full-context so the entire document is injected:
   - **Per user (recommended):** *Settings (gear) → Interface → File →
     Default Upload Mode → "Using Entire Document"*. This sets
     `defaultUploadContext: 'full'`.
   - **Per file:** when attaching a file in the chat input, open the file
     item's modal and toggle the context switch to full ("using entire
     document").
   - **Per deployment (admin):** set the `rag.full_context` config to `true`
     (Admin → Settings → RAG → Full Context). When any attached file is in
     full-context mode, OpenWebUI reads the whole extracted text from the
     file record and injects it as a `<context><source …>…</source></context>`
     block on the user message.

   The tool in section 2 reads that block (and the `__files__` record)
   directly, so the model never re-copies the document and nothing is
   truncated at 10,000 characters by the `view_file` helper.

2. **How content actually reaches the tool (v0.11.3):**
   - With tools enabled, `add_file_context` prepends
     `<attached_files><file id="…" name="…" …/></attached_files>` to the user
     message (metadata only — no body).
   - In full-context mode the RAG layer then wraps the full document text in
     `<context><source …>BODY</source></context>` on the same message.
   - `__files__` passed to the tool is the list of files attached to the
     *current* message; the tool reads each file's stored `data.content`
     in-process (the same text `view_file` shows, but without its 10k cap).

3. **High-Throughput, Structure-Preserving Chunking:**
   The tool automatically activates `split_sentences: True` when texts exceed
   250 characters. The FastAPI gateway splits the document into blocks and
   carries each original separator as data: blocks that fit the token budget
   are whole units, over-budget blocks descend to lines (structural) or
   `pysbd` sentences (prose) with their original gaps, and any remaining
   over-budget unit is hard-sliced at token boundaries. Consecutive units
   separated by plain whitespace are packed into one prompt (fewer requests,
   neighbouring context); a prompt never spans a paragraph break. Code fences
   and `$$` math are verbatim. The translated units are rejoined with their
   original separators in order, so markdown structure survives.

 4. **Long documents end-to-end:**
    - *Input side* is safe: the tool sends the full text (no `view_file` 10k
      cap, no model copy). It first **strips non-translatable artifacts** —
      base64 image data-URIs, markdown images (alt text kept), HTML comments
      and tags (inner text kept) — because OCR-generated markdown embeds image
      payloads inline, and "translating" them produces garbage and inflates
      the token count until the return path overflows. A document that is
      images/binary only returns an explicit "no translatable text" error.
    - *Mid side* (the gateway) is bounded and structure-preserving: with
      `split_sentences` it splits the document into blocks and carries each
      original separator as data — block (whole) → line (structural block) →
      sentence (pysbd; original gaps) → line (languages pysbd lacks a model
      for) → hard token-boundary slices — then packs consecutive
      whitespace-separated units into budget-sized prompts. An oversized
      document degrades to *more chunks*, never a 400, and the rejoin keeps
      the input's blank lines, headings, list lines, and code fences intact. The
      chunk budget is `min(context − max_new_tokens − reserve,
      max_new_tokens // 2)`, so a chunk's translation also cannot be silently
      clipped at the stop (a `finish_reason: length` hit is logged and means
      `MAX_NEW_TOKENS` should go up for that pair).
      - *Output side*: short translations pass through the base model, which
        re-emits them; if one is cut off at the **end**, raise the base model's
        max output tokens (model params → `max_tokens`). **Long translations no
        longer do**: above the `RELAY_MAX_CHARS` valve (default 8000 chars) the
        tool stores the result as a real OpenWebUI file and attaches it to the
        chat (the same `chat:message:files` mechanism OpenWebUI's own
        `generate_image` tool uses), and the base model only receives a
        one-line acknowledgement instruction. This is what stops a
        document-sized tool result from overflowing the base model's own
        context window on the way back. The attached file previews in the chat
        and downloads via its name link (`/files/{id}/content`).
       `MAX_NEW_TOKENS` (512, the gateway default) is per chunk, not per
       document; every extra token of headroom is KV-cache pressure on vLLM
       for *every* chunk of the document, so raise it only if the gateway log
       reports clipped chunks.

---

## 6. Docker Compose Setup for Offline Host

Add OpenWebUI to your `api/docker-compose.yml` so all services communicate through Docker internal DNS:

```yaml
version: "3.8"

services:
  translategemma-vllm:
    image: vllm/vllm-openai:v0.13.0
    container_name: translategemma-vllm
    restart: unless-stopped
    ports:
      - "8001:8000"
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - HF_HUB_OFFLINE=1
      - TRANSFORMERS_OFFLINE=1
    volumes:
      - /models/translategemma-27b-merged:/models:ro
    entrypoint: ["vllm", "serve"]
    command:
      - "/models"
      - "--served-model-name"
      - "model"
      - "--dtype"
      - "bfloat16"
      - "--max-model-len"
      - "8192"
      - "--limit-mm-per-prompt"
      - '{"image": 0}'

  translategemma-api:
    build: .
    container_name: translategemma-api
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - TG_VLLM_BASE_URL=http://translategemma-vllm:8000/v1
      - TG_VLLM_MODEL=model
      - TG_BASE_MODEL_ID=/models/translategemma-27b-merged
      - TG_SERVED_SYSTEM=adapter
      - HF_HUB_OFFLINE=1
      - TRANSFORMERS_OFFLINE=1
    volumes:
      - /models/translategemma-27b-merged:/models:ro
    depends_on:
      - translategemma-vllm

  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: open-webui
    restart: unless-stopped
    ports:
      - "3000:8080"
    environment:
      - OFFLINE_MODE=true
      - WEBUI_AUTH=true
    volumes:
      - open-webui-data:/app/backend/data
    depends_on:
      - translategemma-api

volumes:
  open-webui-data:
```

---

## 7. Verification Checklist

1. **Verify Backend Health**:
   ```bash
   curl -s http://localhost:8000/health-check
   # Expected output: {"translator":"OK"}
   ```

2. **Verify Batch & Sentence Splitting**:
   ```bash
   curl -s http://localhost:8000/translate \
     -H 'Content-Type: application/json' \
     -d '{
       "text": "First sentence to translate. Second sentence to translate.",
       "source_lang": "en",
       "target_lang": "fa",
       "split_sentences": true
     }'
   ```

3. **Verify Context in OpenWebUI**:
   - In OpenWebUI, ask the model: *"Explain CRISPR in two sentences."*
   - Once it replies, type: *"Translate the above into Persian."*
   - Verify that the tool status chip displays `Translating ... words [en ➔ fa] (direct)...` and outputs the Persian translation accurately.

4. **Verify File Upload (full-context) in OpenWebUI**:
   - With *Default Upload Mode = Using Entire Document*, upload a **markdown**
     file and type *"translate this"*. The full document must be translated —
     this is the case that previously reported "no content in the file".
   - Repeat with a **PDF**. Both should reach the gateway with the complete
     extracted text (check the gateway log line for the word count).

5. **Verify Multi-Source Disambiguation**:
   - Upload/translate source **A**, then upload source **B** and type
     *"translate this"*. The tool must translate **B**, not A and not A's
     translation. Repeat by pasting two separate text sources in sequence.

 6. **Verify Long-Document Fidelity**:
    - Translate a document well over 10,000 characters. The gateway receives
      the full text (status chip word count ≈ the document's), and the final
      answer is not cut off mid-sentence.

 7. **Verify the Pipe (deterministic mode, section 2b — only if installed)**:
    - Select the pipe as the model, upload a **markdown** file, and send
      *"translate this"*. The reply must be the full translation itself —
      no base-model chatter, no "full / part / summary?" question — and the
      downloadable file chip must be attached.
    - With a previous translation still in the chat, send *"translate this"*
      with a **new** file: it must translate the new file, never the previous
      translation (pipe replies carry no tool-call marker, so the core
      excludes assistant history by design).

 ---

## 8. Troubleshooting the Previously-Reported Symptoms

Mapping of the five originally observed problems to their v0.11.3 causes and
the fix each relies on:

| # | Symptom | Root cause (v0.11.3) | Fixed by |
|---|---------|----------------------|----------|
| 1 | PDF upload translates fine | Full-context PDF text was in the message; the model could copy a short doc | Still works; now deterministic |
| 2 | OCR extract → switch model → translate works | Source is a prior assistant message; v1's history fallback found it | v2 history fallback explicitly keeps non-relay assistant messages |
| 3 | Long extracted text is shortened | (a) base model re-typing 10k+ chars into a tool argument drops text; (b) `view_file` caps at 10,000 chars; (c) base-model output limit | v2 tool self-reads the full document (no model copy, no 10k cap); raise base-model `max_tokens` for the output relay |
| 4 | Two sources in a row → wrong one translated | v1 fell back to "last non-empty message", which could be a prior translation or an old source | v2 deterministic order: current-message context/`__files__` first; a previous `translate` result is never re-selected |
| 5 | Markdown file → "no content in the file" | Default *focused retrieval* injects top-k chunks (or none below the relevance threshold), so nothing reached the tool | Set upload mode to **Using Entire Document** (`defaultUploadContext: 'full'` / `rag.full_context`); v2 tool reads the full-context block and `__files__` directly |
| 6 | Model asks "full, part, or summary?" | The base chat model, not the tool, adds clarifying questions and offers options | Section-3 system prompt now forbids questions/options and mandates full-content translation; swap to a strictly-instruct-following base model if it persists |
| 7 | File sent, model asks "what to do?" | A file-only message (no instruction) is ambiguous to the base model, so it asks instead of acting | System prompt defines file-attached = a complete translate instruction, default target Persian (fa); model must call `translate()` with empty `text` immediately |
| 8 | The **system prompt** gets translated instead of the document | v2.0's "leave `text` empty" rule pushes the model into the tool's self-resolution; when the file content is absent (focused mode / still extracting) the old history fallback scanned *every* message and returned the system prompt as the source | v2.1 fallback **skips `system`/`developer`/`tool`/`function` roles** and never re-selects a prior `translate` relay; if no real source exists it returns an explicit "file has no readable text yet / use full-context upload mode" error instead of guessing. Also adds a raw on-disk read for text-like files (.md/.txt/…) when extraction hasn't written `data.content` yet |
| 9 | Long document → `Gateway Error (500): Internal Server Error` | `/translate` had no error handler: any upstream failure (vLLM 400 "prompt too long" on a giant unsegmentable chunk; a chunk exceeding `vllm_timeout` under KV pressure from 2048 tokens × hundreds of concurrent segments; API-container OOM) surfaced as a generic 500 whose traceback only lived in a log the container itself corrupted (`fastapi run` = dev mode, reload supervisor shares stdout → torn json-file log) | Gateway now answers **502 with the actual upstream error text** (visible in the chat); Dockerfile runs production `uvicorn` (no reload → intact logs, no mid-request restarts); tool `MAX_NEW_TOKENS` default 2048 → 512 per segment (4× less KV demand). Rebuild both images to apply |
| 10 | vLLM 400: `maximum context length is N … request has M input tokens` | The document (or pysbd's "whole text is one segment" fallback for a language it has no model for — e.g. `pt`/`tr`/`ko`) reached `/completions` as one over-context prompt; `split_sentences` split by sentence but never bounded chunk size to the window | The gateway now chunks **budget-aware**: `min(context − max_new_tokens − 256, max_new_tokens // 2)` source tokens per chunk, sentence → paragraph/line → hard token-boundary slices, greedily re-packed. The window is auto-probed from vLLM `/v1/models` (or set `TG_MAX_CONTEXT_TOKENS`). An oversized document now degrades to more chunks, never a 400 |
| 11 | OCR markdown → garbage words in the output, and the chat then shows the **base model's** own context error (`maximum context length is 262144 … 262145 input tokens`) with no translation shown | Two compounding bugs: (a) OCR markdown embeds full **base64 image payloads** inline — the v2.1 tool sent them to the MT model, which "translated" the blobs into garbage (the split/wrong words); (b) the inflated translation came back as the tool result, which the **base model** must re-emit into its reply — that re-emit overflows the base model's own context window (a different, larger vLLM than the gateway's) and the error replaces the translation | v2.2 tool **strips** base64 data-URIs, markdown images (keeping alt text), HTML comments/tags (keeping inner text) *before* the gateway call — a document that is only images/binary now returns an explicit "no translatable text" error instead of garbage. And translations longer than the `RELAY_MAX_CHARS` valve (default 8000) are stored as a real OpenWebUI file and returned as a **downloadable chat file attachment** (the same `chat:message:files` mechanism OpenWebUI's own `generate_image` uses; `v2.3` adds the file-storage step so the chip previews and the name links to `/files/{id}/content`; `v2.4` attaches a file to **every** translation, small ones alongside the normal reply), with the base model given a one-line acknowledgement instruction it must follow for large results — the model never reads or re-types the document-sized result |
 | 12 | Multi-line markdown input → **one-line** output (all structure gone) | The old chunker flattened the document to a flat sentence list (pysbd drops every newline; the line fallback strips lines) and rejoined translations with `" "`, destroying every paragraph break, heading, and list marker before/after the model saw it | The gateway now chunks **structure-preserving**: blocks (paragraphs/headings/list/table lines) are the unit, each carries its exact original separator, code fences and `$$` math pass through verbatim, and the rejoin is `translated unit + original separator` — so the output keeps the input's markdown layout. A prompt never spans a paragraph break |


### Quick checks when something regresses

- **Tool never fires / model answers itself** → the model isn't bound to the
  tool. Confirm the *Tools* checkbox on the custom model includes
  `TranslateGemma Translation Tool`.
- **Model asks "full / part / summary?" or "what do you want me to do?"** →
  base-model chattiness, not a tool bug. Re-apply the section-3 system prompt
  verbatim (it forbids questions and makes a file attachment a complete
  translate instruction, default target Persian). If it persists, the base
  model is too chatty for a deterministic relay — use a stricter one.
- **"No text or file content found" / "file has no readable text yet"** → the
  current message carries no full-context block and `__files__` had no
  extractable text: the file was uploaded in *focused retrieval* mode or its
  extraction is still running. Re-upload with the full-context toggle
  (**Using Entire Document**) on, or wait a moment and resend.
- **The system prompt shows up in Persian** (regression, v2.0) → the tool's
  fallback latched onto the system message because the document content was
  missing. Upgrade to the v2.1 tool (section 2); it can no longer return the
  system prompt and instead reports the missing content explicitly.
- **Translation cut off at the end** → base-model output limit. For short
  results the base model still re-emits the text, so raise the base model's
  `max_tokens`. Long results no longer pass through the model at all: above
  the `RELAY_MAX_CHARS` valve they arrive as a chat file attachment.
- **No file chip next to a short translation** (v2.4) → the
  `ALWAYS_ATTACH_FILE` valve is off (files are then attached only for large
  results), or the tool has no file-storage access (it then falls back to a
  preview-only data-URI chip — see the download note below). Confirm the
  valve is on and the tool is current (section 2, v2.5.0).
- **Chat shows the base model's context error** (`maximum context length is …
  input tokens`) while the tool panel shows a full translation → the
  document-sized tool result was being relayed through the base model and
  overflowed its window. Upgrade to the v2.3 tool (section 2): it stores long
  translations as files and attaches them. Also check the base model's
  OpenWebUI params: a `max_tokens` of `0` is forwarded to vLLM verbatim and
  rejects any prompt that alone fills the window — set a positive value
  (e.g. 8192).
- **Garbage / split words in the translation of an OCR file** → the file
  carried inline base64 image data or HTML that was "translated" verbatim.
  The v2.3 tool strips these before the gateway call; if the file is images
  only it errors with "no translatable text".
- **Attached translation shows a preview but no way to download it** → the
  tool ran without OpenWebUI file-storage access and fell back to a
  content-only (data-URI) entry, whose name link resolves to
  `/files/undefined/content`. Re-paste the v2.3 tool (section 2); it stores
  the translation in file storage so the chip's name links to
  `/files/{id}/content` (a real download). The file also appears in
  *Workspace → Files*.
- **Wrong language pair** → pass `source_lang`/`target_lang` explicitly, or
  set them in the tool's *User Valves*; `/translate src tgt text` also works.
- **Degenerate / looping output from the gateway** → stop tokens. Confirm the
  gateway's `/model-info` reports `stop_token_ids` containing the
  `<end_of_turn>` id (the merged checkpoint path in section 6 keeps `[1, 106]`).
- **Pipe selected, but the reply is conversational / asks "full / part /
  summary?"** → you are not talking to the pipe: confirm the model selector
  shows the pipe's function name and the function is active
  (Workspace → Functions). A pipe reply is the raw translation — no base
  model, no questions.
- **Pipe replies "Nothing to translate."** → same cause as for the tool: no
  attached file, no pasted text, nothing resolvable in history (pipe mode
  deliberately excludes assistant messages, so a previous pipe reply is never
  re-selected). Attach a document (full-context upload mode) or paste the
  text.
