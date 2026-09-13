# TranslateGemma 27B OpenWebUI (v0.11.3) Integration Guide

This guide details how to integrate the fine-tuned and merged **TranslateGemma-27B** model (served with vLLM on an offline host) into **OpenWebUI v0.11.3**, with complete support for:
1. **Official multimodal template compliance** and stop-token preservation (`[1, 106]`).
2. **Context resolution** (e.g., *"translate this"*, *"translate the above message"*).
3. **Document & file translation** (handling full file uploads with automatic sentence-split batching).
4. **Interactive `/translate` slash command** with default language fallback.

> **Verified against the OpenWebUI v0.11.3 source.** The tool in section 2 is
> version 2.0.0 and is the canonical copy kept in `scripts/openwebui_tool.py`
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
│  │ OpenWebUI Tool: translategemma_tool (Workspace > Tools, v2.0.0)   │  │
│  │ - Self-resolves the text: RAG <context> block of the current      │  │
│  │   message, __files__ full content (in-process, no 10k cap),       │  │
│  │   <attached_files> ids, or conversation history — the base model  │  │
│  │   never has to copy document text into the tool argument          │  │
│  │ - Disambiguates multi-source chats: current message wins; a       │  │
│  │   previous translation is never re-selected as a source           │  │
│  │ - Auto sentence-splitting: split_sentences=True (>250 chars)      │  │
│  │ - Sends live progress status chips via __event_emitter__          │  │
│  └─────────────────────────────────┬─────────────────────────────────┘  │
└────────────────────────────────────┼────────────────────────────────────┘
                                     │ HTTP POST (Internal Docker Network)
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ translategemma-api Gateway (:8000/translate)                            │
│ - Validates inputs (source_lang, target_lang, text)                     │
│ - Splits long documents into sentences using pysbd                      │
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

---

## 2. OpenWebUI Tool Implementation (`Workspace > Tools`)

In OpenWebUI v0.11.3, custom extensibility uses **Tools** (`class Tools:`).

This v2.0.0 tool implementation includes:
- **Self-served content resolution**: The tool reads `__messages__` and `__files__` itself and finds the text to translate. The base model only says *"translate"* — it does **not** retype document content into the tool argument. This removes the two failure modes of v1:
  - *Long documents got shortened*: the model could not reproduce 10k+ chars verbatim in a tool argument (and the built-in `view_file` caps at 10,000 chars by default). The tool now sends the complete text to the gateway.
  - *Mixed-up sources*: with two or more sources in one chat, v1's "last non-empty message" fallback could grab the previous translation or an old source. v2 resolves in a deterministic order (below) and never selects a prior `translate` result as a source.
- **v0.11.3 message-format aware**: handles list-of-parts message content, `<attached_files>`/`<file id=...>` tags, RAG `<context>`/`<source>` blocks, and the fact that the last message at call time is the assistant `tool_call`.
- **Automatic Sentence Splitting**: For inputs exceeding 250 characters or containing newlines, it automatically sets `split_sentences: True`. The backend gateway uses `pysbd` to segment the document into sentences and translates them concurrently in vLLM's continuous batching engine.
- **User & Global Valves**: Preserves default language fallbacks (`en` ➔ `fa`).

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
description: High-accuracy translation using finetuned TranslateGemma 27B. Self-resolves the text to translate from uploaded documents, conversation context, and inline text — no manual copy/paste by the model. Never falls back to the system prompt.
version: 2.1.1
license: MIT
requirements: requests, pydantic
"""

import asyncio
import os
import re
from typing import Any, Callable, List, Optional
from pydantic import BaseModel, Field
import requests


_TEXT_LIKE_EXT = (
    ".md", ".markdown", ".txt", ".text", ".csv", ".tsv", ".json", ".html",
    ".htm", ".xml", ".rst", ".yaml", ".yml", ".ini", ".log",
)
_MAX_RAW_READ_BYTES = 20 * 1024 * 1024


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

    # ------------------------------------------------------------- languages

    def _resolve_languages(
        self,
        source_lang: Optional[str],
        target_lang: Optional[str],
        __user__: Optional[dict],
    ) -> tuple:
        src = (source_lang or "").strip().lower()
        tgt = (target_lang or "").strip().lower()

        user_valves = None
        if __user__ and "valves" in __user__:
            try:
                user_valves = self.UserValves(**__user__["valves"])
            except Exception:
                user_valves = None

        if not src and user_valves and user_valves.source_lang:
            src = user_valves.source_lang.strip().lower()
        if not tgt and user_valves and user_valves.target_lang:
            tgt = user_valves.target_lang.strip().lower()

        if not src:
            src = self.valves.DEFAULT_SOURCE_LANG.strip().lower()
        if not tgt:
            tgt = self.valves.DEFAULT_TARGET_LANG.strip().lower()

        return src, tgt

    # --------------------------------------------------------- message utils

    @staticmethod
    def _msg_text(message: dict) -> str:
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

    @staticmethod
    def _last_user_message(messages: Optional[list]) -> tuple:
        """(index, text) of the most recent user message, or (-1, '')."""
        if not messages:
            return -1, ""
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if isinstance(msg, dict) and msg.get("role") == "user":
                return i, Tools._msg_text(msg)
        return -1, ""

    @staticmethod
    def _context_blocks(text: str) -> list:
        """All OpenWebUI RAG context blocks in `text`, inner <source> tags stripped."""
        blocks = []
        for m in re.finditer(r"<context>(.*?)</context>", text or "", re.DOTALL):
            inner = re.sub(r"</?source[^>]*>", "", m.group(1))
            inner = inner.strip()
            if inner:
                blocks.append(inner)
        return blocks

    @staticmethod
    def _strip_wrappers(text: str) -> str:
        """Remove file-tag wrappers an LLM may have copied verbatim into `text`."""
        text = re.sub(r"<attached_files>.*?</attached_files>", "", text or "", flags=re.DOTALL)
        text = re.sub(r"<knowledge>.*?</knowledge>", "", text, flags=re.DOTALL)
        return text.strip()

    @staticmethod
    def _attached_file_ids(text: str) -> list:
        ids = []
        for m in re.finditer(r"<file\b[^>]*?/?>", text or ""):
            idm = re.search(r'id="([^"]+)"', m.group(0))
            if idm:
                ids.append(idm.group(1))
        return ids

    @staticmethod
    def _is_referential(text: str) -> bool:
        clean = (text or "").strip().lower()
        if clean in _REFERENTIAL:
            return True
        # Prefix matches count only for short phrases: an explicitly passed
        # long argument is content, even one starting with "translate this".
        return any(clean.startswith(prefix) for prefix in _REFERENCE_PREFIXES) and len(clean) < 30

    @staticmethod
    def _parse_command(text: str) -> tuple:
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

    def _current_remainder(self, current_text: str, reference: str) -> str:
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

        if not t or self._is_referential(t):
            return ""
        return t if len(t) >= 8 else ""

    @staticmethod
    def _translation_relay_indices(messages: list) -> set:
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

    # ------------------------------------------------------------ file text

    @staticmethod
    async def _file_record(file_id: str):
        try:
            from open_webui.models.files import Files

            return await Files.get_file_by_id(str(file_id))
        except Exception:
            return None

    async def _raw_disk_text(self, record) -> str:
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
            text_like = name.lower().endswith(_TEXT_LIKE_EXT) or "text/" in content_type
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
            raw = await asyncio.to_thread(self._read_bytes, abs_path)
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

    @staticmethod
    def _read_bytes(path: str) -> bytes:
        with open(path, "rb") as f:
            return f.read()

    @staticmethod
    def _record_content(record) -> str:
        if record is not None:
            data = getattr(record, "data", None)
            if isinstance(data, dict) and data.get("content"):
                return str(data["content"])
        return ""

    async def _file_text(self, item: dict) -> str:
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
        record = await self._file_record(str(file_id))
        content = self._record_content(record)
        if content:
            return content
        return await self._raw_disk_text(record)

    async def _file_text_by_id(self, file_id: str) -> str:
        record = await self._file_record(file_id)
        content = self._record_content(record)
        if content:
            return content
        return await self._raw_disk_text(record)

    # ------------------------------------------------------------- resolve

    def _resolve_from_messages(
        self,
        clean: str,
        __messages__: Optional[list],
        current_idx: int,
        current_text: str,
        file_text: str,
    ) -> tuple:
        """Deterministic resolution of the text to translate.

        Returns (text, source) where source labels where the text came from.
        Order: current message context block -> attached-file text ->
        current message remainder -> earlier context blocks -> last
        substantive history message. Never returns the system prompt, tool
        results, or a prior `translate` relay as a source.
        """
        if __messages__:
            ctx = self._context_blocks(current_text)
            if ctx:
                return ctx[-1], "document context"

        if file_text.strip():
            return file_text.strip(), "attached file"

        remainder = self._current_remainder(current_text, clean)
        if remainder:
            return remainder, "message text"

        if __messages__:
            limit = current_idx if current_idx >= 0 else len(__messages__)
            earlier = []
            for msg in __messages__[:limit]:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    earlier.extend(self._context_blocks(self._msg_text(msg)))
            if earlier:
                return earlier[-1], "earlier document"

            relays = self._translation_relay_indices(__messages__)
            for i in range(len(__messages__) - 1, -1, -1):
                msg = __messages__[i]
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
                if role == "assistant" and msg.get("tool_calls"):
                    continue
                content = self._msg_text(msg).strip()
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
                remainder = self._current_remainder(content, "")
                if not remainder:
                    continue
                return remainder, "conversation history"
        return "", ""

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
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Translates text, documents, or conversation context using finetuned TranslateGemma 27B. ALWAYS translate the ENTIRE content — never a summary or partial excerpt, and never ask the user whether to translate all or part. A file/document attached with no instruction is a complete request to translate it in full (default target: Persian).

        :param text: Text to translate. For uploaded files and references like 'this' or 'the above', leave it empty or pass 'this' — the tool locates the document/message itself. Pass inline/pasted text here verbatim. `/translate [src] [tgt] ...` is also accepted.
        :param source_lang: Source language code (e.g. 'en', 'fa', 'de', 'fr', 'ru').
        :param target_lang: Target language code (e.g. 'fa', 'en', 'de', 'fr', 'ru').
        :param split_sentences: Whether to split long documents into sentences for concurrent batching.
        :return: Translated text.
        """
        # 1. Attached files on the CURRENT message (full text, no viewer caps).
        file_parts = []
        file_names = []
        unreadable_files = []
        for item in __files__ or []:
            if not isinstance(item, dict):
                continue
            if item.get("type") is not None and item.get("type") not in ("file", "doc", "text", "note"):
                continue
            ft = await self._file_text(item)
            label = str(item.get("name") or item.get("id") or "attached file")
            if ft and ft.strip():
                file_parts.append(ft.strip())
                file_names.append(label)
            else:
                unreadable_files.append(label)
        file_text = "\n\n".join(file_parts)

        # 2. Parse /translate command for language overrides.
        clean, cmd_src, cmd_tgt = self._parse_command(text)
        if cmd_src:
            source_lang = cmd_src
        if cmd_tgt:
            target_lang = cmd_tgt

        # 3. Resolve the text. An explicit non-referential `text` wins, unless
        #    it is the model dumping the whole message (then mine the context).
        if not self._is_referential(clean) and clean:
            ctx = self._context_blocks(clean)
            clean = self._strip_wrappers(clean)
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
            if not resolved.strip() and file_text.strip():
                resolved, source = file_text, "attached file"
        else:
            current_idx, current_text = self._last_user_message(__messages__)
            # attached-file ids referenced in the current message
            if not file_text.strip():
                ids = self._attached_file_ids(current_text)
                fetched = []
                for fid in ids:
                    t = await self._file_text_by_id(fid)
                    if t and t.strip():
                        fetched.append(t.strip())
                    else:
                        unreadable_files.append(fid)
                if fetched:
                    file_text = "\n\n".join(fetched)
            resolved, source = self._resolve_from_messages(
                clean, __messages__, current_idx, current_text, file_text
            )

        resolved = (resolved or "").strip()
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

        src, tgt = self._resolve_languages(source_lang, target_lang, __user__)

        if split_sentences is None:
            split_sentences = len(resolved) > 250 or "\n" in resolved

        word_count = len(resolved.split())
        mode_desc = "with sentence-splitting" if split_sentences else "direct"
        if source == "attached file" and file_names:
            source_desc = f"file: {', '.join(file_names[:2])}"
        else:
            source_desc = source or "text"

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
            if self.valves.BACKEND_MODE == "gateway":
                payload = {
                    "text": resolved,
                    "source_lang": src,
                    "target_lang": tgt,
                    "max_new_tokens": self.valves.MAX_NEW_TOKENS,
                    "split_sentences": split_sentences,
                }
                resp = requests.post(
                    self.valves.GATEWAY_URL,
                    json=payload,
                    timeout=self.valves.TIMEOUT_SECONDS,
                )
                if resp.status_code != 200:
                    result = f"Gateway Error ({resp.status_code}): {resp.text}"
                else:
                    result = resp.json().get("translation", "")
            else:
                payload = {
                    "model": self.valves.VLLM_MODEL_NAME,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "source_lang_code": src,
                                    "target_lang_code": tgt,
                                    "text": resolved,
                                }
                            ],
                        }
                    ],
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": self.valves.MAX_NEW_TOKENS,
                    "stop_token_ids": [1, 106],
                }
                resp = requests.post(
                    self.valves.VLLM_URL,
                    json=payload,
                    timeout=self.valves.TIMEOUT_SECONDS,
                )
                if resp.status_code != 200:
                    result = f"vLLM Error ({resp.status_code}): {resp.text}"
                else:
                    choices = resp.json().get("choices", [])
                    result = (
                        choices[0].get("message", {}).get("content", "")
                        if choices
                        else "No translation returned."
                    )

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

      THEN: return the tool's result COMPLETE and VERBATIM and stop. If the
      tool returns an error, show that error and stop.
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

3. **High-Throughput Sentence Splitting:**
   The tool automatically activates `split_sentences: True` when texts exceed
   250 characters. The FastAPI gateway segments the document with `pysbd`,
   dispatches segments concurrently to vLLM's continuous batching engine, and
   rejoins the translated segments in order.

4. **Long documents end-to-end:**
   - *Input side* is safe: the tool sends the full text (no `view_file` 10k
     cap, no model copy).
   - *Output side* still passes through the base model, which must re-emit the
     translation. If a very long translation is cut off at the **end**, raise
      the base model's max output tokens (model params → `max_tokens`, or the
      provider's max completion tokens) so the relay has room. The gateway
      itself never truncates; `MAX_NEW_TOKENS` (512, the gateway default) is
      per sentence-segment, not per document. If a single unsegmentable chunk
      is genuinely longer than 512 output tokens, raise the tool's
      `MAX_NEW_TOKENS` valve — but know that every extra token of headroom is
      KV-cache pressure on vLLM for *every* segment of the document.

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
- **Translation cut off at the end** → base-model output limit. Raise the
  base model's `max_tokens` (the gateway already sends the full text and the
  tool returns the full translation; only the final relay is capped).
- **Wrong language pair** → pass `source_lang`/`target_lang` explicitly, or
  set them in the tool's *User Valves*; `/translate src tgt text` also works.
- **Degenerate / looping output from the gateway** → stop tokens. Confirm the
  gateway's `/model-info` reports `stop_token_ids` containing the
  `<end_of_turn>` id (the merged checkpoint path in section 6 keeps `[1, 106]`).
