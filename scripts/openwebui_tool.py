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
