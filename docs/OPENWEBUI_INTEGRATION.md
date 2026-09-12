# TranslateGemma 27B OpenWebUI (v0.11.3) Integration Guide

This guide details how to integrate the fine-tuned and merged **TranslateGemma-27B** model (served with vLLM on an offline host) into **OpenWebUI v0.11.3**.

---

## 1. Architecture Overview

TranslateGemma requires three mandatory inputs for any translation:
1. **`text`**: The input segment to translate.
2. **`source_lang`**: Source language ISO code (e.g. `en`, `fa`, `de`, `fr`, `ru`).
3. **`target_lang`**: Target language ISO code (e.g. `fa`, `en`, `de`, `fr`, `ru`).

Standard OpenWebUI chat endpoints pass only a flat string (`{"role": "user", "content": "..."}`). Direct connections to standard vLLM fail for three reasons:
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
│  User types: /translate en fa Cellular biology is the study of cells... │
│  or triggers interactive modal via Workspace > Prompts (/translate)     │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ OpenWebUI Tool: TranslateGemmaTool (Workspace > Tools)            │  │
│  │ - Captures: text, source_lang, target_lang                        │  │
│  │ - Configured via Valves (Admin) & UserValves (User Settings)      │  │
│  │ - Sends status events to UI via __event_emitter__                 │  │
│  └─────────────────────────────────┬─────────────────────────────────┘  │
└────────────────────────────────────┼────────────────────────────────────┘
                                     │ HTTP POST (Internal Docker Network)
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ translategemma-api Gateway (:8000/translate)                            │
│ - Validates inputs (source_lang, target_lang, text)                     │
│ - Renders official SFT Jinja template via prompting.py                  │
│ - Resolves stop token IDs: [1, 106] (<end_of_turn>)                     │
│ - Sets greedy decoding: temperature=0.0, top_p=1.0, top_k=-1            │
│ - Encodes to token IDs and posts to vLLM /v1/completions                │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ Token IDs + stop_token_ids
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ translategemma-vllm (:8000)                                             │
│ - vLLM engine serving merged translategemma-27b weights                 │
│ - Decodes tokens using continuous batching & PagedAttention             │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. OpenWebUI Tool Implementation (`Workspace > Tools`)

In OpenWebUI v0.11.3, custom extensibility uses **Tools** (`class Tools:`). Tools:
- Expose clear parameter schemas (`text`, `source_lang`, `target_lang`) to the UI and model agents.
- Can be triggered directly by slash commands or function calling.
- Support **`Valves`** (Admin configuration) and **`UserValves`** (per-user preferences for language pairs).
- Support **`__event_emitter__`** to display live progress chips in the chat UI.

### Tool Code

Navigate in OpenWebUI to **Workspace → Tools → Add Tool (`+`)**, name it `translategemma_tool`, and paste:

```python
"""
title: TranslateGemma Translation Tool
author: TranslateGemma Team
description: High-accuracy translation using finetuned TranslateGemma 27B with official multimodal template formatting and stop-token preservation.
version: 1.1.0
license: MIT
requirements: requests, pydantic
"""

import json
import re
from typing import Any, Callable, Optional
from pydantic import BaseModel, Field
import requests


class Tools:
    class Valves(BaseModel):
        BACKEND_MODE: str = Field(
            default="gateway",
            description="Backend mode: 'gateway' (translategemma-api gateway) or 'vllm' (direct vLLM chat endpoint)",
        )
        GATEWAY_URL: str = Field(
            default="http://translategemma-api:8000/translate",
            description="URL to the TranslateGemma FastAPI gateway (/translate)",
        )
        VLLM_URL: str = Field(
            default="http://translategemma-vllm:8000/v1/chat/completions",
            description="URL to vLLM chat completions if using a patched vLLM endpoint directly",
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
            default=1024,
            description="Maximum new tokens to generate",
        )
        TIMEOUT_SECONDS: int = Field(
            default=120,
            description="Request timeout in seconds",
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

    def _resolve_languages(
        self,
        source_lang: Optional[str],
        target_lang: Optional[str],
        __user__: Optional[dict],
    ) -> tuple[str, str]:
        # 1. Explicit arguments
        src = (source_lang or "").strip().lower()
        tgt = (target_lang or "").strip().lower()

        # 2. User valves
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

        # 3. Global valves
        if not src:
            src = self.valves.DEFAULT_SOURCE_LANG.strip().lower()
        if not tgt:
            tgt = self.valves.DEFAULT_TARGET_LANG.strip().lower()

        return src, tgt

    async def translate(
        self,
        text: str,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Translates text between languages using finetuned TranslateGemma 27B.

        :param text: The source text to translate.
        :param source_lang: Source language code (e.g., 'en', 'fa', 'de', 'fr', 'ru').
        :param target_lang: Target language code (e.g., 'fa', 'en', 'de', 'fr', 'ru').
        :return: Translated text.
        """
        clean_text = (text or "").strip()
        if not clean_text:
            return "Error: No text provided for translation."

        # Parse inline /translate commands if user passed the whole string to 'text'
        m = re.match(
            r"^/translate\s+([a-zA-Z]{2,5}(?:-[a-zA-Z0-9]+)?)(?:->|:|\s+)([a-zA-Z]{2,5}(?:-[a-zA-Z0-9]+)?)\s+(.+)$",
            clean_text,
            re.DOTALL | re.IGNORECASE,
        )
        if m:
            source_lang = m.group(1)
            target_lang = m.group(2)
            clean_text = m.group(3).strip()
        else:
            m_simple = re.match(r"^/translate\s+(.+)$", clean_text, re.DOTALL | re.IGNORECASE)
            if m_simple:
                clean_text = m_simple.group(1).strip()

        src, tgt = self._resolve_languages(source_lang, target_lang, __user__)

        # Emit status indicator to OpenWebUI chat
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Translating [{src} ➔ {tgt}] via TranslateGemma...",
                        "done": False,
                    },
                }
            )

        try:
            if self.valves.BACKEND_MODE == "gateway":
                # Route A: Via translategemma-api gateway
                payload = {
                    "text": clean_text,
                    "source_lang": src,
                    "target_lang": tgt,
                    "max_new_tokens": self.valves.MAX_NEW_TOKENS,
                    "split_sentences": False,
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
                # Route B: Direct vLLM chat endpoint with official multimodal structure
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
                                    "text": clean_text,
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
                            "description": f"Completed translation [{src} ➔ {tgt}]",
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

## 3. Configuring the `/translate` Command (`Workspace > Prompts`)

OpenWebUI v0.11.3 parses prompt variables using simple identifiers `{{variable_name}}` without spaces.

### Option A: Standard Translation (Prompts for Text Only, Defaults to EN ➔ FA)
This is the recommended setup for everyday use so users are not forced to enter `source_lang` and `target_lang` on every translation.

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
- When typing `/translate`, the modal opens with **only one field: `text`** (rendered as a textarea).
- On submission, the tool uses the configured default languages (`en` ➔ `fa`).

---

### Option B: Custom Language Pair Translation
When a user specifically needs to translate between other language pairs (e.g., French or German to Persian):

1. Go to **Workspace → Prompts → Add Prompt (`+`)**.
2. Configure:
   - **Title**: `Translate with Language Pair`
   - **Command**: `/translate-pair`
   - **Content**:
     ```text
     Translate the following text from {{source_lang}} to {{target_lang}}:

     {{text}}
     ```
3. Save.

**Behavior**:
- Opens a modal with 3 clean fields: `source_lang`, `target_lang`, and `text`.

---

### Option C: Direct In-Chat Translation (Zero Modal)
When using the custom model with `translategemma_tool` attached, you can also bypass the modal entirely:
- Type or paste text directly:
  `Cellular biology is the study of cell structure and function.`
- Or use inline command syntax:
  - `/translate Cellular biology is the study of cell structure and function.` (uses default EN ➔ FA)
  - `/translate fr fa Bonjour le monde` (explicit pair)
  - `/translate de->fa Das ist ein Test` (arrow syntax)

---

## 4. Docker Compose Setup for Offline Host

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

## 5. Verification Checklist

1. **Verify Backend Health**:
   ```bash
   curl -s http://localhost:8000/health-check
   # Expected output: {"translator":"OK"}
   ```

2. **Verify Official Template Translation**:
   ```bash
   curl -s http://localhost:8000/translate \
     -H 'Content-Type: application/json' \
     -d '{
       "text": "Cellular biology is the study of cell structure and function.",
       "source_lang": "en",
       "target_lang": "fa"
     }'
   ```
   Confirm that the Persian translation terminates naturally without repetition.

3. **Verify Tool in OpenWebUI**:
   - In OpenWebUI, open a chat with any base model and ensure `translategemma_tool` is toggled on under Tools.
   - Run `/translate en fa CRISPR technology allows targeted genome editing.`
   - Verify that the status chip shows `Translating [en ➔ fa] via TranslateGemma...` followed by `Completed translation [en ➔ fa]`.
