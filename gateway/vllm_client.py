"""Async HTTP client for communicating with the backend vLLM OpenAI server."""

import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import httpx
from fastapi import HTTPException, status

from config import Settings

logger = logging.getLogger("gateway.vllm_client")


class VLLMClient:
    """Manages persistent connection pooling, raw completions, and streaming to vLLM."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = settings.vllm_base_url.rstrip("/")
        timeout = httpx.Timeout(
            timeout=settings.vllm_timeout_seconds,
            connect=settings.vllm_connect_timeout_seconds,
        )
        limits = httpx.Limits(
            max_connections=settings.max_concurrent_requests + 32,
            max_keepalive_connections=settings.max_concurrent_requests,
        )
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            limits=limits,
            headers={"User-Agent": "TranslateGemma-Gateway/1.0.0"},
        )

    async def close(self):
        await self.client.aclose()

    async def check_health(self) -> bool:
        """Check if vLLM server is responsive and models are registered."""
        # 1. Try root /health endpoint
        root_url = self.base_url.removesuffix("/v1") + "/health"
        try:
            resp = await self.client.get(root_url)
            if resp.status_code == 200:
                return True
        except Exception:
            pass

        # 2. Fall back to /models
        try:
            resp = await self.client.get("/models")
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("id") for m in data.get("data", [])]
                return self.settings.vllm_model_name in models or len(models) > 0
        except Exception as e:
            logger.warning("Failed to contact vLLM /models: %s", e)

        return False

    async def generate_raw_completion(
        self,
        prompt: str,
        max_tokens: int,
        stop_token_ids: Optional[List[int]] = None,
        stop_tokens: Optional[List[str]] = None,
        request_id: Optional[str] = None,
    ) -> Tuple[str, str, Dict[str, int]]:
        """Call /v1/completions with rendered prompt and return (text, finish_reason, usage).

        Output text is preserved without stripping to retain stop diagnostics.
        Finish reason 'length' is treated as an explicit error.
        """
        stop_list = stop_tokens or self.settings.stop_tokens
        stop_ids = stop_token_ids or self.settings.stop_token_ids

        payload: Dict[str, Any] = {
            "model": self.settings.vllm_model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "n": 1,
            "stream": False,
            "stop": stop_list,
            "extra_body": {
                "stop_token_ids": stop_ids,
            },
        }

        headers = {}
        if request_id:
            headers["X-Request-ID"] = request_id

        try:
            resp = await self.client.post("/completions", json=payload, headers=headers)
            if resp.status_code != 200:
                logger.error("vLLM error HTTP %d: %s", resp.status_code, resp.text)
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"vLLM backend returned error {resp.status_code}: {resp.text}",
                )

            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="vLLM returned empty completion choices.",
                )

            choice = choices[0]
            text = choice.get("text", "")
            finish_reason = choice.get("finish_reason", "unknown")
            usage = data.get("usage", {})

            if finish_reason == "length":
                logger.error(
                    "Generation truncated by max_tokens limit (%d tokens) without reaching stop token.",
                    max_tokens,
                )
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=(
                        f"Generation truncated: model reached maximum token budget ({max_tokens}) "
                        "without producing an EOS stop token."
                    ),
                )

            # Return raw text without strip() to preserve stop diagnostics
            return text, finish_reason, usage

        except httpx.ConnectError as e:
            logger.error("Cannot connect to vLLM at %s: %s", self.base_url, e)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Inference engine backend is unreachable.",
            ) from e
        except httpx.TimeoutException as e:
            logger.error("vLLM request timed out: %s", e)
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Inference generation timed out.",
            ) from e
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Unexpected error in vLLM communication: %s", e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Inference gateway error: {e}",
            ) from e

    async def generate_stream_completion(
        self,
        prompt: str,
        max_tokens: int,
        stop_token_ids: Optional[List[int]] = None,
        stop_tokens: Optional[List[str]] = None,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream completion chunks via SSE from vLLM without stripping content."""
        stop_list = stop_tokens or self.settings.stop_tokens
        stop_ids = stop_token_ids or self.settings.stop_token_ids

        payload: Dict[str, Any] = {
            "model": self.settings.vllm_model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "n": 1,
            "stream": True,
            "stop": stop_list,
            "extra_body": {
                "stop_token_ids": stop_ids,
            },
        }

        headers = {}
        if request_id:
            headers["X-Request-ID"] = request_id

        try:
            async with self.client.stream("POST", "/completions", json=payload, headers=headers) as resp:
                if resp.status_code != 200:
                    err_text = await resp.aread()
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail=f"vLLM streaming error {resp.status_code}: {err_text.decode('utf-8', errors='replace')}",
                    )

                buffer = ""
                async for raw_bytes in resp.aiter_bytes():
                    buffer += raw_bytes.decode("utf-8", errors="replace")
                    while "\n\n" in buffer:
                        event_block, buffer = buffer.split("\n\n", 1)
                        for line in event_block.splitlines():
                            if line.startswith("data:"):
                                data_str = line[len("data:") :].lstrip()
                                if data_str == "[DONE]":
                                    return
                                try:
                                    chunk = json.loads(data_str)
                                    choices = chunk.get("choices", [])
                                    if choices:
                                        text_part = choices[0].get("text", "")
                                        finish = choices[0].get("finish_reason")
                                        yield {
                                            "text": text_part,
                                            "finish_reason": finish,
                                        }
                                        if finish:
                                            return
                                except Exception:
                                    continue

        except httpx.ConnectError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Inference backend unreachable during stream.",
            ) from e
        except httpx.TimeoutException as e:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Inference stream timed out.",
            ) from e
