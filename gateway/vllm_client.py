"""Async HTTP client for communicating with the backend vLLM OpenAI server."""

from __future__ import annotations

import codecs
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
        """Check if vLLM server is responsive and the exact configured model is registered."""
        # 1. Check root /health endpoint
        root_url = self.base_url.removesuffix("/v1") + "/health"
        try:
            resp = await self.client.get(root_url)
            if resp.status_code != 200:
                logger.warning("vLLM /health returned status %d", resp.status_code)
                return False
        except Exception as e:
            logger.warning("Cannot contact vLLM /health: %s", e)
            return False

        # 2. Verify exact configured model is loaded in /models
        try:
            resp = await self.client.get("/models")
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("id") for m in data.get("data", [])]
                if self.settings.vllm_model_name not in models:
                    logger.error(
                        "Configured model %r not found in vLLM /models: %s",
                        self.settings.vllm_model_name,
                        models,
                    )
                    return False
                return True
        except Exception as e:
            logger.warning("Failed to query vLLM /models: %s", e)

        return False

    def build_raw_completion_payload(
        self,
        prompt: str,
        max_tokens: int,
        stream: bool = False,
        stop_token_ids: Optional[List[int]] = None,
        stop_tokens: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Construct raw JSON request body for vLLM /v1/completions endpoint."""
        stop_list = stop_tokens or self.settings.stop_tokens
        stop_ids = stop_token_ids or self.settings.stop_token_ids

        # vLLM OpenAI completions schema: top-level stop and stop_token_ids
        payload: Dict[str, Any] = {
            "model": self.settings.vllm_model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "n": 1,
            "stream": stream,
            "stop": stop_list,
            "stop_token_ids": stop_ids,
        }
        return payload

    async def generate_raw_completion(
        self,
        prompt: str,
        max_tokens: int,
        stop_token_ids: Optional[List[int]] = None,
        stop_tokens: Optional[List[str]] = None,
        request_id: Optional[str] = None,
    ) -> Tuple[str, str, Dict[str, int]]:
        """Call /v1/completions with rendered prompt and return (text, finish_reason, usage)."""
        payload = self.build_raw_completion_payload(
            prompt=prompt,
            max_tokens=max_tokens,
            stream=False,
            stop_token_ids=stop_token_ids,
            stop_tokens=stop_tokens,
        )

        headers = {}
        if request_id:
            headers["X-Request-ID"] = request_id

        try:
            resp = await self.client.post("/completions", json=payload, headers=headers)
            if resp.status_code != 200:
                logger.error("vLLM error [req_id=%s] HTTP %d: %s", request_id, resp.status_code, resp.text)
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Inference backend returned error status {resp.status_code}.",
                )

            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="Inference backend returned empty completion choices.",
                )

            choice = choices[0]
            text = choice.get("text", "")
            finish_reason = choice.get("finish_reason", "unknown")
            usage = data.get("usage", {})

            if finish_reason == "length":
                logger.error(
                    "Generation truncated by max_tokens limit (%d tokens) [req_id=%s].",
                    max_tokens,
                    request_id,
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
            logger.error("Cannot connect to vLLM at %s [req_id=%s]: %s", self.base_url, request_id, e)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Inference engine backend is unreachable.",
            ) from e
        except httpx.TimeoutException as e:
            logger.error("vLLM request timed out [req_id=%s]: %s", request_id, e)
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Inference generation timed out.",
            ) from e
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Unexpected error in vLLM communication [req_id=%s]: %s", request_id, e)
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
        """Stream completion chunks via SSE from vLLM with stateful UTF-8 decoding and terminal validation."""
        payload = self.build_raw_completion_payload(
            prompt=prompt,
            max_tokens=max_tokens,
            stream=True,
            stop_token_ids=stop_token_ids,
            stop_tokens=stop_tokens,
        )

        headers = {}
        if request_id:
            headers["X-Request-ID"] = request_id

        utf8_decoder = codecs.getincrementaldecoder("utf-8")("strict")
        observed_finish: Optional[str] = None

        try:
            async with self.client.stream("POST", "/completions", json=payload, headers=headers) as resp:
                if resp.status_code != 200:
                    err_bytes = await resp.aread()
                    err_text = err_bytes.decode("utf-8", errors="replace")
                    logger.error("vLLM streaming error [req_id=%s] HTTP %d: %s", request_id, resp.status_code, err_text)
                    yield {
                        "error": f"Inference backend returned error status {resp.status_code}.",
                        "code": "BACKEND_ERROR",
                    }
                    return

                buffer = ""
                async for raw_bytes in resp.aiter_bytes():
                    try:
                        buffer += utf8_decoder.decode(raw_bytes, final=False)
                    except UnicodeDecodeError as ue:
                        logger.error("UTF-8 incremental decoding error in stream [req_id=%s]: %s", request_id, ue)
                        yield {
                            "error": f"UTF-8 decoding failure in backend stream: {ue}",
                            "code": "DECODE_ERROR",
                        }
                        return

                    while "\n\n" in buffer:
                        event_block, buffer = buffer.split("\n\n", 1)
                        for line in event_block.splitlines():
                            line = line.strip()
                            if not line or line.startswith(":"):
                                continue
                            if line.startswith("data:"):
                                data_str = line[len("data:") :].lstrip()
                                if data_str == "[DONE]":
                                    if observed_finish is None:
                                        logger.error(
                                            "Stream protocol error [req_id=%s]: received [DONE] before finish_reason",
                                            request_id,
                                        )
                                        yield {
                                            "error": "Backend sent [DONE] without prior finish_reason.",
                                            "code": "UNEXPECTED_TERMINATION",
                                        }
                                    return
                                try:
                                    chunk = json.loads(data_str)
                                except Exception as je:
                                    logger.error(
                                        "Malformed SSE JSON in stream [req_id=%s]: %r (%s)",
                                        request_id,
                                        data_str[:100],
                                        je,
                                    )
                                    yield {
                                        "error": f"Malformed stream chunk payload: {data_str[:100]}",
                                        "code": "MALFORMED_FRAME",
                                    }
                                    return

                                choices = chunk.get("choices", [])
                                if choices:
                                    text_part = choices[0].get("text", "")
                                    finish = choices[0].get("finish_reason")
                                    if finish is not None:
                                        observed_finish = finish
                                    yield {
                                        "text": text_part,
                                        "finish_reason": finish,
                                    }
                                    if finish:
                                        return

                # Flush any final decoder bytes
                try:
                    buffer += utf8_decoder.decode(b"", final=True)
                except UnicodeDecodeError as ue:
                    logger.error("Incomplete UTF-8 sequence at stream EOF [req_id=%s]: %s", request_id, ue)
                    yield {
                        "error": f"Incomplete UTF-8 sequence at stream EOF: {ue}",
                        "code": "DECODE_ERROR",
                    }
                    return

                # Process remaining buffer if it contains complete event blocks
                while "\n\n" in buffer:
                    event_block, buffer = buffer.split("\n\n", 1)
                    for line in event_block.splitlines():
                        line = line.strip()
                        if line.startswith("data:"):
                            data_str = line[len("data:") :].lstrip()
                            if data_str == "[DONE]":
                                if observed_finish is None:
                                    yield {
                                        "error": "Backend sent [DONE] without prior finish_reason.",
                                        "code": "UNEXPECTED_TERMINATION",
                                    }
                                return
                            try:
                                chunk = json.loads(data_str)
                                choices = chunk.get("choices", [])
                                if choices:
                                    finish = choices[0].get("finish_reason")
                                    if finish is not None:
                                        observed_finish = finish
                                    yield {
                                        "text": choices[0].get("text", ""),
                                        "finish_reason": finish,
                                    }
                                    if finish:
                                        return
                            except Exception:
                                pass

                if observed_finish is None:
                    logger.error("Stream closed prematurely [req_id=%s] without terminal finish_reason", request_id)
                    yield {
                        "error": "Stream closed prematurely by backend without terminal reason.",
                        "code": "PREMATURE_CLOSE",
                    }

        except httpx.ConnectError as e:
            logger.error("Backend unreachable during streaming [req_id=%s]: %s", request_id, e)
            yield {"error": "Inference backend unreachable during stream.", "code": "BACKEND_UNAVAILABLE"}
        except httpx.TimeoutException as e:
            logger.error("Streaming timeout [req_id=%s]: %s", request_id, e)
            yield {"error": "Inference stream timed out.", "code": "STREAM_TIMEOUT"}
        except Exception as e:
            logger.exception("Unexpected error during stream [req_id=%s]: %s", request_id, e)
            yield {"error": f"Stream error: {e}", "code": "STREAM_ERROR"}

