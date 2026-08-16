"""Admission control, exact token counting, and concurrency/queue management for the Gateway."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple
from fastapi import HTTPException, status

from config import Settings

logger = logging.getLogger("gateway.limits")


class TokenEstimator:
    """Exact tokenizer loader with safe heuristic fallback."""

    def __init__(self, tokenizer_path: Optional[str] = None):
        self._tokenizer = None
        self.mode = "heuristic"

        if tokenizer_path:
            p = Path(tokenizer_path)
            if p.is_dir() or p.is_file():
                try:
                    # 1. Try tokenizers Fast Tokenizer
                    from tokenizers import Tokenizer
                    tok_file = p if p.is_file() else (p / "tokenizer.json")
                    if tok_file.is_file():
                        self._tokenizer = Tokenizer.from_file(str(tok_file))
                        self.mode = "exact"
                        logger.info("Loaded exact fast tokenizer from %s", tok_file)
                except Exception as e1:
                    logger.debug("Fast tokenizer load failed (%s); trying transformers AutoTokenizer...", e1)
                    try:
                        from transformers import AutoTokenizer
                        self._tokenizer = AutoTokenizer.from_pretrained(str(p), fix_markdown=False)
                        self.mode = "exact"
                        logger.info("Loaded exact AutoTokenizer from %s", p)
                    except Exception as e2:
                        logger.warning(
                            "Could not load exact tokenizer from %s (Fast: %s, HF: %s). Falling back to heuristic.",
                            tokenizer_path,
                            e1,
                            e2,
                        )

        if self.mode == "heuristic":
            logger.warning(
                "TokenEstimator running in HEURISTIC mode. For production admission accuracy, configure TG_TOKENIZER_PATH."
            )

    def count_tokens(self, text: str) -> int:
        if self._tokenizer is not None:
            try:
                if hasattr(self._tokenizer, "encode"):
                    res = self._tokenizer.encode(text)
                    if hasattr(res, "ids"):
                        return len(res.ids)
                    if isinstance(res, list):
                        return len(res)
            except Exception:
                pass
        # Heuristic fallback: ~3.2 characters per token for mixed English/Persian + 4 token framing
        return max(1, int(len(text) / 3.2) + 4)

    def count_batch_tokens(self, texts: List[str]) -> List[int]:
        return [self.count_tokens(t) for t in texts]


class ConcurrencyManager:
    """Bounded in-flight concurrency limiter with queue depth tracking and fair bulk queuing."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
        self.bulk_semaphore = asyncio.Semaphore(settings.max_bulk_concurrent_requests)
        self.in_flight = 0
        self.queued = 0
        self._lock = asyncio.Lock()

    async def acquire(self, is_bulk: bool = False) -> float:
        """Attempt to enter execution queue with leak-free cancellation handling. Returns wait time."""
        async with self._lock:
            if self.queued >= self.settings.max_queue_depth:
                logger.warning(
                    "Gateway queue saturated (queued=%d, in_flight=%d, max_queue=%d)",
                    self.queued,
                    self.in_flight,
                    self.settings.max_queue_depth,
                )
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=(
                        f"Server is under high load. Waiting queue saturated "
                        f"({self.queued}/{self.settings.max_queue_depth}). Please retry with backoff."
                    ),
                )
            self.queued += 1

        start_wait = time.perf_counter()
        acquired_global = False
        acquired_bulk = False
        try:
            if is_bulk:
                await self.bulk_semaphore.acquire()
                acquired_bulk = True

            await self.semaphore.acquire()
            acquired_global = True
        finally:
            async with self._lock:
                self.queued -= 1
                if acquired_global:
                    self.in_flight += 1
                else:
                    # Clean up bulk semaphore if global acquisition failed or was cancelled
                    if acquired_bulk:
                        self.bulk_semaphore.release()

        wait_time = time.perf_counter() - start_wait
        return wait_time

    def release(self, is_bulk: bool = False):
        """Release acquired concurrency slots."""
        self.semaphore.release()
        if is_bulk:
            self.bulk_semaphore.release()
        self.in_flight = max(0, self.in_flight - 1)


class RequestValidator:
    """Enforces size, character, and context window limits before dispatching to backend."""

    def __init__(self, settings: Settings, estimator: TokenEstimator):
        self.settings = settings
        self.estimator = estimator

    def validate_request(self, raw_text: str, rendered_prompt: str, max_new_tokens: int) -> int:
        stripped = raw_text.strip()
        if not stripped:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Source text cannot be empty or whitespace only.",
            )

        if len(raw_text) > self.settings.max_source_chars_per_text:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    f"Text length ({len(raw_text)} chars) exceeds maximum allowed "
                    f"({self.settings.max_source_chars_per_text} chars)."
                ),
            )

        prompt_tokens = self.estimator.count_tokens(rendered_prompt)
        if prompt_tokens > self.settings.max_estimated_source_tokens:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    f"Prompt tokens ({prompt_tokens}) exceeds maximum prompt token limit "
                    f"({self.settings.max_estimated_source_tokens})."
                ),
            )

        total_context = prompt_tokens + max_new_tokens
        if total_context > self.settings.max_total_context_tokens:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Combined prompt ({prompt_tokens} tokens) + output budget ({max_new_tokens} tokens) = "
                    f"{total_context} tokens, which exceeds the model context window "
                    f"({self.settings.max_total_context_tokens} tokens)."
                ),
            )

        return prompt_tokens

    def validate_batch(self, texts: List[str], max_new_tokens: int) -> None:
        if not texts:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Batch texts list cannot be empty.",
            )

        if len(texts) > self.settings.max_batch_items:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    f"Batch contains {len(texts)} items, exceeding maximum allowed "
                    f"({self.settings.max_batch_items})."
                ),
            )

        total_chars = sum(len(t) for t in texts)
        if total_chars > self.settings.max_batch_total_chars:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    f"Aggregate batch length ({total_chars} chars) exceeds maximum allowed "
                    f"({self.settings.max_batch_total_chars} chars)."
                ),
            )

        for i, text in enumerate(texts):
            if not text.strip():
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Batch item at index {i} is empty or whitespace only.",
                )
