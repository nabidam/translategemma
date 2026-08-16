"""Admission control, token estimation, and concurrency/queue management for the Gateway."""

import asyncio
import logging
import time
from typing import List, Optional
from fastapi import HTTPException, status

from config import Settings

logger = logging.getLogger("gateway.limits")


class TokenEstimator:
    """Estimates token counts for admission control and length bucketing."""

    def __init__(self, tokenizer_path: Optional[str] = None):
        self._tokenizer = None
        if tokenizer_path:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, fix_markdown=False)
                logger.info("Loaded exact tokenizer from %s for estimation.", tokenizer_path)
            except Exception as e:
                logger.warning("Could not load tokenizer from %s (%s); using heuristic estimator.", tokenizer_path, e)

    def estimate_tokens(self, text: str) -> int:
        if self._tokenizer is not None:
            try:
                return len(self._tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                pass
        # Fast heuristic: ~3.5 characters per token for English & Persian mixed scripts + overhead
        return max(1, int(len(text) / 3.2) + 4)

    def estimate_batch_tokens(self, texts: List[str]) -> List[int]:
        return [self.estimate_tokens(t) for t in texts]


class ConcurrencyManager:
    """Bounded in-flight concurrency limiter with queue depth tracking and 429 backpressure."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
        self.in_flight = 0
        self.queued = 0
        self._lock = asyncio.Lock()

    async def acquire(self) -> float:
        """Attempt to enter the execution queue. Returns wait time in seconds."""
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
        try:
            await self.semaphore.acquire()
        finally:
            async with self._lock:
                self.queued -= 1
                self.in_flight += 1

        wait_time = time.perf_counter() - start_wait
        return wait_time

    def release(self):
        """Release concurrency slot."""
        self.semaphore.release()
        self.in_flight = max(0, self.in_flight - 1)


class RequestValidator:
    """Enforces size and token limits before dispatching to backend."""

    def __init__(self, settings: Settings, estimator: TokenEstimator):
        self.settings = settings
        self.estimator = estimator

    def validate_text(self, text: str, max_new_tokens: int) -> int:
        stripped = text.strip()
        if not stripped:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Source text cannot be empty or whitespace only.",
            )

        if len(stripped) > self.settings.max_source_chars_per_text:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    f"Text length ({len(stripped)} chars) exceeds maximum allowed "
                    f"({self.settings.max_source_chars_per_text} chars)."
                ),
            )

        est_tokens = self.estimator.estimate_tokens(stripped)
        if est_tokens > self.settings.max_estimated_source_tokens:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    f"Estimated source tokens ({est_tokens}) exceeds maximum limit "
                    f"({self.settings.max_estimated_source_tokens})."
                ),
            )

        total_context = est_tokens + max_new_tokens + 32
        if total_context > self.settings.max_total_context_tokens:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Combined prompt + output budget ({total_context} tokens) exceeds "
                    f"model context window ({self.settings.max_total_context_tokens})."
                ),
            )

        return est_tokens

    def validate_batch(self, texts: List[str], max_new_tokens: int) -> List[int]:
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

        return [self.validate_text(t, max_new_tokens) for t in texts]
