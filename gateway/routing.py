"""Workload classification, sentence splitting, and structured batch dispatch routing."""

from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import Callable, Coroutine, List, Optional, Tuple

import pysbd

from config import Settings
from limits import TokenEstimator

logger = logging.getLogger("gateway.routing")


class WorkloadClass(str, Enum):
    INTERACTIVE = "interactive"
    DOCUMENT = "document"
    BULK = "bulk"


class WorkloadClassifier:
    """Classifies incoming requests based on size, item count, and token estimates."""

    def __init__(self, settings: Settings, estimator: TokenEstimator):
        self.settings = settings
        self.estimator = estimator

    def classify_single(self, text: str, est_tokens: int) -> WorkloadClass:
        if est_tokens <= self.settings.interactive_max_tokens:
            return WorkloadClass.INTERACTIVE
        return WorkloadClass.DOCUMENT

    def classify_batch(self, texts: List[str]) -> WorkloadClass:
        if len(texts) == 1:
            est = self.estimator.count_tokens(texts[0])
            return self.classify_single(texts[0], est)
        return WorkloadClass.BULK


class SentenceSplitter:
    """Performs language-aware sentence segmentation with deterministic reassembly."""

    def __init__(self):
        self._segmenters = {}

    def _get_segmenter(self, lang: str) -> pysbd.Segmenter:
        if lang not in self._segmenters:
            try:
                self._segmenters[lang] = pysbd.Segmenter(language=lang, clean=False)
            except Exception:
                # Default to English rules if language not explicitly supported by pysbd
                self._segmenters[lang] = pysbd.Segmenter(language="en", clean=False)
        return self._segmenters[lang]

    def split_sentences(self, text: str, lang: str = "en") -> List[str]:
        if not text.strip():
            return []
        segmenter = self._get_segmenter(lang)
        segments = segmenter.segment(text)
        cleaned = [s.strip() for s in segments if s.strip()]
        return cleaned if cleaned else [text.strip()]


async def dispatch_structured_batch(
    items: List[str],
    translate_fn: Callable[[str], Coroutine[None, None, str]],
) -> List[str]:
    """Execute concurrent translations with structured cancellation on failure."""
    tasks = [asyncio.create_task(translate_fn(item)) for item in items]
    try:
        results = await asyncio.gather(*tasks)
        return list(results)
    except Exception as e:
        logger.error("Error during structured batch execution; cancelling pending tasks: %s", e)
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
