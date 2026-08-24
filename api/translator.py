"""Prompt rendering and vLLM-backed translation for the serving API.

The weights live in a vLLM server; this process holds no model. What it does
hold is the *generation contract* — the per-system prompt rendering, the twice
resolved stop set, the greedy decoding parameters — which is still a
line-for-line match with evaluate_translations.generate_translations. Those are
the fixes from docs/2026-08-10_adapter_degeneration_analysis.md, and every one
of them fails silently: an adapter queried after the wrong prefix, or decoded
without <end_of_turn> in the stop set, still returns fluent Farsi.

Two consequences shape the code below.

* Prompts are rendered here, with the checkpoint's own tokenizer, and sent to
  ``/v1/completions`` as **token ids**. Not ``/v1/chat/completions``: that
  applies the chat template with ``add_generation_prompt=True``, which is
  exactly the prefix the SFT adapter was never conditioned on. Not prompt
  strings either: the completions endpoint tokenizes with
  ``add_special_tokens=True``, which would prepend a second <bos> to a rendering
  that already carries one. Sending ids reproduces
  ``tokenize_prompts_for_generation`` exactly.
* The stop set is resolved from the tokenizer and the published
  generation_config.json and passed on every request as ``stop_token_ids``,
  rather than trusting whatever the upstream defaults to.

Throughput comes from vLLM's continuous batching: segments are dispatched
concurrently (chunked into multi-prompt requests) and the GPU lock the
in-process engine needed is gone.
"""

import asyncio
import logging
import random
from dataclasses import dataclass

import httpx
from anyio import to_thread

from config import System
from glossary.apply import Report, apply_preferred
from glossary.matcher import Index, Span, find_spans
from prompting import render_inference_prompts, resolve_stop_token_ids

logger = logging.getLogger("translategemma.api")

# Backoff base for retried requests, in seconds.
_RETRY_BACKOFF_S = 0.5

# Timeout for the startup model check, deliberately not settings.vllm_timeout.
# That defaults to 300s to accommodate a cold vLLM compiling CUDA graphs, and
# inheriting it here would hold startup for five minutes against a dead
# upstream -- the opposite of what a fail-fast check is for.
_MODEL_CHECK_TIMEOUT_S = 10.0


class SentenceSplitter:
    """pysbd segmenters, created lazily and cached per language.

    Falls back to treating the text as one segment for languages pysbd does not
    support: a translation of the whole text is a far better outcome than a 400
    on an otherwise valid request.
    """

    def __init__(self):
        self._segmenters = {}

    def split(self, text: str, language: str) -> list[str]:
        segmenter = self._segmenter(language)
        if segmenter is None:
            return [text]
        segments = [segment.strip() for segment in segmenter.segment(text)]
        return [segment for segment in segments if segment] or [text]

    def _segmenter(self, language: str):
        if language not in self._segmenters:
            import pysbd

            try:
                self._segmenters[language] = pysbd.Segmenter(language=language, clean=False)
            except ValueError:
                logger.warning(
                    "pysbd has no model for language %r; translating the text unsplit.", language
                )
                self._segmenters[language] = None
        return self._segmenters[language]


class _TokenizerProcessor:
    """Adapter giving a bare tokenizer the two attributes prompting.py wants.

    AutoProcessor is preferred because TranslateGemma is a multimodal
    checkpoint whose chat template ships with the processor; this is the
    fallback for a model directory that carries the tokenizer alone.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Bound, not wrapped in a method: rendering belongs to prompting.py, and
        # tests/test_generation_chat_template.py holds this module to that by
        # failing on any apply_chat_template call site it contains.
        self.apply_chat_template = tokenizer.apply_chat_template


def load_processor(model_path: str):
    """Load the rendering front end: processor if there is one, tokenizer if not."""
    from transformers import AutoProcessor, AutoTokenizer

    try:
        processor = AutoProcessor.from_pretrained(
            model_path, use_fast=True, fix_mistral_regex=False
        )
    except Exception as error:
        logger.warning(
            "AutoProcessor unavailable for %s (%r); falling back to AutoTokenizer.",
            model_path,
            error,
        )
        return _TokenizerProcessor(AutoTokenizer.from_pretrained(model_path, use_fast=True))
    return processor


@dataclass(frozen=True)
class TranslationResult:
    """Translations plus, per text, the pre-glossary model output and what the
    glossary did — or None for the report when off.

    A result object rather than a bare list because the glossary's report has to
    reach the response, and threading it through a second return value would put
    the two out of step on the first refactor. `raw_translations` is kept
    index-aligned with `translations` even when the glossary is off (the two
    are then equal), so a caller can always compare the pair without branching
    on whether the feature is enabled.
    """

    translations: list[str]
    raw_translations: list[str]
    reports: list["Report | None"]


class TranslationEngine:
    """Renders prompts locally and generates them on a vLLM server.

    Safe to use concurrently: it owns no mutable per-request state, and the only
    shared resources are an httpx.AsyncClient (concurrency-safe by design) and a
    semaphore bounding in-flight requests.
    """

    def __init__(self, settings):
        self.settings = settings
        self.splitter = SentenceSplitter()
        self.processor = None
        self.stop_token_ids = []
        self._client: httpx.AsyncClient | None = None
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_requests)

    # ------------------------------------------------------------------ load

    def load(self):
        """Load the tokenizer and resolve the stop set. No weights, no GPU.

        Blocking (file I/O plus a tokenizer build), so main.py still runs it in
        a worker thread; it now takes a second rather than minutes.
        """
        settings = self.settings
        tokenizer_path = settings.resolved_tokenizer_path
        logger.info(
            "Gateway starting: upstream=%s model=%s tokenizer=%s",
            settings.vllm_base_url,
            settings.vllm_model,
            tokenizer_path,
        )

        processor = load_processor(tokenizer_path)
        # No padding configuration: prompts are tokenized one rendering at a
        # time and padded by vLLM, so this tokenizer is never asked to build a
        # padded batch.

        # Resolved locally and sent on every request: a stop set configured on
        # the vLLM side, or inherited from a config.json, must not be able to
        # drop <end_of_turn>.
        self.stop_token_ids = resolve_stop_token_ids(
            processor.tokenizer, base_model_id=tokenizer_path
        )
        logger.info(
            "Stop tokens for generation: %s -> %s",
            processor.tokenizer.convert_ids_to_tokens(self.stop_token_ids),
            self.stop_token_ids,
        )
        headers = {"Content-Type": "application/json"}
        if settings.vllm_api_key:
            headers["Authorization"] = f"Bearer {settings.vllm_api_key}"
        self._client = httpx.AsyncClient(
            base_url=settings.vllm_base_url,
            timeout=settings.vllm_timeout,
            headers=headers,
            # Enough sockets for the concurrency cap, so requests queue on the
            # semaphore (and then in vLLM's scheduler) rather than on a pool.
            limits=httpx.Limits(
                max_connections=settings.max_concurrent_requests,
                max_keepalive_connections=settings.max_concurrent_requests,
            ),
        )
        self.processor = processor

    async def verify_upstream_model(self):
        """Fail fast when TG_VLLM_MODEL names something the upstream does not serve.

        A wrong model name is otherwise invisible until the first translation,
        where vLLM answers 4xx, `_post` correctly refuses to retry, and the
        caller receives a bare 500 with nothing actionable in it.

        Only a *definite* mismatch is fatal: we asked, we got an answer, and the
        configured name was not in it. An upstream we could not reach is not a
        misconfiguration -- it is a restart, a slow start, or a network blip --
        and crash-looping the gateway for it would be a worse outage than the
        one this check exists to prevent. The retry logic in `_post` already
        covers a briefly absent upstream.
        """
        settings = self.settings
        if self._client is None:
            # No client means no way to ask, which is the "could not verify"
            # case, not a definite mismatch — the same reasoning as an
            # unreachable upstream below. Nothing is hidden by being lenient
            # here: translate() already refuses to run without a client.
            logger.warning(
                "Could not verify TG_VLLM_MODEL: no upstream client. Continuing."
            )
            return
        try:
            response = await self._client.get("/models", timeout=_MODEL_CHECK_TIMEOUT_S)
        except httpx.HTTPError as error:
            logger.warning(
                "Could not verify TG_VLLM_MODEL against %s (%s). Continuing: an "
                "unreachable upstream is a restart, not a misconfiguration.",
                settings.vllm_base_url,
                error,
            )
            return

        if response.status_code != 200:
            logger.warning(
                "Could not verify TG_VLLM_MODEL: %s/models returned %s. Continuing.",
                settings.vllm_base_url,
                response.status_code,
            )
            return

        try:
            served = [entry["id"] for entry in response.json()["data"]]
        except (ValueError, KeyError, TypeError) as error:
            logger.warning(
                "Could not verify TG_VLLM_MODEL: unexpected /models body from %s (%s). "
                "Continuing.",
                settings.vllm_base_url,
                error,
            )
            return

        if settings.vllm_model not in served:
            raise ValueError(
                f"TG_VLLM_MODEL={settings.vllm_model!r} is not served by "
                f"{settings.vllm_base_url}, which serves {served}. This name must match "
                "vLLM's --served-model-name. Left unfixed, every translation fails with "
                "an opaque 500."
            )
        logger.info("Upstream at %s serves %r, as configured.", settings.vllm_base_url,
                    settings.vllm_model)

    async def aclose(self):
        """Release the upstream client. There are no weights to free."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        self.processor = None

    @property
    def is_loaded(self) -> bool:
        return self.processor is not None and self._client is not None

    @property
    def upstream(self) -> str:
        """Which vLLM answered the request, in the shape /model-info reports."""
        return f"vllm:{self.settings.vllm_base_url}"

    # ------------------------------------------------------------- translate

    async def translate(
        self,
        texts: list[str],
        system: System,
        source_lang: str,
        target_lang: str,
        max_new_tokens: int,
        split_sentences: bool,
        *,
        glossary_index: Index | None = None,
    ) -> TranslationResult:
        """Translate texts, preserving order. One output per input.

        Glossary matching runs on the FULL text before splitting: a term
        straddling a sentence boundary would otherwise be invisible to every
        segment. The spans are then applied to the rejoined translation.
        """
        if not self.is_loaded:
            raise RuntimeError("Gateway is not ready.")
        if system is not self.settings.served_system:
            raise ValueError(
                f"System {system!r} is not what this upstream serves "
                f"({self.settings.served_system})."
            )

        spans_per_text: list[list[Span]] = [
            [] if glossary_index is None else find_spans(glossary_index, text) for text in texts
        ]

        if split_sentences:
            segments_per_text = [self.splitter.split(text, source_lang) for text in texts]
        else:
            segments_per_text = [[text] for text in texts]

        flat_segments = [segment for segments in segments_per_text for segment in segments]
        flat_translations = await self._generate(
            flat_segments, system, source_lang, target_lang, max_new_tokens
        )

        translations: list[str] = []
        raw_translations: list[str] = []
        reports: list[Report | None] = []
        cursor = 0
        for text_index, segments in enumerate(segments_per_text):
            chunk = flat_translations[cursor : cursor + len(segments)]
            cursor += len(segments)
            joined = " ".join(part for part in chunk if part)
            raw_translations.append(joined)
            if glossary_index is None:
                translations.append(joined)
                reports.append(None)
                continue
            rewritten, report = apply_preferred(joined, spans_per_text[text_index])
            translations.append(rewritten)
            reports.append(report)
        return TranslationResult(
            translations=translations, raw_translations=raw_translations, reports=reports
        )

    async def _generate(
        self,
        segments: list[str],
        system: System,
        source_lang: str,
        target_lang: str,
        max_new_tokens: int,
    ) -> list[str]:
        if not segments:
            return []

        # Rendering and tokenization are pure CPU work that scales with the
        # request, so they run off the event loop, as generation used to.
        prompt_ids = await to_thread.run_sync(
            self._encode, segments, system, source_lang, target_lang
        )

        batch_size = self.settings.batch_size
        chunks = [
            prompt_ids[start : start + batch_size]
            for start in range(0, len(prompt_ids), batch_size)
        ]
        # Chunks are independent requests dispatched at once; vLLM merges them
        # with every other in-flight request into its own running batch.
        results = await asyncio.gather(
            *(self._complete(chunk, max_new_tokens) for chunk in chunks)
        )
        return [text for chunk_texts in results for text in chunk_texts]

    def _encode(
        self, segments: list[str], system: System, source_lang: str, target_lang: str
    ) -> list[list[int]]:
        """Render each segment the way its system was trained, then tokenize.

        The adapter is conditioned on the SFT rendering, which
        add_generation_prompt=True does not reproduce; the untouched base model
        is conditioned on the generation prompt. add_special_tokens=False
        matches train.py: the chat template already emits the leading special
        tokens, and adding them twice shifts every position by one.
        """
        user_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": source_lang,
                        "target_lang_code": target_lang,
                        "text": segment,
                    }
                ],
            }
            for segment in segments
        ]
        prompts = render_inference_prompts(
            self.processor, user_messages, self.settings.use_training_rendering(system)
        )
        return self.processor.tokenizer(prompts, add_special_tokens=False)["input_ids"]

    def _sampling_params(self, max_new_tokens: int) -> dict:
        """The decoding settings, in vLLM's vocabulary.

        Every sampling knob is sent explicitly, none left to default. vLLM reads
        the model directory's generation_config.json and uses it as the default
        sampling parameters (``--generation-config auto``), so anything omitted
        here is silently supplied by that file. For a checkpoint merged by
        scripts/merge_lora_adapter.py those defaults happen to be the right ones
        -- the same make_deterministic_generation_config wrote them -- but a
        checkpoint merged elsewhere, or a base model whose config.json carries
        TranslateGemma's invalid sampling defaults, would change the decoding
        without changing anything here. That is the failure this module exists
        to prevent, so the request states the whole set.

        Greedy is expressed as temperature=0 rather than do_sample=False plus a
        neutral temperature, which is the same distribution the HF path took.
        """
        settings = self.settings
        params = {
            "max_tokens": max_new_tokens,
            # Belt and braces with the generation_config.json baked into the
            # merged checkpoint: <end_of_turn> must end a generation whichever
            # of the two the upstream honours first.
            "stop_token_ids": self.stop_token_ids,
            # Matches the harness's batch_decode(skip_special_tokens=True).
            "skip_special_tokens": True,
            "include_stop_str_in_output": False,
        }
        if settings.do_sample:
            # top_k mirrors the harness's explicit 50; unset, it would come from
            # generation_config.json instead.
            params.update(temperature=settings.temperature, top_p=settings.top_p, top_k=50)
        else:
            # Under temperature=0 vLLM takes the argmax and top_p/top_k do not
            # apply, but they are neutralised anyway so the request never
            # depends on that being true.
            params.update(temperature=0.0, top_p=1.0, top_k=-1)
        return params

    async def _complete(self, prompt_ids: list[list[int]], max_new_tokens: int) -> list[str]:
        payload = {
            "model": self.settings.vllm_model,
            "prompt": prompt_ids,
            **self._sampling_params(max_new_tokens),
        }
        data = await self._post("/completions", payload)
        choices = data.get("choices", [])
        if len(choices) != len(prompt_ids):
            raise RuntimeError(
                f"vLLM returned {len(choices)} choices for {len(prompt_ids)} prompts."
            )
        # Order is not part of the OpenAI contract; index is.
        texts = [""] * len(prompt_ids)
        for choice in choices:
            texts[int(choice["index"])] = choice.get("text", "")
        # Deliberately not stripped, as the harness does not strip. Trailing
        # whitespace is the visible signature of an unstopped decoder (70% of
        # rows in the 2026-08-10 run); trimming it here would hide a regression
        # from whoever is reading the output.
        return texts

    async def _post(self, path: str, payload: dict) -> dict:
        """POST with a bounded retry on the failures a restart looks like."""
        attempts = self.settings.vllm_max_retries + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                async with self._semaphore:
                    response = await self._client.post(path, json=payload)
                if response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"vLLM returned {response.status_code}: {response.text[:500]}",
                        request=response.request,
                        response=response,
                    )
                if response.status_code >= 400:
                    # A 4xx is this gateway's bug (bad model name, prompt too
                    # long for the context window); retrying cannot help.
                    # Name the configured model: a wrong TG_VLLM_MODEL is the
                    # most common cause, and this message may be all a reader
                    # gets if the startup check could not reach the upstream.
                    raise RuntimeError(
                        f"vLLM rejected the request ({response.status_code}) for "
                        f"model {self.settings.vllm_model!r}: {response.text[:500]}"
                    )
                return response.json()
            except (httpx.TransportError, httpx.HTTPStatusError) as error:
                last_error = error
                if attempt == attempts - 1:
                    break
                # Jittered backoff: a fleet of gateway workers retrying a
                # restarting upstream in lockstep is how a restart becomes an
                # outage.
                delay = _RETRY_BACKOFF_S * (2**attempt) * (0.5 + random.random())
                logger.warning(
                    "vLLM request failed (%s); retrying in %.2fs (%d/%d).",
                    error,
                    delay,
                    attempt + 1,
                    attempts - 1,
                )
                await asyncio.sleep(delay)
        raise RuntimeError(f"vLLM request failed after {attempts} attempts: {last_error}")
