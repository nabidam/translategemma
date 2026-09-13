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
import re

import httpx
from anyio import to_thread

from config import System
from prompting import render_inference_prompts, resolve_stop_token_ids

logger = logging.getLogger("translategemma.api")

# Backoff base for retried requests, in seconds.
_RETRY_BACKOFF_S = 0.5

# Reserve of tokens above prompt + max_new_tokens: the rendered chat template
# plus boundary slack when packed sentences re-tokenise.
_CONTEXT_RESERVE_TOKENS = 256
# Used when the upstream does not report its context length. Deliberately
# small: it only has to keep prompt + max_new_tokens legal, and the output
# budget (max_new_tokens // 2) is the tighter bound in practice.
_MAX_CONTEXT_FALLBACK = 8192


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

    def segmenter(self, language: str):
        """The pysbd segmenter for `language`, or None when pysbd lacks one.

        Exposed so the budget-aware chunker can reuse the same per-language
        cache instead of building a second set of segmenters.
        """
        return self._segmenter(language)

    def _segmenter(self, language: str):
        if language not in self._segmenters:
            try:
                import pysbd
            except ImportError:
                logger.warning(
                    "pysbd is not installed; falling back to paragraph/line "
                    "chunking for language %r.",
                    language,
                )
                self._segmenters[language] = None
                return None
            try:
                self._segmenters[language] = pysbd.Segmenter(language=language, clean=False)
            except ValueError:
                logger.warning(
                    "pysbd has no model for language %r; falling back to "
                    "paragraph/line chunking.",
                    language,
                )
                self._segmenters[language] = None
        return self._segmenters[language]


# ---------------------------------------------------------------------------
# Budget-aware chunking.
#
# vLLM rejects a request whose prompt plus max_new_tokens exceeds the model's
# context length, and a segment whose translation outgrows max_new_tokens is
# silently clipped at the stop. Both are properties of the *chunk*, so the
# chunk size must be bounded before dispatch. The hierarchy mirrors the
# proven quick_pipeline chunker (paragraph -> sentence -> hard slice) and the
# greedy sentence packing it uses, with the budget expressed in tokens:
#
#   1. pysbd sentences for the source language;
#   2. paragraph and line boundaries when pysbd has no model for the language
#      (its fallback of "the whole text is one segment" is exactly how a
#      long document becomes a single over-context prompt);
#   3. hard slices at token boundaries for a unit that still exceeds the
#      budget (a punctuation-free wall of text, a table row, a code block).
#
# The pure helpers below carry the logic; the tokenizer only measures and
# decodes, which keeps the packing testable without one.
# ---------------------------------------------------------------------------


def pack_units(units: list[str], costs: list[int], limit: int) -> list[list[int]]:
    """Greedily pack unit indices into groups whose summed cost stays <= limit.

    A unit that alone exceeds `limit` is emitted as its own group (the caller
    pre-splits such units; if one slips through it is still isolated, so no
    *packed* group ever crosses the limit). Empty units are dropped. Joining
    the packed units is the caller's job.
    """
    groups: list[list[int]] = []
    current: list[int] = []
    current_cost = 0
    for index, (unit, cost) in enumerate(zip(units, costs)):
        if not unit:
            continue
        if cost > limit:
            if current:
                groups.append(current)
                current, current_cost = [], 0
            groups.append([index])
            continue
        if current and current_cost + cost > limit:
            groups.append(current)
            current, current_cost = [], 0
        current.append(index)
        current_cost += cost
    if current:
        groups.append(current)
    return groups


def fallback_units(text: str) -> list[str]:
    """Paragraph/line units for languages pysbd has no model for."""
    units = []
    for paragraph in re.split(r"\n\s*\n", text):
        units.extend(line.strip() for line in paragraph.splitlines() if line.strip())
    return units


def token_windows(token_ids: list[int], limit: int) -> list[list[int]]:
    """Slice a token sequence into windows of at most `limit` tokens."""
    return [
        token_ids[start : start + limit] for start in range(0, len(token_ids), limit)
    ]


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
        self.max_context_tokens = 0
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

        # Resolved once at startup: chunk sizes are derived from it, and a
        # request that ignores it dies with a 400 deep in a long document.
        self.max_context_tokens = self._detect_max_context_tokens(headers)
        logger.info("Upstream max context length: %d tokens.", self.max_context_tokens)

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

    # --------------------------------------------------------------- chunking

    def _detect_max_context_tokens(self, headers: dict) -> int:
        """The upstream's context length: configured, or probed, or a safe floor.

        vLLM reports ``max_model_len`` on /v1/models; a probe that fails or a
        server that omits the field must not break startup, so it degrades to
        ``_MAX_CONTEXT_FALLBACK`` (small on purpose -- see the constant).
        """
        configured = self.settings.max_context_tokens
        if configured > 0:
            return configured
        try:
            response = httpx.get(
                f"{self.settings.vllm_base_url}/models",
                headers=headers,
                timeout=10.0,
            )
            if response.status_code == 200:
                data = response.json().get("data") or []
                for model in data:
                    length = model.get("max_model_len")
                    if isinstance(length, int) and length > 0:
                        return length
        except Exception as error:
            logger.warning(
                "Could not probe the upstream's max context length (%s); "
                "using %d.",
                error,
                _MAX_CONTEXT_FALLBACK,
            )
        return _MAX_CONTEXT_FALLBACK

    def _source_budget(self, max_new_tokens: int) -> int:
        """Max source tokens per chunk, under both ceilings that can bite.

        * The *context* ceiling: the rendered prompt (source plus template
          overhead) plus the whole output must fit the upstream window.
        * The *output* ceiling: a chunk whose translation outgrows
          ``max_new_tokens`` is clipped at the stop, mid-sentence. Cross-
          language output is at most a small multiple of the source token
          count, so capping the source at half the output budget keeps normal
          pairs (e.g. EN->FA, where the target runs longer) safely unclipped
          while a pathologically long single sentence is still sliced.
        """
        context_room = self.max_context_tokens - max_new_tokens - _CONTEXT_RESERVE_TOKENS
        budget = min(context_room, max_new_tokens // 2)
        # Never so small that a whole word no longer fits: a floor keeps the
        # hard-slicer from emitting empty windows on a tiny context.
        return max(budget, 16)

    def _chunk_text(
        self, text: str, language: str, source_budget: int
    ) -> list[str]:
        """Split one text into chunks whose source stays within `source_budget`.

        Sentence-level (pysbd) when the language is covered, paragraph/line
        otherwise, then hard-sliced at token boundaries as a last resort, then
        greedily re-packed so a chunk carries as many whole sentences as the
        budget allows. Packing (rather than one prompt per sentence) is what
        quick_pipeline does, and it both cuts the request count and gives the
        model neighbouring context for a better translation.
        """
        if not text.strip():
            return []

        segmenter = self.splitter.segmenter(language)
        if segmenter is not None:
            units = [u.strip() for u in segmenter.segment(text) if u.strip()]
        else:
            units = fallback_units(text)
        if not units:
            return [text]

        tokenizer = self.processor.tokenizer
        # Tokenize all units in one call: the fast tokenizer batches in C, and
        # per-unit Python calls would dominate on a long document.
        unit_ids = tokenizer(units, add_special_tokens=False)["input_ids"]

        flat_units: list[str] = []
        flat_costs: list[int] = []
        for unit, ids in zip(units, unit_ids):
            if len(ids) <= source_budget:
                flat_units.append(unit)
                flat_costs.append(len(ids))
                continue
            # One unit over budget: slice at token boundaries so a slice never
            # cuts a token, then let the packer group the slices like any other
            # units. Decoding a slice and re-encoding it in _encode can shift a
            # token or two at the seam; _CONTEXT_RESERVE_TOKENS absorbs that.
            for window in token_windows(ids, source_budget):
                flat_units.append(tokenizer.decode(window).strip())
                flat_costs.append(len(window))
        flat_units = [u for u in flat_units if u]
        flat_costs = [c for c, u in zip(flat_costs, flat_units) if u]

        return [
            " ".join(flat_units[index] for index in group)
            for group in pack_units(flat_units, flat_costs, source_budget)
        ]

    def _chunk_texts(
        self, texts: list[str], language: str, source_budget: int
    ) -> list[list[str]]:
        """_chunk_text for a list of texts, preserving order. Sync on purpose:
        it runs in a worker thread via to_thread from translate()."""
        return [self._chunk_text(text, language, source_budget) for text in texts]

    # ------------------------------------------------------------- translate

    async def translate(
        self,
        texts: list[str],
        system: System,
        source_lang: str,
        target_lang: str,
        max_new_tokens: int,
        split_sentences: bool,
    ) -> list[str]:
        """Translate texts, preserving order. One output per input.

        With split_sentences, each text is chunked to fit the upstream context
        window and the output budget (see _chunk_text), every chunk of every
        text is sent to the shared upstream, and the chunks are rejoined per
        text. That keeps the server busy even when one request carries a single
        long document, and an oversized document degrades to more chunks
        instead of a 400.
        """
        if not self.is_loaded:
            raise RuntimeError("Gateway is not ready.")
        if system is not self.settings.served_system:
            raise ValueError(
                f"System {system!r} is not what this upstream serves "
                f"({self.settings.served_system})."
            )

        if split_sentences:
            source_budget = self._source_budget(max_new_tokens)
            # Segmentation, tokenization and packing are pure CPU work that
            # scales with the document, so they run off the event loop, as
            # _encode does.
            segments_per_text = await to_thread.run_sync(
                self._chunk_texts, texts, source_lang, source_budget
            )
        else:
            segments_per_text = [[text] for text in texts]

        flat_segments = [segment for segments in segments_per_text for segment in segments]
        flat_translations = await self._generate(
            flat_segments, system, source_lang, target_lang, max_new_tokens
        )

        translations = []
        cursor = 0
        for segments in segments_per_text:
            chunk = flat_translations[cursor : cursor + len(segments)]
            cursor += len(segments)
            translations.append(" ".join(part for part in chunk if part))
        return translations

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
        clipped = 0
        for choice in choices:
            texts[int(choice["index"])] = choice.get("text", "")
            if choice.get("finish_reason") == "length":
                clipped += 1
        if clipped:
            # The chunk's translation hit max_new_tokens: the end of the
            # translation is missing, silently. The budget in _source_budget
            # exists to prevent this; a hit means a pair expands more than
            # expected, and MAX_NEW_TOKENS should go up.
            logger.warning(
                "%d of %d segments hit max_new_tokens=%d and are clipped; "
                "raise MAX_NEW_TOKENS for this language pair.",
                clipped,
                len(prompt_ids),
                max_new_tokens,
            )
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
                    raise RuntimeError(
                        f"vLLM rejected the request ({response.status_code}): "
                        f"{response.text[:500]}"
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
