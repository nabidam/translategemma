"""TranslateGemma FastAPI Serving Gateway.

High-throughput, continuous-batching proxy in front of vLLM OpenAI server.
Preserves canonical SFT training prompt rendering, stop contracts, and response schemas.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from config import Settings, get_settings
from limits import ConcurrencyManager, RequestValidator, TokenEstimator
from metrics import MetricsCollector, get_metrics
from prompting import (
    CHAT_TURN_END_TOKEN,
    TARGET_BOUNDARY_MARKER,
    render_training_prompt,
)
from routing import SentenceSplitter, WorkloadClass, WorkloadClassifier, dispatch_structured_batch
from schemas import (
    BatchPrompt,
    BatchTranslationResponse,
    HealthResponse,
    ModelInfoResponse,
    Prompt,
    TranslationResponse,
)
from vllm_client import VLLMClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("gateway.main")


class CanonicalPromptRenderer:
    """Renders prompts using canonical chat template and assistant-turn boundary marker."""

    def __init__(self, processor_or_tokenizer: Any = None):
        self.processor = processor_or_tokenizer

    def render(self, source_lang: str, target_lang: str, text: str) -> str:
        user_message = {
            "role": "user",
            "content": f"<<<source>>>{source_lang}<<<target>>>{target_lang}<<<text>>>{text}",
        }
        if self.processor is not None and hasattr(self.processor, "apply_chat_template"):
            return render_training_prompt(self.processor, user_message)

        # Canonical fallback matching exact chat_template.jinja SFT block formatting
        # with the exact TARGET_BOUNDARY_MARKER cut
        marker_template = (
            f"<start_of_turn>user\n"
            f"{user_message['content']}<end_of_turn>\n"
            f"<start_of_turn>model\n\n        {TARGET_BOUNDARY_MARKER}<end_of_turn>\n"
        )
        boundary = marker_template.rindex(TARGET_BOUNDARY_MARKER)
        return marker_template[:boundary]


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Initializing TranslateGemma Serving Gateway (1 worker per process)...")

    # Startup validation: ensure context limits align
    if settings.max_total_context_tokens > settings.vllm_max_model_len:
        logger.warning(
            "TG_MAX_TOTAL_CONTEXT_TOKENS (%d) exceeds TG_VLLM_MAX_MODEL_LEN (%d). Clamping to %d.",
            settings.max_total_context_tokens,
            settings.vllm_max_model_len,
            settings.vllm_max_model_len,
        )
        settings.max_total_context_tokens = settings.vllm_max_model_len

    # Load processor/tokenizer for exact rendering & admission token counting
    processor_or_tok = None
    tok_path = settings.tokenizer_path or settings.model_dir
    if tok_path:
        try:
            from transformers import AutoProcessor
            processor_or_tok = AutoProcessor.from_pretrained(tok_path, fix_markdown=False)
            logger.info("Loaded exact AutoProcessor from %s for prompt rendering.", tok_path)
        except Exception as e:
            logger.debug("AutoProcessor not loaded from %s (%s); checking TokenEstimator...", tok_path, e)

    estimator = TokenEstimator(tokenizer_path=tok_path)
    renderer = CanonicalPromptRenderer(processor_or_tok)
    vllm_client = VLLMClient(settings)
    concurrency_mgr = ConcurrencyManager(settings)
    validator = RequestValidator(settings, estimator)
    splitter = SentenceSplitter()
    classifier = WorkloadClassifier(settings, estimator)
    metrics = get_metrics()

    # Initial probe
    backend_ready = False
    for attempt in range(1, 4):
        if await vllm_client.check_health():
            logger.info("Successfully verified backend vLLM server at %s", settings.vllm_base_url)
            backend_ready = True
            break
        logger.warning("Backend vLLM not yet ready (attempt %d/3). Retrying in 1s...", attempt)
        await asyncio.sleep(1.0)

    app.state.settings = settings
    app.state.vllm_client = vllm_client
    app.state.estimator = estimator
    app.state.renderer = renderer
    app.state.concurrency_mgr = concurrency_mgr
    app.state.validator = validator
    app.state.splitter = splitter
    app.state.classifier = classifier
    app.state.metrics = metrics

    try:
        yield
    finally:
        logger.info("Shutting down TranslateGemma Gateway...")
        await vllm_client.close()


app = FastAPI(
    title="TranslateGemma Serving Gateway",
    description="High-performance, continuous-batching translation gateway backed by vLLM.",
    version="1.0.0",
    lifespan=lifespan,
)

_initial_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_initial_settings.cors_origins,
    allow_credentials=_initial_settings.cors_allow_credentials,
    allow_methods=_initial_settings.cors_allow_methods,
    allow_headers=_initial_settings.cors_allow_headers,
)


def get_vllm_client(request: Request) -> VLLMClient:
    return request.app.state.vllm_client


def get_validator(request: Request) -> RequestValidator:
    return request.app.state.validator


def get_renderer(request: Request) -> CanonicalPromptRenderer:
    return request.app.state.renderer


def get_concurrency_mgr(request: Request) -> ConcurrencyManager:
    return request.app.state.concurrency_mgr


def get_splitter(request: Request) -> SentenceSplitter:
    return request.app.state.splitter


def get_classifier(request: Request) -> WorkloadClassifier:
    return request.app.state.classifier


def validate_system_option(requested_system: Optional[str], default_system: str) -> str:
    """Ensure requested system matches loaded merged adapter system, refusing unsupported values."""
    if requested_system is None or requested_system == default_system:
        return default_system
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=(
            f"System {requested_system!r} is not loaded on this merged-checkpoint gateway. "
            f"Only '{default_system}' is available."
        ),
    )


async def _execute_single_translation(
    text: str,
    source_lang: str,
    target_lang: str,
    max_new_tokens: int,
    renderer: CanonicalPromptRenderer,
    validator: RequestValidator,
    vllm_client: VLLMClient,
    concurrency_mgr: ConcurrencyManager,
    metrics: MetricsCollector,
    workload_class: str,
    request_id: Optional[str] = None,
) -> str:
    """Render canonical prompt, validate context, acquire concurrency slot, call vLLM."""
    raw_prompt = renderer.render(source_lang, target_lang, text)
    validator.validate_request(raw_text=text, rendered_prompt=raw_prompt, max_new_tokens=max_new_tokens)

    queue_wait = await concurrency_mgr.acquire()
    start_infer = time.perf_counter()
    try:
        completion_text, finish_reason, usage = await vllm_client.generate_raw_completion(
            prompt=raw_prompt,
            max_tokens=max_new_tokens,
            request_id=request_id,
        )
        infer_latency = time.perf_counter() - start_infer

        metrics.record_completion(
            endpoint="/translate",
            workload_class=workload_class,
            latency=infer_latency,
            queue_wait=queue_wait,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            finish_reason=finish_reason,
        )
        # Return completion text without strip() to preserve stop diagnostics
        return completion_text
    finally:
        concurrency_mgr.release()


@app.get("/live")
async def live_check():
    """Liveness probe: verifies process is alive."""
    return {"status": "ALIVE"}


@app.get("/ready")
@app.get("/health-check")
async def health_check(
    vllm_client: VLLMClient = Depends(get_vllm_client),
):
    """Readiness probe: returns 200 OK when vLLM is ready, 503 when backend is down."""
    is_healthy = await vllm_client.check_health()
    if is_healthy:
        return HealthResponse(translator="OK", ready=True)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"translator": "FAIL", "ready": False, "detail": "Backend vLLM server is unreachable or unready."},
    )


@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info(
    request: Request,
    settings: Settings = Depends(get_settings),
):
    """Reports provenance, exact context limits, and runtime configuration."""
    estimator = request.app.state.estimator
    return ModelInfoResponse(
        model_release_id=settings.model_release_id,
        base_model_id=settings.base_model_id,
        source_adapter_path=settings.source_adapter_path,
        is_merged_checkpoint=True,
        default_system=settings.default_system,
        loaded_systems=[settings.default_system],
        stop_token_ids=settings.stop_token_ids,
        stop_tokens=settings.stop_tokens,
        default_source_lang=settings.source_lang,
        default_target_lang=settings.target_lang,
        max_new_tokens=settings.max_new_tokens,
        max_total_context_tokens=settings.max_total_context_tokens,
        vllm_model_name=settings.vllm_model_name,
        vllm_base_url=settings.vllm_base_url,
        estimator_mode=estimator.mode,
    )


@app.post("/translate", response_model=TranslationResponse)
async def translate(
    prompt: Prompt,
    request: Request,
    settings: Settings = Depends(get_settings),
    vllm_client: VLLMClient = Depends(get_vllm_client),
    validator: RequestValidator = Depends(get_validator),
    renderer: CanonicalPromptRenderer = Depends(get_renderer),
    concurrency_mgr: ConcurrencyManager = Depends(get_concurrency_mgr),
    splitter: SentenceSplitter = Depends(get_splitter),
    classifier: WorkloadClassifier = Depends(get_classifier),
):
    """Translate a single text segment with admission control and sentence splitting."""
    req_id = str(uuid.uuid4())
    system_name = validate_system_option(prompt.system, settings.default_system)
    source_lang = prompt.source_lang or settings.source_lang
    target_lang = prompt.target_lang or settings.target_lang
    max_new_tokens = prompt.max_new_tokens or settings.max_new_tokens
    do_split = settings.split_sentences if prompt.split_sentences is None else prompt.split_sentences
    metrics = request.app.state.metrics

    est_tokens = request.app.state.estimator.count_tokens(prompt.text)
    workload_class = classifier.classify_single(prompt.text, est_tokens).value
    metrics.record_request("/translate", workload_class)

    if do_split:
        sentences = splitter.split_sentences(prompt.text, source_lang)
        if len(sentences) > 1:
            async def _tr_sentence(s: str) -> str:
                return await _execute_single_translation(
                    text=s,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    max_new_tokens=max_new_tokens,
                    renderer=renderer,
                    validator=validator,
                    vllm_client=vllm_client,
                    concurrency_mgr=concurrency_mgr,
                    metrics=metrics,
                    workload_class=WorkloadClass.DOCUMENT.value,
                    request_id=req_id,
                )

            translated_sentences = await dispatch_structured_batch(sentences, _tr_sentence)
            final_translation = " ".join(translated_sentences)
            return TranslationResponse(
                translation=final_translation,
                system=system_name,
                source_lang=source_lang,
                target_lang=target_lang,
            )

    translation = await _execute_single_translation(
        text=prompt.text,
        source_lang=source_lang,
        target_lang=target_lang,
        max_new_tokens=max_new_tokens,
        renderer=renderer,
        validator=validator,
        vllm_client=vllm_client,
        concurrency_mgr=concurrency_mgr,
        metrics=metrics,
        workload_class=workload_class,
        request_id=req_id,
    )

    return TranslationResponse(
        translation=translation,
        system=system_name,
        source_lang=source_lang,
        target_lang=target_lang,
    )


@app.post("/translate/batch", response_model=BatchTranslationResponse)
async def translate_batch(
    prompt: BatchPrompt,
    request: Request,
    settings: Settings = Depends(get_settings),
    vllm_client: VLLMClient = Depends(get_vllm_client),
    validator: RequestValidator = Depends(get_validator),
    renderer: CanonicalPromptRenderer = Depends(get_renderer),
    concurrency_mgr: ConcurrencyManager = Depends(get_concurrency_mgr),
    classifier: WorkloadClassifier = Depends(get_classifier),
):
    """Translate multiple texts concurrently using structured batching, preserving input ordering."""
    req_id = str(uuid.uuid4())
    system_name = validate_system_option(prompt.system, settings.default_system)
    source_lang = prompt.source_lang or settings.source_lang
    target_lang = prompt.target_lang or settings.target_lang
    max_new_tokens = prompt.max_new_tokens or settings.max_new_tokens
    metrics = request.app.state.metrics

    validator.validate_batch(prompt.texts, max_new_tokens)
    workload_class = classifier.classify_batch(prompt.texts).value
    metrics.record_request("/translate/batch", workload_class)

    async def _tr_item(text: str) -> str:
        return await _execute_single_translation(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            max_new_tokens=max_new_tokens,
            renderer=renderer,
            validator=validator,
            vllm_client=vllm_client,
            concurrency_mgr=concurrency_mgr,
            metrics=metrics,
            workload_class=workload_class,
            request_id=req_id,
        )

    translations = await dispatch_structured_batch(prompt.texts, _tr_item)
    return BatchTranslationResponse(
        translations=translations,
        system=system_name,
        source_lang=source_lang,
        target_lang=target_lang,
    )


@app.post("/translate/stream")
async def translate_stream(
    prompt: Prompt,
    request: Request,
    settings: Settings = Depends(get_settings),
    vllm_client: VLLMClient = Depends(get_vllm_client),
    renderer: CanonicalPromptRenderer = Depends(get_renderer),
    validator: RequestValidator = Depends(get_validator),
    concurrency_mgr: ConcurrencyManager = Depends(get_concurrency_mgr),
):
    """Stream translation tokens using Server-Sent Events (SSE) without stripping chunk content."""
    validate_system_option(prompt.system, settings.default_system)
    source_lang = prompt.source_lang or settings.source_lang
    target_lang = prompt.target_lang or settings.target_lang
    max_new_tokens = prompt.max_new_tokens or settings.max_new_tokens
    req_id = str(uuid.uuid4())

    raw_prompt = renderer.render(source_lang, target_lang, prompt.text)
    validator.validate_request(raw_text=prompt.text, rendered_prompt=raw_prompt, max_new_tokens=max_new_tokens)

    async def _event_generator():
        await concurrency_mgr.acquire()
        try:
            async for chunk in vllm_client.generate_stream_completion(
                prompt=raw_prompt,
                max_tokens=max_new_tokens,
                request_id=req_id,
            ):
                text_chunk = chunk.get("text", "")
                finish = chunk.get("finish_reason")
                if text_chunk:
                    yield f"data: {text_chunk}\n\n"
                if finish:
                    if finish == "length":
                        yield f"event: error\ndata: Generation truncated by max_new_tokens\n\n"
                    else:
                        yield f"event: done\ndata: {finish}\n\n"
                    break
        finally:
            concurrency_mgr.release()

    return StreamingResponse(_event_generator(), media_type="text/event-stream")


@app.get("/metrics")
async def metrics_endpoint(
    metrics: MetricsCollector = Depends(get_metrics),
):
    """Observability metrics summary."""
    return JSONResponse(content=metrics.get_summary())
