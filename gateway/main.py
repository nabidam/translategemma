"""TranslateGemma FastAPI Serving Gateway.

High-throughput, continuous-batching proxy in front of vLLM OpenAI server.
Preserves exact SFT training prompt rendering, stop contracts, and response schemas.
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from config import Settings, get_settings
from limits import ConcurrencyManager, RequestValidator, TokenEstimator
from metrics import MetricsCollector, get_metrics
from prompting import CHAT_TURN_END_TOKEN
from routing import SentenceSplitter, WorkloadClass, WorkloadClassifier
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


def render_exact_training_prompt(source_lang: str, target_lang: str, text: str) -> str:
    """Format prompt with exact SFT Jinja indentation prefix."""
    user_payload = f"<<<source>>>{source_lang}<<<target>>>{target_lang}<<<text>>>{text}"
    return (
        f"<start_of_turn>user\n"
        f"{user_payload}<end_of_turn>\n"
        f"<start_of_turn>model\n\n        "
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Initializing TranslateGemma Serving Gateway...")

    vllm_client = VLLMClient(settings)
    estimator = TokenEstimator()
    concurrency_mgr = ConcurrencyManager(settings)
    validator = RequestValidator(settings, estimator)
    splitter = SentenceSplitter()
    classifier = WorkloadClassifier(settings, estimator)
    metrics = get_metrics()

    # Check vLLM backend readiness (retry briefly if starting concurrently)
    backend_ready = False
    for attempt in range(1, 6):
        if await vllm_client.check_health():
            logger.info("Successfully connected to backend vLLM server at %s", settings.vllm_base_url)
            backend_ready = True
            break
        logger.warning(
            "Backend vLLM not yet ready (attempt %d/5). Retrying in 2s...",
            attempt,
        )
        await asyncio.sleep(2.0)

    if not backend_ready:
        logger.warning("Gateway started while backend vLLM is not yet ready. Requests will return 503 until backend is up.")

    app.state.settings = settings
    app.state.vllm_client = vllm_client
    app.state.estimator = estimator
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


def get_concurrency_mgr(request: Request) -> ConcurrencyManager:
    return request.app.state.concurrency_mgr


def get_splitter(request: Request) -> SentenceSplitter:
    return request.app.state.splitter


def get_classifier(request: Request) -> WorkloadClassifier:
    return request.app.state.classifier


async def _execute_single_translation(
    text: str,
    source_lang: str,
    target_lang: str,
    max_new_tokens: int,
    vllm_client: VLLMClient,
    concurrency_mgr: ConcurrencyManager,
    metrics: MetricsCollector,
    workload_class: str,
    request_id: Optional[str] = None,
) -> str:
    """Acquire concurrency slot, format raw prompt, call vLLM, and track latency."""
    queue_wait = await concurrency_mgr.acquire()
    start_infer = time.perf_counter()
    try:
        raw_prompt = render_exact_training_prompt(source_lang, target_lang, text)
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
        return completion_text.strip()
    finally:
        concurrency_mgr.release()


@app.get("/health-check", response_model=HealthResponse)
async def health_check(
    vllm_client: VLLMClient = Depends(get_vllm_client),
):
    """Probes backend vLLM health. Returns translator: 'OK' or 'FAIL'."""
    is_healthy = await vllm_client.check_health()
    return HealthResponse(translator="OK" if is_healthy else "FAIL")


@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info(
    settings: Settings = Depends(get_settings),
):
    """Reports provenance, stop tokens, and runtime configuration."""
    return ModelInfoResponse(
        model_release_id=settings.model_release_id,
        base_model_id=settings.base_model_id,
        adapter_path=settings.adapter_path,
        default_system=settings.default_system,
        loaded_systems=[settings.default_system],
        stop_token_ids=settings.stop_token_ids,
        stop_tokens=settings.stop_tokens,
        default_source_lang=settings.source_lang,
        default_target_lang=settings.target_lang,
        max_new_tokens=settings.max_new_tokens,
        vllm_model_name=settings.vllm_model_name,
        vllm_base_url=settings.vllm_base_url,
    )


@app.post("/translate", response_model=TranslationResponse)
async def translate(
    prompt: Prompt,
    request: Request,
    settings: Settings = Depends(get_settings),
    vllm_client: VLLMClient = Depends(get_vllm_client),
    validator: RequestValidator = Depends(get_validator),
    concurrency_mgr: ConcurrencyManager = Depends(get_concurrency_mgr),
    splitter: SentenceSplitter = Depends(get_splitter),
    classifier: WorkloadClassifier = Depends(get_classifier),
):
    """Translate a single text segment with admission control and sentence splitting."""
    req_id = str(uuid.uuid4())
    source_lang = prompt.source_lang or settings.source_lang
    target_lang = prompt.target_lang or settings.target_lang
    max_new_tokens = prompt.max_new_tokens or settings.max_new_tokens
    do_split = settings.split_sentences if prompt.split_sentences is None else prompt.split_sentences
    system_name = prompt.system or settings.default_system
    metrics = request.app.state.metrics

    # Validate limits
    est_tokens = validator.validate_text(prompt.text, max_new_tokens)
    workload_class = classifier.classify_single(prompt.text, est_tokens).value
    metrics.record_request("/translate", workload_class)

    if do_split:
        sentences = splitter.split_sentences(prompt.text, source_lang)
        if len(sentences) > 1:
            tasks = [
                _execute_single_translation(
                    text=s,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    max_new_tokens=max_new_tokens,
                    vllm_client=vllm_client,
                    concurrency_mgr=concurrency_mgr,
                    metrics=metrics,
                    workload_class=WorkloadClass.DOCUMENT.value,
                    request_id=req_id,
                )
                for s in sentences
            ]
            translated_sentences = await asyncio.gather(*tasks)
            final_translation = " ".join(translated_sentences)
            return TranslationResponse(
                translation=final_translation,
                system=system_name,
                source_lang=source_lang,
                target_lang=target_lang,
            )

    translation = await _execute_single_translation(
        text=prompt.text.strip(),
        source_lang=source_lang,
        target_lang=target_lang,
        max_new_tokens=max_new_tokens,
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
    concurrency_mgr: ConcurrencyManager = Depends(get_concurrency_mgr),
    classifier: WorkloadClassifier = Depends(get_classifier),
):
    """Translate multiple texts concurrently, preserving exact input ordering."""
    req_id = str(uuid.uuid4())
    source_lang = prompt.source_lang or settings.source_lang
    target_lang = prompt.target_lang or settings.target_lang
    max_new_tokens = prompt.max_new_tokens or settings.max_new_tokens
    system_name = prompt.system or settings.default_system
    metrics = request.app.state.metrics

    # Validate batch
    validator.validate_batch(prompt.texts, max_new_tokens)
    workload_class = classifier.classify_batch(prompt.texts).value
    metrics.record_request("/translate/batch", workload_class)

    tasks = [
        _execute_single_translation(
            text=text.strip(),
            source_lang=source_lang,
            target_lang=target_lang,
            max_new_tokens=max_new_tokens,
            vllm_client=vllm_client,
            concurrency_mgr=concurrency_mgr,
            metrics=metrics,
            workload_class=workload_class,
            request_id=req_id,
        )
        for text in prompt.texts
    ]

    translations = await asyncio.gather(*tasks)
    return BatchTranslationResponse(
        translations=list(translations),
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
    validator: RequestValidator = Depends(get_validator),
    concurrency_mgr: ConcurrencyManager = Depends(get_concurrency_mgr),
):
    """Stream translation tokens using Server-Sent Events (SSE)."""
    source_lang = prompt.source_lang or settings.source_lang
    target_lang = prompt.target_lang or settings.target_lang
    max_new_tokens = prompt.max_new_tokens or settings.max_new_tokens
    req_id = str(uuid.uuid4())

    validator.validate_text(prompt.text, max_new_tokens)
    raw_prompt = render_exact_training_prompt(source_lang, target_lang, prompt.text.strip())

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
