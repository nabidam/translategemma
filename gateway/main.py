"""TranslateGemma FastAPI Serving Gateway.

High-throughput, continuous-batching proxy in front of vLLM OpenAI server.
Preserves canonical SFT training prompt rendering, stop contracts, and response schemas.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware

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
    ReadyResponse,
    TranslationResponse,
)
from vllm_client import VLLMClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("gateway.main")


import hashlib
from pathlib import Path

class _BodyTooLargeException(Exception):
    def __init__(self, detail: str):
        self.detail = detail


class BodySizeLimitMiddleware:
    """Rejects oversized requests before/during body reading without re-invoking broken receive callable."""

    def __init__(self, app, max_body_bytes: int):
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        cl_header = headers.get(b"content-length")
        if cl_header is not None:
            try:
                length = int(cl_header.decode("latin1"))
                if length < 0:
                    await self._send_json_response(
                        send,
                        status.HTTP_400_BAD_REQUEST,
                        {"detail": "Invalid negative Content-Length header."},
                    )
                    return
                if length > self.max_body_bytes:
                    await self._send_json_response(
                        send,
                        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        {"detail": f"Request body ({length} bytes) exceeds limit ({self.max_body_bytes} bytes)."},
                    )
                    return
            except ValueError:
                await self._send_json_response(
                    send,
                    status.HTTP_400_BAD_REQUEST,
                    {"detail": "Invalid non-integer Content-Length header."},
                )
                return

        total_received = 0

        async def limited_receive():
            nonlocal total_received
            message = await receive()
            if message["type"] == "http.request":
                body = message.get("body", b"")
                total_received += len(body)
                if total_received > self.max_body_bytes:
                    raise _BodyTooLargeException(
                        f"Request body exceeded limit of {self.max_body_bytes} bytes during stream transfer."
                    )
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _BodyTooLargeException as exc:
            await self._send_json_response(
                send,
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                {"detail": exc.detail},
            )

    async def _send_json_response(self, send, status_code: int, content: dict):
        body_bytes = json.dumps(content).encode("utf-8")
        await send({
            "type": "http.response.start",
            "status": status_code,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body_bytes)).encode("latin1")),
            ],
        })
        await send({
            "type": "http.response.body",
            "body": body_bytes,
            "more_body": False,
        })


class ManifestState:
    """Manages loaded merge manifest provenance, checksum verification, and authenticity."""

    def __init__(
        self,
        is_loaded: bool = False,
        is_verified: bool = False,
        authenticity_status: str = "unverified",
        manifest_sha256: Optional[str] = None,
        data: Optional[dict] = None,
        error: Optional[str] = None,
    ):
        self.is_loaded = is_loaded
        self.is_verified = is_verified
        self.authenticity_status = authenticity_status
        self.manifest_sha256 = manifest_sha256
        self.data = data or {}
        self.error = error

    @classmethod
    def load_and_verify(cls, settings: Settings) -> ManifestState:
        model_dir_str = settings.model_dir
        if not model_dir_str:
            return cls(error="No model_dir configured.")

        model_dir = Path(model_dir_str)
        manifest_file = model_dir / "merge_manifest.json"
        if not manifest_file.is_file():
            return cls(error=f"merge_manifest.json not found in {model_dir}")

        try:
            content_bytes = manifest_file.read_bytes()
            manifest_sha256 = hashlib.sha256(content_bytes).hexdigest()
            data = json.loads(content_bytes.decode("utf-8"))
        except Exception as e:
            return cls(error=f"Failed to read/parse merge_manifest.json: {e}")

        authenticity = "unverified"
        is_verified = False

        if settings.trusted_manifest_sha256:
            expected_hash = settings.trusted_manifest_sha256.strip().lower()
            if manifest_sha256.lower() == expected_hash:
                authenticity = "trusted_external_anchor"
                is_verified = True
                logger.info("Manifest verified against operator trusted SHA256 anchor (%s).", expected_hash)
            else:
                msg = f"Manifest SHA256 mismatch with TG_TRUSTED_MANIFEST_SHA256: computed {manifest_sha256}, expected {expected_hash}"
                logger.error(msg)
                return cls(
                    is_loaded=True,
                    is_verified=False,
                    authenticity_status="unverified",
                    manifest_sha256=manifest_sha256,
                    data=data,
                    error=msg,
                )
        elif settings.trusted_anchor_file:
            anchor_p = Path(settings.trusted_anchor_file)
            if not anchor_p.is_file():
                msg = f"Configured TG_TRUSTED_ANCHOR_FILE not found: {anchor_p}"
                logger.error(msg)
                return cls(
                    is_loaded=True,
                    is_verified=False,
                    authenticity_status="unverified",
                    manifest_sha256=manifest_sha256,
                    data=data,
                    error=msg,
                )
            try:
                anchor_text = anchor_p.read_text(encoding="utf-8").strip()
                expected_hash = anchor_text.split()[0].strip().lower()
                if manifest_sha256.lower() == expected_hash:
                    authenticity = "trusted_external_anchor"
                    is_verified = True
                    logger.info("Manifest verified against external trusted anchor file %s (%s).", anchor_p, expected_hash)
                else:
                    msg = f"Manifest SHA256 mismatch with trusted anchor file: computed {manifest_sha256}, expected {expected_hash}"
                    logger.error(msg)
                    return cls(
                        is_loaded=True,
                        is_verified=False,
                        authenticity_status="unverified",
                        manifest_sha256=manifest_sha256,
                        data=data,
                        error=msg,
                    )
            except Exception as e:
                msg = f"Failed to read trusted anchor file {anchor_p}: {e}"
                logger.error(msg)
                return cls(
                    is_loaded=True,
                    is_verified=False,
                    authenticity_status="unverified",
                    manifest_sha256=manifest_sha256,
                    data=data,
                    error=msg,
                )
        else:
            local_anchor = model_dir / "merge_manifest.sha256"
            if local_anchor.is_file():
                try:
                    local_text = local_anchor.read_text(encoding="utf-8").strip()
                    expected_local_hash = local_text.split()[0].strip().lower()
                    if manifest_sha256.lower() == expected_local_hash:
                        authenticity = "colocated_checksum_only"
                        is_verified = True
                        logger.info("Manifest matches co-located merge_manifest.sha256 integrity anchor.")
                    else:
                        msg = f"Co-located merge_manifest.sha256 mismatch: computed {manifest_sha256}, expected {expected_local_hash}"
                        logger.error(msg)
                        return cls(
                            is_loaded=True,
                            is_verified=False,
                            authenticity_status="unverified",
                            manifest_sha256=manifest_sha256,
                            data=data,
                            error=msg,
                        )
                except Exception as e:
                    logger.warning("Could not read local merge_manifest.sha256: %s", e)
            else:
                logger.warning("No integrity anchor found for merge_manifest.json.")

        return cls(
            is_loaded=True,
            is_verified=is_verified,
            authenticity_status=authenticity,
            manifest_sha256=manifest_sha256,
            data=data,
            error=None,
        )



class CanonicalPromptRenderer:
    """Renders prompts using the canonical TranslateGemma chat template and boundary cut."""

    def __init__(self, processor_or_tokenizer: Any = None, allow_fallback: bool = False):
        self.processor = processor_or_tokenizer
        self.allow_fallback = allow_fallback
        if self.processor is not None and hasattr(self.processor, "apply_chat_template"):
            self.mode = "canonical"
        elif allow_fallback:
            self.mode = "fallback"
        else:
            self.mode = "none"

    def render(self, source_lang: str, target_lang: str, text: str) -> str:
        # Structured user message matching train.py and api/translator.py exactly
        user_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": source_lang,
                    "target_lang_code": target_lang,
                    "text": text,
                }
            ],
        }

        if self.processor is not None and hasattr(self.processor, "apply_chat_template"):
            return render_training_prompt(self.processor, user_message)

        if not self.allow_fallback:
            raise RuntimeError(
                "Canonical processor rendering is required in production, but AutoProcessor with apply_chat_template "
                "is not initialized. Check model mount and dependencies."
            )

        # Fallback only when explicitly permitted in isolated unit-test execution
        marker_template = (
            f"<start_of_turn>user\n"
            f"<<<source>>>{source_lang}<<<target>>>{target_lang}<<<text>>>{text}<end_of_turn>\n"
            f"<start_of_turn>model\n\n        {TARGET_BOUNDARY_MARKER}<end_of_turn>\n"
        )
        boundary = marker_template.rindex(TARGET_BOUNDARY_MARKER)
        return marker_template[:boundary]


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Initializing TranslateGemma Serving Gateway (1 worker per process)...")

    # Startup validation: ensure context limits strictly align
    if settings.max_total_context_tokens != settings.vllm_max_model_len:
        raise ValueError(
            f"Configuration mismatch: TG_MAX_TOTAL_CONTEXT_TOKENS ({settings.max_total_context_tokens}) "
            f"must equal TG_VLLM_MAX_MODEL_LEN ({settings.vllm_max_model_len}) for single authoritative context alignment."
        )

    # Load and verify merge manifest
    manifest_state = ManifestState.load_and_verify(settings)
    if settings.require_verified_manifest:
        if not manifest_state.is_verified:
            raise RuntimeError(
                f"TG_REQUIRE_VERIFIED_MANIFEST=true, but manifest verification failed: {manifest_state.error}"
            )
        if manifest_state.authenticity_status == "unverified":
            raise RuntimeError(
                "TG_REQUIRE_VERIFIED_MANIFEST=true, but manifest authenticity cannot be verified."
            )

    # Derive runtime settings from verified manifest if available
    if manifest_state.is_loaded and manifest_state.data:
        m = manifest_state.data
        if m.get("release_id"):
            settings.model_release_id = str(m["release_id"])
        if m.get("base_model"):
            settings.base_model_id = str(m["base_model"])
        if m.get("adapter"):
            settings.source_adapter_path = str(m["adapter"])
        if m.get("stop_token_ids"):
            settings.stop_token_ids = list(m["stop_token_ids"])
        if m.get("stop_tokens"):
            settings.stop_tokens = list(m["stop_tokens"])

    # Load processor/tokenizer for exact rendering & admission token counting
    processor_or_tok = None
    tok_path = settings.tokenizer_path or settings.model_dir
    processor_mode = "none"
    if tok_path:
        try:
            from transformers import AutoProcessor
            processor_or_tok = AutoProcessor.from_pretrained(tok_path, fix_markdown=False)
            if hasattr(processor_or_tok, "apply_chat_template"):
                processor_mode = "canonical"
                logger.info("Loaded exact AutoProcessor from %s for prompt rendering.", tok_path)
        except Exception as e:
            logger.debug("AutoProcessor not loaded from %s (%s); trying TokenEstimator...", tok_path, e)

    estimator = TokenEstimator(tokenizer_path=tok_path)
    if settings.require_exact_tokenizer:
        if estimator.mode != "exact":
            raise RuntimeError(
                f"TG_REQUIRE_EXACT_TOKENIZER=true, but exact tokenizer could not be loaded from {tok_path}. "
                "Mount the verified model artifact or check tokenizer dependencies."
            )
        if processor_mode != "canonical":
            raise RuntimeError(
                f"TG_REQUIRE_EXACT_TOKENIZER=true, but canonical AutoProcessor chat template could not be loaded from {tok_path}. "
                "Mount the verified model artifact or check transformers dependencies."
            )

    renderer = CanonicalPromptRenderer(
        processor_or_tok,
        allow_fallback=(not settings.require_exact_tokenizer),
    )
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
    app.state.manifest_state = manifest_state
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
    BodySizeLimitMiddleware,
    max_body_bytes=_initial_settings.max_request_body_bytes,
)
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
    is_bulk: bool = False,
) -> str:
    """Render canonical prompt, validate context, acquire concurrency slot, call vLLM."""
    raw_prompt = renderer.render(source_lang, target_lang, text)
    validator.validate_request(raw_text=text, rendered_prompt=raw_prompt, max_new_tokens=max_new_tokens)

    queue_wait = await concurrency_mgr.acquire(is_bulk=is_bulk)
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
        concurrency_mgr.release(is_bulk=is_bulk)


@app.get("/live")
async def live_check():
    """Liveness probe: verifies gateway process is up."""
    return {"status": "ALIVE"}


@app.get("/health-check", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
@app.get("/healthz", response_model=HealthResponse)
async def health_check_legacy(
    vllm_client: VLLMClient = Depends(get_vllm_client),
):
    """Strict legacy health check response shape: {'translator': 'OK'|'FAIL'}."""
    is_healthy = await vllm_client.check_health()
    if is_healthy:
        return HealthResponse(translator="OK")
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"translator": "FAIL"},
    )


@app.get("/ready", response_model=ReadyResponse)
@app.get("/health/ready", response_model=ReadyResponse)
async def ready_check(
    request: Request,
    settings: Settings = Depends(get_settings),
    vllm_client: VLLMClient = Depends(get_vllm_client),
):
    """Extended readiness probe verifying engine, exact tokenizer, canonical processor, and manifest."""
    is_healthy = await vllm_client.check_health()
    estimator = request.app.state.estimator
    renderer = request.app.state.renderer
    manifest_state: ManifestState = getattr(request.app.state, "manifest_state", ManifestState())

    if settings.require_verified_manifest and not manifest_state.is_verified:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "translator": "FAIL",
                "ready": False,
                "model_name": settings.vllm_model_name,
                "estimator_mode": estimator.mode,
                "processor_mode": renderer.mode,
                "manifest_verified": False,
                "authenticity_status": manifest_state.authenticity_status,
                "detail": f"Manifest verification failed: {manifest_state.error}",
            },
        )

    if is_healthy:
        return ReadyResponse(
            translator="OK",
            ready=True,
            model_name=settings.vllm_model_name,
            estimator_mode=estimator.mode,
            processor_mode=renderer.mode,
            manifest_verified=manifest_state.is_verified,
            authenticity_status=manifest_state.authenticity_status,
        )
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "translator": "FAIL",
            "ready": False,
            "model_name": settings.vllm_model_name,
            "estimator_mode": estimator.mode,
            "processor_mode": renderer.mode,
            "manifest_verified": manifest_state.is_verified,
            "authenticity_status": manifest_state.authenticity_status,
            "detail": "Backend vLLM server is unreachable or exact model is not loaded.",
        },
    )


@app.get("/model-info", response_model=ModelInfoResponse)
@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info(
    request: Request,
    settings: Settings = Depends(get_settings),
):
    """Reports provenance, exact context limits, runtime configuration, and verified manifest."""
    estimator = request.app.state.estimator
    renderer = request.app.state.renderer
    manifest_state: ManifestState = getattr(request.app.state, "manifest_state", ManifestState())
    m_data = manifest_state.data or {}

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
        processor_mode=renderer.mode,
        prompt_contract_version=m_data.get("prompt_contract_version", "2026-08-10"),
        routing_policy_version="v1",
        manifest_verified=manifest_state.is_verified,
        manifest_sha256=manifest_state.manifest_sha256,
        authenticity_status=manifest_state.authenticity_status,
        resolved_base_commit=m_data.get("resolved_base_commit") or (m_data.get("base_model_info", {}).get("resolved_revision")),
        resolved_adapter_commit=m_data.get("resolved_adapter_commit") or (m_data.get("adapter_info", {}).get("resolved_revision")),
        base_fingerprint=m_data.get("base_fingerprint"),
        adapter_fingerprint=m_data.get("adapter_fingerprint"),
        architecture_fingerprint=m_data.get("architecture_fingerprint"),
        manifest_metadata=m_data if manifest_state.is_loaded else None,
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
                    is_bulk=False,
                )

            translated_sentences = await dispatch_structured_batch(
                sentences,
                _tr_sentence,
                max_concurrency=settings.max_concurrent_requests,
            )
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
        is_bulk=False,
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
    """Translate multiple texts concurrently using structured batching and bulk fairness."""
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
            is_bulk=True,
        )

    translations = await dispatch_structured_batch(
        prompt.texts,
        _tr_item,
        max_concurrency=settings.max_bulk_concurrent_requests,
    )
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
    """Stream translation tokens using structured JSON SSE records to prevent framing corruption."""
    validate_system_option(prompt.system, settings.default_system)
    source_lang = prompt.source_lang or settings.source_lang
    target_lang = prompt.target_lang or settings.target_lang
    max_new_tokens = prompt.max_new_tokens or settings.max_new_tokens
    req_id = str(uuid.uuid4())

    raw_prompt = renderer.render(source_lang, target_lang, prompt.text)
    validator.validate_request(raw_text=prompt.text, rendered_prompt=raw_prompt, max_new_tokens=max_new_tokens)

    async def _event_generator():
        await concurrency_mgr.acquire(is_bulk=False)
        emitted_terminal = False
        try:
            async for chunk in vllm_client.generate_stream_completion(
                prompt=raw_prompt,
                max_tokens=max_new_tokens,
                request_id=req_id,
            ):
                if "error" in chunk:
                    err_event = {
                        "error": chunk["error"],
                        "code": chunk.get("code", "STREAM_ERROR"),
                        "request_id": req_id,
                    }
                    yield f"event: error\ndata: {json.dumps(err_event)}\n\n"
                    emitted_terminal = True
                    break

                text_chunk = chunk.get("text", "")
                finish = chunk.get("finish_reason")

                if text_chunk:
                    event_data = {
                        "text": text_chunk,
                        "finish_reason": finish,
                        "request_id": req_id,
                    }
                    yield f"data: {json.dumps(event_data)}\n\n"

                if finish:
                    if finish == "length":
                        len_err = {
                            "error": "Generation truncated by max_new_tokens budget.",
                            "code": "TRUNCATED_LENGTH",
                            "request_id": req_id,
                        }
                        yield f"event: error\ndata: {json.dumps(len_err)}\n\n"
                    else:
                        done_event = {
                            "text": "",
                            "finish_reason": finish,
                            "request_id": req_id,
                        }
                        yield f"event: done\ndata: {json.dumps(done_event)}\n\n"
                    emitted_terminal = True
                    break

            if not emitted_terminal:
                incomplete_err = {
                    "error": "Stream closed prematurely without terminal completion event.",
                    "code": "INCOMPLETE_STREAM",
                    "request_id": req_id,
                }
                yield f"event: error\ndata: {json.dumps(incomplete_err)}\n\n"

        finally:
            concurrency_mgr.release(is_bulk=False)

    return StreamingResponse(_event_generator(), media_type="text/event-stream")


@app.get("/metrics")
async def metrics_endpoint(
    metrics: MetricsCollector = Depends(get_metrics),
):
    """Observability metrics summary."""
    return JSONResponse(content=metrics.get_summary())
