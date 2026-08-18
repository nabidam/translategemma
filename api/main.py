"""FastAPI service for TranslateGemma.

Endpoint shapes mirror the NLLB service (POST body with a `text` field, a
`{"translator": "OK"}` health check) so existing callers move over with a URL
change, plus a batch endpoint, a per-request system selector, and a /model-info
that reports exactly which checkpoint answered.

Generation happens in a vLLM server (see translator.py); this process renders
prompts and forwards them. Requests are therefore concurrent end to end — no
GPU lock — and vLLM's continuous batching merges whatever is in flight,
including the segments of a single sentence-split request.
"""

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass

from anyio import to_thread
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from config import Settings, System, get_settings
from schemas import (
    BatchPrompt,
    BatchTranslationResponse,
    HealthResponse,
    ModelInfoResponse,
    Prompt,
    TranslationResponse,
)
from translator import TranslationEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("translategemma.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    engine = TranslationEngine(settings)
    # Tokenizer only, but still blocking file I/O: keep it off the event loop.
    await to_thread.run_sync(engine.load)
    app.state.engine = engine
    try:
        yield
    finally:
        await engine.aclose()


app = FastAPI(
    title="TranslateGemma API",
    description=(
        "Translation gateway for TranslateGemma: renders prompts and forwards "
        "generation to a vLLM server."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origins,
    allow_credentials=_settings.cors_allow_credentials,
    allow_methods=_settings.cors_allow_methods,
    allow_headers=_settings.cors_allow_headers,
)


def get_engine() -> TranslationEngine:
    engine = getattr(app.state, "engine", None)
    if engine is None or not engine.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Gateway is not ready."
        )
    return engine


@dataclass(frozen=True)
class ResolvedOptions:
    system: System
    source_lang: str
    target_lang: str
    max_new_tokens: int
    split_sentences: bool


def _resolve(options, settings: Settings) -> ResolvedOptions:
    """Apply server defaults to a request, rejecting a system this upstream is not.

    `system` is an assertion, not a selector: one vLLM serves one set of weights.
    A caller that names the other one is asking for a checkpoint this deployment
    cannot produce, and gets a 400 rather than a translation from the wrong one.
    """
    system = options.system or settings.served_system
    if system is not settings.served_system:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"System {system!r} is not served here: this gateway fronts the "
            f"{settings.served_system!s} system (TG_SERVED_SYSTEM).",
        )
    return ResolvedOptions(
        system=system,
        source_lang=options.source_lang or settings.source_lang,
        target_lang=options.target_lang or settings.target_lang,
        max_new_tokens=options.max_new_tokens or settings.max_new_tokens,
        split_sentences=(
            settings.split_sentences
            if options.split_sentences is None
            else options.split_sentences
        ),
    )


async def _translate(engine, texts: list[str], resolved: ResolvedOptions) -> list[str]:
    return await engine.translate(
        texts,
        resolved.system,
        resolved.source_lang,
        resolved.target_lang,
        resolved.max_new_tokens,
        resolved.split_sentences,
    )


@app.get("/health-check", response_model=HealthResponse)
async def health_check(
    engine: TranslationEngine = Depends(get_engine),
    settings: Settings = Depends(get_settings),
):
    """Prove the decoder still produces text, not just that the process is up."""
    resolved = ResolvedOptions(
        system=settings.served_system,
        source_lang=settings.source_lang,
        target_lang=settings.target_lang,
        max_new_tokens=16,
        split_sentences=False,
    )
    try:
        translations = await _translate(engine, ["Hello."], resolved)
    except Exception:
        logger.exception("Health check translation failed.")
        return HealthResponse(translator="FAIL")
    return HealthResponse(translator="OK" if translations and translations[0].strip() else "FAIL")


@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info(
    engine: TranslationEngine = Depends(get_engine),
    settings: Settings = Depends(get_settings),
):
    tokenizer = engine.processor.tokenizer
    return ModelInfoResponse(
        base_model_id=settings.base_model_id,
        served_system=settings.served_system,
        adapter_path=settings.adapter_path,
        upstream=engine.upstream,
        use_training_rendering=settings.use_training_rendering(settings.served_system),
        stop_token_ids=engine.stop_token_ids,
        stop_tokens=tokenizer.convert_ids_to_tokens(engine.stop_token_ids),
        default_source_lang=settings.source_lang,
        default_target_lang=settings.target_lang,
        max_new_tokens=settings.max_new_tokens,
        do_sample=settings.do_sample,
        batch_size=settings.batch_size,
    )


@app.post("/translate", response_model=TranslationResponse)
async def translate(
    prompt: Prompt,
    engine: TranslationEngine = Depends(get_engine),
    settings: Settings = Depends(get_settings),
):
    text = prompt.text.strip()
    if not text:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "text is empty.")
    resolved = _resolve(prompt, settings)
    translations = await _translate(engine, [text], resolved)
    return TranslationResponse(
        translation=translations[0],
        system=resolved.system,
        source_lang=resolved.source_lang,
        target_lang=resolved.target_lang,
    )


@app.post("/translate/batch", response_model=BatchTranslationResponse)
async def translate_batch(
    prompt: BatchPrompt,
    engine: TranslationEngine = Depends(get_engine),
    settings: Settings = Depends(get_settings),
):
    if len(prompt.texts) > settings.max_batch_items:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"At most {settings.max_batch_items} texts per request; got {len(prompt.texts)}.",
        )
    texts = [text.strip() for text in prompt.texts]
    if any(not text for text in texts):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "texts contains an empty item.")
    resolved = _resolve(prompt, settings)
    translations = await _translate(engine, texts, resolved)
    return BatchTranslationResponse(
        translations=translations,
        system=resolved.system,
        source_lang=resolved.source_lang,
        target_lang=resolved.target_lang,
    )
