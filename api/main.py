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
from glossary.serializers import to_report
from glossary.service import GlossaryService, UnknownDomainError
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
    glossary = None
    if settings.glossary_enabled:
        glossary = GlossaryService(
            database_url=settings.glossary_db_url,
            default_domain=settings.glossary_default_domain,
            unknown_domain=settings.glossary_unknown_domain,
        )
        await glossary.start()
        logger.info("Glossary enabled: %s", settings.glossary_db_url)
    app.state.glossary = glossary
    try:
        yield
    finally:
        if glossary is not None:
            await glossary.aclose()
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


def get_glossary() -> GlossaryService | None:
    """None when the feature is off. Callers must not branch on settings instead."""
    return getattr(app.state, "glossary", None)


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


async def _resolve_index(glossary, prompt, resolved, settings):
    """The snapshot for this request, or None when the glossary does not apply."""
    if glossary is None:
        return None
    mode = prompt.terminology_mode or settings.terminology_mode
    if mode == "off":
        return None
    try:
        return await glossary.resolve(resolved.source_lang, resolved.target_lang, prompt.domain)
    except UnknownDomainError as error:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Unknown glossary domain {error.name!r}. Available: {error.available}",
        ) from error


async def _translate(engine, texts, resolved, glossary_index=None):
    return await engine.translate(
        texts,
        resolved.system,
        resolved.source_lang,
        resolved.target_lang,
        resolved.max_new_tokens,
        resolved.split_sentences,
        glossary_index=glossary_index,
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
        result = await _translate(engine, ["Hello."], resolved)
        translations = result.translations
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
        glossary_enabled=settings.glossary_enabled,
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


@app.post("/translate", response_model=TranslationResponse, response_model_exclude_none=True)
async def translate(
    prompt: Prompt,
    engine: TranslationEngine = Depends(get_engine),
    settings: Settings = Depends(get_settings),
):
    text = prompt.text.strip()
    if not text:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "text is empty.")
    resolved = _resolve(prompt, settings)
    glossary = get_glossary()
    index = await _resolve_index(glossary, prompt, resolved, settings)
    result = await _translate(engine, [text], resolved, index)
    response = TranslationResponse(
        translation=result.translations[0],
        system=resolved.system,
        source_lang=resolved.source_lang,
        target_lang=resolved.target_lang,
    )
    if index is not None:
        response.raw_translation = result.raw_translations[0]
        response.glossary_version = index.version
        response.glossary = to_report(result.reports[0])
    return response


@app.post(
    "/translate/batch",
    response_model=BatchTranslationResponse,
    response_model_exclude_none=True,
)
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
    glossary = get_glossary()
    index = await _resolve_index(glossary, prompt, resolved, settings)
    result = await _translate(engine, texts, resolved, index)
    response = BatchTranslationResponse(
        translations=result.translations,
        system=resolved.system,
        source_lang=resolved.source_lang,
        target_lang=resolved.target_lang,
    )
    if index is not None:
        response.raw_translations = result.raw_translations
        response.glossary_version = index.version
        response.glossary = [to_report(report) for report in result.reports]
    return response
