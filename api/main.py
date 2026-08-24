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
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.exc import SQLAlchemyError

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

# Declared on a router, not on an app instance: create_app() must be able to
# build more than one app, and a decorator binds to whichever object it names.
router = APIRouter()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = app.state.settings
    engine = TranslationEngine(settings)
    # Tokenizer only, but still blocking file I/O: keep it off the event loop.
    await to_thread.run_sync(engine.load)
    try:
        # Before anything else: a TG_VLLM_MODEL that names a model this upstream
        # does not serve is fatal here rather than an opaque 500 on every
        # translation later. Closing the client on the way out keeps a
        # misconfigured start from leaking a connection pool.
        await engine.verify_upstream_model()
    except Exception:
        await engine.aclose()
        raise
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
        app.state.glossary = None
        app.state.engine = None
        if glossary is not None:
            await glossary.aclose()
        await engine.aclose()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build an app bound to one Settings object.

    Routes are mounted here, at construction, rather than inside `lifespan`.
    Lifespan is a re-entrant context manager: mounting from it appends another
    copy of every route each time it runs, so a process that starts the app
    twice (any test that drives startup more than once) accumulates duplicates
    and the first, already-closed registration shadows the live one.

    The admin router still only exists when the feature is enabled, so a
    disabled deployment answers 404 rather than 401 -- there is nothing there
    to authorize against. What moved is *when* it is mounted, not *whether*.
    """
    settings = settings or get_settings()
    app = FastAPI(
        title="TranslateGemma API",
        description=(
            "Translation gateway for TranslateGemma: renders prompts and forwards "
            "generation to a vLLM server."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    # Read by lifespan and by the request-scoped dependencies below, so an app
    # never consults the module-level cache and two apps can differ.
    app.state.settings = settings
    app.state.engine = None
    app.state.glossary = None
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )
    app.include_router(router)
    if settings.glossary_enabled:
        from glossary.router import build_router

        app.include_router(
            build_router(lambda: app.state.glossary, settings.admin_api_key)
        )
    return app


app = create_app()


def get_engine(request: Request) -> TranslationEngine:
    engine = getattr(request.app.state, "engine", None)
    if engine is None or not engine.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Gateway is not ready."
        )
    return engine


def get_glossary(request: Request) -> GlossaryService | None:
    """None when the feature is off. Callers must not branch on settings instead.

    Read from `request.app`, never the module-level `app`: a test (or any host
    that builds more than one app in a process) must not have its requests
    answered from another app's state.
    """
    return getattr(request.app.state, "glossary", None)


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


_VALID_TERMINOLOGY_MODES = ("off", "enforce")


async def _resolve_index(glossary, prompt, resolved, settings):
    """The snapshot for this request, or None when the glossary does not apply.

    terminology_mode is validated here rather than as a Pydantic Literal on
    the request schema: this branch only runs when the glossary is enabled
    (see the `glossary is None` guard immediately below), which keeps a
    disabled deployment permissive about the field's value -- a caller who
    learned to send `terminology_mode` must not start getting 422s the
    instant an operator flips TG_GLOSSARY_ENABLED off, since the value is
    then inert anyway.
    """
    if glossary is None:
        return None
    mode = prompt.terminology_mode or settings.terminology_mode
    if mode not in _VALID_TERMINOLOGY_MODES:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"terminology_mode must be one of {_VALID_TERMINOLOGY_MODES}; got {mode!r}.",
        )
    if mode == "off":
        return None
    try:
        return await glossary.resolve(resolved.source_lang, resolved.target_lang, prompt.domain)
    except UnknownDomainError as error:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"{error.reason} Available: {error.available}",
        ) from error
    except (SQLAlchemyError, OSError):
        # The glossary's own premise: a database problem (unmounted volume,
        # unreadable file) must degrade to plain translation, not take down a
        # healthy model. UnknownDomainError is a caller error and is handled
        # above, not here -- it must keep its 404.
        logger.error(
            "Glossary store failed while resolving a snapshot; continuing without it.",
            exc_info=True,
        )
        return None


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


@router.get("/health-check", response_model=HealthResponse)
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


@router.get("/model-info", response_model=ModelInfoResponse)
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


@router.post("/translate", response_model=TranslationResponse, response_model_exclude_none=True)
async def translate(
    prompt: Prompt,
    engine: TranslationEngine = Depends(get_engine),
    settings: Settings = Depends(get_settings),
    glossary: GlossaryService | None = Depends(get_glossary),
):
    text = prompt.text.strip()
    if not text:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "text is empty.")
    resolved = _resolve(prompt, settings)
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


@router.post(
    "/translate/batch",
    response_model=BatchTranslationResponse,
    response_model_exclude_none=True,
)
async def translate_batch(
    prompt: BatchPrompt,
    engine: TranslationEngine = Depends(get_engine),
    settings: Settings = Depends(get_settings),
    glossary: GlossaryService | None = Depends(get_glossary),
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
