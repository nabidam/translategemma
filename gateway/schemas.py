"""Request and response schemas for the TranslateGemma FastAPI Gateway."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

EXAMPLE_SOURCE = "The model relies on multi-query attention to process the genome sequence."
EXAMPLE_TARGET = "این مدل برای پردازش توالی ژنوم بر توجه چندپرسشی تکیه می‌کند."


class TranslationOptions(BaseModel):
    """Per-request options falling back to server configuration."""

    system: Optional[str] = Field(
        default=None,
        description=(
            "Target system identifier. Merged checkpoint gateway serves 'adapter'. "
            "Specifying an unsupported system (such as 'base') returns 400 Bad Request."
        ),
    )
    source_lang: Optional[str] = Field(
        default=None,
        description="Source language ISO code (e.g. 'en'). Defaults to TG_SOURCE_LANG.",
    )
    target_lang: Optional[str] = Field(
        default=None,
        description="Target language ISO code (e.g. 'fa'). Defaults to TG_TARGET_LANG.",
    )
    max_new_tokens: Optional[int] = Field(
        default=None,
        gt=0,
        le=4096,
        description="Maximum generation token budget. Defaults to TG_MAX_NEW_TOKENS.",
    )
    split_sentences: Optional[bool] = Field(
        default=None,
        description="Split multi-sentence input with pysbd, translate chunks, and rejoin.",
    )


class Prompt(TranslationOptions):
    """Single-segment translation request."""

    text: str = Field(min_length=1, description="Source text to translate.")

    model_config = {
        "json_schema_extra": {
            "examples": [{"text": EXAMPLE_SOURCE, "source_lang": "en", "target_lang": "fa"}]
        }
    }


class BatchPrompt(TranslationOptions):
    """Batch translation request for multiple independent texts."""

    texts: List[str] = Field(min_length=1, description="List of source texts to translate.")

    model_config = {
        "json_schema_extra": {
            "examples": [{"texts": [EXAMPLE_SOURCE], "source_lang": "en", "target_lang": "fa"}]
        }
    }


class TranslationResponse(BaseModel):
    """Single translation response, maintaining compatibility with existing API callers."""

    translation: str
    system: str
    source_lang: str
    target_lang: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "translation": EXAMPLE_TARGET,
                    "system": "adapter",
                    "source_lang": "en",
                    "target_lang": "fa",
                }
            ]
        }
    }


class BatchTranslationResponse(BaseModel):
    """Batch translation response, guaranteeing identical output ordering."""

    translations: List[str]
    system: str
    source_lang: str
    target_lang: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "translations": [EXAMPLE_TARGET],
                    "system": "adapter",
                    "source_lang": "en",
                    "target_lang": "fa",
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Legacy health check response format: exactly {'translator': 'OK'|'FAIL'}."""

    translator: str

    model_config = {"json_schema_extra": {"examples": [{"translator": "OK"}]}}


class ReadyResponse(BaseModel):
    """Extended readiness probe response."""

    translator: str
    ready: bool
    model_name: str
    estimator_mode: str
    detail: Optional[str] = None


class ModelInfoResponse(BaseModel):
    """Provenance and runtime configuration information."""

    model_release_id: str
    base_model_id: str
    source_adapter_path: Optional[str]
    is_merged_checkpoint: bool = True
    default_system: str
    loaded_systems: List[str]
    stop_token_ids: List[int]
    stop_tokens: List[str]
    default_source_lang: str
    default_target_lang: str
    max_new_tokens: int
    max_total_context_tokens: int
    vllm_model_name: str
    vllm_base_url: str
    estimator_mode: str
    prompt_contract_version: str = "2026-08-10"
    routing_policy_version: str = "v1"
