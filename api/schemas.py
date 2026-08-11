"""Request and response schemas for the TranslateGemma serving API."""

from pydantic import BaseModel, Field

from config import System

EXAMPLE_SOURCE = "The model relies on multi-query attention to process the genome sequence."
EXAMPLE_TARGET = "این مدل برای پردازش توالی ژنوم بر توجه چندپرسشی تکیه می‌کند."


class TranslationOptions(BaseModel):
    """Per-request overrides. Every field falls back to the server settings."""

    system: System | None = Field(
        default=None,
        description=(
            "Which loaded system answers: 'base' or 'adapter'. Defaults to "
            "TG_DEFAULT_SYSTEM. Only selectable when TG_MODEL_MODE=both; asking "
            "for a system that is not loaded returns 400."
        ),
    )
    source_lang: str | None = Field(
        default=None, description="Source language code, e.g. 'en'. Defaults to TG_SOURCE_LANG."
    )
    target_lang: str | None = Field(
        default=None, description="Target language code, e.g. 'fa'. Defaults to TG_TARGET_LANG."
    )
    max_new_tokens: int | None = Field(
        default=None,
        gt=0,
        le=4096,
        description="Generation length cap. Defaults to TG_MAX_NEW_TOKENS.",
    )
    split_sentences: bool | None = Field(
        default=None,
        description=(
            "Split the input into sentences with pysbd, translate each, and rejoin. "
            "Defaults to TG_SPLIT_SENTENCES. Leave off for single segments: the "
            "adapter was trained on whole segments."
        ),
    )


class Prompt(TranslationOptions):
    """A single text to translate."""

    text: str = Field(min_length=1)

    model_config = {
        "json_schema_extra": {
            "examples": [{"text": EXAMPLE_SOURCE, "source_lang": "en", "target_lang": "fa"}]
        }
    }


class BatchPrompt(TranslationOptions):
    """Several texts translated in one request, batched into shared generate() calls."""

    texts: list[str] = Field(min_length=1)

    model_config = {
        "json_schema_extra": {
            "examples": [{"texts": [EXAMPLE_SOURCE], "source_lang": "en", "target_lang": "fa"}]
        }
    }


class TranslationResponse(BaseModel):
    translation: str
    system: System
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
    translations: list[str]
    system: System
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
    """Kept key-compatible with the NLLB API so existing probes keep working."""

    translator: str

    model_config = {"json_schema_extra": {"examples": [{"translator": "OK"}]}}


class ModelInfoResponse(BaseModel):
    """What is actually loaded, so a translation can be attributed to a checkpoint."""

    base_model_id: str
    model_mode: str
    loaded_systems: list[System]
    default_system: System
    adapter_path: str | None
    dtype: str
    attn_implementation: str
    load_in_4bit: bool
    device: str
    # Per system, because the two are prompted differently on purpose.
    use_training_rendering: dict[str, bool]
    stop_token_ids: list[int]
    stop_tokens: list[str]
    default_source_lang: str
    default_target_lang: str
    max_new_tokens: int
    do_sample: bool
    num_beams: int
    batch_size: int
