"""Runtime configuration for the TranslateGemma serving API.

Everything is settable from the environment (or an api/.env file) so the same
image fronts any vLLM deployment without a rebuild. See api/.env.example.

Generation no longer happens in this process: vLLM owns the weights and this
service is a gateway (see translator.py). The model-shaped settings below
(dtype, attention implementation, 4-bit) therefore describe how the *upstream*
was launched and are reported by /model-info unchanged; they are not applied
here. Everything under "Decoding" is still applied, translated into the
sampling parameters of the vLLM request.
"""

import json
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelMode(StrEnum):
    """Which system(s) the server loads and exposes.

    BOTH loads one copy of the weights and toggles the adapter per request
    (PEFT's disable_adapter), so serving two systems costs the adapter's few
    hundred MB rather than a second 12B model.
    """

    BASE = "base"
    ADAPTER = "adapter"
    BOTH = "both"


class System(StrEnum):
    """The system a single request is answered by."""

    BASE = "base"
    ADAPTER = "adapter"


class Settings(BaseSettings):
    """Server configuration, read from the environment with `TG_` prefixed names."""

    model_config = SettingsConfigDict(
        env_prefix="TG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # `model_` is a Pydantic-protected namespace; model_mode below would
        # otherwise warn. Serving fields keep the training vocabulary, so the
        # protection is dropped rather than the fields renamed.
        protected_namespaces=(),
    )

    # --- Upstream vLLM ----------------------------------------------------
    # OpenAI-compatible base URL of the vLLM server holding the weights. The
    # gateway talks to /completions under it (never /chat/completions: the chat
    # endpoint renders add_generation_prompt=True, which is the off-distribution
    # prefix docs/2026-08-10_adapter_degeneration_analysis.md is about).
    vllm_base_url: str = "http://translategemma-vllm:8000/v1"
    # Must match vLLM's --served-model-name.
    vllm_model: str = "model"
    vllm_api_key: str | None = None
    # Seconds. A cold vLLM still compiling CUDA graphs, or a 512-token
    # generation on a busy server, both live well inside this.
    vllm_timeout: float = Field(default=300.0, gt=0)
    # Retries for connection errors and 5xx, which is what a restarting or
    # briefly overloaded upstream looks like. Generation is greedy by default,
    # so a retried request returns the same text.
    vllm_max_retries: int = Field(default=2, ge=0)
    # In-flight /completions requests across all callers. vLLM does its own
    # continuous batching, so this is a politeness cap on queue depth, not the
    # thing that creates batches.
    max_concurrent_requests: int = Field(default=32, gt=0)

    # --- What is served ---------------------------------------------------
    # The checkpoint vLLM serves. Read locally for the tokenizer and chat
    # template only (no weights are loaded), and reported by /model-info.
    base_model_id: str = "google/translategemma-12b-it"
    # base    : untouched checkpoint only.
    # adapter : base + LoRA adapter, the fine-tuned system only.
    # both    : one set of weights, adapter toggled per request, so /translate
    #           can answer as either system. This is the baseline-vs-adapter
    #           comparison evaluate_translations.py makes, served live.
    model_mode: ModelMode = ModelMode.ADAPTER
    # Provenance only, reported by /model-info: with a merged checkpoint the
    # adapter is already folded into the weights vLLM serves, and nothing is
    # loaded from this path. Left settable so a served translation can still be
    # attributed to the adapter it came from.
    adapter_path: str | None = None
    # Where the tokenizer and chat template are read from. Defaults to
    # base_model_id; override when the merged checkpoint vLLM serves is not the
    # directory this container has mounted.
    tokenizer_path: str | None = None
    # Which system answers a request that does not name one. Must be loaded.
    default_system: System | None = None

    # --- How the upstream was loaded (reported, not applied) --------------
    dtype: str = "bfloat16"
    # "sdpa" is the portable default. "flash_attention_3" requires the image to
    # carry the FA3 wheel and a Hopper (sm_90) GPU.
    attn_implementation: str = "sdpa"
    # 4-bit weights trade throughput for VRAM. Off by default: a 12B model in
    # bf16 fits a single 80 GB card comfortably.
    load_in_4bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    # Where vLLM was told to place the model. Reported, never applied.
    device: str | None = None

    # --- Prompt rendering -------------------------------------------------
    # Each system is queried the way it was trained: the SFT adapter after the
    # training rendering, an untouched upstream checkpoint after the generation
    # prompt. Mixing them is silent -- generation still returns fluent text from
    # a prefix the model never saw. See prompting.py and
    # docs/2026-08-10_adapter_degeneration_analysis.md section B.
    #
    # Override only for a MERGED adapter, which is served as the base system
    # (model_mode="base") but still expects the training rendering.
    base_use_training_rendering: bool = False
    adapter_use_training_rendering: bool = True

    # --- Defaults for a request that does not override them ---------------
    source_lang: str = "en"
    target_lang: str = "fa"

    # --- Decoding ---------------------------------------------------------
    # Mirrors the `evaluation:` block of config.yaml. Greedy by default:
    # translation does not benefit from sampling, and determinism keeps a served
    # translation comparable with the evaluated one.
    max_new_tokens: int = Field(default=512, gt=0)
    do_sample: bool = False
    temperature: float = Field(default=1.0, gt=0)
    top_p: float = Field(default=1.0, gt=0, le=1.0)
    num_beams: int = Field(default=1, ge=1)

    # --- Batching ---------------------------------------------------------
    # Segments per /completions request. vLLM accepts a list of prompts in one
    # request and schedules them itself, so this controls HTTP round-trips (and
    # how far a single slow segment can hold up its neighbours), not GPU
    # batching. Chunks are dispatched concurrently up to
    # max_concurrent_requests.
    batch_size: int = Field(default=8, gt=0)
    # Hard cap on inputs per /translate/batch request, so one caller cannot
    # occupy the single GPU worker indefinitely.
    max_batch_items: int = Field(default=128, gt=0)

    # --- Sentence splitting (optional, off by default) --------------------
    # TranslateGemma was fine-tuned on whole segments, so splitting is opt-in:
    # useful for long free text, off-distribution for short ones.
    split_sentences: bool = False

    # --- CORS -------------------------------------------------------------
    # Origins permitted by CORSMiddleware. Comma-separated string or JSON list
    # in environment variable (e.g. TG_CORS_ORIGINS="http://localhost:3000,http://localhost:8000").
    # Defaults to allowing all origins.
    cors_origins: list[str] = ["*"]
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] = ["*"]
    cors_allow_headers: list[str] = ["*"]

    @field_validator(
        "adapter_path", "tokenizer_path", "default_system", "device", "vllm_api_key", mode="before"
    )
    @classmethod
    def _convert_empty_str_to_none(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("cors_origins", "cors_allow_methods", "cors_allow_headers", mode="before")
    @classmethod
    def _parse_cors_list(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return []
            if value.startswith("[") and value.endswith("]"):
                try:
                    return json.loads(value)
                except Exception:
                    pass
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


    @model_validator(mode="after")
    def _validate_and_resolve(self):
        self.base_model_id = _resolve_base_model_path(self.base_model_id)
        # Not validated as a directory: with a merged checkpoint the adapter is
        # already in vLLM's weights and this is a label, which may well name a
        # path that only the machine that ran the merge ever had.
        self.vllm_base_url = self.vllm_base_url.rstrip("/")

        if self.default_system is None:
            self.default_system = (
                System.BASE if self.model_mode is ModelMode.BASE else System.ADAPTER
            )
        if self.default_system not in self.loaded_systems:
            raise ValueError(
                f"TG_DEFAULT_SYSTEM={self.default_system} is not loaded under "
                f"TG_MODEL_MODE={self.model_mode}."
            )
        return self

    @property
    def resolved_tokenizer_path(self) -> str:
        return self.tokenizer_path or self.base_model_id

    @property
    def loaded_systems(self) -> tuple[System, ...]:
        if self.model_mode is ModelMode.BASE:
            return (System.BASE,)
        if self.model_mode is ModelMode.ADAPTER:
            return (System.ADAPTER,)
        return (System.BASE, System.ADAPTER)

    def use_training_rendering(self, system: System) -> bool:
        return (
            self.adapter_use_training_rendering
            if system is System.ADAPTER
            else self.base_use_training_rendering
        )


def _resolve_base_model_path(base_model_id: str) -> str:
    """Return an absolute local base model directory, or a hub id unchanged.

    Transformers treats any local path without config.json as a Hub repo id,
    which in offline mode fails with an opaque LocalEntryNotFoundError.
    """
    if _looks_like_hub_id(base_model_id):
        return base_model_id
    path = Path(base_model_id).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"Base model path does not exist: {path} (from {base_model_id!r})")
    if not (path / "config.json").is_file():
        available = sorted(
            child.parent.relative_to(path).as_posix()
            for child in path.rglob("config.json")
        )
        hint = f" Models found below it: {available[:10]}" if available else ""
        raise ValueError(f"No config.json in {path}.{hint}")
    return str(path)


def _looks_like_hub_id(value: str) -> bool:
    """True for "org/name" style ids that are not filesystem paths."""
    return "/" in value and not value.startswith((".", "/", "~")) and value.count("/") == 1


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
