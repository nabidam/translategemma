"""Runtime configuration for the TranslateGemma serving API.

Everything is settable from the environment (or an api/.env file) so the same
image serves a base checkpoint, a base + LoRA adapter, or both side by side
without a rebuild. See api/.env.example.
"""

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

    # --- What to load -----------------------------------------------------
    base_model_id: str = "google/translategemma-12b-it"
    # base    : untouched checkpoint only.
    # adapter : base + LoRA adapter, the fine-tuned system only.
    # both    : one set of weights, adapter toggled per request, so /translate
    #           can answer as either system. This is the baseline-vs-adapter
    #           comparison evaluate_translations.py makes, served live.
    model_mode: ModelMode = ModelMode.ADAPTER
    # Directory holding adapter_config.json, or a hub id. Required unless
    # model_mode is "base".
    adapter_path: str | None = None
    # Which system answers a request that does not name one. Must be loaded.
    default_system: System | None = None

    # --- How to load ------------------------------------------------------
    dtype: str = "bfloat16"
    # "sdpa" is the portable default. "flash_attention_3" requires the image to
    # carry the FA3 wheel and a Hopper (sm_90) GPU.
    attn_implementation: str = "sdpa"
    # 4-bit weights trade throughput for VRAM. Off by default: a 12B model in
    # bf16 fits a single 80 GB card comfortably.
    load_in_4bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    device: str | None = None  # None -> cuda if available, else cpu.

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
    # Segments per generate() call.
    batch_size: int = Field(default=8, gt=0)
    # Hard cap on inputs per /translate/batch request, so one caller cannot
    # occupy the single GPU worker indefinitely.
    max_batch_items: int = Field(default=128, gt=0)

    # --- Sentence splitting (optional, off by default) --------------------
    # TranslateGemma was fine-tuned on whole segments, so splitting is opt-in:
    # useful for long free text, off-distribution for short ones.
    split_sentences: bool = False

    @field_validator("adapter_path", "default_system", "device", mode="before")
    @classmethod
    def _convert_empty_str_to_none(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value


    @model_validator(mode="after")
    def _validate_and_resolve(self):
        self.base_model_id = _resolve_base_model_path(self.base_model_id)
        if self.model_mode is not ModelMode.BASE:
            if not self.adapter_path:
                raise ValueError(f"TG_MODEL_MODE={self.model_mode} requires TG_ADAPTER_PATH.")
            self.adapter_path = _resolve_adapter_path(self.adapter_path)

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


def _resolve_adapter_path(adapter_path: str) -> str:
    """Return an absolute local adapter directory, or a hub id unchanged.

    Mirrors evaluate_translations.resolve_adapter_path. PeftModel.from_pretrained
    treats any local path without adapter_config.json as a Hub repo id, so a
    wrong directory surfaces as an opaque HFValidationError minutes into startup
    instead of as a path problem.
    """
    if _looks_like_hub_id(adapter_path):
        return adapter_path
    path = Path(adapter_path).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"Adapter path does not exist: {path} (from {adapter_path!r})")
    if not (path / "adapter_config.json").is_file():
        available = sorted(
            child.parent.relative_to(path).as_posix()
            for child in path.rglob("adapter_config.json")
        )
        hint = f" Adapters found below it: {available[:10]}" if available else ""
        raise ValueError(f"No adapter_config.json in {path}.{hint}")
    return str(path)


def _looks_like_hub_id(value: str) -> bool:
    """True for "org/name" style ids that are not filesystem paths."""
    return "/" in value and not value.startswith((".", "/", "~")) and value.count("/") == 1


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
