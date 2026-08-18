"""Runtime configuration for the TranslateGemma serving API.

Everything is settable from the environment (or an api/.env file) so the same
image fronts any vLLM deployment without a rebuild. See api/.env.example.

Generation no longer happens in this process: vLLM owns the weights and this
service is a gateway (see translator.py). Nothing here describes how the
upstream was launched -- dtype, attention implementation and quantization are
vLLM's flags and are configured on that service, not duplicated here where
nothing could enforce the copy. What remains is what this process actually
applies: where the upstream is, how prompts are rendered, and the sampling
parameters sent with every request.
"""

import json
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class System(StrEnum):
    """Which system the upstream answers as.

    One vLLM server holds one set of weights, so one gateway serves exactly one
    system. The distinction is not cosmetic: the two are prompted differently
    (see `use_training_rendering`), and a translation is only attributable to a
    checkpoint if the server states which of the two it is.

    To compare the two, run a gateway and a vLLM per system. There is no
    in-process adapter toggle any more: the weights are not in this process.
    """

    BASE = "base"
    ADAPTER = "adapter"


class Settings(BaseSettings):
    """Server configuration, read from the environment with `TG_` prefixed names."""

    model_config = SettingsConfigDict(
        env_prefix="TG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # `model_` is a Pydantic-protected namespace, which base_model_id's
        # neighbours (vllm_model) would otherwise warn about. Serving fields
        # keep the training vocabulary, so the protection is dropped rather
        # than the fields renamed.
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
    # Which system the upstream weights are: an untouched checkpoint ("base") or
    # one carrying this repository's SFT adapter, merged or otherwise
    # ("adapter"). Selects the prompt rendering, and is reported by /model-info
    # so a served translation can be attributed.
    served_system: System = System.ADAPTER
    # Provenance only, reported by /model-info: with a merged checkpoint the
    # adapter is already folded into the weights vLLM serves, and nothing is
    # loaded from this path. Left settable so a served translation can still be
    # attributed to the adapter it came from.
    adapter_path: str | None = None
    # Where the tokenizer and chat template are read from. Defaults to
    # base_model_id; override when the merged checkpoint vLLM serves is not the
    # directory this container has mounted.
    tokenizer_path: str | None = None

    # --- Prompt rendering -------------------------------------------------
    # The system is queried the way it was trained: the SFT adapter after the
    # training rendering, an untouched upstream checkpoint after the generation
    # prompt. Mixing them is silent -- generation still returns fluent text from
    # a prefix the model never saw. See prompting.py and
    # docs/2026-08-10_adapter_degeneration_analysis.md section B.
    #
    # Override only for a MERGED adapter that is labelled as the base system
    # (served_system="base") but still expects the training rendering.
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
    cors_origins: list[str] | str = ["*"]
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] | str = ["*"]
    cors_allow_headers: list[str] | str = ["*"]

    @field_validator("adapter_path", "tokenizer_path", "vllm_api_key", mode="before")
    @classmethod
    def _convert_empty_str_to_none(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("cors_origins", "cors_allow_methods", "cors_allow_headers", mode="before")
    @classmethod
    def _parse_cors_list(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return []
            if value.startswith("[") and value.endswith("]"):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        return [str(item) for item in parsed]
                except Exception:
                    pass
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, (list, tuple, set)):
            return [str(item) for item in value]
        return value

    @model_validator(mode="after")
    def _validate_and_resolve(self):
        # Only the path this process actually opens is checked against the
        # filesystem. The gateway reads a tokenizer and a chat template, so when
        # TG_TOKENIZER_PATH names where those live, TG_BASE_MODEL_ID is a label
        # for /model-info and must not have to be mounted -- it may well name a
        # path that only the machine vLLM runs on ever had.
        #
        # adapter_path is never checked for the same reason: with a merged
        # checkpoint the adapter is already in vLLM's weights, and the path is
        # provenance that may only have existed on the machine that merged it.
        if self.tokenizer_path is None:
            self.base_model_id = _resolve_local_checkpoint(self.base_model_id, "TG_BASE_MODEL_ID")
        else:
            self.tokenizer_path = _resolve_local_checkpoint(
                self.tokenizer_path, "TG_TOKENIZER_PATH"
            )
        self.vllm_base_url = self.vllm_base_url.rstrip("/")
        return self

    @property
    def resolved_tokenizer_path(self) -> str:
        return self.tokenizer_path or self.base_model_id

    def use_training_rendering(self, system: System) -> bool:
        return (
            self.adapter_use_training_rendering
            if system is System.ADAPTER
            else self.base_use_training_rendering
        )


def _resolve_local_checkpoint(value: str, env_name: str) -> str:
    """Return an absolute local checkpoint directory, or a hub id unchanged.

    Transformers treats any local path without config.json as a Hub repo id,
    which in offline mode fails with an opaque LocalEntryNotFoundError. Failing
    here instead means the message names the environment variable and says what
    is wrong with the path, which in a container is nearly always the bind mount
    that should have supplied it.
    """
    if _looks_like_hub_id(value):
        return value
    path = Path(value).expanduser().resolve()
    if not path.exists():
        parent = path.parent
        siblings = sorted(child.name for child in parent.iterdir()) if parent.is_dir() else []
        hint = (
            f" {parent} contains: {siblings[:10]}"
            if siblings
            else f" {parent} is empty."
            if parent.is_dir()
            else f" {parent} does not exist either."
        )
        raise ValueError(
            f"{env_name}={value!r} does not exist. Inside a container this is a missing "
            f"bind mount, not a bad setting: check that a volume supplies {path}.{hint}"
        )
    if not path.is_dir():
        raise ValueError(
            f"{env_name}={value!r} is a file, not a checkpoint directory ({path}). A bind "
            "mount whose source is a file mounts as a file."
        )
    if not (path / "config.json").is_file():
        # A mount that landed one level off is the common cause, so point at the
        # checkpoints below it before falling back to a plain listing.
        below = sorted(
            child.parent.relative_to(path).as_posix() for child in path.rglob("config.json")
        )
        if below:
            hint = f" Checkpoints below it: {below[:10]}"
        else:
            contents = sorted(child.name for child in path.iterdir())
            hint = f" It contains: {contents[:10]}" if contents else " It is empty."
        raise ValueError(f"{env_name}={value!r} has no config.json ({path}).{hint}")
    return str(path)


def _looks_like_hub_id(value: str) -> bool:
    """True for "org/name" style ids that are not filesystem paths."""
    return "/" in value and not value.startswith((".", "/", "~")) and value.count("/") == 1


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
