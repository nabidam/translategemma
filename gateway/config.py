"""Configuration settings for the TranslateGemma FastAPI Serving Gateway."""

from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Backend vLLM connection
    vllm_base_url: str = "http://127.0.0.1:8000/v1"
    vllm_model_name: str = "translategemma"
    vllm_timeout_seconds: float = 60.0
    vllm_connect_timeout_seconds: float = 5.0
    vllm_max_model_len: int = 4096

    # Model & Tokenizer directory (for offline exact prompt rendering and token counting)
    model_dir: Optional[str] = "/models/model"
    tokenizer_path: Optional[str] = None

    # Model identity & provenance
    model_release_id: str = "translategemma-12b-it-merged"
    base_model_id: str = "google/translategemma-12b-it"
    source_adapter_path: Optional[str] = "checkpoints/sft-translategemma-12b-it"
    default_system: str = "adapter"
    source_lang: str = "en"
    target_lang: str = "fa"
    max_new_tokens: int = 512
    split_sentences: bool = False

    # Stop tokens
    stop_token_ids: List[int] = [1, 106]
    stop_tokens: List[str] = ["<eos>", "<end_of_turn>"]

    # Admission & Concurrency Controls
    max_concurrent_requests: int = 64
    max_queue_depth: int = 128
    max_request_body_bytes: int = 1_000_000  # 1MB
    max_batch_items: int = 128
    max_source_chars_per_text: int = 50_000
    max_estimated_source_tokens: int = 3584
    max_total_context_tokens: int = 4096  # strictly aligned with vllm_max_model_len

    # Workload routing thresholds (in estimated tokens)
    interactive_max_tokens: int = 128
    document_max_tokens: int = 2048

    # CORS configuration
    cors_origins: List[str] = ["*"]
    cors_allow_credentials: bool = False
    cors_allow_methods: List[str] = ["GET", "POST", "OPTIONS"]
    cors_allow_headers: List[str] = ["*"]

    # Gateway Server (1 worker per container for global in-process admission control)
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_prefix="TG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
