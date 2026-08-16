"""Unit and integration tests for TranslateGemma Gateway."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from gateway.config import Settings
from gateway.limits import ConcurrencyManager, RequestValidator, TokenEstimator
from gateway.main import CanonicalPromptRenderer, app, validate_system_option
from gateway.metrics import MetricsCollector
from gateway.prompting import TARGET_BOUNDARY_MARKER, render_training_prompt
from gateway.routing import SentenceSplitter, WorkloadClass, WorkloadClassifier, dispatch_structured_batch
from gateway.schemas import BatchPrompt, Prompt


def test_canonical_prompt_renderer_parity():
    renderer = CanonicalPromptRenderer()
    rendered = renderer.render("en", "fa", "Cellular biology is the study of cell structure.")

    assert rendered.startswith("<start_of_turn>user\n<<<source>>>en<<<target>>>fa<<<text>>>Cellular biology is the study of cell structure.<end_of_turn>\n<start_of_turn>model\n\n        ")
    assert rendered.endswith("\n\n        ")
    assert TARGET_BOUNDARY_MARKER not in rendered


def test_system_option_validation():
    # Default adapter system succeeds
    assert validate_system_option(None, "adapter") == "adapter"
    assert validate_system_option("adapter", "adapter") == "adapter"

    # Unsupported systems like 'base' or random string return 400
    with pytest.raises(HTTPException) as exc1:
        validate_system_option("base", "adapter")
    assert exc1.value.status_code == 400
    assert "System 'base' is not loaded" in exc1.value.detail

    with pytest.raises(HTTPException) as exc2:
        validate_system_option("custom", "adapter")
    assert exc2.value.status_code == 400


def test_token_estimator():
    estimator = TokenEstimator()
    tokens = estimator.count_tokens("The model relies on attention.")
    assert tokens > 0
    assert isinstance(tokens, int)

    batch_tokens = estimator.count_batch_tokens(["Hello", "World"])
    assert len(batch_tokens) == 2


def test_request_validator_context_limits():
    settings = Settings(
        max_source_chars_per_text=100,
        max_estimated_source_tokens=50,
        max_total_context_tokens=60,
    )
    estimator = TokenEstimator()
    validator = RequestValidator(settings, estimator)

    # Empty text -> 422
    with pytest.raises(HTTPException) as exc:
        validator.validate_request("   ", "prompt", 10)
    assert exc.value.status_code == 422

    # Oversized raw text -> 413
    with pytest.raises(HTTPException) as exc:
        validator.validate_request("a" * 150, "prompt", 10)
    assert exc.value.status_code == 413

    # Context limit exceeded (prompt tokens + max_new_tokens > max_total_context_tokens) -> 422
    with pytest.raises(HTTPException) as exc:
        validator.validate_request("Valid text", "Rendered prompt with many tokens...", 100)
    assert exc.value.status_code == 422
    assert "exceeds the model context window" in exc.value.detail


def test_sentence_splitter():
    splitter = SentenceSplitter()
    text = "First sentence. Second sentence! Third one?"
    sentences = splitter.split_sentences(text, "en")
    assert len(sentences) == 3
    assert sentences[0] == "First sentence."
    assert sentences[1] == "Second sentence!"
    assert sentences[2] == "Third one?"


def test_workload_classifier():
    settings = Settings(interactive_max_tokens=10)
    estimator = TokenEstimator()
    classifier = WorkloadClassifier(settings, estimator)

    assert classifier.classify_single("Hi", 3) == WorkloadClass.INTERACTIVE
    assert classifier.classify_single("Long text...", 50) == WorkloadClass.DOCUMENT
    assert classifier.classify_batch(["Hi", "There"]) == WorkloadClass.BULK


@pytest.mark.asyncio
async def test_structured_batch_cancellation_on_failure():
    async def _failing_task(text: str) -> str:
        if text == "fail":
            raise RuntimeError("Backend error")
        await asyncio.sleep(0.05)
        return f"translated_{text}"

    with pytest.raises(RuntimeError, match="Backend error"):
        await dispatch_structured_batch(["ok1", "fail", "ok2"], _failing_task)


@pytest.mark.asyncio
async def test_concurrency_manager_saturation():
    settings = Settings(max_concurrent_requests=1, max_queue_depth=1)
    mgr = ConcurrencyManager(settings)

    wait1 = await mgr.acquire()
    assert wait1 >= 0
    assert mgr.in_flight == 1

    task2 = asyncio.create_task(mgr.acquire())
    await asyncio.sleep(0.01)
    assert mgr.queued == 1

    with pytest.raises(HTTPException) as exc:
        await mgr.acquire()
    assert exc.value.status_code == 429

    mgr.release()
    await task2
    mgr.release()


def test_gateway_api_endpoints_health_and_translation():
    mock_vllm_client = MagicMock()
    mock_vllm_client.check_health = AsyncMock(return_value=True)
    mock_vllm_client.generate_raw_completion = AsyncMock(
        return_value=("  ترجمه تست \n", "stop", {"prompt_tokens": 10, "completion_tokens": 5})
    )

    settings = Settings(default_system="adapter")
    estimator = TokenEstimator()
    renderer = CanonicalPromptRenderer()
    concurrency_mgr = ConcurrencyManager(settings)
    validator = RequestValidator(settings, estimator)
    splitter = SentenceSplitter()
    classifier = WorkloadClassifier(settings, estimator)
    metrics = MetricsCollector()

    app.state.settings = settings
    app.state.vllm_client = mock_vllm_client
    app.state.estimator = estimator
    app.state.renderer = renderer
    app.state.concurrency_mgr = concurrency_mgr
    app.state.validator = validator
    app.state.splitter = splitter
    app.state.classifier = classifier
    app.state.metrics = metrics

    client = TestClient(app)

    # 1. Liveness check
    resp = client.get("/live")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ALIVE"}

    # 2. Readiness check (Healthy)
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json() == {"translator": "OK", "ready": True, "detail": None}

    # 3. Model info (provenance)
    resp = client.get("/model-info")
    assert resp.status_code == 200
    info = resp.json()
    assert info["is_merged_checkpoint"] is True
    assert info["default_system"] == "adapter"
    assert info["loaded_systems"] == ["adapter"]
    assert 106 in info["stop_token_ids"]

    # 4. Single translate preserving trailing whitespace without strip()
    resp = client.post("/translate", json={"text": "Hello world", "source_lang": "en", "target_lang": "fa"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["translation"] == "  ترجمه تست \n"
    assert data["system"] == "adapter"

    # 5. Translate with invalid system returns 400
    resp_bad = client.post("/translate", json={"text": "Hello", "system": "base"})
    assert resp_bad.status_code == 400
    assert "System 'base' is not loaded" in resp_bad.json()["detail"]

    # 6. Backend failure in readiness returns 503
    mock_vllm_client.check_health = AsyncMock(return_value=False)
    resp_unready = client.get("/ready")
    assert resp_unready.status_code == 503
    assert resp_unready.json()["ready"] is False
