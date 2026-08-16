"""Unit and integration tests for TranslateGemma Gateway."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from gateway.config import Settings
from gateway.limits import ConcurrencyManager, RequestValidator, TokenEstimator
from gateway.main import app, render_exact_training_prompt
from gateway.metrics import MetricsCollector
from gateway.routing import SentenceSplitter, WorkloadClass, WorkloadClassifier
from gateway.schemas import BatchPrompt, Prompt


def test_render_exact_training_prompt():
    prompt = render_exact_training_prompt("en", "fa", "Hello world")
    assert prompt.startswith("<start_of_turn>user\n<<<source>>>en<<<target>>>fa<<<text>>>Hello world<end_of_turn>\n<start_of_turn>model\n\n        ")
    assert prompt.endswith("\n\n        ")


def test_token_estimator():
    estimator = TokenEstimator()
    tokens = estimator.estimate_tokens("The model relies on attention.")
    assert tokens > 0
    assert isinstance(tokens, int)

    batch_tokens = estimator.estimate_batch_tokens(["Hello", "World"])
    assert len(batch_tokens) == 2


def test_request_validator():
    settings = Settings(
        max_source_chars_per_text=100,
        max_estimated_source_tokens=50,
        max_total_context_tokens=100,
    )
    estimator = TokenEstimator()
    validator = RequestValidator(settings, estimator)

    # Empty text
    with pytest.raises(HTTPException) as exc:
        validator.validate_text("   ", 10)
    assert exc.value.status_code == 422

    # Oversized char text
    with pytest.raises(HTTPException) as exc:
        validator.validate_text("a" * 150, 10)
    assert exc.value.status_code == 413

    # Valid text
    tokens = validator.validate_text("Valid text", 10)
    assert tokens > 0


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
async def test_concurrency_manager_saturation():
    settings = Settings(max_concurrent_requests=1, max_queue_depth=1)
    mgr = ConcurrencyManager(settings)

    # First acquires slot
    wait1 = await mgr.acquire()
    assert wait1 >= 0
    assert mgr.in_flight == 1

    # Second waits in queue (depth 1)
    task2 = asyncio.create_task(mgr.acquire())
    await asyncio.sleep(0.01)
    assert mgr.queued == 1

    # Third should get 429 immediately because queue is saturated
    with pytest.raises(HTTPException) as exc:
        await mgr.acquire()
    assert exc.value.status_code == 429

    # Release slot
    mgr.release()
    await task2
    mgr.release()


def test_metrics_collector():
    collector = MetricsCollector()
    collector.record_request("/translate", "interactive")
    collector.record_completion(
        endpoint="/translate",
        workload_class="interactive",
        latency=0.15,
        queue_wait=0.01,
        prompt_tokens=20,
        completion_tokens=25,
        finish_reason="stop",
    )

    summary = collector.get_summary()
    assert summary["requests"]["/translate:interactive"] == 1
    assert summary["total_prompt_tokens"] == 20
    assert summary["total_completion_tokens"] == 25
    assert summary["finish_reasons"]["stop"] == 1


def test_gateway_api_endpoints():
    mock_vllm_client = MagicMock()
    mock_vllm_client.check_health = AsyncMock(return_value=True)
    mock_vllm_client.generate_raw_completion = AsyncMock(
        return_value=("ترجمه تست", "stop", {"prompt_tokens": 10, "completion_tokens": 5})
    )

    settings = Settings()
    estimator = TokenEstimator()
    concurrency_mgr = ConcurrencyManager(settings)
    validator = RequestValidator(settings, estimator)
    splitter = SentenceSplitter()
    classifier = WorkloadClassifier(settings, estimator)
    metrics = MetricsCollector()

    app.state.settings = settings
    app.state.vllm_client = mock_vllm_client
    app.state.estimator = estimator
    app.state.concurrency_mgr = concurrency_mgr
    app.state.validator = validator
    app.state.splitter = splitter
    app.state.classifier = classifier
    app.state.metrics = metrics

    client = TestClient(app)

    # 1. Health check
    resp = client.get("/health-check")
    assert resp.status_code == 200
    assert resp.json() == {"translator": "OK"}

    # 2. Model info
    resp = client.get("/model-info")
    assert resp.status_code == 200
    info = resp.json()
    assert "stop_token_ids" in info
    assert 106 in info["stop_token_ids"]

    # 3. Single translate
    resp = client.post("/translate", json={"text": "Hello world", "source_lang": "en", "target_lang": "fa"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["translation"] == "ترجمه تست"
    assert data["source_lang"] == "en"
    assert data["target_lang"] == "fa"

    # 4. Batch translate
    resp = client.post("/translate/batch", json={"texts": ["Text 1", "Text 2"], "source_lang": "en", "target_lang": "fa"})
    assert resp.status_code == 200
    b_data = resp.json()
    assert len(b_data["translations"]) == 2
    assert b_data["translations"] == ["ترجمه تست", "ترجمه تست"]

    # 5. Metrics
    resp = client.get("/metrics")
    assert resp.status_code == 200
    m_data = resp.json()
    assert "requests" in m_data
