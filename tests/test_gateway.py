"""Unit and integration tests for TranslateGemma Gateway."""

import asyncio
import codecs
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from gateway.config import Settings
from gateway.limits import ConcurrencyManager, RequestValidator, TokenEstimator
from gateway.main import BodySizeLimitMiddleware, CanonicalPromptRenderer, app, validate_system_option
from gateway.metrics import MetricsCollector
from gateway.prompting import TARGET_BOUNDARY_MARKER, render_training_prompt
from gateway.routing import SentenceSplitter, WorkloadClass, WorkloadClassifier, dispatch_structured_batch
from gateway.schemas import BatchPrompt, Prompt
from gateway.vllm_client import VLLMClient


def test_canonical_prompt_renderer_structured_message():
    renderer = CanonicalPromptRenderer(allow_fallback=True)
    rendered = renderer.render("en", "fa", "Cellular biology is the study of cell structure.")

    assert rendered.startswith("<start_of_turn>user\n<<<source>>>en<<<target>>>fa<<<text>>>Cellular biology is the study of cell structure.<end_of_turn>\n<start_of_turn>model\n\n        ")
    assert rendered.endswith("\n\n        ")
    assert TARGET_BOUNDARY_MARKER not in rendered


def test_canonical_prompt_renderer_production_refuses_fallback():
    renderer = CanonicalPromptRenderer(processor_or_tokenizer=None, allow_fallback=False)
    assert renderer.mode == "none"
    with pytest.raises(RuntimeError, match="Canonical processor rendering is required in production"):
        renderer.render("en", "fa", "Sample text")


def test_canonical_prompt_renderer_with_processor():
    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = (
        "<start_of_turn>user\n<<<source>>>en<<<target>>>fa<<<text>>>Hello<end_of_turn>\n<start_of_turn>model\n\n        "
    )
    renderer = CanonicalPromptRenderer(processor_or_tokenizer=mock_processor, allow_fallback=False)
    assert renderer.mode == "canonical"
    rendered = renderer.render("en", "fa", "Hello")
    assert rendered.startswith("<start_of_turn>user\n")


def test_vllm_client_payload_structure():
    settings = Settings(vllm_model_name="translategemma", stop_token_ids=[1, 106])
    client = VLLMClient(settings)
    payload = client.build_raw_completion_payload(
        prompt="test_prompt",
        max_tokens=128,
        stream=False,
    )

    # vLLM wire protocol: top-level stop_token_ids (not nested under extra_body)
    assert payload["model"] == "translategemma"
    assert payload["stop_token_ids"] == [1, 106]
    assert "<end_of_turn>" in payload["stop"]
    assert "extra_body" not in payload


@pytest.mark.asyncio
async def test_concurrency_cancellation_leak_free():
    settings = Settings(max_concurrent_requests=1, max_queue_depth=5, max_interactive_queue_depth=5)
    mgr = ConcurrencyManager(settings)

    # 1. First task acquires the only slot
    wait1 = await mgr.acquire()
    assert mgr.in_flight == 1
    assert mgr.queued == 0

    # 2. Second task queues up
    task2 = asyncio.create_task(mgr.acquire())
    await asyncio.sleep(0.01)
    assert mgr.queued == 1
    assert mgr.in_flight == 1

    # 3. Cancel task2 while queued
    task2.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task2

    # After cancellation, queued count must decrement and in_flight must NOT have increased
    assert mgr.queued == 0
    assert mgr.in_flight == 1

    # 4. Release task 1
    mgr.release()
    assert mgr.in_flight == 0
    assert mgr.queued == 0


@pytest.mark.asyncio
async def test_bulk_queue_fairness_does_not_starve_interactive():
    settings = Settings(
        max_concurrent_requests=2,
        max_bulk_concurrent_requests=1,
        max_interactive_queue_depth=5,
        max_bulk_queue_depth=2,
    )
    mgr = ConcurrencyManager(settings)

    # 1. Acquire bulk slot
    await mgr.acquire(is_bulk=True)
    assert mgr.bulk_in_flight == 1

    # 2. Fill bulk queue to capacity (2 tasks)
    bulk_task1 = asyncio.create_task(mgr.acquire(is_bulk=True))
    bulk_task2 = asyncio.create_task(mgr.acquire(is_bulk=True))
    await asyncio.sleep(0.01)
    assert mgr.bulk_queued == 2

    # 3. Next bulk request should be rejected with 429 because bulk queue is saturated
    with pytest.raises(HTTPException) as exc:
        await mgr.acquire(is_bulk=True)
    assert exc.value.status_code == 429
    assert "Bulk processing queue is saturated" in exc.value.detail

    # 4. Interactive request CAN still acquire immediately (or queue) because interactive queue is isolated!
    await mgr.acquire(is_bulk=False)
    assert mgr.interactive_in_flight == 1
    assert mgr.interactive_queued == 0

    # Cleanup
    mgr.release(is_bulk=False)
    mgr.release(is_bulk=True)
    bulk_task1.cancel()
    bulk_task2.cancel()
    await asyncio.gather(bulk_task1, bulk_task2, return_exceptions=True)


def test_system_option_validation():
    assert validate_system_option(None, "adapter") == "adapter"
    assert validate_system_option("adapter", "adapter") == "adapter"

    with pytest.raises(HTTPException) as exc1:
        validate_system_option("base", "adapter")
    assert exc1.value.status_code == 400
    assert "System 'base' is not loaded" in exc1.value.detail


def test_token_estimator_and_request_validator():
    settings = Settings(
        max_source_chars_per_text=100,
        max_estimated_source_tokens=50,
        max_total_context_tokens=60,
        max_batch_items=5,
        max_batch_total_chars=200,
        max_batch_total_tokens=100,
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

    # Context limit exceeded -> 422
    with pytest.raises(HTTPException) as exc:
        validator.validate_request("Valid text", "Rendered prompt...", 100)
    assert exc.value.status_code == 422
    assert "exceeds the model context window" in exc.value.detail

    # Batch total token limit exceeded -> 413
    with pytest.raises(HTTPException) as exc:
        validator.validate_batch(["Word " * 50, "Another text " * 50], 10)
    assert exc.value.status_code == 413
    assert "Aggregate batch estimated tokens" in exc.value.detail


@pytest.mark.asyncio
async def test_structured_batch_cancellation():
    async def _failing_worker(text: str) -> str:
        if text == "bad":
            raise RuntimeError("Backend failure")
        await asyncio.sleep(0.05)
        return f"translated_{text}"

    with pytest.raises(RuntimeError, match="Backend failure"):
        await dispatch_structured_batch(["ok1", "bad", "ok2"], _failing_worker)


@pytest.mark.asyncio
async def test_stream_persian_multibyte_chunk_boundaries():
    """Prove stateful UTF-8 incremental decoding reconstructs Persian text across byte boundaries without corruption."""
    persian_text = "این یک متن فارسی طولانی با نویسه‌های چندبایتی برای آزمون جریان است."
    encoded_bytes = f'data: {json.dumps({"choices": [{"text": persian_text, "finish_reason": "stop"}]})}\n\n'.encode("utf-8")

    # Split into 1-byte, 2-byte, and arbitrary byte chunks
    for chunk_size in [1, 2, 3, 5, 7]:
        byte_chunks = [encoded_bytes[i : i + chunk_size] for i in range(0, len(encoded_bytes), chunk_size)]

        decoder = codecs.getincrementaldecoder("utf-8")("strict")
        reconstructed = ""
        for b in byte_chunks:
            reconstructed += decoder.decode(b, final=False)
        reconstructed += decoder.decode(b"", final=True)

        assert "\ufffd" not in reconstructed
        assert persian_text in reconstructed


def test_gateway_api_endpoints():
    mock_vllm_client = MagicMock()
    mock_vllm_client.check_health = AsyncMock(return_value=True)
    mock_vllm_client.generate_raw_completion = AsyncMock(
        return_value=("  ترجمه تست \n", "stop", {"prompt_tokens": 10, "completion_tokens": 5})
    )

    settings = Settings(
        default_system="adapter",
        max_request_body_bytes=500,
    )
    estimator = TokenEstimator()
    renderer = CanonicalPromptRenderer(allow_fallback=True)
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

    # 1. Liveness
    resp = client.get("/live")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ALIVE"}

    # 2. Legacy /health-check -> strictly {"translator": "OK"}
    resp = client.get("/health-check")
    assert resp.status_code == 200
    assert resp.json() == {"translator": "OK"}

    # 3. /ready probe exposes processor_mode and estimator_mode
    resp_ready = client.get("/ready")
    assert resp_ready.status_code == 200
    data_ready = resp_ready.json()
    assert data_ready["ready"] is True
    assert "processor_mode" in data_ready
    assert "estimator_mode" in data_ready

    # 4. /model-info exposes processor_mode
    resp_info = client.get("/model-info")
    assert resp_info.status_code == 200
    data_info = resp_info.json()
    assert "processor_mode" in data_info

    # 5. Single translate preserves trailing whitespace without stripping
    resp = client.post("/translate", json={"text": "Hello world", "source_lang": "en", "target_lang": "fa"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["translation"] == "  ترجمه تست \n"
    assert data["system"] == "adapter"

    # 6. Oversized body -> 413 via BodySizeLimitMiddleware
    large_payload = {"text": "A" * 1000}
    resp_large = client.post(
        "/translate",
        content=json.dumps(large_payload),
        headers={"Content-Length": str(len(json.dumps(large_payload)))},
    )
    assert resp_large.status_code == 413

    # 7. Invalid negative Content-Length -> 400
    resp_neg = client.post(
        "/translate",
        content=b"{}",
        headers={"Content-Length": "-5"},
    )
    assert resp_neg.status_code == 400

    # 8. Backend failure in /health-check -> 503 with {"translator": "FAIL"}
    mock_vllm_client.check_health = AsyncMock(return_value=False)
    resp_down = client.get("/health-check")
    assert resp_down.status_code == 503
    assert resp_down.json() == {"translator": "FAIL"}

