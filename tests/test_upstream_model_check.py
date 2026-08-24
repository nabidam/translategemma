"""The startup check that TG_VLLM_MODEL names something the upstream serves.

A wrong model name used to be invisible until the first translation, where
vLLM answers 4xx and the gateway turned it into a bare 500 with nothing
actionable in it. That cost a real debugging session against a live
deployment, which is why this check exists.

The asymmetry these tests pin is the whole design: a definite mismatch is
fatal at startup, but an upstream we could not reach is NOT — that is a
restart or a slow start, and crash-looping the gateway for it would be a
worse outage than the one the check prevents.
"""

import logging

import httpx
import pytest

from translator import TranslationEngine


class FakeSettings:
    """Only the fields the check and the client touch."""

    max_concurrent_requests = 4
    vllm_model = "translategemma"
    vllm_base_url = "http://upstream/v1"
    vllm_max_retries = 0


def engine_with(handler) -> TranslationEngine:
    """An engine wired to a fake upstream, bypassing load().

    load() builds a tokenizer, which needs transformers and a checkpoint —
    neither is available here, and neither is what these tests are about.
    """
    engine = TranslationEngine(FakeSettings())
    engine._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://upstream/v1"
    )
    return engine


def serving(*model_ids: str):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": name} for name in model_ids]})

    return handler


async def test_a_served_model_passes_the_check():
    engine = engine_with(serving("translategemma"))
    await engine.verify_upstream_model()  # must not raise
    await engine.aclose()


async def test_a_model_the_upstream_does_not_serve_fails_startup():
    engine = engine_with(serving("qwen", "some-other-model"))
    with pytest.raises(ValueError) as error:
        await engine.verify_upstream_model()
    message = str(error.value)
    # The message has to name both halves, or it is no more actionable than
    # the 500 it replaces.
    assert "translategemma" in message
    assert "qwen" in message
    assert "TG_VLLM_MODEL" in message
    await engine.aclose()


async def test_an_unreachable_upstream_warns_but_does_not_fail(caplog):
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    engine = engine_with(refuse)
    with caplog.at_level(logging.WARNING, logger="translategemma.api"):
        await engine.verify_upstream_model()  # must not raise
    assert any("could not verify" in record.message.lower() for record in caplog.records)
    await engine.aclose()


async def test_a_non_200_from_the_models_endpoint_warns_but_does_not_fail(caplog):
    engine = engine_with(lambda request: httpx.Response(503, text="starting up"))
    with caplog.at_level(logging.WARNING, logger="translategemma.api"):
        await engine.verify_upstream_model()  # must not raise
    assert caplog.records
    await engine.aclose()


async def test_an_unexpected_body_warns_but_does_not_fail(caplog):
    engine = engine_with(lambda request: httpx.Response(200, json={"unexpected": True}))
    with caplog.at_level(logging.WARNING, logger="translategemma.api"):
        await engine.verify_upstream_model()  # must not raise
    assert caplog.records
    await engine.aclose()


async def test_a_rejected_request_names_the_configured_model():
    """Belt and braces for a deployment whose startup check was skipped."""
    engine = engine_with(
        lambda request: httpx.Response(400, text="The model `x` does not exist.")
    )
    with pytest.raises(RuntimeError) as error:
        await engine._post("/completions", {"prompt": [[1, 2]]})
    assert "translategemma" in str(error.value)
    await engine.aclose()


async def test_an_engine_with_no_client_warns_but_does_not_fail(caplog):
    """No client means no way to ask — the same case as an unreachable upstream.

    This is not hypothetical: the kill-switch tests drive the real lifespan with
    a stubbed load(), so no client is ever built. An earlier version of this
    check raised here and broke five of them, which was the check contradicting
    its own rule that only a definite mismatch is fatal.
    """
    engine = TranslationEngine(FakeSettings())
    with caplog.at_level(logging.WARNING, logger="translategemma.api"):
        await engine.verify_upstream_model()  # must not raise
    assert any("could not verify" in record.message.lower() for record in caplog.records)
