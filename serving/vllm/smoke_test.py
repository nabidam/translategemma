#!/usr/bin/env python
"""Smoke test script for verifying vLLM OpenAI-compatible server before routing traffic.

Runs on the serving machine to validate:
  1. vLLM health and model availability (/health, /v1/models).
  2. Exact raw prompt completion endpoint (/v1/completions).
  3. Stop token handling (<end_of_turn> token 106, finish_reason="stop").
  4. Determinism of greedy decoding (temperature=0.0).
  5. Concurrent request batching.

Usage:
  python serving/vllm/smoke_test.py --base-url http://localhost:8000/v1 --model-name translategemma
"""

import argparse
import json
import logging
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vllm_smoke_test")

# Exact training prompt format for TranslateGemma
SAMPLE_USER_CONTENT = "<<<source>>>en<<<target>>>fa<<<text>>>Cellular biology is the study of cell structure."
PROMPT_TEMPLATE = (
    "<start_of_turn>user\n"
    f"{SAMPLE_USER_CONTENT}<end_of_turn>\n"
    "<start_of_turn>model\n\n        "
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000/v1",
        help="Base URL of the vLLM OpenAI-compatible server (e.g. http://localhost:8000/v1).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="translategemma",
        help="Served model name configured in vLLM.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP request timeout in seconds.",
    )
    return parser.parse_args(argv)


def http_get(url: str, timeout: float = 10.0) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "TranslateGemma-SmokeTest/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_post(url: str, payload: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "TranslateGemma-SmokeTest/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def test_health_and_models(base_url: str, model_name: str, timeout: float) -> bool:
    logger.info("Checking server health and model registration...")
    # Health endpoint is usually at root /health
    root_url = base_url.rstrip("/").removesuffix("/v1")
    health_url = f"{root_url}/health"

    try:
        req = urllib.request.Request(health_url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                logger.error("Health check failed with status %d", resp.status)
                return False
        logger.info("vLLM /health returned 200 OK.")
    except Exception as e:
        logger.warning("Could not probe %s (%s); proceeding to /v1/models check.", health_url, e)

    models_url = f"{base_url.rstrip('/')}/models"
    try:
        data = http_get(models_url, timeout=timeout)
        models = [m.get("id") for m in data.get("data", [])]
        logger.info("Available models in vLLM: %s", models)
        if model_name not in models:
            logger.error("Expected model %r not found in loaded models: %s", model_name, models)
            return False
    except Exception as e:
        logger.error("Failed to query /v1/models: %s", e)
        return False

    return True


def test_completion_and_stop(base_url: str, model_name: str, timeout: float) -> Optional[str]:
    completions_url = f"{base_url.rstrip('/')}/completions"
    payload = {
        "model": model_name,
        "prompt": PROMPT_TEMPLATE,
        "max_tokens": 128,
        "temperature": 0.0,
        "top_p": 1.0,
        "stop": ["<end_of_turn>"],
        "extra_body": {
            "stop_token_ids": [1, 106],
        },
    }

    logger.info("Sending raw completion request to %s...", completions_url)
    start_t = time.perf_counter()
    try:
        resp = http_post(completions_url, payload, timeout=timeout)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        logger.error("vLLM returned HTTP %d: %s", e.code, err_body)
        return None
    except Exception as e:
        logger.error("Error connecting to vLLM: %s", e)
        return None

    elapsed = time.perf_counter() - start_t
    choices = resp.get("choices", [])
    if not choices:
        logger.error("No choices returned in completion response: %s", resp)
        return None

    choice = choices[0]
    text = choice.get("text", "")
    finish_reason = choice.get("finish_reason")

    logger.info("Generation completed in %.3fs. Finish reason: %r", elapsed, finish_reason)
    logger.info("Generated translation: %r", text.strip())

    if finish_reason != "stop":
        logger.error("Expected finish_reason='stop', got %r. Generation did not stop on token.", finish_reason)
        return None

    if "<end_of_turn>" in text or "<start_of_turn>" in text:
        logger.error("Special turn markers leaked into visible completion text: %r", text)
        return None

    if not text.strip():
        logger.error("Empty completion text returned.")
        return None

    return text.strip()


def test_determinism(base_url: str, model_name: str, timeout: float, expected_text: str) -> bool:
    logger.info("Testing greedy determinism across 3 repeated queries...")
    for i in range(3):
        res = test_completion_and_stop(base_url, model_name, timeout)
        if res != expected_text:
            logger.error("Determinism violation on run %d: expected %r, got %r", i + 1, expected_text, res)
            return False
    logger.info("Determinism test PASSED: 3/3 queries matched exact output.")
    return True


def test_concurrency(base_url: str, model_name: str, timeout: float, num_workers: int = 8) -> bool:
    logger.info("Testing continuous batching concurrency with %d parallel requests...", num_workers)
    completions_url = f"{base_url.rstrip('/')}/completions"
    payload = {
        "model": model_name,
        "prompt": PROMPT_TEMPLATE,
        "max_tokens": 128,
        "temperature": 0.0,
        "stop": ["<end_of_turn>"],
    }

    start_all = time.perf_counter()
    errors = 0

    def _worker(idx: int):
        try:
            resp = http_post(completions_url, payload, timeout=timeout)
            choices = resp.get("choices", [])
            return bool(choices and choices[0].get("finish_reason") == "stop")
        except Exception as ex:
            logger.error("Concurrent worker %d failed: %s", idx, ex)
            return False

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_worker, i) for i in range(num_workers)]
        for f in as_completed(futures):
            if not f.result():
                errors += 1

    total_time = time.perf_counter() - start_all
    logger.info(
        "Concurrency test finished in %.3fs. Success: %d/%d (Errors: %d)",
        total_time,
        num_workers - errors,
        num_workers,
        errors,
    )
    return errors == 0


def main() -> int:
    args = parse_args()
    logger.info("Starting vLLM compatibility smoke test against %s", args.base_url)

    if not test_health_and_models(args.base_url, args.model_name, args.timeout):
        logger.error("FAIL: vLLM health or model registration check failed.")
        return 1

    text = test_completion_and_stop(args.base_url, args.model_name, args.timeout)
    if text is None:
        logger.error("FAIL: vLLM raw completion or stop token test failed.")
        return 1

    if not test_determinism(args.base_url, args.model_name, args.timeout, text):
        logger.error("FAIL: Determinism test failed.")
        return 1

    if not test_concurrency(args.base_url, args.model_name, args.timeout, num_workers=8):
        logger.error("FAIL: Concurrency test failed.")
        return 1

    logger.info("ALL vLLM SMOKE TESTS PASSED SUCCESSFULLY.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
