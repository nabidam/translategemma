# Inference Serving Implementation Review 2

Review scope: static code review only. No tests, scripts, Docker builds, model loads, or runtime checks were executed on this machine.

Reviewed fix commit: `40034ff fix(serving): address static review findings across gateway and export tooling`

Previous review: `docs/INFERENCE_SERVING_REVIEW.md`

Reference plan: `docs/INFERENCE_SERVING_PLAN.md`

## Findings

### 1. High: Production gateway cannot use canonical processor or exact tokenizer

Locations:

- `gateway/main.py:86-98`
- `gateway/config.py:15-17`
- `gateway/requirements.txt:1-6`
- `serving/docker-compose.yml:55-74`

Gateway now attempts `AutoProcessor.from_pretrained()` and `TokenEstimator` attempts `tokenizers.Tokenizer` / `transformers.AutoTokenizer`, but gateway requirements contain neither `transformers` nor `tokenizers`. Serving compose also does not mount merged model directory into gateway, even though default `TG_MODEL_DIR` is `/models/model`. In deployed composition, processor/tokenizer loading therefore fails and gateway silently uses manual prompt fallback plus heuristic counting. This leaves previous prompt-parity and exact-admission findings unresolved in production.

Required fix:

- Choose one supported production design and complete it:
  - install pinned tokenizer/Transformers dependencies in gateway and mount merged model/tokenizer directory read-only; or
  - package a lightweight, tested tokenizer/prompt renderer that does not require Transformers.
- Pass `TG_MODEL_DIR`/`TG_TOKENIZER_PATH` explicitly in compose.
- Fail startup/readiness when exact rendering or exact tokenization cannot initialize. Do not silently downgrade in production.
- Add a production-composition test that builds gateway image with declared requirements and loads artifacts from the actual mounted path.

### 2. High: Processor-backed canonical rendering uses wrong message content shape

Location: `gateway/main.py:52-58`

Gateway builds `user_message["content"]` as one preformatted string. Training and reference serving build content as a list containing a structured text item with `type`, `source_lang_code`, `target_lang_code`, and `text`. Passing the preformatted string through `apply_chat_template()` is not the same contract and can fail or double/mis-render language markers depending on template behavior.

Required fix:

- Build the exact structured user message used by `api/translator.py` and evaluation code.
- Call vendored canonical `render_training_prompt()` with that message.
- Add byte and token-ID parity fixtures comparing gateway output against root/reference rendering, not a hand-written expected prefix.
- Remove test coverage that validates only gateway's own fallback string against itself.

### 3. High: Direct vLLM wire payload still nests `stop_token_ids` under `extra_body`

Locations:

- `gateway/vllm_client.py:78-90`
- `gateway/vllm_client.py:167-179`
- `serving/vllm/smoke_test.py:135-143`
- `serving/vllm/COMPATIBILITY.md:24-29`

`extra_body` is commonly an OpenAI SDK argument used to merge extension fields into the HTTP body. Gateway and smoke script construct raw JSON requests themselves, but send `{"extra_body":{"stop_token_ids":[1,106]}}`. Compatibility documentation says raw completion request uses top-level `stop_token_ids`. Unless vLLM 0.13.0 explicitly defines nested `extra_body` in its HTTP schema, token-ID stopping is not being requested and the key may be rejected or ignored.

Required fix:

- Send vLLM extension fields in exact raw HTTP schema verified for pinned vLLM version, normally top-level `"stop_token_ids": [1, 106]`.
- Use one request-builder function shared by gateway protocol tests and smoke test, or a contract fixture that asserts exact JSON.
- Record real serving-host response evidence before claiming stop compatibility.

### 4. High: Export verifier rejects exports produced by merge script

Locations:

- `scripts/merge_lora_adapter.py:349-395`
- `scripts/verify_model_export.py:182-218`
- `tests/test_verify_model_export.py:97-111`

Merge script computes `SHA256SUMS` before writing `merge_manifest.json`; it also does not include `SHA256SUMS` in its own checksum entries. Verifier defines every disk file absent from `SHA256SUMS` as untracked, so both `merge_manifest.json` and `SHA256SUMS` make a real export fail. Added end-to-end test attempts to checksum `SHA256SUMS` inside itself, then rewrites it, making its recorded self-hash stale; this cannot produce a stable self-checksum.

Required fix:

- Define non-self-referential integrity format.
- Recommended: `SHA256SUMS` covers all payload files except itself; manifest contains payload inventory and is either covered by `SHA256SUMS` after manifest creation or separately signed/checksummed outside release directory.
- Verifier must explicitly exempt only defined metadata files and require exact set equality for all payload files.
- Rewrite end-to-end test to construct an artifact using the same export helper/order as production and verify that artifact.

### 5. High: Concurrency cancellation corrupts queue counters and capacity accounting

Locations:

- `gateway/limits.py:83-109`
- `gateway/routing.py:67-82`

`ConcurrencyManager.acquire()` increments `in_flight` in a `finally` block even when `semaphore.acquire()` is cancelled or raises. Structured batch failure cancels queued tasks, triggering this path. Cancelled tasks that never acquired semaphore become counted as in-flight and do not call `release()` because `_execute_single_translation()` never passed `await acquire()`. Queue and in-flight metrics drift, and capacity behavior becomes unreliable after failed batches.

Required fix:

- Increment `in_flight` only after successful semaphore acquisition.
- Decrement `queued` on every exit without pretending a slot was acquired.
- Track an acquired flag and release only acquired slots.
- Add cancellation tests where queued tasks are cancelled before semaphore acquisition and assert semaphore value, `queued`, and `in_flight` return to baseline.

### 6. High: SSE gateway still corrupts multiline and framing-sensitive output

Location: `gateway/main.py:425-433`

Gateway forwards raw generated text with `yield f"data: {text_chunk}\n\n"`. If chunk contains newline, subsequent lines are not prefixed with `data:` and blank lines terminate SSE event early. Generated text containing `event:`, `id:`, carriage returns, or multiple newlines can be reinterpreted as SSE framing rather than translation data. This violates output-preservation goal and can lose or alter translation text.

Required fix:

- JSON-encode each text chunk and send encoded object as one SSE data record, or prefix every physical line according to SSE specification.
- Define stable stream event schema containing `text`, `finish_reason`, and request ID.
- Add tests assembling multiline/whitespace-heavy chunks and proving final text exactly equals non-streaming output.

### 7. High: Request-body byte limit remains unenforced

Locations:

- `gateway/config.py:36`
- `gateway/main.py:276-438`

`max_request_body_bytes` exists but no middleware, content-length check, or bounded body reader uses it. FastAPI parses request JSON before endpoint validation, allowing oversized payloads to consume memory and CPU. Per-text and batch-item limits do not enforce total request bytes.

Required fix:

- Enforce body size before JSON parsing through trusted reverse proxy and/or ASGI middleware with bounded streaming reads.
- Treat missing/chunked `Content-Length` safely.
- Add aggregate batch character/token budget, not only per-item limits.
- Return 413 before creating translation tasks.

### 8. Medium: Batch fairness and atomic admission remain unresolved

Locations:

- `gateway/main.py:369-388`
- `gateway/routing.py:67-82`
- `gateway/limits.py:73-116`

Structured cancellation fixes one failure mode, but batch handler still immediately creates up to 128 tasks and each independently enters same queue as interactive requests. One bulk request can fill all in-flight and queued capacity. No atomic reservation, bulk limit, token budget, weighted fairness, or separate route capacity was added.

Required fix:

- Add aggregate batch token budget and bounded bulk concurrency.
- Reserve capacity or reject batch before creating all tasks.
- Separate or weight interactive and bulk admission so one bulk request cannot monopolize queue.
- Keep vLLM continuous batching, but control gateway fan-out fairly.

### 9. Medium: Health check can report ready for wrong loaded model

Location: `gateway/vllm_client.py:39-57`

`check_health()` returns true immediately when root `/health` returns 200, without checking `/models`. Fallback `/models` also returns true when any model exists because of `or len(models) > 0`. Gateway can therefore report ready while configured `vllm_model_name` is absent, causing requests to fail later.

Required fix:

- Require both engine health and exact configured model registration, or make `/models` authoritative after health succeeds.
- Remove `or len(models) > 0`.
- Add wrong-model readiness test.

### 10. Medium: vLLM/gateway context configuration can still diverge

Locations:

- `gateway/main.py:76-84`
- `gateway/config.py:13,40`
- `serving/docker-compose.yml:25-26,64-71`
- `gateway/.env.example:23-24`

Compose passes `TG_MAX_TOTAL_CONTEXT_TOKENS` from `MAX_MODEL_LEN` but does not pass `TG_VLLM_MAX_MODEL_LEN`. If operator sets `MAX_MODEL_LEN` above 4096, gateway clamps total context back to its default 4096. If settings disagree, startup mutates one setting and logs warning instead of failing configuration. Gateway `.env.example` remains stale at 8192 and uses old source-token limits.

Required fix:

- Pass same `MAX_MODEL_LEN` into both `TG_VLLM_MAX_MODEL_LEN` and `TG_MAX_TOTAL_CONTEXT_TOKENS`.
- Fail startup on mismatch rather than silently clamp.
- Update `.env.example` to current field names and values (`TG_SOURCE_ADAPTER_PATH`, tokenizer/model paths, 4096 context unless deliberately changed).
- Keep one authoritative environment variable where possible.

### 11. Medium: Exact-tokenizer fallback is silent readiness degradation

Locations:

- `gateway/main.py:86-98`
- `gateway/limits.py:20-53`

Even after exact loading is implemented, failures only log warning/debug and continue in heuristic mode. Plan required readiness failure when exact tokenizer is required. Heuristic context admission can undercount prompts and allows vLLM rejections at boundary.

Required fix:

- Add `require_exact_tokenizer=true` production default.
- Fail lifespan or readiness when estimator mode is not exact.
- Permit heuristic only in explicitly configured development mode with conservative limits.

### 12. Medium: Adapter compatibility check accepts unrelated repositories sharing basename

Location: `scripts/merge_lora_adapter.py:172-186`

Compatibility accepts any identifiers with matching trailing basename, and also broad `endswith` matches. For example, `organization-a/model` and `organization-b/model` pass. This is not reliable base identity verification and revisions/config hashes remain unchecked.

Required fix:

- Require exact normalized Hub repository ID equality for remote models.
- Require exact resolved path equality for local models unless explicit override.
- Verify architecture/config fingerprint and requested base revision where available.
- Keep override explicit and manifest-recorded.

### 13. Medium: Forced export still deletes previous release after successful promotion

Location: `scripts/merge_lora_adapter.py:397-415`

Backup protects only failed rename. On successful promotion, old release is deleted immediately. This does not satisfy immutable release and rollback requirement; an artifact that promotes successfully but later fails verification leaves no previous artifact at that path.

Required fix:

- Do not overwrite release directories.
- Export each release to unique immutable directory and atomically switch a `current` symlink/pointer only after verification.
- Retain at least one previous verified release for rollback.
- Remove `--force` for release directories, or scope it only to abandoned temporary output.

### 14. Medium: Checksum verifier computes unused manifest inventory and does not compare hashes

Location: `scripts/verify_model_export.py:189-193`

`manifest_inventory` is built but never used. Verifier does not compare manifest file set/hashes against `SHA256SUMS`, despite comments and review requirement claiming strict inventory coverage. Manifest can disagree with checksum file without detection.

Required fix:

- Require manifest payload set and hashes to equal checksum payload set and hashes exactly.
- Validate duplicate paths, duplicate checksum lines, malformed hashes, path normalization, and file sizes.
- Ensure manifest provenance itself has an integrity anchor.

### 15. Medium: Streaming errors after response starts are not mapped safely

Locations:

- `gateway/main.py:417-438`
- `gateway/vllm_client.py:185-228`

Once `StreamingResponse` begins, `HTTPException` from backend stream cannot change HTTP status. Event generator does not catch backend exceptions and emit a defined terminal SSE error. Clients may receive an abruptly closed 200 stream with no structured failure or finish reason.

Required fix:

- Catch backend/protocol exceptions inside event generator and emit documented `error` event before closing when possible.
- Include request ID and machine-readable error code.
- Record streaming errors and release concurrency slot.

### 16. Low: `/health-check` response shape is no longer strictly compatible

Locations:

- `gateway/main.py:235-247`
- `gateway/schemas.py:108-115`

Existing endpoint returned only `{"translator":"OK"|"FAIL"}`. New success response includes `ready` and `detail`. Additive fields often work, but this is a contract change for strict clients despite plan saying endpoint shape remains compatible.

Required fix:

- Keep `/health-check` legacy response body and use `/ready` for expanded readiness payload, while still returning non-2xx on legacy failure if required by health semantics.
- Document compatibility decision and test exact legacy response.

### 17. Low: vLLM image remains unpinned until manual host action

Locations:

- `serving/docker-compose.yml:3`
- `serving/vllm/docker-compose.yml:3`
- `serving/vllm/COMPATIBILITY.md:3-5,49-61`

Documentation now correctly says host verification is required, but active deployment still defaults to mutable tag. Prior finding is not closed until digest and evidence are recorded on target machine.

Required fix:

- Treat this as release blocker, not completed code fix.
- Require `VLLM_IMAGE` with digest for production profile and fail if placeholder/tag-only value is used.
- Store host verification evidence in designated release artifact location.

## Prior Findings Status

- Finding 1, canonical prompt: partially fixed in code; unresolved in production composition and structured message shape.
- Finding 2, output stripping: non-streaming fixed; streaming framing remains incorrect.
- Finding 3, system validation: fixed.
- Finding 4, liveness/readiness: mostly fixed; exact-model readiness remains broken.
- Finding 5, multi-worker limits: fixed by one worker.
- Findings 6-7, context/tokenizer: partially fixed; production lacks dependencies/mount and mismatch handling remains weak.
- Finding 8, batch fan-out/fairness: cancellation added; fairness and atomic admission unresolved.
- Finding 9, adapter/base compatibility: partially fixed; matching remains too permissive.
- Finding 10, immutable rollback: partially fixed for promotion failure; previous verified release still deleted.
- Findings 11-12, export integrity/token mapping: token mapping improved; checksum design is currently self-inconsistent and manifest comparison incomplete.
- Finding 13, missing `Any`: fixed.
- Finding 14, image evidence: status wording fixed; digest/evidence remains release blocker.
- Finding 15, health tool assumption: improved with Python fallback.
- Finding 16, smoke contract consistency: request builder unified, but wire shape remains questionable.
- Finding 17, provenance naming: improved; environment example remains stale and manifest provenance is not loaded.

## Required Fix Order

1. Make production gateway load exact processor/tokenizer and use exact structured message rendering.
2. Correct raw vLLM stop-token request schema and prove it against pinned image.
3. Repair export checksum/manifest format so produced artifacts pass verifier.
4. Fix concurrency cancellation accounting.
5. Encode SSE safely and handle stream failures.
6. Enforce body and aggregate batch limits; add fairness.
7. Require exact configured model for readiness.
8. Unify context-limit configuration and fail on mismatch.
9. Tighten adapter/base identity and immutable release promotion.
10. Complete manifest integrity checks, image digest pinning, and release evidence.

## Review Conclusion

Fix commit closes several visible issues, especially system validation, non-streaming output preservation, readiness status, and single-worker admission control. Production path is still not ready: compose cannot provide exact prompt/token behavior, stop-token JSON likely does not match raw vLLM schema, artifact verifier rejects its own exporter output, and batch cancellation corrupts concurrency accounting. Resolve high-severity items before any serving-host benchmark or cutover.
