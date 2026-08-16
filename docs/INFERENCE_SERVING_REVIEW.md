# Inference Serving Implementation Review

Review scope: static code review only. No tests, scripts, Docker builds, or runtime checks were executed on this machine.

Reviewed commit: `59259d9 feat(serving): implement vllm serving stack and lora merge export`

Reference plan: `docs/INFERENCE_SERVING_PLAN.md`

## Findings

### 1. High: Gateway bypasses canonical prompt contract

Location: `gateway/main.py:40-47`

`render_exact_training_prompt()` manually concatenates the prompt instead of calling the canonical `render_training_prompt()` / `render_training_prompts()` implementation. This violates the plan's single-source-of-truth requirement and makes the gateway's prompt behavior depend on a guessed string layout. The existing repository specifically introduced marker-based chat-template rendering because the template is not safely reducible to hand-written text. Any future template, processor, language-content, or special-token change can silently desynchronize gateway inference from training/evaluation.

Required fix:

- Use the canonical prompt helper through a shared package or explicitly synchronized gateway vendoring.
- Build the same structured user message used by training, including the expected `content` shape.
- Add a parity fixture that compares gateway prompt bytes and token IDs with the root/reference rendering for every supported language pair.
- Do not retain a second manual renderer.

### 2. High: Gateway strips model output and hides stop failures

Locations:

- `gateway/main.py:164`
- `gateway/main.py:343-346`
- `gateway/vllm_client.py:114-120`

Non-streaming responses return `completion_text.strip()`. The streaming path strips each SSE line before forwarding it. The plan explicitly requires preserving output for stop diagnostics and forbids hiding runaway decoding through whitespace cleanup. A backend that ignores token 106 can therefore appear healthy after the gateway removes trailing whitespace. Streaming line stripping also modifies content and can corrupt meaningful whitespace/newline chunks.

Required fix:

- Return completion text without `strip()`.
- Preserve SSE payload content exactly; parse framing separately from payload data.
- Make finish reason `length` a controlled error or explicit response status, according to the documented API policy. Do not log it and return a normal translation.
- Add tests for trailing whitespace, turn restart, `stop`, and `length` responses.

### 3. High: Public `system` option is accepted but ignored

Locations:

- `gateway/main.py:216-217`
- `gateway/main.py:245-246`
- `gateway/main.py:284-285`
- `gateway/schemas.py:13-16`

Gateway accepts arbitrary `system` values and returns the requested value, but always sends the one merged model to vLLM. A request with `{"system":"base"}` or any arbitrary string receives adapter output labeled as that system. The old API rejected systems that were not loaded, and the serving plan requires truthful model identity.

Required fix:

- Remove `system` from public requests for merged-only serving, or validate it against exactly one configured value, normally `adapter`.
- Return HTTP 400 for unavailable or unsupported systems.
- Never echo an unvalidated client value in the response.
- Keep `/model-info.loaded_systems` and response `system` consistent with actual loaded model behavior.

### 4. High: Gateway health check reports HTTP success while backend is down

Locations:

- `gateway/main.py:169-175`
- `gateway/Dockerfile:20-21`
- `serving/docker-compose.yml:61-63`

`/health-check` always returns HTTP 200 with body `{"translator":"FAIL"}` when vLLM is unavailable. The Docker `HEALTHCHECK` only checks whether `urlopen()` returns successfully, so it marks the gateway healthy even when the inference backend is down. The compose readiness chain can therefore report a broken gateway as healthy after startup or backend failure.

Required fix:

- Separate liveness from readiness.
- Return a non-2xx status for readiness failure, or make the container health command assert the response body is `OK`.
- Add `/live` for process health and `/ready` for backend/model readiness.
- Make external traffic depend on readiness, not merely process startup.
- Ensure backend failure after startup changes readiness state.

### 5. High: Multi-worker gateway multiplies concurrency limits and state

Locations:

- `gateway/Dockerfile:23`
- `gateway/limits.py:43-47`
- `gateway/metrics.py:81-88`

The gateway starts with two Uvicorn workers. Each worker creates its own `ConcurrencyManager`, allowing up to 64 in-flight requests and 128 queued requests. Effective limits become approximately two independent sets, not the configured global limits. Metrics are also process-local, and each worker has its own backend client and readiness lifecycle. This can overload vLLM and makes queue metrics and 429 behavior misleading.

Required fix:

- Run one gateway worker per container when using in-process admission control, then scale gateway replicas only with an intentional shared/global limiter strategy; or
- Move admission control to a shared external mechanism and explicitly aggregate metrics.
- Do not claim `max_concurrent_requests` and `max_queue_depth` are global while they are process-local.
- Keep backend connection and request limits aligned with the actual number of workers/replicas.

### 6. High: Admission context limit is inconsistent with vLLM model limit

Locations:

- `gateway/config.py:35`
- `gateway/limits.py:120-127`
- `serving/docker-compose.yml:25-26`

Gateway permits `max_total_context_tokens=8192`, and validates only an estimate of source tokens plus output budget. vLLM is configured with `--max-model-len 4096`. Valid gateway requests can therefore reach vLLM with a context budget that the engine rejects. The estimate also omits the actual rendered prompt length and uses a fixed `+32` overhead.

Required fix:

- Define one authoritative context limit and inject it into both gateway and vLLM configuration.
- Estimate tokens using the exact merged-model tokenizer, including the fully rendered prompt, not a character heuristic.
- Validate `prompt_tokens + max_new_tokens <= max_model_len`.
- Reject before backend dispatch with a clear 413/422 response.
- Add a startup/config consistency check that fails if gateway limits exceed vLLM limits.

### 7. Medium: Gateway does not load the exact tokenizer despite claiming token-based admission

Locations:

- `gateway/main.py:56`
- `gateway/limits.py:17-34`
- `gateway/config.py:7-57`

`TokenEstimator` is constructed without `tokenizer_path`, so it always uses the heuristic estimator in normal deployment. The plan requires estimates from the same tokenizer artifact where practical. The heuristic can undercount Persian, special prompt markers, and scientific text, allowing requests beyond context limits and producing bad workload classification.

Required fix:

- Add a configured local tokenizer path/release directory and load it at startup in offline mode.
- Fail readiness if exact tokenizer loading is required but unavailable; use heuristic mode only as an explicit, bounded fallback.
- Count the actual rendered prompt, not only raw source text.
- Expose estimator mode and tokenizer release in `/model-info` and metrics.

### 8. Medium: Batch endpoint can create unbounded task fan-out relative to backend policy

Locations:

- `gateway/main.py:293-308`
- `gateway/limits.py:132-148`

The batch limit is 128, and the handler creates one asyncio task per input immediately. Each task then waits on the per-worker semaphore. A single batch can occupy the entire queue and starve unrelated interactive requests. If one task fails, `asyncio.gather()` raises while other tasks can continue consuming backend capacity, and the request has no structured cancellation/drain policy.

Required fix:

- Classify and schedule batch work through an explicit bulk queue with a per-request token/item budget.
- Reserve capacity atomically for a batch or reject it before creating tasks if it cannot fit.
- Define failure semantics: cancel remaining work on failure, await cancellation, release every slot, and return a controlled error.
- Add separate interactive and bulk capacity or weighted fairness so bulk cannot monopolize the gateway.

### 9. Medium: Merge script does not enforce adapter/base compatibility

Location: `scripts/merge_lora_adapter.py:142-160`

`validate_adapter_compatibility()` reads `base_model_name_or_path` and logs it, but never compares it with the requested base model. An adapter trained against a different model/revision can be merged into the requested checkpoint if PEFT accepts the module names, producing an invalid or silently incorrect release.

Required fix:

- Normalize local paths and repository IDs before comparison.
- Compare configured base identity and, where available, revision/config/model architecture hashes against the requested base.
- Require an explicit override flag for intentional mismatch.
- Record the verified comparison and override state in the manifest.

### 10. Medium: Merge export is not safely atomic when replacing an existing release

Location: `scripts/merge_lora_adapter.py:353-356`

With `--force`, the script deletes the existing output directory before renaming the temporary export. If rename fails, the last-known-good artifact is gone. This conflicts with immutable release and rollback requirements.

Required fix:

- Never mutate an existing release directory.
- Require a new release ID/output directory for each export.
- If replacement is explicitly needed, rename old output to a retained backup and promote new output first, or use an atomic symlink/current-pointer switch.
- Ensure rollback can select the previous artifact without rebuilding or restoring deleted files.

### 11. Medium: Export integrity manifest does not cover itself and verifier accepts incomplete checksum coverage

Locations:

- `scripts/merge_lora_adapter.py:306-351`
- `scripts/verify_model_export.py:114-150`

`SHA256SUMS` is generated before `merge_manifest.json`, so the manifest is not covered by the checksum file. The verifier checks every entry present in `SHA256SUMS`, but does not require every artifact and manifest-inventory entry to be listed and verified. An artifact can therefore contain changed/unlisted files without failing integrity verification.

Required fix:

- Define a checksum scheme that covers all release files, including manifest, without a self-referential checksum problem. One option is a manifest containing checksums for payload files and a separately signed/checksummed manifest; another is checksum all payloads in `SHA256SUMS` and verify manifest inventory matches the payload set exactly.
- Reject absolute paths and `..` traversal in checksum entries.
- Require exact set equality between inventory, checksum entries, and allowed release files.
- Verify tokenizer special-token mapping, not only generation config IDs.

### 12. Medium: Export verifier does not verify that token ID 106 is actually `<end_of_turn>`

Locations:

- `scripts/verify_model_export.py:87-111`
- `scripts/verify_model_export.py:31-41`

The verifier hardcodes that EOS contains IDs 1 and 106, but does not load or inspect tokenizer files to confirm ID 106 maps to `<end_of_turn>` and ID 1 maps to the expected EOS token. A corrupted or incompatible tokenizer could pass the metadata check while changing token semantics.

Required fix:

- Add a tokenizer-aware verification path on the serving host, or generate a signed token mapping during merge and verify it against tokenizer files.
- Fail unless the resolved token strings match the manifest and expected contract.

### 13. High: Export verifier cannot be imported because `Any` is undefined

Location: `scripts/verify_model_export.py:153`

`verify_manifest()` annotates its return type as `Tuple[bool, Dict[str, Any]]`, but `Any` is not imported from `typing`. Because this module does not enable postponed annotation evaluation, importing the verifier evaluates that annotation and raises `NameError` before `main()` can run. The serving artifact verification command therefore fails at startup.

Required fix:

- Import `Any` from `typing`, or enable postponed annotations consistently.
- Keep an import-level test for every offline verifier script.

### 14. Medium: vLLM compatibility document claims verification without pinned image digest or evidence

Locations:

- `serving/vllm/COMPATIBILITY.md:3-4`
- `serving/docker-compose.yml:3`
- `serving/vllm/docker-compose.yml:3`

The image is controlled by a mutable tag and the compatibility document says `Status: Verified Reference`, but no digest, runtime evidence, or versioned smoke-test output is stored. The plan required pinning the image digest after compatibility validation. A later tag mutation or different host pull can change protocol and model behavior.

Required fix:

- Pin an immutable image digest in deployment configuration.
- Record the tag, digest, GPU, driver, CUDA/runtime, and smoke-test result artifact.
- Make deployment fail or warn loudly when configured image digest differs from the recorded compatibility manifest.
- Do not label compatibility verified based only on source documentation.

### 15. Medium: vLLM health check depends on `curl` without declaring it available

Location: `serving/docker-compose.yml:46-50` and `serving/vllm/docker-compose.yml:46-50`

Both vLLM healthchecks invoke `curl`, but the deployment relies on the contents of the external `vllm/vllm-openai:v0.13.0` image. If that image does not include `curl`, the service remains unhealthy regardless of model state, and gateway startup is blocked.

Required fix:

- Confirm the pinned image includes `curl`, or use a healthcheck mechanism guaranteed by the image, such as Python if present.
- Store the healthcheck assumption in the compatibility evidence.

### 16. Low: Smoke concurrency test does not send the same stop contract as gateway

Location: `serving/vllm/smoke_test.py:179-188`

The primary completion test sends `extra_body.stop_token_ids`, but the concurrency test sends only the string stop. It therefore does not validate the same request contract used by the gateway and can pass while token-ID stopping is broken or unsupported.

Required fix:

- Use one canonical request builder in the smoke test for all completion modes.
- Include the same model, stop strings, stop IDs, and deterministic parameters in serial, repeated, and concurrent tests.

### 17. Low: Model provenance reports stale adapter path for merged model

Locations:

- `gateway/config.py:14-18`
- `gateway/main.py:183-196`

`/model-info` reports `adapter_path="checkpoints/sft-translategemma-12b-it"` while the running backend serves a merged checkpoint. The plan explicitly requires merged releases to report provenance without claiming an active adapter. This can mislead operators about what answered a request and makes release auditing ambiguous.

Required fix:

- Report merged release ID/path and manifest provenance.
- Rename adapter field to `source_adapter` or make it clearly archival provenance.
- Include base revision, adapter revision, merge manifest hash, image digest, and prompt contract hash.

## Missing Coverage

These are acceptance requirements with no meaningful implementation evidence in the reviewed change:

- No gateway test proves prompt token-ID parity against root/reference rendering.
- No integration test proves vLLM accepts the exact `extra_body.stop_token_ids` wire shape for the pinned image.
- No test proves finish reason `length` is rejected or surfaced safely by the gateway.
- No test covers gateway readiness failure through the Docker healthcheck.
- No test covers two gateway workers and global concurrency behavior.
- No test covers batch failure cancellation and slot release.
- No test covers exact tokenizer-based context accounting.
- No test covers adapter/base mismatch rejection.
- No test covers atomic replacement failure and rollback preservation.
- No test proves checksum inventory completeness or path traversal rejection.
- No evidence artifact is linked from `COMPATIBILITY.md` despite its verified status.

## Required Fix Order

1. Replace manual prompt rendering with canonical rendering and add token parity coverage.
2. Remove output stripping and make stop/length behavior explicit.
3. Fix system validation so responses cannot mislabel merged adapter output.
4. Fix readiness HTTP status and healthcheck semantics.
5. Resolve multi-worker/global admission control design.
6. Align gateway context limits with vLLM and use exact rendered-prompt tokenization.
7. Fix batch queue/failure cancellation behavior.
8. Enforce adapter/base compatibility during merge.
9. Make release promotion rollback-safe and integrity verification complete.
10. Pin vLLM digest and attach actual compatibility evidence.
11. Correct tokenizer stop-token verification and provenance reporting.
12. Close remaining smoke and unit/integration coverage gaps.

## Review Conclusion

Implementation has substantial serving surface, but it is not ready for production or parity sign-off. Highest risk is correctness drift: gateway hand-renders prompts, strips output, and accepts arbitrary system labels. Next highest risk is operational: health and concurrency limits are not truthful with two workers, and context admission can exceed vLLM's configured model length. Resolve those before tuning throughput or retiring the current Transformers reference service.
