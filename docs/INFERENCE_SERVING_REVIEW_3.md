# Inference Serving Implementation Review 3

Review scope: static code review only. No tests, scripts, Docker builds, model loads, or runtime checks were executed on this machine.

Reviewed fix commit: `5285b01 fix(serving): address second review findings across gateway and export tooling`

Previous review: `docs/INFERENCE_SERVING_REVIEW_2.md`

Reference plan: `docs/INFERENCE_SERVING_PLAN.md`

## Findings

### 1. High: Exact-tokenizer readiness can pass while canonical processor rendering failed

Locations:

- `gateway/main.py:118-136`
- `gateway/limits.py:20-53`

Startup requires only `TokenEstimator.mode == "exact"`. `TokenEstimator` can load `tokenizer.json` directly while `AutoProcessor.from_pretrained()` fails. Gateway then passes readiness but `CanonicalPromptRenderer` uses its hand-written fallback instead of canonical `apply_chat_template()` rendering. Exact admission counting therefore does not prove exact prompt rendering, and production can silently return to the prompt drift that this migration must prevent.

Required fix:

- Track processor/template readiness separately from tokenizer readiness.
- With production exact mode enabled, fail startup unless both canonical processor rendering and exact tokenizer counting initialize from the mounted release.
- Remove manual prompt fallback from production path; retain only in isolated unit-test fixtures if needed.
- Expose processor/template mode in `/ready` and `/model-info`.
- Add production-composition test proving mounted artifact renders byte/token-identical prompts through `render_training_prompt()`.

### 2. High: Streaming UTF-8 decoder can corrupt Persian text at network chunk boundaries

Location: `gateway/vllm_client.py:209-212`

Each arbitrary byte chunk is decoded independently with `errors="replace"`. HTTP chunks can split a multibyte UTF-8 Persian character. First half and second half are then replaced with Unicode replacement characters, permanently corrupting translation output.

Required fix:

- Use `httpx.Response.aiter_text()` or a stateful incremental UTF-8 decoder across byte chunks.
- Do not use replacement decoding for successful model output.
- Add a stream protocol test that splits every multibyte Persian character at each possible byte boundary and proves reconstructed text is exact.

### 3. High: Merge revision arguments are provenance labels only and do not select revisions

Locations:

- `scripts/merge_lora_adapter.py:225-267`
- `scripts/merge_lora_adapter.py:282-305`
- `scripts/merge_lora_adapter.py:366-380`

`--base-revision` and `--adapter-revision` are written to manifest, but are not passed to `AutoProcessor.from_pretrained()`, model config loading, `AutoModelForCausalLM.from_pretrained()`, `PeftConfig.from_pretrained()`, or `PeftModel.from_pretrained()`. Export can therefore merge whatever local/cache/default revisions load while manifest claims different immutable revisions.

Required fix:

- Pass base revision consistently to processor, config, generation config resolution, and base model loading.
- Pass adapter revision to PEFT config and adapter loading when adapter is a Hub ID.
- For local artifacts, compute and record artifact/config/weight hashes instead of accepting an unverifiable revision label.
- Record resolved Hub commit SHA returned by loaded artifacts, not only requested revision.
- Reject manifest revision claims that cannot be tied to loaded bytes.

### 4. High: Request body limit is bypassed when `Content-Length` is absent or invalid

Location: `gateway/main.py:49-68`

Middleware checks only declared `Content-Length`. Chunked requests, HTTP/2 bodies, omitted headers, and invalid values pass through unbounded; FastAPI then buffers/parses body. This does not satisfy prior requirement to treat missing/chunked length safely or enforce limit before parsing.

Required fix:

- Enforce limit while reading ASGI `receive` chunks and stop once cumulative bytes exceed cap, or enforce at a mandatory trusted reverse proxy plus defense-in-depth middleware.
- Reject invalid/negative `Content-Length` instead of ignoring it.
- Ensure compressed request bodies are limited after decompression if compression is accepted.
- Add missing-length and chunked-body tests.

### 5. High: Batch traffic can still fill shared queue and reject interactive requests

Locations:

- `gateway/main.py:441-461`
- `gateway/limits.py:73-131`
- `gateway/routing.py:67-82`

Bulk semaphore limits active bulk requests to 16, but handler still creates up to 128 tasks immediately. Remaining bulk tasks wait while incrementing the same global `queued` counter used by interactive requests. One maximum batch can consume all 128 queue positions, causing later interactive calls to receive 429. Bulk concurrency cap alone does not provide queue fairness or atomic admission.

Required fix:

- Use separate bounded bulk queue/accounting or reserve queue capacity for interactive traffic.
- Do not create one task per item before capacity exists; feed items through fixed worker pool or bounded producer.
- Apply aggregate batch token budget, not only aggregate characters.
- Define and test fairness SLA: saturated bulk load must leave configured interactive queue/in-flight capacity available.

### 6. Medium: Adapter with missing base identity is still accepted without override

Location: `scripts/merge_lora_adapter.py:172-193`

Exact comparison is improved, but if adapter config omits `base_model_name_or_path`, function only logs warning and continues with `base_match_verified=false`. This bypasses stated compatibility enforcement without requiring `--allow-base-mismatch`.

Required fix:

- Treat missing base identity as mismatch and fail unless explicit override is supplied.
- Verify model architecture/config fingerprint before merge even when names match.
- Record override reason, not only boolean, in manifest.

### 7. Medium: Manifest provenance remains outside checksum coverage

Locations:

- `scripts/merge_lora_adapter.py:344-384`
- `scripts/verify_model_export.py:189-218`

Payload checksum/inventory consistency is now coherent, but `merge_manifest.json` is explicitly excluded from checksum coverage. Accidental or malicious changes to release ID, base/adapter revisions, dtype, prompt contract, package versions, or command arguments are not detected as long as payload inventory still matches. Plan requires trustworthy provenance.

Required fix:

- Add external integrity anchor for manifest, such as detached `merge_manifest.sha256` stored outside manifest inventory, detached signature, or release metadata signed by transfer pipeline.
- Verify anchor before trusting provenance fields.
- Keep non-self-referential payload checksum design.

### 8. Medium: Stream can end silently without terminal event

Locations:

- `gateway/vllm_client.py:209-242`
- `gateway/main.py:490-535`

Malformed JSON lines are silently ignored. If backend sends `[DONE]` without a choice carrying `finish_reason`, closes connection, leaves partial buffer, or all terminal frames fail parsing, client generator returns normally. Gateway emits neither `done` nor `error`, leaving clients with an ambiguous successful HTTP 200 stream.

Required fix:

- Track whether terminal finish event was observed.
- Treat malformed protocol frames, incomplete trailing buffer, `[DONE]` without prior finish reason, and EOF without terminal reason as structured stream errors.
- Never silently swallow JSON/protocol errors; log request ID and emit machine-readable terminal error event.

### 9. Medium: Dependency ranges can load incompatible Transformers behavior

Location: `gateway/requirements.txt:7-8`

Gateway now depends on unbounded `transformers>=4.40.0` and loosely bounded tokenizer. Repository serving/evaluation contract uses Transformers `4.57.6` and intentionally excludes v5. Gateway rebuild can install v5 or an older v4 with different TranslateGemma processor/chat-template behavior, breaking exact prompt parity.

Required fix:

- Pin gateway model-side dependencies to validated versions, at minimum `transformers==4.57.6` and compatible exact `tokenizers` version used by artifact/evaluation environment.
- Record resolved gateway dependencies in image.
- Treat prompt parity as dependency upgrade gate.

### 10. Medium: Context equality check accepts mismatched lower gateway limit silently

Location: `gateway/main.py:110-116`

Code enters mismatch branch when values differ, but raises only when gateway limit exceeds vLLM limit. Lower gateway limit silently passes despite stated single authoritative configuration and comments saying strict alignment. This creates confusing rejection behavior and drift between `/model-info` and backend capacity.

Required fix:

- Require equality and fail startup on any mismatch, or remove duplicate setting and derive gateway limit directly from one value.

### 11. Medium: Release overwrite still mutates release directories instead of switching verified pointer

Location: `scripts/merge_lora_adapter.py:386-404`

Prior release is now retained, which improves rollback. However `--force` still renames active release before new artifact has passed `verify_model_export.py` and merged-model quality verification. New unverified export becomes active path immediately. Rollback requires manual discovery of timestamped backup and backup naming can collide for repeated operations in same second.

Required fix:

- Export to unique immutable release directory.
- Run integrity and quality gates before switching `current` symlink/pointer.
- Atomically switch pointer after verification; retain explicit last-known-good pointer.
- Avoid `--force` mutation of release directories.

### 12. Low: Active vLLM image remains mutable tag-only

Locations:

- `serving/docker-compose.yml:3`
- `serving/vllm/docker-compose.yml:3`

Release blocker remains: default production composition accepts mutable `vllm/vllm-openai:v0.13.0` tag. Documentation says pin digest later, but compose does not require it.

Required fix:

- Require digest-pinned `VLLM_IMAGE` in production deployment and keep tag-only value only in explicitly named development profile.
- Store serving-host compatibility evidence with release.

### 13. Low: Backend errors can expose internal response bodies to public clients

Location: `gateway/vllm_client.py:121-127`

Gateway includes complete vLLM response text in public 502 detail. Backend errors can reveal model paths, engine internals, configuration, or stack details.

Required fix:

- Log detailed backend response server-side with request ID.
- Return stable sanitized public error code/message.

## Prior Findings Status

- Production model mount/dependencies: fixed, but exact processor failure still silently falls back.
- Structured message shape: fixed.
- Top-level `stop_token_ids`: fixed in gateway client; target-host verification still required.
- Export checksum payload format: substantially fixed; manifest provenance remains unanchored.
- Cancellation accounting: fixed for queued cancellation.
- SSE outbound JSON framing: fixed; inbound UTF-8 and terminal protocol handling remain broken.
- Body limit: partially fixed; chunked/missing length bypass remains.
- Batch fairness: bulk concurrency cap added; shared queue starvation remains.
- Exact model readiness: fixed.
- Context environment alignment: compose fixed; startup equality enforcement remains partial.
- Exact tokenizer production requirement: fixed in compose; processor/template exactness not required.
- Adapter identity comparison: exact names fixed; missing identity and config/revision verification remain.
- Previous release retention: improved; verification-before-activation remains unresolved.
- Manifest inventory comparison: fixed for payload files.
- Legacy health response shape: fixed.
- Image digest pinning: unresolved release blocker.

## Required Fix Order

1. Require exact processor/template initialization, not only exact tokenizer.
2. Fix incremental UTF-8 stream decoding and terminal protocol validation.
3. Make revision arguments select and verify actual model/adapter bytes.
4. Enforce body size for chunked/missing-length requests.
5. Prevent bulk queue starvation and add aggregate token budget.
6. Fail missing adapter base identity and verify architecture/config fingerprint.
7. Anchor manifest provenance and switch only verified immutable releases.
8. Pin gateway dependency versions and vLLM image digest.
9. Sanitize backend errors and enforce one context setting.

## Review Conclusion

Third fix round closes many prior defects: production mounts model artifacts, exact tokenizer is required by compose, structured prompt message shape is correct, vLLM stop IDs are top-level, exact model readiness is checked, cancellation accounting is improved, and payload checksum verification is coherent. Remaining high-severity issues center on silent prompt fallback, streamed Persian corruption, unverifiable revision provenance, body-limit bypass, and bulk queue starvation. Resolve these before serving-host validation or traffic cutover.
