# TranslateGemma Adapter Inference Serving Plan

Status: draft
Owner: implementation agent
Scope: merge LoRA adapter on fine-tune machine, serve merged checkpoint through vLLM, and place a FastAPI gateway in front of vLLM on serving machine.

## 1. Objective

Replace the current single-process Transformers/PEFT API with a two-service serving stack:

```text
client
  |
  v
FastAPI gateway
  |  exact prompt rendering, request validation, routing, response compatibility
  v
vLLM OpenAI-compatible server
  |  continuous batching, paged KV cache, GPU scheduling, generation
  v
merged TranslateGemma checkpoint
```

The LoRA adapter is merged into base weights on the fine-tune machine. Only the resulting immutable model artifact, tokenizer files, manifest, and serving configuration move to the serving machine.

The system must preserve the behavior that made current adapter evaluation correct:

- Adapter prompt must use exact SFT rendering, not generic `add_generation_prompt=True` rendering.
- Stop contract must include `<eos>` id `1` and `<end_of_turn>` id `106`, subject to runtime verification from tokenizer/model artifacts.
- Generation must be deterministic greedy decoding by default.
- Gateway output must preserve current `/translate` and `/translate/batch` response shapes.
- Trailing output must not be silently stripped as a workaround for bad stopping. Stop behavior must be fixed at generation boundary and tested.

The old Transformers API remains the correctness oracle during migration. It is not removed until parity and load gates pass.

## 2. Important Corrections To Proposed Approach

### 2.1 Merge before serving

Serving one adapter does not require runtime LoRA support. Use `PeftModel.merge_and_unload()` on the fine-tune machine and save a complete merged checkpoint.

Benefits:

- Removes PEFT runtime wrapper and adapter dispatch.
- Avoids runtime adapter selection/toggling.
- Simplifies vLLM deployment and model identity.
- Enables future FP8/INT8/TensorRT conversion against one immutable model.
- Avoids shipping training dependencies or adapter paths to serving host.

Merge does not reduce base model parameter count. BF16 merged weights remain approximately the same size as base BF16 weights.

### 2.2 Gateway must own prompt rendering

Do not send ordinary OpenAI `messages` to vLLM and assume vLLM will reproduce the adapter's training prompt. The current adapter was trained with a prompt prefix that differs from `add_generation_prompt=True` by assistant-turn whitespace.

Gateway must render the exact raw prompt using the shared prompting contract, tokenize or otherwise inspect it as needed, and call vLLM's raw completion path. The gateway may use chat messages only if an automated parity test proves byte/token-equivalent rendering for the exact pinned vLLM version and model template.

### 2.3 vLLM is the batch scheduler

The gateway must not recreate the current global GPU lock or implement synchronous request-local batching as its main performance mechanism. vLLM should receive individual or appropriately grouped requests and perform continuous batching.

Gateway workload routing is for admission control, length classes, priority, and backend selection. It is not a replacement for vLLM's scheduler.

### 2.4 vLLM 0.13.0 support must be verified

The chosen image is `vllm/vllm-openai:v0.13.0`. Do not assume TranslateGemma architecture, tokenizer behavior, raw completion requests, stop-token IDs, or merged checkpoint loading work without validation. First task must be a compatibility smoke test. If the image cannot serve the architecture or exact stop contract, record the failure and evaluate a compatible vLLM image or another engine before implementation proceeds.

## 3. Machine Boundaries

### 3.1 Fine-tune machine

Contains the parent repository, base checkpoint, adapter, training environment, and merge script.

Responsibilities:

- Validate adapter/base compatibility.
- Merge adapter into base in BF16 by default.
- Preserve tokenizer and generation metadata.
- Run merged-model correctness and quality checks.
- Produce a versioned export directory and manifest.
- Transfer export to serving machine through the existing artifact process.

Must not be required by serving machine:

- Parent repository.
- Training data.
- `trl`, `datasets`, evaluation models, or optimizer state.
- LoRA adapter directory after merge, except retained archival copy.

### 3.2 Serving machine

Contains only deployment repository/artifacts:

- Merged model export mounted read-only.
- vLLM container using `vllm/vllm-openai:v0.13.0` or approved compatible version.
- Gateway source, image, configuration, and tests.
- Optional benchmark/load-test tooling.

Serving machine must not download from Hugging Face at runtime in offline deployment. Model and tokenizer files must be staged before startup.

## 4. Target Contracts

### 4.1 External gateway API

Keep current endpoint contracts unless a concrete compatibility need requires an additive change:

```text
GET  /health-check
GET  /model-info
POST /translate
POST /translate/batch
```

Current response fields remain:

- `/translate`: `translation`, `system`, `source_lang`, `target_lang`
- `/translate/batch`: `translations`, `system`, `source_lang`, `target_lang`

Gateway may add internal timing/request metadata to logs and metrics, but must not expose vLLM response details as the primary public schema.

### 4.2 Internal gateway-to-vLLM contract

Preferred protocol: vLLM OpenAI-compatible raw completion endpoint with a fully rendered prompt.

Request requirements:

- `model`: configured vLLM model name, not arbitrary client-supplied model.
- `prompt`: exact rendered TranslateGemma prompt.
- `max_tokens`: resolved output budget.
- `temperature: 0` or equivalent deterministic greedy setting.
- `top_p: 1` where accepted.
- `n: 1`.
- `stream`: configurable; default false until streaming parity is tested.
- Stop configuration that demonstrably stops on `<end_of_turn>`.

The implementation must determine the supported vLLM mechanism for token stops. Candidate mechanisms are:

1. Model `generation_config.json` containing both stop IDs.
2. Request `stop_token_ids` through vLLM's supported `extra_body` field.
3. Request `stop` with the literal `<end_of_turn>` string, if vLLM recognizes it before special-token decoding.
4. A gateway-side stop-token validator that rejects/truncates only when runtime metadata proves the boundary was emitted, never blind whitespace trimming.

Select one mechanism only after an integration test proves it stops at token 106 and returns the same visible text as the reference server.

### 4.3 Model-info contract

Gateway `/model-info` should report:

- merged model path or release ID
- base model revision
- adapter revision used to build it
- merge timestamp/export version
- dtype
- vLLM image/version
- configured model name
- source and target defaults
- max output policy
- stop token IDs and token strings
- prompt contract version/hash
- backend URL/name
- routing policy version

Never report a merged model as an active LoRA adapter. Preserve provenance explicitly.

## 5. Prompt and Stop Correctness Contract

### 5.1 Single source of truth

Root `prompting.py` remains canonical. The serving gateway must either:

- import a packaged copy of the canonical module in its own deployment unit, or
- contain a vendored copy generated by an explicit sync script and checked for drift.

The gateway must not implement a second hand-written chat-template path.

The current `api/prompting.py` and `api/model_loading.py` are deployment copies for the old local model server. The new gateway needs the prompt/stop subset, but it must retain the same implementation and tests. Decide whether to move the shared code into a small source-machine package or continue explicit vendoring; do not allow silent divergence.

### 5.2 Merged model prompt mode

Merged adapter is still adapter behavior. It must use `adapter_use_training_rendering=True` semantics even though it is now a complete model checkpoint.

Do not set `base_use_training_rendering` merely because the merged directory is called a model. Naming does not change conditioning distribution.

### 5.3 Stop token verification

At merge/export time and gateway startup:

- Load tokenizer from merged export.
- Resolve `<end_of_turn>` token ID.
- Confirm expected ID is 106 for this artifact, or fail with an explicit artifact mismatch.
- Resolve tokenizer/model generation EOS set.
- Ensure expected stop set contains 1 and 106.
- Record exact set in manifest and `/model-info`.

At runtime smoke test:

- Submit a short known translation.
- Confirm output ends at first generated turn boundary.
- Confirm no newline flood, assistant-turn restart, or max-token truncation.
- Confirm visible translation matches reference output within the accepted merge tolerance.

Do not use `rstrip()`, regex cleanup, repetition penalties, or `no_repeat_ngram_size` to hide a stop failure.

## 6. Workload Routing Design

### 6.1 Request classes

Gateway classifies requests into at least:

- `interactive`: one or few short texts, latency-sensitive.
- `document`: one long text or sentence-split document.
- `bulk`: many independent segments, throughput-sensitive.

Classification must be deterministic and based on request metadata:

- number of texts
- source character count
- estimated source token count
- requested output budget
- explicit client priority, if later supported

Never trust client-provided class without applying server limits.

### 6.2 Admission controls

Gateway must enforce:

- maximum request body bytes
- maximum number of texts
- maximum source tokens per request
- maximum estimated prompt plus output tokens
- maximum `max_new_tokens`
- bounded in-flight request count
- bounded waiting queue
- request timeout and cancellation behavior

Return explicit 413/422/429/504 errors as appropriate. Do not allow one page request to consume unlimited GPU scheduler capacity.

### 6.3 Routing options

Implement initial routing as one vLLM backend with request metadata and per-class limits. Keep backend selection abstract enough to support later deployment of:

- one shared vLLM instance
- separate interactive and bulk vLLM instances on separate GPUs
- separate model replicas behind a load balancer

Do not deploy separate vLLM instances on one GPU unless memory and contention are measured. Each instance loads a full 12B model.

### 6.4 Long-document handling

Preserve sentence-aware splitting as an explicit gateway option. Improve it with bounded chunks where required:

- preserve sentence boundaries
- avoid tiny one-sentence calls when batching is possible
- group similar estimated lengths
- keep ordering and reassembly exact
- isolate tables, formulas, headings, and reference blocks where splitting could damage meaning

For very long documents, prefer an asynchronous bulk endpoint/job mode in a later task rather than keeping one synchronous HTTP connection open for many minutes. Initial implementation may keep existing synchronous endpoints, but must enforce timeout and output limits.

## 7. Batching Strategy

### 7.1 vLLM batching

Use vLLM continuous batching, paged KV cache, and configured maximum model length/output limits. Tune:

- `--max-num-seqs`
- `--max-num-batched-tokens`
- GPU memory utilization
- maximum model length
- chunked prefill settings if supported
- prefix caching if supported and beneficial

Exact flags must be checked against vLLM 0.13.0 help output, not copied from another version.

### 7.2 Gateway batching

Do not hold requests for arbitrary time. If gateway micro-batching is needed for a legacy-compatible batch endpoint, use a short bounded window and token budget:

- flush on max wait, max items, or max estimated tokens
- never mix incompatible source/target language or generation policy
- preserve response order
- reject a batch exceeding model context limits

Prefer forwarding individual requests to vLLM for interactive traffic so vLLM can schedule continuously. The existing `/translate/batch` endpoint can fan out inputs concurrently or submit a controlled group, but must not serialize each text.

### 7.3 Length bucketing

For client bulk workloads, bucket by estimated prompt/output length. This reduces scheduler padding and slowest-sequence effects. Use token estimates from the same tokenizer artifact as the merged model. Character count alone is insufficient for Arabic-script and scientific text.

## 8. Files And Atomic Tasks

Tasks are ordered. Each task has one implementation result and one verification gate. Do not combine tasks merely because they touch the same deployment area.

### Task 1: Freeze baseline and artifact contract

Purpose: establish current behavior and exact comparison data before changing serving architecture.

Files to add/change:

- Add `docs/INFERENCE_SERVING_BASELINE.md`.
- Add benchmark output under an ignored or explicitly designated benchmark directory.
- Review, but do not casually rewrite, `docs/2026-08-10_adapter_degeneration_analysis.md` and `api/README.md`.

Solution:

- Record base model revision, adapter revision/path, tokenizer revision, dtype, GPU, driver, CUDA, Torch, Transformers, PEFT, prompt contract hash, stop IDs, and decoding settings.
- Run current API engine and HTTP benchmarks.
- Run concurrency load against current API and record queue behavior, p50/p95/p99 latency, output tokens/s, and truncation/degeneration rates.
- Freeze a small parity corpus and full quality corpus.

Verification:

- Baseline artifact is reproducible.
- At least one short, medium, long, batch, and concurrent workload is recorded.

### Task 2: Add merge/export script on fine-tune machine

Purpose: create complete merged model release from base plus LoRA adapter.

Files to add/change:

- Add `scripts/merge_lora_adapter.py` at repository root.
- Add focused tests under `tests/`, for argument validation and manifest logic.
- Update root `README.md` with source-machine usage.
- Update `docs/OFFLINE_DEPLOYMENT.md` or add a merge section to this plan's eventual operational guide.

Solution:

- Accept explicit `--base-model`, `--adapter`, `--output-dir`.
- Accept optional immutable base/adapter revisions and release ID.
- Load base in BF16 for normal merge; do not load 4-bit for the canonical merged export unless an explicitly tested dequantize/merge path is needed.
- Load `PeftModel` with adapter attached.
- Verify adapter base model identity and architecture before merge.
- Call `merge_and_unload()`.
- Set evaluation mode and save with safe serialization.
- Copy/save tokenizer and processor files from the base model.
- Preserve or explicitly write `generation_config.json` with deterministic settings and stop IDs.
- Resolve and validate `<end_of_turn>` and expected EOS IDs before writing export.
- Write `merge_manifest.json` containing source paths/revisions, hashes where practical, package versions, dtype, stop IDs/tokens, prompt contract version/hash, command arguments, timestamp, and output file inventory.
- Refuse to overwrite an existing output directory unless `--force` is explicit.
- Write to a temporary directory and atomically rename on success so interrupted exports cannot look complete.

Verification:

- Export loads with `AutoProcessor` and `AutoModelForCausalLM`.
- No adapter-only files are required for loading.
- Manifest is complete and identifies exact source adapter.
- Merged model fits target serving GPU memory in BF16.

### Task 3: Add merge correctness and quality gate

Purpose: prove merge did not change translation behavior materially.

Files to add/change:

- Add `tests/test_merged_checkpoint.py` or a root `scripts/verify_merged_checkpoint.py`.
- Update evaluation documentation with merged-model commands.

Solution:

- Load original base + adapter and merged model under identical dtype/device.
- Use canonical prompt rendering and stop set for both.
- Run the frozen parity corpus with greedy decoding.
- Compare visible translations, generated token lengths, stop behavior, and truncation flags.
- Run full quality metrics and degeneration classifier.
- Define explicit tolerance for numerical differences. Exact string equality is preferred for deterministic greedy output, but the acceptance policy must allow documented BF16 merge differences if quality remains within threshold.

Verification gate:

- No missing stop tokens.
- No new whitespace floods, turn restarts, or max-token truncations.
- No material regression on COMET/MetricX/chrF++ or domain slices.
- Human-review sample covers numbers, units, formulas, acronyms, names, and mixed scripts.

### Task 4: Create serving artifact staging contract

Purpose: transfer only required files from fine-tune machine to serving machine.

Files to add/change:

- Add `docs/INFERENCE_ARTIFACT_TRANSFER.md`.
- Add optional `scripts/verify_model_export.py`.
- Update `.gitignore`/deployment ignore rules only if required; never commit weights or secrets.

Solution:

- Define export directory layout:

```text
exports/<release-id>/
  config.json
  generation_config.json
  model-*.safetensors
  tokenizer.json / tokenizer.model / tokenizer_config.json
  special_tokens_map.json
  preprocessor_config.json (if required)
  merge_manifest.json
  SHA256SUMS
```

- Generate checksums.
- Verify checksums after transfer.
- Verify required files before container startup.
- Mount export read-only in serving stack.
- Keep release directories immutable and switch releases by configuration/symlink, never mutate active files in place.

Verification:

- Offline load succeeds on serving host.
- Checksum and manifest validation fail closed.

### Task 5: Validate vLLM image and TranslateGemma compatibility

Purpose: prove the chosen runtime supports this exact merged architecture and protocol.

Files to add/change:

- Add `serving/vllm/COMPATIBILITY.md` or `docs/VLLM_COMPATIBILITY.md`.
- Add `serving/vllm/smoke_test.py` or shell test script.
- Add pinned image digest alongside tag `vllm/vllm-openai:v0.13.0` after pulling it.

Solution:

- Run `--help` and record supported flags.
- Start vLLM against a test merged export.
- Verify model loads without runtime Hub access.
- Verify raw completion endpoint.
- Verify tokenizer and prompt behavior.
- Verify stop behavior for token 106.
- Verify deterministic decoding across repeated calls.
- Verify batch/concurrent requests.
- Verify OpenAI response fields needed by gateway.
- Check whether vLLM supports model's multimodal/processor requirements even though this service uses text content.

Decision:

- If all gates pass, pin image tag plus digest.
- If not, stop implementation at this task and document exact failure. Evaluate a compatible vLLM release or SGLang rather than weakening prompt/stop correctness.

### Task 6: Add gateway deployment unit

Purpose: create FastAPI proxy independent from the old local-model API.

Files to add/change:

- Add `gateway/` or rename the current deployment directory only after migration decision.
- Add `gateway/main.py`.
- Add `gateway/config.py`.
- Add `gateway/schemas.py`.
- Add `gateway/prompting.py` as synchronized canonical copy or shared package.
- Add `gateway/vllm_client.py`.
- Add `gateway/requirements.txt`.
- Add `gateway/Dockerfile`.
- Add `gateway/.env.example`.

Solution:

- Preserve `/translate` and `/translate/batch` schemas.
- Validate and normalize text without changing meaningful content beyond current contract.
- Render exact adapter training prompt through canonical helper.
- Resolve language defaults and validate supported pair policy.
- Resolve bounded output budget.
- Send raw prompt to vLLM through a persistent async HTTP client with connection pooling.
- Configure connect/read/write/pool timeouts separately.
- Propagate request IDs.
- Translate vLLM errors into stable gateway errors.
- Do not expose unrestricted vLLM model selection or arbitrary generation parameters.
- Add graceful startup readiness check against vLLM `/health` and model endpoint.
- Add liveness endpoint that does not perform inference.
- Add deep translation probe as a separate optional endpoint/job.

Verification:

- Gateway can run without Torch, Transformers model weights, PEFT, or CUDA.
- Gateway starts before vLLM but remains unready until backend is ready.
- Existing client request/response examples continue to work.

### Task 7: Implement stop and output extraction in gateway

Purpose: make vLLM output termination explicit and safe.

Files to change:

- `gateway/vllm_client.py`.
- `gateway/prompting.py`.
- `gateway/main.py`.
- Add protocol tests under `gateway/tests/`.

Solution:

- Use the stop mechanism selected by Task 5.
- Extract only completion text, preserving visible output semantics of current API.
- Detect finish reasons `stop`, `length`, and backend errors.
- Return or log truncation status internally.
- Treat `length` as a controlled failure or response metadata according to compatibility policy; never silently present a likely truncated translation as normal.
- Preserve trailing whitespace for parity diagnostics. If product API eventually strips it, do so only after stop correctness is separately verified and document that it changes the old diagnostic behavior.

Verification:

- Test stop at EOS and turn end.
- Test no stop and max-token finish.
- Test special token visibility.
- Test repeated generation determinism.
- Test vLLM errors, timeout, cancellation, and malformed response.

### Task 8: Implement workload routing and admission control

Purpose: prevent long/bulk traffic from degrading interactive latency while allowing vLLM to batch efficiently.

Files to add/change:

- `gateway/routing.py`.
- `gateway/limits.py`.
- `gateway/config.py`.
- `gateway/main.py`.
- `gateway/schemas.py`.
- Add routing tests.

Solution:

- Estimate source tokens using the staged tokenizer or a lightweight tokenizer service/package. Use exact model tokenizer where practical.
- Classify interactive/document/bulk.
- Enforce body, item, source-token, context-token, and output-token limits.
- Add bounded concurrency and queue limits at gateway.
- Return 429 when queue is full.
- Use request priority only within documented policy.
- Keep one backend initially; add backend abstraction for future replicas/queues.
- Do not create separate vLLM containers on same GPU by default.
- Add route labels to logs/metrics.
- Maintain response ordering for batch requests.
- Support cancellation when client disconnects where HTTP client library permits.

Verification:

- Short requests are not blocked indefinitely by long requests under load.
- Oversized requests fail before reaching vLLM.
- Queue saturation returns predictable 429.
- Batch output order is unchanged.

### Task 9: Configure vLLM continuous batching and cache

Purpose: tune runtime for actual workload after compatibility is proven.

Files to add/change:

- `serving/vllm/docker-compose.yml`.
- `serving/vllm/.env.example`.
- `serving/vllm/README.md`.
- Gateway compose/network configuration.

Solution:

- Pin image digest.
- Mount merged export read-only.
- Set model name and local model path.
- Set offline/cache variables.
- Configure GPU reservation and `NVIDIA_VISIBLE_DEVICES`/GPU count correctly.
- Set maximum model length from actual model context and product limits.
- Tune GPU memory utilization, max batched tokens, max sequences, and chunked prefill using benchmark results.
- Enable prefix caching only after measuring benefit and memory cost.
- Keep vLLM internal metrics available on private network.
- Do not publish vLLM directly to public clients; expose gateway only.

Verification:

- vLLM and gateway start in correct dependency order.
- Gateway readiness reflects backend readiness.
- No host network or public port accidentally bypasses gateway policy.

### Task 10: Replace old compose/API deployment documentation

Purpose: prevent operators from accidentally deploying old PEFT server as production path.

Files to change:

- `api/docker-compose.yml` or replace with new serving compose layout.
- `api/README.md`.
- Root `README.md`.
- Add/update `docs/INFERENCE_SERVING_RUNBOOK.md`.
- Update `docs/DEPLOYMENT_BACKLOG.md`.

Solution:

- Clearly label old local Transformers API as reference/fallback or remove only after migration gate.
- Document two-machine workflow.
- Document merge command, export verification, transfer, startup, health checks, rollback, and release switching.
- Document exact vLLM image/digest and GPU prerequisites.
- Document no multiple Uvicorn workers for GPU model service; gateway workers are safe only within resource limits.
- Document secret handling, private backend network, CORS policy, request limits, and logs.
- Remove stale adapter-mode production defaults from active deployment instructions.
- Resolve `.env.example` mismatch between old base/4-bit values and production adapter/BF16 assumptions.

Verification:

- A new operator can deploy from clean serving host using only runbook and transferred artifact.
- Runbook does not require parent repository on serving machine.

### Task 11: Add production observability

Purpose: identify whether speed problem is queueing, prefill, decode, batching, network, or model failure.

Files to add/change:

- `gateway/metrics.py` or equivalent.
- Gateway logging modules.
- Runbook and compose environment.

Solution:

Record structured metrics:

- request count/error count by route and class
- queue wait time
- gateway processing time
- backend request time
- time to first token if streaming is enabled
- end-to-end latency p50/p95/p99
- input tokens and output tokens
- finish reason
- truncation count
- vLLM batch/queue metrics where available
- HTTP status and error category
- model release ID and prompt contract hash

Never log source text or translations by default if content may be sensitive. Use request IDs and sampled hashes.

Verification:

- A load test can distinguish queue delay from GPU generation time.
- Metrics do not contain credentials or full document text.

### Task 12: Add streaming only after non-streaming parity

Purpose: improve perceived latency without weakening correctness.

Files to change:

- `gateway/vllm_client.py`.
- `gateway/main.py`.
- `gateway/schemas.py` if additive streaming endpoint is chosen.
- Documentation and tests.

Solution:

- Prefer additive streaming endpoint or explicit request flag.
- Forward vLLM SSE safely.
- Ensure special stop tokens are not emitted to clients.
- Ensure final chunk carries finish reason and request ID.
- Handle disconnect cancellation.
- Keep existing non-streaming endpoints unchanged.

Verification:

- First token arrives earlier than full response.
- Final assembled text equals non-streaming text.
- Stop and error events are correctly represented.

### Task 13: Benchmark engine, scheduler, and routing matrix

Purpose: choose settings from evidence rather than assumptions.

Files to add/change:

- Add `serving/benchmarks/benchmark_serving.py` or use approved external tools with checked-in command manifests.
- Add `docs/INFERENCE_SERVING_BENCHMARKS.md`.
- Add benchmark output outside source control or under designated artifacts.

Solution:

Compare:

- old Transformers reference
- vLLM merged BF16
- vLLM merged FP8/other quantization only after quality gate
- vLLM adapter mode only if merge is rejected
- SDPA/engine default/FA3 where selectable
- concurrency 1, 2, 4, 8, 16, 32, 64
- short, median, p95, page, and mixed-length workloads
- interactive, document, and bulk routes

Record:

- TTFT
- inter-token latency
- end-to-end p50/p95/p99
- requests/s
- source tokens/s
- output tokens/s
- queue time
- GPU utilization/power
- peak VRAM
- vLLM scheduler metrics
- error/timeout/truncation rate
- quality and degeneration results

Use `guidellm`, `vllm bench serve`, NVIDIA GenAI-Perf, Locust, k6, or equivalent. Pin tool versions and retain workload files.

Verification:

- Selected runtime/config beats old reference on agreed SLA without quality regression.
- Results are repeatable across at least three runs.

### Task 14: Add rollout and rollback procedure

Purpose: deploy without losing known-good adapter service.

Files to add/change:

- `docs/INFERENCE_SERVING_RUNBOOK.md`.
- Compose files and environment examples.
- Optional deployment scripts under `serving/scripts/`.

Solution:

- Keep old API image available.
- Start vLLM privately.
- Validate vLLM directly with smoke tests.
- Start gateway pointed at vLLM.
- Run gateway parity suite and small load test.
- Switch external traffic to gateway.
- Monitor errors, p99, finish reasons, and quality sample.
- Roll back by switching gateway/backend or image/artifact release, without deleting previous model export.
- Use immutable release IDs and retain at least one last-known-good release.

Verification:

- Rollback completes without rebuilding model image.
- Previous service remains deployable.

## 9. Testing Matrix

### Unit tests

- Prompt rendering exactness.
- Prompt token IDs against known fixture.
- Stop ID resolution.
- Request option resolution.
- Token-budget calculation.
- Routing classification.
- Admission limits.
- Batch ordering.
- Error mapping.
- Manifest/checksum validation.

### Integration tests

- Gateway to vLLM raw completion.
- Offline startup.
- vLLM model load.
- Exact prompt parity against reference Transformers path.
- EOS and turn-end stopping.
- Deterministic repeated calls.
- Batch request behavior.
- Backend timeout and unavailable states.
- Client cancellation.

### Quality tests

- Full frozen evaluation set.
- Degeneration classifier.
- Numeric/unit/formula/acronym preservation.
- Mixed-script and scientific terminology slices.
- Human review sample.

### Performance tests

- Concurrency sweep.
- Mixed-length load.
- Long-document isolation.
- Batch API throughput.
- Queue saturation.
- GPU memory boundary.
- Cold start and warm steady state.

## 10. Security And Reliability

- Do not expose vLLM port publicly.
- Restrict gateway CORS; do not retain wildcard credentials in production.
- Authenticate gateway if service is not fully private.
- Enforce request size and token limits before backend call.
- Never allow client to choose arbitrary model path/name.
- Never mount model export writable.
- Pin image digest, Python dependencies, and benchmark tool versions.
- Keep model provenance and checksums.
- Protect `/model-info`, metrics, and deep health endpoints if they reveal internal paths.
- Use graceful shutdown and bounded backend connection pools.
- Ensure one gateway process does not create unbounded asynchronous tasks.
- Use separate GPU replicas for throughput scaling; do not use multiple model workers on one GPU.

## 11. Explicit Non-Goals

- No retraining solely for serving speed.
- No blind prompt-template change.
- No adapter merge performed from 4-bit weights for canonical export.
- No runtime repetition penalty as a stop workaround.
- No public exposure of raw vLLM endpoint.
- No separate vLLM replicas on one GPU without measurement.
- No promise that FlashAttention alone fixes request-level serialization.
- No removal of reference Transformers path before parity evidence exists.

## 12. Acceptance Gates

The implementation is complete only when all gates pass:

1. Merge export loads offline and manifest/checksums validate.
2. Merged model quality and degeneration metrics meet agreed tolerance against base-plus-adapter reference.
3. Gateway renders identical prompt token sequences for parity fixtures.
4. Runtime stops on `<end_of_turn>` and EOS without whitespace flood or turn restart.
5. Gateway preserves existing endpoint schemas and output ordering.
6. Oversized and overloaded requests fail predictably.
7. vLLM is private and gateway readiness tracks backend readiness.
8. Continuous-batching load test demonstrates improvement over current serialized API.
9. p50/p95/p99, queue time, TTFT, output throughput, finish reasons, and GPU metrics are recorded.
10. Rollback to previous serving path is documented and tested.

Do not declare success from a single curl request or batch-size-1 benchmark. The central success criterion is correct merged-adapter translation under realistic concurrent workload with measurable latency and throughput improvement.

## 13. Recommended Execution Order

```text
1. Freeze baseline
2. Run preliminary vLLM compatibility smoke test against the unmerged base architecture
3. Implement merge/export script
4. Verify merged quality and stop behavior
5. Stage immutable artifact and run the full vLLM compatibility gate against it
6. Build gateway with exact prompt contract
7. Prove gateway-to-vLLM parity
8. Add limits and workload routing
9. Configure vLLM continuous batching
10. Add observability
11. Run performance matrix
12. Document and test rollout/rollback
13. Canary production traffic
14. Retire old API only after rollback window
```

The implementation agent must stop and report when a compatibility gate fails. It must not solve vLLM incompatibility by changing prompt rendering, removing stop IDs, stripping output, or silently falling back to a different model.
