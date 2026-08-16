# Inference Serving Implementation Review 4

Review scope: static code review only. No tests, scripts, Docker builds, model loads, or runtime checks were executed on this machine.

Reviewed state: working tree after `5285b01`, including uncommitted fourth-round changes.

Previous review: `docs/INFERENCE_SERVING_REVIEW_3.md`

Reference plan: `docs/INFERENCE_SERVING_PLAN.md`

## Findings

### 1. High: Manifest checksum anchor is co-located and not an authenticity boundary

Locations:

- `scripts/merge_lora_adapter.py:440-443`
- `scripts/verify_model_export.py:255-274`

`merge_manifest.sha256` is written beside `merge_manifest.json` and verified from that same directory. Anyone who can alter the release can alter both files, recompute the hash, and make changed provenance appear valid. This detects accidental corruption but does not provide the external integrity anchor required by Review 3 or the plan's trusted provenance requirement.

Required fix:

- Store the manifest hash/signature in a trusted release index, transfer manifest, deployment metadata store, or detached signature distributed separately from the model directory.
- Make serving verification accept the expected hash/signature as an operator-supplied or trusted-release input, not derive trust solely from files under `/models/model`.
- Distinguish checksum integrity from authenticity in documentation and `/model-info`.

### 2. High: Remote revision provenance is still not tied to resolved loaded bytes

Locations:

- `scripts/merge_lora_adapter.py:190`
- `scripts/merge_lora_adapter.py:301`
- `scripts/merge_lora_adapter.py:343-345`
- `scripts/merge_lora_adapter.py:418-424`

The script now passes requested revisions into loaders, but `resolved_base_commit` is captured only from `base_model.config._commit_hash`, and there is no equivalent resolved adapter commit. For remote adapters, `adapter_fingerprint` is `None` because `adapter_path` is resolved as a local path and `is_dir()` is false. Manifest can still claim `adapter_revision=X` without recording the commit actually loaded by PEFT. For local models, the requested `base_revision` can also be a label while only file fingerprints prove bytes.

Required fix:

- Capture resolved commit SHA from the actual Hugging Face config/download metadata for both base and adapter.
- For local artifacts, record complete deterministic file fingerprints and reject unverifiable revision claims or mark them explicitly as labels.
- Add manifest fields distinguishing `requested_revision`, `resolved_revision`, and `artifact_fingerprint`.
- Fail export when an immutable revision was requested but cannot be verified against loaded content.

### 3. High: Release promotion still makes unverified output the active directory before serving-side verification

Locations:

- `scripts/merge_lora_adapter.py:445-469`
- `serving/docker-compose.yml:13`

The merge script verifies its staged artifact with the lightweight export verifier, then renames the existing `output_dir` to a backup and renames the new directory into `output_dir`. The serving machine mounts `/opt/models/translategemma/current`, but there is no atomic `current` symlink/pointer switch tied to a verified release directory. Any serving-specific validation, vLLM compatibility test, or quality gate happens after this active path has changed. A failed serving validation leaves the new artifact active and requires manual rollback.

Required fix:

- Publish each export under a unique immutable release directory.
- Run export integrity, merged-vs-adapter quality, serving compatibility, and smoke gates before activation.
- Atomically switch `current` only after all required gates pass.
- Retain explicit `previous`/last-known-good pointer and document rollback as pointer switch.
- Do not use `--force` to mutate an active release directory.

### 4. Medium: Stream error paths still expose internal details

Locations:

- `gateway/vllm_client.py:221-226`
- `gateway/vllm_client.py:249-260`
- `gateway/vllm_client.py:330-332`
- `gateway/main.py:580-586`

Non-streaming backend errors were sanitized, but streaming errors still include raw exception text, raw malformed payload prefixes, and UTF-8 decoder details in yielded public events. The gateway forwards these values to clients as SSE `error` payloads. This can reveal backend internals, malformed response content, or deployment details.

Required fix:

- Log detailed exception/frame data server-side with request ID.
- Yield stable public error codes/messages only: `BACKEND_ERROR`, `MALFORMED_FRAME`, `DECODE_ERROR`, `PREMATURE_CLOSE`, etc.
- Never include raw backend response, exception string, or payload excerpt in public stream events.

### 5. Medium: Stream parser still silently ignores malformed final frames

Location: `gateway/vllm_client.py:287-315`

The main parser reports malformed JSON, but the EOF flush path catches all exceptions and executes `pass`. A malformed final event can therefore be discarded and then produce a generic premature-close error, losing the actual protocol failure and making diagnostics inconsistent. Lines without `data:` and malformed event structure are also not validated in the flush path.

Required fix:

- Reuse one SSE event parser for normal and EOF-flush paths.
- Never `except Exception: pass` on protocol frames.
- Emit one stable `MALFORMED_FRAME`/`INCOMPLETE_STREAM` error with request ID and log the detailed cause server-side.
- Define behavior for a final event without trailing blank line according to SSE framing rules.

### 6. Medium: Body middleware raises during receive after partial body consumption

Location: `gateway/main.py:90-112`

`limited_receive()` raises `HTTPException` when cumulative body bytes exceed the cap. The outer handler then invokes a new `JSONResponse` using the original `receive` callable after the request body may already be partially consumed. ASGI middleware should generally terminate with a response directly from the wrapper rather than raising an application exception across a partially consumed receive stream. Depending on server behavior, this can produce duplicate response attempts, hanging clients, or unhandled exception logs.

Required fix:

- When limit is exceeded, send a single 413 response directly from middleware and drain/close the input according to ASGI server expectations.
- Avoid raising FastAPI `HTTPException` from the low-level receive wrapper.
- Test chunked and oversized bodies through the actual ASGI composition, not only direct `Content-Length` requests.

### 7. Medium: Exact processor readiness still does not prove prompt token parity

Locations:

- `gateway/main.py:174-204`
- `tests/test_gateway.py:22-46`

Gateway now requires `AutoProcessor.apply_chat_template`, which closes silent fallback in production. However the test only checks a hand-built fallback and a mocked processor return value; it never compares gateway output against the root canonical renderer with the real TranslateGemma processor or token IDs. A copied gateway prompting module can drift while both gateway tests and startup continue to pass.

Required fix:

- Add a parity test using the same processor fixture/artifact as root evaluation.
- Compare rendered prompt bytes and token IDs against `render_training_prompt()` from canonical source.
- Add a vendored-module synchronization check for gateway prompt helpers, or package the shared helper instead of duplicating it.

### 8. Medium: Gateway model identity/provenance does not load or verify merge manifest

Locations:

- `gateway/main.py:391-416`
- `gateway/config.py:20-24`
- `serving/docker-compose.yml:65-76`

Gateway reports `model_release_id`, `base_model_id`, and `source_adapter_path` from environment defaults. It does not read `merge_manifest.json`, verify its anchor, compare release ID/fingerprints, or expose the actual artifact hash. Operator configuration can claim one release while mounted files belong to another release.

Required fix:

- Load and verify model manifest at gateway startup from mounted artifact.
- Derive release ID, source adapter, base revision, merge provenance, prompt contract, and artifact fingerprint from verified manifest rather than free-form environment values.
- Fail readiness on manifest mismatch.
- Expose manifest/artifact identity in `/model-info`.

### 9. Medium: Adapter architecture fingerprint is collected but not validated against base architecture

Locations:

- `scripts/merge_lora_adapter.py:221-230`
- `scripts/merge_lora_adapter.py:325-345`

The script records PEFT fields such as rank, alpha, and target modules, but does not compare adapter target modules against actual base model module names or verify model architecture/config fingerprint. An adapter can have a matching repository name yet target missing/wrong modules or an incompatible architecture and still proceed to merge if PEFT does not fail immediately.

Required fix:

- Validate target modules exist in loaded base model and match expected module types.
- Compare adapter/base architecture and configuration fingerprints.
- Fail before expensive merge on mismatch; require explicit override with reason for intentional exceptions.
- Record validation result in manifest.

### 10. Low: Gateway dependency lock is still not reproducible

Locations:

- `gateway/requirements.txt:1-10`
- `gateway/Dockerfile:11-13`

Model-side versions are pinned, but FastAPI, Uvicorn, Pydantic, pydantic-settings, and HTTPX remain range-pinned. `pip freeze` is written into the image but no lock/hash file controls future builds. A gateway rebuild can change HTTP/SSE behavior without changing source.

Required fix:

- Commit a gateway lock/constraints file with hashes or pin all direct/transitive production dependencies through the chosen package workflow.
- Build from locked inputs and record image digest alongside vLLM digest.

### 11. Low: Active vLLM image digest is hard-coded without documented provenance

Locations:

- `serving/docker-compose.yml:4`
- `serving/vllm/docker-compose.yml:4`

Digest pinning is now present, but the digest has no source/provenance record or association with target GPU/CUDA compatibility evidence in deployment configuration. A copied digest can be wrong for the claimed tag or unsupported host while appearing production-ready.

Required fix:

- Store image inspect output, tag-to-digest mapping, GPU/driver/CUDA data, and smoke evidence in a versioned compatibility release record.
- Add deployment preflight that checks configured digest against approved compatibility record.

## Prior Findings Status

- Processor fallback: fixed for `require_exact_tokenizer=true`; parity evidence still missing.
- UTF-8 incremental decoding: fixed for primary parser; final parser handling and public error leakage remain.
- Revision selection: requested revisions now passed to loaders; resolved adapter commit/byte provenance remains incomplete.
- Body size: chunked handling added; low-level ASGI response semantics remain unverified/risky.
- Bulk fairness: separate queues and bounded worker fan-out added; aggregate fairness is improved but actual ASGI/load behavior remains unverified.
- Missing adapter identity: fixed with explicit override reason.
- Manifest anchor: added, but co-located anchor is not trusted external authenticity.
- Dependency pinning: model-side dependencies pinned; gateway web dependencies remain ranges.
- Context equality: now fails on any mismatch.
- Release promotion: staged export is verified before promotion; serving compatibility and immutable pointer activation remain unresolved.
- Backend error sanitization: non-streaming fixed; streaming still leaks details.
- vLLM digest: added, but evidence/preflight association remains incomplete.

## Required Fix Order

1. Load and verify manifest/artifact identity in gateway.
2. Complete immutable release pointer and post-serving-gate activation flow.
3. Finish revision/commit and adapter architecture verification.
4. Sanitize all streaming errors and unify final-frame parser.
5. Harden ASGI body-limit middleware and test actual chunked composition.
6. Add real processor/token parity and vendored synchronization coverage.
7. Lock gateway dependencies and record approved image compatibility evidence.

## Review Conclusion

Fourth-round changes close most prior functional findings: production exact mode now requires both tokenizer and processor, UTF-8 decoding is stateful, requested revisions reach loaders, body limits handle chunked input, queues are separated, missing adapter identity requires explicit override, manifest anchors exist, and vLLM is digest-pinned. Remaining release blockers are trust and activation concerns: gateway does not verify mounted artifact provenance, the manifest anchor is not external authenticity, remote revision resolution is incomplete, serving activation is not a verified pointer switch, and streaming/ASGI error paths still need hardening.
