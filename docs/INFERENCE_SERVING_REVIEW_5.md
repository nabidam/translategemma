# Inference Serving Implementation Review 5

Review scope: static code review only. No tests, scripts, Docker builds, model loads, or runtime checks were executed on this machine.

Reviewed commit: `77626b9 fix(serving): address fourth review findings across provenance, promotion, streaming, and gateway`

Previous review: `docs/INFERENCE_SERVING_REVIEW_4.md`

Reference plan: `docs/INFERENCE_SERVING_PLAN.md`

## Findings

### 1. High: Production compose does not enable trusted manifest verification

Locations:

- `gateway/config.py:18-21`
- `serving/docker-compose.yml:65-80`
- `gateway/.env.example:8-21`

`require_verified_manifest` defaults to false. Production compose sets exact-tokenizer mode but does not set `TG_REQUIRE_VERIFIED_MANIFEST`, does not mount an external anchor, and does not provide `TG_TRUSTED_MANIFEST_SHA256` or `TG_TRUSTED_ANCHOR_FILE`. Gateway therefore starts and reports readiness using only optional co-located metadata. Newly added external trust path is inactive in deployed composition.

Required fix:

- Set `TG_REQUIRE_VERIFIED_MANIFEST=true` in production compose.
- Mount trusted anchor from a separate read-only host path outside model release and set `TG_TRUSTED_ANCHOR_FILE`, or inject expected hash through protected deployment configuration.
- Make trusted verification default fail-closed for production profile; allow unverified mode only in explicit development profile.
- Update `.env.example` and runbook with required trust settings.

### 2. High: “Required verified manifest” accepts co-located checksum as verified authenticity

Locations:

- `gateway/main.py:235-268`
- `gateway/main.py:330-340`

When no external trust input is configured, a matching `merge_manifest.sha256` sets `is_verified=True` and `authenticity_status="colocated_checksum_only"`. Startup with `require_verified_manifest=true` rejects only `is_verified=false` or status `unverified`; `colocated_checksum_only` passes. Anyone able to modify model directory can alter manifest and co-located checksum together, defeating required authenticity.

Required fix:

- Split integrity and authenticity into separate booleans.
- `require_verified_manifest=true` must require `authenticity_status == "trusted_external_anchor"` exactly.
- Co-located checksum may satisfy integrity diagnostics only, never production authenticity.
- Add test proving required mode rejects co-located-only anchor.

### 3. High: Gateway authenticates manifest but never verifies model payload against it

Locations:

- `gateway/main.py:155-268`
- `gateway/main.py:330-355`

Gateway hashes and authenticates `merge_manifest.json`, then derives runtime stop IDs and identity from it. It does not run `verify_checksums_and_inventory()` or otherwise hash mounted config, tokenizer, and weight shards. Model payload can be changed after promotion while trusted manifest remains unchanged; gateway still reports manifest verified and vLLM can load altered weights.

Required fix:

- Verify full export payload against authenticated manifest/SHA256SUMS at gateway startup or mandatory deployment preflight before containers start.
- Record payload verification state and timestamp separately from manifest authenticity.
- Fail readiness when authenticated manifest does not match mounted payload.
- Ensure vLLM and gateway mount same immutable resolved release directory.

### 4. High: Promotion command does not run claimed serving and quality gates

Locations:

- `scripts/promote_model_release.py:64-107`
- `scripts/promote_model_release.py:165-208`
- `serving/COMPATIBILITY.md:40-47`

Promotion docstring says active pointer changes only after “all gates,” but implementation runs only `verify_export()`. It does not run merged-vs-adapter quality verification, vLLM model load, stop-token smoke test, deployment compatibility preflight, or gateway parity checks. A structurally valid but behaviorally broken model can be promoted.

Required fix:

- Require signed evidence/results for merged quality gate, degeneration gate, vLLM compatibility smoke test, and deployment preflight before pointer switch.
- Either execute configured gate commands or consume a signed release-attestation file binding all gate results to manifest hash.
- Refuse promotion when any required evidence is absent, stale, or bound to different artifact hash/image digest.
- Update CLI/docs so “verified promotion” matches actual behavior.

### 5. High: Merge script retains a second non-atomic activation path that bypasses promotion gates

Location: `scripts/merge_lora_adapter.py:627-644`

Merge script accepts `--current-symlink` and directly unlinks/recreates `current` after only export verification. This bypasses `promote_model_release.py`, external anchor requirement, serving compatibility checks, and quality evidence. `unlink()` plus `symlink_to()` is not atomic, creating interval with no active pointer.

Required fix:

- Remove symlink activation options from merge script.
- Make merge/export produce immutable release only.
- Centralize all activation through `promote_model_release.py` using atomic replace and complete release gates.

### 6. High: Rollback verification does not require external trusted anchor

Location: `scripts/promote_model_release.py:110-139`

Rollback calls `verify_export()` without expected manifest hash or external trusted anchor. It accepts co-located checksum only. A tampered previous release can be activated during incident response, exactly when controls matter most. Rollback also leaves `previous` pointing to same release now selected as current, so subsequent rollback cannot atomically return to release that was active before rollback.

Required fix:

- Store trusted manifest hash/attestation per release in release index and require it during rollback.
- Atomically swap current and previous pointers, or maintain explicit release history.
- Verify payload and trusted authenticity before rollback activation.
- Add repeated rollback/roll-forward tests.

### 7. High: Immutable requested revisions can still be claimed when Hub resolution failed

Location: `scripts/merge_lora_adapter.py:256-296`

For remote model, `hf_model_info()` failure is logged only at debug. If requested revision is a 40-character SHA and no remote commit resolves, result sets `resolved_revision` to requested SHA and marks `revision_type="commit_sha"`. Export proceeds even though loaded bytes were not verified against that SHA, violating Review 4 requirement to fail when immutable revision cannot be resolved.

Required fix:

- If immutable revision is requested and Hub resolution fails, abort export unless local snapshot metadata proves exact commit.
- Capture resolved adapter commit from actual PEFT/Hugging Face cached snapshot, not requested string fallback.
- Bind loaded config/weight hashes to resolved revision in manifest.

### 8. High: Stop-token generation config lookup ignores selected base revision

Locations:

- `scripts/merge_lora_adapter.py:438-445`
- `scripts/merge_lora_adapter.py:501-506`
- `prompting.py:55-67`

Processor, config, and base weights now load with `base_revision`, but `resolve_stop_token_ids(..., base_model_id=base_model_id)` calls `GenerationConfig.from_pretrained(base_model_id)` without revision. `build_deterministic_generation_config()` repeats same default-revision lookup. A merge pinned to older/specific revision can combine its weights with generation config from current default revision.

Required fix:

- Thread `base_revision` through stop-token and deterministic generation-config helpers.
- Load every base artifact from same resolved revision/snapshot.
- Prefer local staged snapshot path after resolution so all reads bind to identical bytes.

### 9. Medium: Deployment preflight validates source text, not effective runtime configuration or host

Location: `scripts/verify_deployment_compatibility.py:52-112`

Preflight regexes first SHA256 in compose source and checks membership in matrix. It does not resolve `${VLLM_IMAGE}` environment override, parse Compose YAML, inspect actual pulled image digest, inspect running image, verify GPU architecture/driver/CUDA, compare runtime flags, or verify context values. Presence of variable names is treated as context alignment. An unsupported host or overridden image can pass.

Required fix:

- Parse rendered Compose configuration (`docker compose config`) or require resolved config input.
- Inspect actual image RepoDigest and host GPU/driver/CUDA capabilities.
- Compare effective flags, model length, dtype, tensor parallelism, and GPU architecture against approved entry.
- Matrix evidence must reference real archived smoke output, not self-asserted booleans.

### 10. Medium: Compatibility matrix claims production approval without evidence artifact

Locations:

- `serving/vllm_compatibility_matrix.json:25-31`
- `serving/COMPATIBILITY.md:5-8`

Matrix states `APPROVED_FOR_PRODUCTION`, date, stop-token verification, and token parity true, but provides no evidence file path, command output hash, host identity, image inspect output, model release hash, or reviewer. These claims cannot be audited and contradict required target-host verification workflow.

Required fix:

- Mark status `UNVERIFIED` until serving-host evidence exists.
- Add evidence artifact paths/hashes, host GPU/driver/CUDA, image RepoDigest output, model manifest hash, smoke report hash, and verifier identity/date.
- Preflight must reject entries lacking verifiable evidence.

### 11. Medium: Gateway derives security-critical stop tokens from unverified manifest

Location: `gateway/main.py:342-354`

Runtime settings are overwritten from any loaded manifest, regardless of `is_verified` or authenticity. In default compose, an untrusted manifest controls `stop_token_ids` and `stop_tokens`, potentially removing `<end_of_turn>` protection or adding premature stops. This can reintroduce runaway decoding or truncate translations.

Required fix:

- Derive runtime settings only from externally authenticated manifest whose payload has been verified.
- Independently resolve and assert required stop IDs from mounted tokenizer/generation config.
- Fail startup unless manifest, tokenizer, and generation config agree and contain IDs 1 and 106.

### 12. Medium: Model-info exposes complete manifest and internal provenance paths

Location: `gateway/main.py:598-638`

`/model-info` returns entire `manifest_metadata`, including command arguments, local adapter paths, package details, override reasons, and fingerprints. Endpoint is public in current gateway and plan warned against revealing internal paths.

Required fix:

- Return curated public provenance fields only.
- Protect detailed provenance endpoint with authentication/private network or remove it.
- Never expose command arguments or filesystem paths publicly.

### 13. Medium: Body-limit middleware can send second response after downstream started

Location: `gateway/main.py:96-117`

Middleware still raises from wrapped receive and catches outside `self.app()`. If downstream middleware/application sends response headers before consuming full request body, limit exception then sends a new 413 response after response start, violating ASGI protocol. It also does not drain remaining input, harming keep-alive reuse.

Required fix:

- Track whether `http.response.start` has been sent via wrapped `send`; never send a second response.
- Prefer pre-buffering bounded body before invoking application for JSON translation endpoints, then replay accepted body through receive wrapper.
- Drain or close oversized request safely according to ASGI server behavior.

### 14. Medium: Architecture validation checks names only, not module types or dimensions

Location: `scripts/merge_lora_adapter.py:299-350`

Validation considers target valid when suffix/name exists. It does not confirm target is compatible linear/projection module, compare LoRA tensor shapes to base weight dimensions, or verify expected model architecture/config. A same-name incompatible module can pass until merge failure or produce incorrect behavior.

Required fix:

- Verify target module classes and base weight shapes against adapter A/B tensors.
- Compare model type/config fingerprint with adapter expectations.
- Record exact matched module count and shape validation result.

### 15. Low: Gateway lockfile has exact versions but no hashes and Docker image is unpinned

Locations:

- `gateway/requirements-lock.txt:1-50`
- `gateway/Dockerfile:1,11-13`

Lock uses exact versions but no artifact hashes; package index compromise or republished files are not detected. Base image `python:3.11-slim` is mutable, and resulting gateway image digest is not recorded despite compatibility doc claiming it is production-pinned through lockfile.

Required fix:

- Generate hash-locked requirements and install with `--require-hashes`.
- Pin Python base image digest.
- Build, sign, and record gateway image digest in compatibility matrix.

### 16. Low: Prompt synchronization test compares signatures, not implementation bytes

Location: `tests/test_prompt_parity.py:13-27`

Gateway and root prompting copies can diverge internally while constants and function signatures remain identical. Mock-based parity then calls gateway renderer using gateway helper only; it does not prove copied helper implementation matches root.

Required fix:

- Reuse existing byte-identical vendoring sync mechanism for gateway prompting or package one shared module.
- Compare file bytes/source hashes, not only signatures.
- Keep real artifact processor/token-ID parity gate on serving host.

## Prior Findings Status

- External manifest trust: APIs added, but production compose does not enable/mount it and required mode accepts co-located checksum.
- Gateway manifest identity: loaded and exposed, but payload is not verified and untrusted manifest controls runtime stop settings.
- Immutable promotion: dedicated atomic tool added, but merge retains bypass path and promotion lacks quality/serving gates.
- Remote revisions: metadata resolver added, but unresolved immutable SHA still accepted and adapter loaded-byte binding remains weak.
- Architecture validation: module-name existence added; type/shape/config validation remains incomplete.
- Streaming sanitization/parser: substantially fixed.
- Body middleware: improved response generation; response-start/draining edge remains.
- Prompt parity: new test added; synchronization and real processor evidence remain incomplete.
- Dependency reproducibility: exact lock added; hashes/base/image digest remain.
- Compatibility evidence: matrix/preflight added, but evidence is self-asserted and preflight checks source text only.

## Required Fix Order

1. Enable fail-closed external manifest trust in production compose and reject co-located-only authenticity.
2. Verify model payload at gateway/deployment startup and prevent untrusted manifest from controlling stop settings.
3. Make promotion require full release attestation; remove merge-script activation bypass and secure rollback.
4. Fail unresolved immutable revisions and bind all generation artifacts to same base revision.
5. Replace compatibility self-claims with real host evidence and effective runtime inspection.
6. Protect model provenance output and complete architecture shape validation.
7. Finish ASGI body-limit hardening, prompt vendoring sync, and hash-locked gateway image.

## Review Conclusion

Fifth-round implementation adds substantial missing infrastructure: external anchors, gateway manifest state, atomic promotion/rollback tooling, compatibility matrix/preflight, architecture checks, sanitized stream parser, and dependency lock. However production still runs with trust disabled, co-located checksum is accepted as verified, mounted model payload is not checked, promotion does not enforce behavior gates, and compatibility approval is self-declared without evidence. Resolve these trust-chain issues before calling deployment production-ready.
