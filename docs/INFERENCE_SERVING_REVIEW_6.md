# Inference Serving Implementation Review 6

Review scope: static code review only. No tests, scripts, Docker builds, model loads, or runtime checks were executed on this machine.

Reviewed commit: `12f7a4e fix(serving): close release trust chain and replace self-asserted deployment claims`

Previous review: `docs/INFERENCE_SERVING_REVIEW_5.md`

Reference plan: `docs/INFERENCE_SERVING_PLAN.md`

## Findings

### 1. High: Release attestation is still self-asserted and unauthenticated

Location: `scripts/promote_model_release.py:137-226`

Promotion verifies that attestation JSON says each gate passed and that referenced evidence files match hashes written in that same attestation. Anyone able to write attestation and evidence can create arbitrary files, hash them, set `passed: true`, and authorize promotion. Attestation has no signature, trusted expected hash, issuer allowlist, or external authenticity anchor.

Required fix:

- Require signed attestation or an operator-supplied trusted attestation SHA256 from protected configuration.
- Verify signature against pinned public key or compare attestation hash against external release authorization store.
- Record issuer identity and reject unsigned/self-issued production attestations.
- Bind attestation signature to manifest hash, vLLM image digest, gateway image digest, and all evidence hashes.

### 2. High: Promotion validates evidence file hashes but not evidence semantics

Location: `scripts/promote_model_release.py:180-219`

Each gate needs `passed: true`, timestamp, evidence path, and matching evidence hash, but promotion does not parse evidence content or execute gate-specific validators. Empty or unrelated files can satisfy evidence requirement if attestation lists their hashes. `passed` remains self-declared.

Required fix:

- Define versioned schema per gate evidence.
- Parse each report and verify required outcomes: quality thresholds, zero disallowed degeneration, exact model/manifest identity, stop reason, token IDs, image digest, host compatibility, and parity result.
- Prefer generating attestation from gate runners rather than hand-writing JSON.
- Reject unknown evidence schema versions and reports not bound internally to same manifest hash.

### 3. High: vLLM image digest binding is optional during promotion

Locations:

- `scripts/promote_model_release.py:165-171`
- `scripts/promote_model_release.py:274-355`
- `docs/INFERENCE_SERVING_RUNBOOK.md:96-101`

Attestation image digest is checked only when caller passes `--expected-image-digest`. CLI makes it optional, and documented production promotion command omits it. A release can therefore be promoted using smoke evidence from a different vLLM image than production deployment.

Required fix:

- Make `--expected-image-digest` mandatory for production promotion.
- Also bind and require gateway image digest.
- Update runbook promotion command.
- Cross-check expected digests against approved compatibility record and resolved deployment configuration.

### 4. High: Rollback trusts unsigned, mutable release index as authenticity source

Location: `scripts/promote_model_release.py:390-480`

Without explicit anchor, rollback uses manifest hash from `release_index.json`. Index is ordinary writable JSON beside symlinks, with no signature or external anchor. An attacker can modify target path and hash, then rewrite release payload and co-located metadata to make rollback activate tampered model.

Required fix:

- Require per-release external trusted anchor or signed release index during rollback.
- Sign/hash-chain release index and store trust root outside writable model/pointer directory.
- Never treat mutable local index as sole authenticity source.
- Validate recorded attestation authenticity again during rollback.

### 5. High: Anchor rotation is separate and non-atomic from release pointer promotion

Locations:

- `scripts/promote_model_release.py:368-386`
- `docs/INFERENCE_SERVING_RUNBOOK.md:96-106`

Promotion atomically switches `current`, but trusted gateway anchor `/trust/current.sha256` is updated later through manual `cp`. Between operations, active model and trusted anchor disagree. Crash, operator interruption, or automation failure leaves gateway unable to start. Same split operation exists during rollback.

Required fix:

- Promotion tool must atomically update release pointer and trusted current-anchor pointer as one recoverable transaction.
- Prefer immutable per-release anchor path selected by deployment configuration, avoiding mutable `current.sha256` copy.
- Record transaction state and recover safely after interruption.
- Include anchor update in rollback tool.

### 6. High: `docker compose restart` may continue using old bind-mounted symlink target

Locations:

- `serving/docker-compose.yml:13,70`
- `docs/INFERENCE_SERVING_RUNBOOK.md:176-190`

Both containers bind-mount host path `/opt/models/translategemma/current`, which is a symlink switched by promotion. Docker resolves bind-mount source when container is created. Restarting existing container generally reuses original mount configuration/inode rather than recreating bind mount against new symlink target. Runbook uses `docker compose restart`, so vLLM/gateway may continue serving old release after promotion or rollback.

Required fix:

- Recreate containers after pointer switch: `docker compose up -d --force-recreate vllm gateway`, not restart.
- Better: resolve immutable release directory into rendered Compose environment and recreate against that path.
- Verify mounted release ID inside both containers after recreation before traffic.
- Add deployment test proving promoted release hash equals model mounted by both services.

### 7. High: Gateway and vLLM release identity is not mutually verified

Locations:

- `gateway/main.py:247-317`
- `gateway/vllm_client.py:42-72`
- `serving/docker-compose.yml:13,70`

Gateway verifies its mounted model payload and checks only vLLM served model alias through `/models`. It cannot prove vLLM loaded same release bytes/manifest as gateway. Different bind resolution, stale vLLM container, or operator path mismatch can lead gateway reporting release B while vLLM generates with release A.

Required fix:

- Expose immutable release identity from vLLM deployment, such as served model name containing release ID/hash or sidecar metadata bound to container mount.
- Gateway readiness must compare backend release ID/manifest hash with its own verified manifest.
- Deployment smoke must verify generated backend process was started from same immutable release path/hash.

### 8. High: Stop-token revision fix still does not bind all generation artifacts to one resolved snapshot

Locations:

- `scripts/merge_lora_adapter.py`
- `prompting.py`
- `model_loading.py`

Revision parameters were threaded into helpers, but merge still loads model artifacts through multiple independent Hub calls using repository ID/revision. Provenance resolver may query Hub separately from actual cached load. There is no single resolved local snapshot path used for processor, config, generation config, and weights. Branch/tag movement or inconsistent cache state can still produce mixed bytes.

Required fix:

- Resolve base and adapter once to immutable local snapshot directories.
- Perform every subsequent load from those local paths in offline mode.
- Record snapshot commit and complete file inventory hashes.
- Reject mutable branch/tag revisions unless resolved and pinned before loading.

### 9. Medium: Compatibility evidence remains unauthenticated

Locations:

- `scripts/verify_deployment_compatibility.py:266-301`
- `serving/vllm_compatibility_matrix.json`

Preflight now validates evidence files against hashes stored in compatibility matrix, but matrix itself is unsigned and repository-writable. An attacker or accidental edit can replace evidence files and update hashes/status in matrix. This is integrity within one mutable trust domain, not approval authenticity.

Required fix:

- Sign compatibility matrix or require trusted expected matrix hash from deployment configuration.
- Separate candidate matrix from approved signed release record.
- Verify approval issuer and signature before accepting `APPROVED_FOR_PRODUCTION`.

### 10. Medium: Host report and image-inspect reports are accepted as files, not live host facts

Location: `scripts/verify_deployment_compatibility.py:383-455`

Preflight parses supplied JSON reports but does not prove they were generated on current host or correspond to currently pulled/running image. Stale/copied evidence from another host can pass if hashes match approved matrix.

Required fix:

- Collect live `docker image inspect`, `nvidia-smi`, driver, and CUDA data inside preflight by default.
- Allow file input only for offline audit mode and label it non-live.
- Bind host identity and collection timestamp; reject stale evidence for deployment gate.

### 11. Medium: Payload verification hashes full model at every gateway startup

Locations:

- `gateway/manifest_verification.py:61-143`
- `gateway/main.py:247-275`

Production defaults verify all weight shards in gateway lifespan. For 12B BF16 model this reads roughly tens of GB before gateway starts, after vLLM already loaded same model. Compose healthcheck start period remains 5 seconds. This can create long unavailable startup, heavy shared-disk contention with vLLM, and misleading unhealthy status during expected verification.

Required fix:

- Perform full payload hashing as mandatory pre-deployment gate and persist signed verification receipt bound to manifest hash.
- At gateway startup verify receipt plus lightweight metadata/inode checks, or increase readiness/start-period based on measured hashing time.
- Do not concurrently hash model while vLLM loads from same storage.
- Keep optional full re-hash for periodic integrity audit.

### 12. Medium: Gateway trust anchor filename defaults to mutable generic `current.sha256`

Location: `serving/docker-compose.yml:71-90`

Gateway always reads `/trust/current.sha256`. This mutable filename is not intrinsically bound to release ID. Wrong copy can authorize wrong release or block startup, and audit trail loses which immutable anchor deployment selected.

Required fix:

- Select immutable release-specific anchor path, e.g. `/trust/<release-id>.sha256`.
- Bind selected anchor filename/hash to rendered deployment config and attestation.
- Avoid mutable generic aliases for trust roots.

### 13. Medium: Promotion can skip full payload checks via production CLI flag

Location: `scripts/promote_model_release.py:274-355,590-593`

`--skip-checksums` remains accepted by promotion and rollback. Help says not for production, but no production mode prevents use. Operator can promote based on authenticated manifest without verifying model weights match it.

Required fix:

- Remove `--skip-checksums` from promotion/rollback commands.
- Keep quick verification only in separate diagnostic command that cannot switch pointers.

### 14. Medium: Release index update occurs after current pointer switch

Location: `scripts/promote_model_release.py:368-384`

Current symlink switches before `record_activation()` writes index. Crash or disk error after pointer switch leaves active release changed but index stale. Rollback then may lack trusted hash/attestation record for actual active state.

Required fix:

- Implement transactional journal: write pending activation record, atomically switch pointers, then mark committed.
- Recovery command must reconcile pointers and journal after crash.
- Do not report success until index and pointers agree.

### 15. Medium: Gateway startup mutates cached global Settings from manifest

Location: `gateway/main.py:277-285`

`get_settings()` returns global singleton. Lifespan mutates release/base/adapter fields in-place. Tests, multiple app instances in one process, lifespan restarts, or reload can retain old artifact-derived values. This can misreport new mounted release or contaminate subsequent test state.

Required fix:

- Keep immutable environment settings separate from verified runtime provenance.
- Store artifact identity in app state, not mutate cached settings object.
- Build responses from verified manifest state.

### 16. Low: Hash-lock file is required by Dockerfile but absent from reviewed file list

Locations:

- `gateway/Dockerfile:17-22`
- `scripts/generate_gateway_hash_lock.sh`

Dockerfile copies `requirements-hashes.txt`, but repository file inventory shown for gateway contains no such file. If absent in commit, gateway image cannot build. Generator script exists but generated artifact must be committed or generated before Docker COPY through defined build stage.

Required fix:

- Commit `gateway/requirements-hashes.txt`, or generate it in a prior reproducible build stage before COPY dependency.
- Add source-control/build preflight checking file exists and matches lock.

### 17. Low: Prompt parity still uses mock processor instead of real artifact

Location: `tests/test_prompt_parity.py`

Synchronization checks improved, but parity test still uses mocked `apply_chat_template`. It cannot detect actual TranslateGemma processor/template behavior changes in pinned artifact.

Required fix:

- Add offline integration test using staged real processor/tokenizer artifact and compare prompt bytes/token IDs across root and gateway paths.
- Keep mock test as unit coverage only.

## Prior Findings Status

- Production external manifest trust: fixed and fail-closed in compose.
- Co-located checksum authenticity: fixed; no longer accepted as production authenticity.
- Gateway payload verification: implemented.
- Promotion behavioral gates: attestation requirement added, but attestation remains unauthenticated and semantically unchecked.
- Merge activation bypass: removed from primary flow.
- Rollback trust: improved with release index, but index itself is unsigned and mutable.
- Immutable revisions: resolver improved; one-snapshot binding remains incomplete.
- Stop-token revision threading: improved; multiple independent loads remain.
- Effective deployment preflight: substantially improved; evidence authenticity/live-host binding remains.
- Compatibility matrix status: now correctly `UNVERIFIED` by default.
- Runtime stop settings: derived independently from mounted payload.
- Public provenance exposure: curated by default.
- Body middleware: pre-buffered safely before app response.
- Architecture validation: improved, but tensor shape checks remain incomplete.
- Dependency hashing/base-image pin: build controls added; hash file presence must be ensured.
- Prompt vendoring sync: improved; real artifact parity remains missing.

## Required Fix Order

1. Authenticate release attestations and validate evidence semantics.
2. Make image digests mandatory and bind both vLLM/gateway images to attestation.
3. Secure rollback index and atomically coordinate pointer plus trust-anchor activation.
4. Recreate containers against immutable release path and verify gateway/vLLM release identity match.
5. Resolve all model artifacts to one immutable local snapshot before merge.
6. Sign compatibility approval and collect live host evidence.
7. Remove checksum-skip activation paths and add crash-safe activation journal.
8. Optimize startup payload verification without weakening pre-deployment integrity.
9. Finish settings immutability, hash-lock artifact, tensor-shape validation, and real prompt parity.

## Review Conclusion

Sixth-round implementation materially closes previous trust-chain gaps: production requires external anchors and payload verification, compatibility status defaults unverified, deployment parsing checks effective flags and host reports, promotion requires gate attestation, rollback tracks history, public provenance is curated, and build inputs are moving toward hash pinning. Remaining blockers concern authenticity and atomic operations: attestations and compatibility approvals are still editable self-claims, rollback index is unsigned, anchor rotation is separate from pointer switch, containers may remain bound to old symlink targets, and gateway cannot prove vLLM loaded same release. Address these before production cutover.
