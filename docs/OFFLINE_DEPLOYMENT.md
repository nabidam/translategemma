# Offline deployment guide

How to run the TranslateGemma fine-tuning pipeline on an air-gapped GPU machine
using a shipped Docker image, with the source, models and outputs bind-mounted
so everything stays visible and debuggable from the host.

Target GPUs: **RTX 6000 Ada (sm_89)**, **H100 / H100 NVL (sm_90)**, **RTX 5090
(sm_120)** and **RTX PRO 6000 Blackwell (sm_120)**. The default `cu128` image
covers all of them with an NVIDIA r570+ driver. The retained, named `cu130`
image covers the same GPUs but needs an r580+ driver.

---

## 1. What runs where

| Phase | Machine | Network | Produces |
| --- | --- | --- | --- |
| A. Staging | build machine | online | image tarball + model tarball + toolkit packages |
| B. Transfer | — | — | files on removable media |
| C. Install | GPU host | offline | working container environment |
| D. Run | GPU host | offline | adapters, evaluation, logs |

The build machine does **not** need a GPU. It needs Docker, disk space (~40 GB)
and internet.

---

## 2. Required host versions

These are the only things the container cannot provide. Check them **before**
building anything.

| Component | Minimum | Recommended | Why |
| --- | --- | --- | --- |
| NVIDIA driver | **570.x** for `cu128`; **580.65.06** for `cu130` | Latest supported driver in the required branch | The default CUDA 12.8 wheels require r570+. The CUDA 13.0 image refuses to initialise below r580. |
| nvidia-container-toolkit | **1.17.8** | 1.18.x | Supports Blackwell and current CUDA device/library injection. |
| Docker Engine | 25.0 | 27.x or newer | Stable `--gpus` / CDI handling. |
| Docker Compose plugin | 2.30 | 2.35+ | `deploy.resources.reservations.devices` behaviour used in `docker-compose.yml`. |
| Host kernel | 5.15 | 6.x | Meets the r570/r580 driver requirements. |

Verify on the GPU host:

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
docker --version && docker compose version
```

Choose the image that matches the installed driver:

| Image tag | CUDA / PyTorch | Minimum driver | Compose selection |
| --- | --- | --- | --- |
| `translategemma:cu128-py312` **(default)** | CUDA 12.8 / PyTorch 2.8.0 | r570 | No changes: use `.env.example` as supplied. |
| `translategemma:cu128-fa3-py312` | CUDA 12.8 / PyTorch 2.8.0 + FlashAttention 3 | r570 | Set `IMAGE_TAG=cu128-fa3-py312` and `INSTALL_FLASH_ATTN3=1`. **Hopper hosts only** — see §6.7. |
| `translategemma:cu130-py312` | CUDA 13.0 / PyTorch 2.13.0 | r580.65.06 | Set `IMAGE_TAG=cu130-py312` and `PYTORCH_CUDA=cu130` when building. |

Do not build the `cu130` tag for an r570-only host: an image can build
successfully but cannot initialise CUDA at runtime on that driver.

The `cu128-fa3` tag is a superset of the default one and is interchangeable
with it on a Hopper host, since `model.attn_implementation` still defaults to
`sdpa`. It is a separate tag rather than a new default because the wheel it
carries contains sm_90 kernels only, and because it is not yet benchmarked
against `sdpa` on this pipeline.

---

## 3. Phase A — staging (online machine)

### 3.1 Wheel availability (already verified)

`pyproject.toml` pins the default image to `torch==2.8.0` from the `cu128`
index. The pipeline is text-only, so it deliberately does not install the
optional TorchVision or TorchAudio binary extensions. This avoids unrelated
vision-operator import failures during `transformers` initialisation.
`torch-2.8.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl` is published. When
`PYTORCH_CUDA=cu130`, the Docker build selects the retained Torch 2.13.0 CUDA
13.0 configuration. Both wheel variants use `manylinux_2_28`, which needs glibc
≥ 2.28; `python:3.12-slim-bookworm` ships 2.36.

To re-check after a pin change:

```bash
curl -s https://download.pytorch.org/whl/cu128/torch/ | grep -c 'torch-2.8.0+cu128-cp312.*manylinux'
curl -s https://download.pytorch.org/whl/cu130/torch/ | grep -c 'torch-2.13.0+cu130-cp312.*manylinux'
```

### 3.2 Build the image

```bash
cd translategemma
cp .env.example .env
printf 'HOST_UID=%s\nHOST_GID=%s\n' "$(id -u)" "$(id -g)" >> .env

# The default cu128 Docker build installs exactly this lockfile. Create or
# refresh it on the GPU staging host only after the import preflight passes,
# then commit it before creating the git archive transferred offline.
uv lock
git add uv.lock && git commit -m "Lock CUDA 12.8 training dependencies"

docker compose build trainer
```

**A stale lock is now a hard build failure, not a warning.** `liger-kernel` was
added to `pyproject.toml` for `training.use_liger_kernel`, so a lockfile created
before that change no longer matches and `uv sync --locked` aborts. Any
dependency change means: `uv lock`, commit, rebuild, re-export, re-transfer.

`flash-attn-3` is not in this image. It is declared as an optional `speed` extra
and compiles against `nvcc`, which `python:3.12-slim` does not carry;
`uv sync --no-dev` does not install extras, so the build is unaffected.
`model.attn_implementation` must stay `sdpa` here — see §6.7 for the
Hopper-only variant that does carry it.

This builds the default `translategemma:cu128-py312` image. To deliberately
build the retained CUDA 13.0 image for an r580+ host:

```bash
IMAGE_TAG=cu130-py312 PYTORCH_CUDA=cu130 docker compose build trainer
```

To build the FlashAttention 3 variant for a Hopper host, first produce the
wheel (once, ~1–3 hours), then build against it (seconds):

```bash
scripts/build_flash_attn3_wheel.sh
IMAGE_TAG=cu128-fa3-py312 INSTALL_FLASH_ATTN3=1 docker compose build trainer
```

The build fails fast with a clear message if `INSTALL_FLASH_ATTN3=1` is set but
`wheels/` holds no wheel.

**Verify that image before exporting it.** The build host is not a Hopper
machine, so a successful build does not mean a working wheel — but the two
worst outcomes (glibc and Torch-ABI mismatches) are detectable there anyway:

```bash
docker run --rm -v "$PWD/scripts:/scripts:ro" \
    translategemma:cu128-fa3-py312 python /scripts/verify_flash_attn3.py
```

Full procedure, including the architecture check and what must wait for the
H100, is in §6.7.

Expect roughly **9–11 GB** uncompressed. The build context contains only
`pyproject.toml`, `uv.lock` and `wheels/`, so rebuilds after a dependency
change are the only slow ones. The default `cu128` build uses `uv sync --locked`; it fails if
`uv.lock` does not match `pyproject.toml` rather than silently resolving newer
packages.

Sanity-check the resolved environment (no GPU needed):

```bash
docker run --rm translategemma:cu128-py312 \
  python -c "import torch, peft, trl, bitsandbytes, liger_kernel; print(torch.__version__, torch.version.cuda)"
docker run --rm translategemma:cu128-py312 cat /opt/resolved-requirements.txt
```

`torch.version.cuda` must print `12.8`. For the named CUDA 13.0 image, replace
the tag in both commands with `cu130-py312`; it must print `13.0`.

### 3.3 Export the image

```bash
docker save translategemma:cu128-py312 | zstd -19 -T0 -o translategemma-cu128-image.tar.zst
```

For the CUDA 13.0 image, save `translategemma:cu130-py312` as
`translategemma-cu130-image.tar.zst`; for the FlashAttention 3 variant, save
`translategemma:cu128-fa3-py312` as `translategemma-cu128-fa3-image.tar.zst`.
Transfer only the image matching the GPU host's driver and architecture. The
wheel itself does not travel separately — it is already inside that image.

Roughly 5–6 GB compressed. Use `gzip` if `zstd` is unavailable on either side.

### 3.4 Stage the model weights

Six repositories are needed, not one. `scripts/fetch_offline_assets.py` reads
the ids out of `config.yaml` and `testset_config.yaml`, so the list follows your
run configuration; the two indirect dependencies are resolved automatically.

| Repository | Used by | Staged | Approx. size |
| --- | --- | --- | --- |
| `google/translategemma-12b-it` | `train.py`, `evaluate_translations.py`, `inference.py` | full | ~24 GB |
| `google/metricx-24-hybrid-large-v2p6` | `evaluate_translations.py` (`metricx_enabled`) | full | ~4.9 GB |
| `google/mt5-xl` | MetricX tokenizer (`metricx_tokenizer_id`) | tokenizer only | ~20 MB |
| `Unbabel/wmt22-comet-da` | `evaluate_translations.py` (`comet_enabled`) | full | ~2.3 GB |
| `xlm-roberta-large` | COMET's encoder — **indirect**, see below | tokenizer only | ~20 MB |
| `sentence-transformers/LaBSE` | `build_test_set.py` embeddings | full | ~1.8 GB |

Weights are skipped for the two tokenizer-only repositories: `google/mt5-xl`
alone would otherwise pull a 15 GB checkpoint that is never loaded.

**The COMET encoder is the easiest thing to miss.** `load_from_checkpoint()`
constructs an XLM-R encoder and calls `XLMRobertaTokenizerFast.from_pretrained()`
plus `XLMRobertaConfig.from_pretrained()` on whatever `hparams.yaml` names in
`pretrained_model` (`xlm-roberta-large` for `wmt22-comet-da`). The encoder
*weights* are not fetched — COMET passes `load_pretrained_weights=False` — but
the tokenizer and config are, and an offline run dies without them. The script
reads that id from the downloaded checkpoint's `hparams.yaml`, so it stays
correct if you switch COMET models. It stages the id **literally as COMET
requests it**: the hub cache is keyed by the exact string, so staging the
canonical alias `FacebookAI/xlm-roberta-large` would miss.

Only `google/translategemma-12b-it` is gated, and it is gated **manually** —
request access on huggingface.co and wait for approval, which is not instant.
The other five are open.

The staging script needs nothing from the project environment, so run it without
installing the full dependency set:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
uv run --no-project --with huggingface_hub --with pyyaml \
    python scripts/fetch_offline_assets.py --dest offline_assets/models

tar -I 'zstd -10 -T0' -cf translategemma-models.tar.zst offline_assets/models
```

The script prints every repository it staged and exits non-zero if any failed.
`--dest` becomes `HF_HOME` in the container and must match `MODELS_DIR` in
`.env`; snapshots land in `<dest>/hub`.

### 3.5 Stage the runtime packages (only if the GPU host lacks them)

On a machine matching the GPU host's distro:

```bash
# Debian / Ubuntu
apt-get download nvidia-container-toolkit nvidia-container-toolkit-base \
                 libnvidia-container1 libnvidia-container-tools
# RHEL / Rocky
dnf download --resolve nvidia-container-toolkit
```

Driver: download the matching `NVIDIA-Linux-x86_64-<version>.run` installer from
nvidia.com — it is self-contained and installs without network access.

---

## 4. Phase B — transfer

Copy to removable media:

```
translategemma-cu128-image.tar.zst (or translategemma-cu130-image.tar.zst) ~5-6 GB
translategemma-models.tar.zst     ~30 GB
translategemma-src.tar            (git archive HEAD, a few MB)
data/sft_farsi_science.jsonl      ~140 MB  (corpus, not in the git archive)
nvidia-*.deb / *.rpm / *.run      (only if needed)
```

**`offline_assets/hf_cache` is deliberately not on that list.** It holds only
machine-generated scratch — Triton's JIT kernel cache, Torch inductor artifacts,
matplotlib's font cache, and the tokenized-dataset Arrow cache
(`HF_DATASETS_CACHE`) — all of it reproducible on the target. The Triton kernels
in particular are compiled for the *building* GPU's architecture, so they are
useless or actively wrong on a different card. The container rebuilds them on
first use, which is why the image carries `gcc` (§6.3). Nothing is ever
downloaded into it: every network-sourced artefact goes through
`HF_HOME=/models`. Create it empty on the target and leave it alone.

Size it accordingly on the GPU host. The tokenized cache is the large tenant:
the 2.7M-row corpus produces tens of GB of Arrow shards, one set per worker in
`training.tokenize_num_proc`. It scales with the *measured* mean length (336
tokens), not with `training.max_length`, since only truncated rows are affected
by the cap.

If you want to confirm that on the staging machine before wiping it:

```bash
du -sh offline_assets/hf_cache
find offline_assets/hf_cache -type f | head -20
```

Expect `.triton/`, similar compiled caches, and `datasets/`. Anything resembling
a *model* snapshot there means a library bypassed `HF_HOME` and the transfer
list needs revisiting.

Source snapshot. This directory is its own git repository, so `git archive`
bundles exactly the pipeline and nothing from any parent repo. Ignored paths
(`offline_assets/`, `data/`, run outputs) are excluded automatically because
`git archive` only exports tracked files:

```bash
git archive --format=tar --prefix=translategemma/ HEAD -o translategemma-src.tar
```

The corpus is not in that archive by design — 140 MB+ of JSONL does not belong
in git. Transfer `data/` separately, next to the model tarball.

---

## 5. Phase C+D — install and run (offline GPU host)

### 5.1 Runtime prerequisites

Only if not already present:

```bash
# Default cu128 image: install a matching r570+ driver. The named cu130 image
# instead requires r580.65.06 or newer.
sudo sh NVIDIA-Linux-x86_64-570.xx.xx.run      # driver
sudo dpkg -i ./*.deb                            # container toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify the GPU is visible to containers:

```bash
docker run --rm --gpus all translategemma:cu128-py312 nvidia-smi
```

### 5.2 Unpack

```bash
zstd -d -c translategemma-cu128-image.tar.zst | docker load
tar -xf translategemma-src.tar
cd translategemma
tar -I zstd -xf ../translategemma-models.tar.zst      # creates offline_assets/models

cp .env.example .env
printf 'HOST_UID=%s\nHOST_GID=%s\n' "$(id -u)" "$(id -g)" >> .env

# Both bind targets must exist and be owned by HOST_UID *before* the first run.
# Docker auto-creates missing bind targets as root:root, and the container runs
# as a non-root uid, so anything writing a cache then fails with EACCES.
# hf_cache starts EMPTY on purpose -- it is regenerated scratch, not shipped.
mkdir -p offline_assets/hf_cache offline_assets/models
sudo chown -R "$(id -u):$(id -g)" offline_assets
```

For the named CUDA 13.0 image, load `translategemma-cu130-image.tar.zst` and
set the following in `.env` before using Compose:

```dotenv
IMAGE_TAG=cu130-py312
PYTORCH_CUDA=cu130
```

The `PYTORCH_CUDA` setting only affects image builds; retaining it alongside
the tag makes a later `docker compose build` select the correct image variant.

Also place the corpus, which travels outside the git archive:

```bash
mkdir -p data
cp /media/usb/sft_farsi_science.jsonl data/
```

If the weights live elsewhere on the host (a shared NFS mount, a second disk),
point `MODELS_DIR` in `.env` at that path instead of moving anything.

### 5.3 Smoke-test the environment

```bash
docker compose run --rm trainer python -c \
  "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

On the `cu128-fa3-py312` image, verify the FlashAttention 3 wheel first — this
is the machine that can finally execute it, and it fails in one second rather
than after a model load:

```bash
docker compose run --rm trainer python scripts/verify_flash_attn3.py
```

Then, in order — each step is cheap and fails fast:

```bash
# 1. Config, paths and chat template only. No weights loaded.
docker compose run --rm trainer python train.py --config config.yaml --dry-run

# 2. Every enabled stage, <=10 rows per split, one optimiser step, temp outputs.
docker compose run --rm trainer python train.py --config config.yaml --smoke-test

# 3. Token-length distribution of the real corpus. Decide training.max_length
#    from this before committing to a multi-day run.
docker compose run --rm trainer \
    python scripts/analyze_token_lengths.py --config config.yaml
```

A successful `--smoke-test` proves the model loads from the offline cache, the
configured precision path (`model.use_4bit`, `model.dtype`) runs on this GPU,
Triton compiled the Liger kernels against this architecture, and the evaluation
path is importable.

The first `--smoke-test` after loading a new image is slower than later runs:
Triton is compiling the fused kernels into `/hf_cache/.triton` for this GPU. A
compiler error here, rather than a CUDA error, points at §6.3.

Step 3 has already been run against the current corpus and its conclusions are
baked into `config.yaml`: mean 336 tokens, p95 1072, a thin tail to 6509, so
`training.max_length` stays at 2048 and `training.group_by_length` is on. Re-run
it whenever the corpus changes materially — it is cheap relative to what it
decides, and both of those settings follow from the shape of the distribution
rather than from a default. Full reasoning, including why lowering `max_length`
is *not* the lever it looks like, is in
`docs/2026-08-03_training_speed_tier1_tier2_applied.md`.

The setting that still needs a decision on this host is `training.batch_size`.
It is 4, which underuses an H100 now that the memory fixes have landed; with
grouping on, a typical batch is far narrower than the 2048 cap implies.

### 5.4 Real runs

```bash
docker compose run --rm trainer python split_dataset.py --config config.yaml
docker compose run --rm trainer python train.py       --config config.yaml
docker compose run --rm trainer python evaluate_translations.py \
    --config config.yaml --adapter-path translategemma-farsi-science/sft_final
```

Or take an interactive shell and work normally — `pdb`, re-running scripts after
editing them on the host, inspecting `logs/`:

```bash
docker compose run --rm trainer          # drops into bash in /workspace
```

Because `.` is bind-mounted at `/workspace`, edits made with your host editor
take effect on the next `python …` invocation. Nothing is baked into the image
except dependencies.

For a long run, detach it from your SSH session:

```bash
docker compose run --rm -d --name tg-train trainer \
    python train.py --config config.yaml
docker logs -f tg-train
```

---

## 6. Known issues to expect

### 6.1 Per-GPU memory settings

`model.use_4bit` now defaults to **false** — bf16 LoRA, not QLoRA. On a large
card the 4-bit weights bought VRAM that was not scarce while adding a
dequantisation step to every matmul. That changes the memory picture: bf16
weights are ~24 GB before activations, against ~7 GB for the same model in NF4.

Rough figures at `max_length: 2048` with gradient checkpointing on:

| Configuration | Weights | Typical peak |
| --- | --- | --- |
| `use_4bit: true` (QLoRA) | ~7 GB | 20–26 GB |
| `use_4bit: false` (bf16 LoRA) | ~24 GB | 34–42 GB |

| GPU | VRAM | Guidance |
| --- | --- | --- |
| H100 / H100 NVL | 80–94 GB | Defaults fine. Prefer `use_4bit: false`. Raise `batch_size` and consider `gradient_checkpointing: false`; see the speed document. |
| RTX PRO 6000 Blackwell | 96 GB | Defaults fine. `batch_size` can go to 8–16. |
| RTX 6000 Ada | 48 GB | bf16 LoRA fits but leaves little headroom for long batches. Either lower `batch_size` or set `use_4bit: true`. |
| RTX 5090 | 32 GB | Set `use_4bit: true`. Even then the defaults are close: on OOM use `batch_size: 2` and `gradient_accumulation_steps: 8` — the effective batch is unchanged. |

`training.use_liger_kernel: true` cuts peak memory substantially on top of the
above, because the `batch × seq × 262144` logit tensor is never materialised.
The figures assume it is on. Turning it off may reintroduce OOMs that the
old QLoRA defaults did not hit.

Evaluation loads the base model, then the adapter, then MetricX, then COMET.
On a 32 GB card, run evaluation as a separate step rather than via
`evaluation.run_after_training: true`, so training memory is fully released.

### 6.2 MetricX-24 integration

`evaluate_metricx()` was aligned with the reference implementation
(`metricx24/predict.py`, pinned at commit `fc4978eb`). Four things had to match,
and all four fail *silently* — wrong scores, not exceptions — if changed:

| Aspect | Value | Why |
| --- | --- | --- |
| Module | `metricx24.models` | The checkpoint is MetricX-**24**; `metricx23` is a different input format. |
| Input string | `source: … candidate: … reference: …` | MetricX-24's serialisation. MetricX-23 used a different, source-free one. |
| Trailing EOS | stripped (`[:, :-1]`) | The models were trained on inputs without it. |
| Tokenizer | `google/mt5-xl` | MetricX repos contain `config.json` + `pytorch_model.bin` only — no tokenizer. |

`metricx_max_length` was raised from `1024` (the MetricX-23 value) to `1536`,
which is what MetricX-24 was trained at.

MetricX is vendored into the image as a source checkout at `/opt/metricx` on
`PYTHONPATH`, not pip-installed: the repository has no `pyproject.toml` or
`setup.py`, so `pip install git+…` fails. Its `requirements.txt` is also not
installed — it pins `transformers==4.30.2`, which would wreck the training
environment. Only `metricx24/models.py` is imported.

Verify after loading the image:

```bash
docker compose run --rm trainer python -c \
  "from metricx24.models import MT5ForRegression; print('ok')"
```

If `metricx24/models.py` ever breaks against a newer `transformers` (it
subclasses `MT5PreTrainedModel`), change the commit SHA in the Dockerfile's
`ADD https://codeload.github.com/...` line, or set
`evaluation.metricx_enabled: false` and rely on COMET.

### 6.3 The image needs a C compiler at runtime

Triton (pulled in by `torch`, and now used on every step by
`training.use_liger_kernel`) JIT-compiles GPU kernels **while training runs**,
not at build time, and shells out to a C compiler to build the launcher stubs.
Without it:

```
RuntimeError: Failed to find C compiler. Please specify via CC environment
variable or set triton.knobs.build.impl.
```

This used to be an occasional path. With Liger enabled it is guaranteed: the
fused cross-entropy, RMSNorm, RoPE and SwiGLU kernels are all Triton, and they
compile on the first step of every run whose cache is cold. Symptoms of a
missing or unwritable cache therefore appear at step 0, not at import.

The image therefore installs `gcc` and `libc6-dev`. That is the entire apt
footprint: `ca-certificates` and `tar` come with the base, `torch` and
`scikit-learn` vendor their own `libgomp`, and MetricX is fetched as an HTTPS
tarball rather than cloned, so no `git` is needed. apt is configured to download
serially with retries, because the default parallel pipeline is what draws 502s
from a loaded proxy.

`ps`, `top` and `less` are deliberately absent; use `docker stats` and
`nvidia-smi` from the host.

### 6.4 Offline flags are strict by design

`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` make any missed download raise
immediately instead of hanging on a connection timeout. If a run dies with
"Cannot find the requested files in the disk cache", a repository was missed in
§3.4 — add it with `--repo <id>` and re-stage.

### 6.5 The model mount is writable on purpose

`/models` is **not** mounted read-only. `huggingface_hub` takes file locks under
`<cache>/.locks`, and COMET calls `snapshot_download()` on every evaluation run
rather than reading the cache directly. A read-only mount turns both into
permission errors. `MODELS_DIR` must therefore be writable by `HOST_UID`.

Path wiring, which must agree in three places:

```
fetch_offline_assets.py --dest offline_assets/models   →  writes <dest>/hub
.env          MODELS_DIR=./offline_assets/models       →  mounted at /models
compose       HF_HOME=/models, HUGGINGFACE_HUB_CACHE=/models/hub
```

### 6.6 `bitsandbytes` on Blackwell

The image requires and locks `bitsandbytes>=0.48.0`. Older versions lack sm_120
kernels and fail on RTX 5090 / RTX PRO 6000 with
`no kernel image is available for execution on the device`. Do not pin it lower.

This only bites when `model.use_4bit: true`. With the current bf16 default,
`bitsandbytes` is imported but never asked for a quantised kernel, so a
Blackwell host can appear healthy right up until someone re-enables QLoRA.

### 6.7 FlashAttention 3 lives in a separate image tag

`model.attn_implementation` must remain `sdpa` in the **default** image. Setting
`flash_attention_3` there raises at model load:

```
ImportError: FlashAttention3 has been toggled on, but it cannot be used due to
the following error: the package flash_attn_3 seems to be not installed.
```

FlashAttention 3 is not a major-version upgrade of the `flash-attn` PyPI
package. The official distribution is `flash-attn-3==3.0.0b1`, built from the
upstream repository's `hopper/` directory; the `flash-attn-3` project currently
visible on PyPI is an empty, yanked release and must not be used. The `speed`
extra pins the source to upstream tag `v2.8.3.post1` and compiles it against the
project's Torch. That requires `nvcc` — the full CUDA toolkit, roughly 3 GB,
which `python:3.12-slim` deliberately does not carry.

`uv lock` itself does not require CUDA. `[tool.uv.dependency-metadata]` records
the metadata from the pinned `hopper/setup.py`, preventing uv from executing
that CUDA-sensitive setup script during resolution. The CUDA requirement begins
at `uv sync --extra speed`, when the extension is actually built.

FA3 requires a Hopper-class GPU (H100/H800; compute capability >= 9.0) and CUDA
>=12.3; upstream recommends CUDA 12.8. Keep `sdpa` on Ada/Ampere/Blackwell
hardware — the compiled kernels are sm_90 only.

This costs less than it sounds. On Hopper, SDPA already dispatches to fused
flash-attention kernels; the meaningful gap opens only with packed or
padding-free batches, which this pipeline does not yet do.

**Nothing in the smoke test needs it.** `--dry-run`, `--smoke-test` and the
length analysis all run on `sdpa`. FA3 is an optimisation to benchmark later,
not a prerequisite.

#### The `cu128-fa3-py312` variant

For a Hopper host, build the separate tag rather than changing the default one:

```bash
scripts/build_flash_attn3_wheel.sh                                     # once, 1-3 h
IMAGE_TAG=cu128-fa3-py312 INSTALL_FLASH_ATTN3=1 docker compose build trainer
```

Three deliberate choices there, each of which has a failure mode attached:

**The wheel is built outside the image.** `scripts/build_flash_attn3_wheel.sh`
runs the compile inside `nvidia/cuda:12.8.1-devel-ubuntu22.04` and drops a wheel
into `wheels/`. The Dockerfile then installs that wheel in seconds. Compiling in
a build stage instead would repeat 1–3 hours of nvcc on every rebuild and pull a
~3 GB toolkit into the build. The build host needs Docker and internet; it needs
neither a GPU nor a local toolkit.

**The base image is Ubuntu 22.04, not 24.04.** The wheel is installed into
`python:3.12-slim-bookworm` (glibc 2.36). Ubuntu 22.04 is glibc 2.35, so the
wheel links against an older glibc than the runtime image. Building on 24.04
(glibc 2.39) produces a wheel that fails to load in the shipped image.

**The install is `--no-index --no-deps`, after `uv sync --locked`.** The locked
environment is the validated one and nothing may re-resolve it. FA3's runtime
dependencies — torch, einops, packaging, ninja — are already in `uv.lock`, so
`--no-deps` drops nothing. The consequence is that FA3 is *out of lock* by
design; `/opt/resolved-requirements.txt` is written afterwards and records it,
so the shipped image still describes itself accurately.

`TORCH_VERSION` and `FA3_TAG` in the script duplicate the pins in
`pyproject.toml`. Change all of them together. A wheel built against a different
Torch installs cleanly and then fails at import with an undefined-symbol error,
which is much harder to diagnose than a version mismatch.

The image is a superset of the default one and still defaults to `sdpa`, so it
is safe to run everything through it on a Hopper host. Flip
`model.attn_implementation: "flash_attention_3"` per run and benchmark it
against `sdpa` — measuring non-padding tokens/second, not samples/second —
rather than assuming it wins.

#### Verifying the wheel when the build host is not a Hopper machine

This is the normal case: the wheel is compiled on a machine with a CUDA toolkit
and internet, and executed on an H100 that has neither. The build host cannot
run the kernels it just produced, so the build "succeeding" proves very little.

Two failure modes are silent until the image reaches the H100, and each costs a
full rebuild-and-transfer cycle to fix:

| Failure | Where it comes from | How it looks on the H100 |
| --- | --- | --- |
| glibc mismatch | wheel built on a newer distro than `bookworm` | `version 'GLIBC_2.38' not found` at import |
| Torch ABI mismatch | wheel built against a Torch other than the pinned one | `undefined symbol: _ZN3c10...` at import |
| Wrong architecture | kernels compiled without `sm_90a` | `no kernel image is available for execution on the device` at the first attention call |

The first two are **fully detectable without a GPU**, because they are dynamic
linking problems, not execution problems. Do not defer them to the H100.

Run this on the build host, immediately after building the image and *before*
`docker save`:

```bash
docker run --rm -v "$PWD/scripts:/scripts:ro" \
    translategemma:cu128-fa3-py312 python /scripts/verify_flash_attn3.py
```

`scripts/verify_flash_attn3.py` splits its checks by what they require. On a
non-Hopper host it loads the compiled extension, resolves its symbols against
the installed Torch, and reports the kernel checks as `SKIP` rather than
failing. A `PASSED, with N check(s) skipped` line there means the image is worth
transferring; a `FAILED` line means rebuild now, while the toolkit is still in
reach.

Also confirm the binary actually contains Hopper kernels. This needs
`cuobjdump`, which lives in the CUDA toolkit image rather than the training
image, so run it against the wheel directly:

```bash
docker run --rm -v "$PWD/wheels:/w:ro" nvidia/cuda:12.8.1-devel-ubuntu22.04 \
    bash -c 'cd /tmp && python3 -m zipfile -e /w/flash_attn_3-*.whl . \
             && cuobjdump --list-elf $(find . -name "*.so" | head -1) | sort -u'
```

Every listed entry should be `sm_90a`. An empty listing, or one naming only
other architectures, is the third failure mode caught before transfer instead of
after.

What genuinely cannot be checked before the H100:

- that a kernel launches and returns correct numbers,
- that `transformers.utils.is_flash_attn_3_available()` returns `True` (the
  probe inspects the current device, so it is legitimately `False` on the build
  host — the script reports this as `SKIP`, not `FAIL`),
- throughput.

So re-run the same script on the offline host after loading the image, where it
executes every tier:

```bash
docker compose run --rm trainer python scripts/verify_flash_attn3.py
```

There it launches `flash_attn_func` on a small causal batch and compares the
output against SDPA. Comparing rather than merely catching exceptions matters: a
numerically wrong build produces plausible-looking training that quietly
converges worse, which is far more expensive than a crash.

Only after that is clean, set `model.attn_implementation: "flash_attention_3"`
and run the normal `--smoke-test` (§5.3) before committing to a long run.

#### Building the extra directly on a host that has a toolkit

`uv sync --extra speed` remains available for a developer environment with a
local CUDA toolkit, and is what the wheel build performs in miniature. Two
distinct failures, in the order they appear:

```
ModuleNotFoundError: No module named 'torch'
```

flash-attn-3's `setup.py` imports torch, packaging, wheel and ninja before its
runtime dependencies can be processed, so an isolated PEP 517 build does not
otherwise contain them.
`pyproject.toml` now supplies it via `[tool.uv.extra-build-dependencies]` with
an explicit `torch==2.8.0` build pin identical to the runtime pin. Do not replace
this with `match-runtime = true`: FA3 exposes dynamic rather than static package
metadata, so uv cannot determine the runtime requirement early enough and aborts
before compilation. Building against a different Torch can succeed and then fail
at import with an undefined-symbol error, which is much harder to diagnose.
Whenever the project Torch pin changes, update all three occurrences together:
`[project.dependencies]`, `[tool.uv.extra-build-dependencies]`, and
`TORCH_VERSION` in `scripts/build_flash_attn3_wheel.sh`.

```
RuntimeError: The current installed version of nvcc ... / nvcc: not found
```

The torch wheels vendor the CUDA *runtime*, not the compiler. Install a CUDA
toolkit whose major version matches the wheels (12.x for `cu128`) and ensure
`nvcc --version` works before retrying. This is not fixable from `pyproject.toml`.

The build compiles a large set of CUDA kernels: expect 1–3 hours, and cap
parallelism so it is not OOM-killed on a machine with limited RAM.

```bash
uv lock                              # works without a CUDA toolkit or GPU
nvcc --version                       # must report CUDA >=12.3 before sync
MAX_JOBS=4 uv sync --extra speed
```

Only then set `model.attn_implementation: "flash_attention_3"`, and benchmark it
against `sdpa` rather than assuming it wins. Note that this path installs into
the *developer* environment only; the offline image gets FA3 through the wheel
route above, never through `uv sync --extra speed`.

### 6.8 vLLM serving

The vLLM command in `README.md` is not part of this image — `vllm` is not in
`pyproject.toml`. Serving offline needs its own image, staged the same way.

### 6.9 Transformers version boundary

The image pins `transformers>=4.57.6,<5.0`. Without the upper bound, a fresh
build can resolve Transformers v5 while other training dependencies still use
v4 import paths, producing a misleading `Could not import module
'BloomPreTrainedModel'` error during `import peft`. This is unrelated to CUDA,
the NVIDIA driver, or the staged model files. Keep the v4 bound until the full
training dependency set is validated against Transformers v5.

The dependency set also pins `pyarrow<21.0.0`: `unbabel-comet` currently
resolves `datasets` 2.14.x, whose extension APIs predate newer PyArrow
releases. Do not remove that bound independently; upgrade and validate
`datasets`/COMET together first.

`unbabel-comet` also resolves `torchmetrics` 0.10.x through PyTorch Lightning.
That release imports `pkg_resources`, which Setuptools 82+ no longer ships, so
the image pins `setuptools<82.0.0`. Removing it produces
`ModuleNotFoundError: No module named 'pkg_resources'` when importing COMET.

---

## 7. Build-time troubleshooting

Failures seen while building this image on a proxied WSL/Docker Desktop host.

### 7.1 Building through a proxy

The staging build needs ~4 GB from PyPI, `download.pytorch.org`, `deb.debian.org`
and `codeload.github.com`. The default download is `cu128`; a deliberate CUDA
13.0 build downloads from `cu130`. Symptoms and meanings differ:

| Message | Meaning |
| --- | --- |
| `dial tcp <ip>:<port>: connection attempt failed` | Proxy configured but unreachable from the build network. |
| `502 Bad Gateway [IP: 192.168.65.254 …]` | Proxy reached; **its** upstream failed, usually under parallel load. |
| `failed to resolve source metadata` / `manifest unknown` | Registry lookup, not proxy — check the image tag. |

`192.168.65.254` is Docker Desktop's internal gateway, so seeing it confirms the
proxy is being used. Set it per build with predefined BuildKit args, which are
**not** baked into the image:

```bash
docker compose build \
  --build-arg HTTP_PROXY=http://host.docker.internal:PORT \
  --build-arg HTTPS_PROXY=http://host.docker.internal:PORT \
  --build-arg NO_PROXY=localhost,127.0.0.1 \
  trainer
```

From WSL with a proxy on the Windows host, use the default gateway
(`ip route show default | awk '{print $3}'`) rather than the LAN address, and
allow the port through the Windows Firewall. Note that apt cannot use a SOCKS
proxy through `HTTP_PROXY`; port 1080 is usually SOCKS, so find the proxy's
separate HTTP/mixed port.

The Dockerfile already tells apt to fetch serially with retries
(`Acquire::Queue-Mode=access`, `Acquire::Retries=5`), which is what makes the
502-under-load case survivable. If the build still dies during the ~3 GB torch
download, the proxy is the wall — build on a machine with direct internet. The
result ships as a tarball either way, so the build host is not constrained.

### 7.2 `Permission denied` on `/hf_cache` or `/models`

```
PermissionError: [Errno 13] Permission denied: '/hf_cache/.triton'
```

Docker creates missing bind-mount targets as `root:root`, and the container runs
as `HOST_UID:HOST_GID`. Create and chown both directories before the first run
(§5.2). Triton's kernel cache and Hugging Face's lock files both need to write.

On a Windows drive under WSL (`/mnt/c/...`), `chown` does not stick because
ownership comes from the drvfs mount options. Move the project into the WSL
filesystem — which is also much faster for I/O — rather than setting
`HOST_UID=0`, since running as root leaves root-owned adapters and logs behind.

### 7.3 `failed to parse stage name`

A variable in a `COPY --from=` stage name is expanded by BuildKit but not by the
classic builder. The Dockerfile hardcodes the `uv` image tag to avoid this
entirely. If you reintroduce a variable there, ensure BuildKit is active
(`DOCKER_BUILDKIT` unset or `1`).
