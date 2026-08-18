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
This default image cannot run the checked-in packed recipe. To use it, set
`training.packing: false` and `model.attn_implementation: sdpa`. Use the
Hopper-only FA3 variant in §6.7 for the production packed configuration.

This builds the default `translategemma:cu128-py312` image. To deliberately
build the retained CUDA 13.0 image for an r580+ host:

```bash
IMAGE_TAG=cu130-py312 PYTORCH_CUDA=cu130 docker compose build trainer
```

To build the FlashAttention 3 variant for a Hopper host, first fetch the pinned
matching wheel, then build against it:

```bash
scripts/fetch_flash_attn3_wheel.sh
IMAGE_TAG=cu128-fa3-py312 INSTALL_FLASH_ATTN3=1 docker compose build trainer
```

The fetch is resumable and verifies a pinned SHA-256. The wheel is a reviewed
community build, not an official Dao-AILab binary. Use
`scripts/build_flash_attn3_wheel.sh` to compile the pinned official source when
third-party binaries are unacceptable or an ABI pin changes. The image build
requires exactly one wheel, validates its Torch/CUDA/package versions, and runs
the no-GPU import/ABI checks before completing.

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
`pyproject.toml`, `uv.lock`, `wheels/`, and the FA3 verifier, so rebuilds after
a dependency change are the only slow ones. The default `cu128` build uses
`uv sync --locked`; it fails if `uv.lock` does not match `pyproject.toml`
rather than silently resolving newer packages.

Sanity-check the resolved environment (no GPU needed):

```bash
docker run --rm translategemma:cu128-py312 \
  python -c "import torch, peft, trl, bitsandbytes, liger_kernel, sacrebleu; print(torch.__version__, torch.version.cuda, sacrebleu.__version__)"
docker run --rm translategemma:cu128-py312 cat /opt/resolved-requirements.txt
```

`torch.version.cuda` must print `12.8`. For the named CUDA 13.0 image, replace
the tag in both commands with `cu130-py312`; it must print `13.0`.

Translation-benchmark compatibility by component:

| Capability | Docker image status | Additional offline requirement |
| --- | --- | --- |
| CSV/JSON import, BLEU, chrF++, preservation metrics, HTML report | Included | Frozen dataset and imported files under `/workspace` |
| TranslateGemma generation | Included through Transformers/PEFT | Every enabled size staged under `/models`; local LoRA directory under `/workspace` |
| NLLB generation/fine-tuning comparison | Included through Transformers/PEFT | Enabled NLLB checkpoint staged under `/models`; adapter transferred or staged |
| COMET scoring | Included | COMET checkpoint and indirect XLM-R tokenizer/config staged |
| MetricX scoring | Included when `INSTALL_METRICX=1` | MetricX checkpoint and mT5 tokenizer staged |

Thus the existing image design is compatible with the benchmark. The image
provides code dependencies; §3.4 is what makes each configured model and metric
actually runnable without a network.

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

The required repository count follows the enabled training, test-set, and
multi-model benchmark configuration. `scripts/fetch_offline_assets.py` reads
IDs from `config.yaml`, `testset_config.yaml`, and `benchmark_config.yaml`; the
indirect evaluator dependencies are resolved automatically. Enable every
generated benchmark candidate that must be runnable on the offline host before
staging. Disabled candidates are intentionally skipped to avoid transferring
unused multi-gigabyte checkpoints; use repeated `--repo` arguments when a
disabled candidate must still travel.

| Repository | Used by | Staged | Approx. size |
| --- | --- | --- | --- |
| `google/translategemma-12b-it` | training, quick evaluation, and configured benchmark candidates | full | ~24 GB |
| Other enabled TranslateGemma sizes | `benchmark_translations.py` | full | size-dependent |
| Enabled NLLB checkpoints | `benchmark_translations.py` | full | size-dependent |
| `google/metricx-24-hybrid-large-v2p6` | quick evaluation or benchmark MetricX | full | ~4.9 GB |
| `google/mt5-xl` | MetricX tokenizer (`metricx_tokenizer_id`) | tokenizer only | ~20 MB |
| Configured COMET checkpoint (`Unbabel/XCOMET-XL` in `config.yaml`; `Unbabel/wmt22-comet-da` for the benchmark default) | quick evaluation or benchmark COMET | full | ~2.3 GB (`wmt22-comet-da`), ~14 GB (`XCOMET-XL`) |
| That checkpoint's encoder — `facebook/xlm-roberta-xl` for `XCOMET-XL`, `xlm-roberta-large` for `wmt22-comet-da` | COMET's encoder — **indirect**, see below | tokenizer only | ~20 MB |
| `sentence-transformers/LaBSE` | `build_test_set.py` embeddings | full | ~1.8 GB |

Weights are skipped for the two tokenizer-only repositories: `google/mt5-xl`
alone would otherwise pull a 15 GB checkpoint that is never loaded.

**The COMET encoder is the easiest thing to miss.** `load_from_checkpoint()`
constructs an XLM-R encoder and calls `XLMRobertaTokenizerFast.from_pretrained()`
plus `XLMRobertaConfig.from_pretrained()` on whatever `hparams.yaml` names in
`pretrained_model` (`facebook/xlm-roberta-xl` for `XCOMET-XL`,
`xlm-roberta-large` for `wmt22-comet-da`). **Changing `comet_model_id` changes
this repository too**, so a models tarball staged for one checkpoint is
incomplete for another even though the checkpoint itself downloads fine. The encoder
*weights* are not fetched — COMET passes `load_pretrained_weights=False` — but
the tokenizer and config are, and an offline run dies without them. The script
reads that id from the downloaded checkpoint's `hparams.yaml`, so it stays
correct if you switch COMET models. It stages the id **literally as COMET
requests it**: the hub cache is keyed by the exact string, so staging the
canonical alias `FacebookAI/xlm-roberta-large` would miss.

TranslateGemma repositories may require manual access approval on
huggingface.co. Request access for every configured size before staging and
export an authorized `HF_TOKEN`. NLLB and the listed evaluator repositories are
open at the time of writing.

The staging script needs nothing from the project environment, so run it without
installing the full dependency set:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
uv run --no-project --with huggingface_hub --with pyyaml \
    python scripts/fetch_offline_assets.py \
    --config config.yaml \
    --testset-config testset_config.yaml \
    --benchmark-config benchmark_config.yaml \
    --dest offline_assets/models

tar -I 'zstd -10 -T0' -cf translategemma-models.tar.zst offline_assets/models
```

The script prints every repository it staged and exits non-zero if any failed.
`--dest` becomes `HF_HOME` in the container and must match `MODELS_DIR` in
`.env`; snapshots land in `<dest>/hub`. Explicit `processor`, `tokenizer`, and
`adapter_repo` values on enabled generated candidates are also staged, including
their configured model/adapter revision pins. A local
`adapter` path is not a Hub asset and must travel with the run artifacts.

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
data/.../frozen_test.jsonl         evaluation set (if not already in data/)
translategemma-farsi-science/...  local LoRA adapters used by benchmark candidates
existing_translations/...         imported model outputs, if configured
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

Place the frozen benchmark dataset, local LoRA adapters, and imported
translation files at the paths configured in `benchmark_config.yaml`. All must
live inside the project tree (or another explicitly added bind mount) so they
appear below `/workspace` in the container. The default
`benchmark_output/` directory also lives there, so CSV and HTML reports remain
on the host after each `docker compose run --rm` container exits.

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
# 0. Import every multi-model benchmark runtime dependency without loading
#    model weights. This checks TranslateGemma/NLLB runners, CSV/HTML reporting,
#    transparent metrics, COMET, and the vendored MetricX source.
docker compose run --rm trainer python -c \
  "import pandas, sacrebleu, torch, yaml; from comet import download_model, load_from_checkpoint; from metricx24.models import MT5ForRegression; from peft import PeftModel; from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoProcessor, AutoTokenizer; import translation_benchmark.config, translation_benchmark.generation, translation_benchmark.io, translation_benchmark.metrics, translation_benchmark.pipeline, translation_benchmark.report; print('translation benchmark dependency preflight: OK')"

# 1. Config, paths and chat template only. No weights loaded.
docker compose run --rm trainer python train.py --config config.yaml --dry-run

# 2. Every enabled stage, <=10 rows per split, one optimiser step, temp outputs.
docker compose run --rm trainer python train.py --config config.yaml --smoke-test

# 3. Token-length distribution of the real corpus. Decide training.max_length
#    from this before committing to a multi-day run.
docker compose run --rm trainer \
    python scripts/analyze_token_lengths.py --config config.yaml

# 4. Benchmark config, frozen dataset columns/IDs, and candidate definitions.
#    No translation or evaluator weights are loaded.
docker compose run --rm trainer python benchmark_translations.py \
    --config benchmark_config.yaml validate
```

The dependency preflight assumes the standard `INSTALL_METRICX=1` image. If the
image was deliberately built with `INSTALL_METRICX=0`, remove only
`from metricx24.models import MT5ForRegression` from that command and keep
`metrics.metricx.enabled: false` in `benchmark_config.yaml`. An import failure
here is an image/dependency problem; a missing Hugging Face snapshot appears
later during generation or scoring and points to the staging procedure in
§3.4.

A successful `--smoke-test` proves the model loads from the offline cache, the
configured precision path (`model.use_4bit`, `model.dtype`) runs on this GPU,
Triton compiled the Liger kernels against this architecture, and the evaluation
path is importable.

The first `--smoke-test` after loading a new image is slower than later runs:
Triton is compiling the fused kernels into `/hf_cache/.triton` for this GPU. A
compiler error here, rather than a CUDA error, points at §6.3.

Step 3 has already been run against the current corpus and its conclusions are
baked into `config.yaml`: mean 336 tokens, p95 1072, a thin tail to 6509, so
`training.max_length` stays at 2048. Length grouping is disabled because its
sampler stalled production startup; rank zero now builds and caches BFD-packed
training blocks instead. Re-run the analysis whenever the corpus changes
materially. Full reasoning, including why lowering `max_length` is *not* the
lever it looks like, is in
`docs/2026-08-03_training_speed_tier1_tier2_applied.md`.

The checked-in H200 starting point is a per-device batch of 6 and effective
global batch of 48. Packing makes blocks close to 2048 tokens, so validate peak
memory before raising the per-device batch.

### 5.4 Real runs

```bash
docker compose run --rm trainer python split_dataset.py --config config.yaml
docker compose run --rm trainer python train.py       --config config.yaml
# Evaluation features Rich progress bars and line-flushed crash-proof resumption
docker compose run --rm trainer python evaluate_translations.py \
    --config config.yaml --adapter-path translategemma-farsi-science/sft_final

# Multi-GPU data-parallel evaluation. Each rank keeps its own resumable cache
# shard, so an interrupted run continues where it stopped.
docker compose run --rm trainer accelerate launch \
    --config_file accelerate_configs/h200_8gpu.yaml evaluate_translations.py \
    --config config.yaml --adapter-path translategemma-farsi-science/sft_final

# CPU-only. Renders evaluation/report.html: metric summary, per-domain
# breakdown, and base-versus-adapter translations side by side for review.
docker compose run --rm trainer python report_evaluation.py --config config.yaml
```

Run the final multi-model benchmark as separate containers. This guarantees
that translation-model VRAM is released before learned evaluators are loaded:

```bash
# Generate enabled TranslateGemma/NLLB candidates and canonicalize enabled
# imported outputs. Existing candidate artifacts with matching hashes are reused.
docker compose run --rm trainer python benchmark_translations.py \
    --config benchmark_config.yaml collect

# Score all collected outputs. Enable COMET/MetricX in benchmark_config.yaml
# only when their checkpoints were staged in §3.4.
docker compose run --rm trainer python benchmark_translations.py \
    --config benchmark_config.yaml score

# CPU-only report rendering; writes report.html plus CSV/Markdown artifacts.
docker compose run --rm trainer python benchmark_translations.py \
    --config benchmark_config.yaml report
```

For models with different VRAM requirements, collect one candidate at a time:

```bash
docker compose run --rm trainer python benchmark_translations.py \
    --config benchmark_config.yaml generate \
    --candidates translategemma-12b-base
docker compose run --rm trainer python benchmark_translations.py \
    --config benchmark_config.yaml generate \
    --candidates nllb-600m-base
docker compose run --rm trainer python benchmark_translations.py \
    --config benchmark_config.yaml import
```

Generation can use the same Accelerate profiles as training. For example, to
split one candidate's evaluation rows over four GPUs:

```bash
docker compose run --rm trainer accelerate launch \
    --config_file accelerate_configs/h200_4gpu.yaml \
    benchmark_translations.py --config benchmark_config.yaml generate \
    --candidates translategemma-finetuned
```

The 2-, 4-, and 8-GPU profiles provide single-node data-parallel inference:
one complete model replica per GPU, deterministic row shards, and rank-zero
assembly. They improve throughput but do not pool VRAM for an oversized model.
Run `score` and `report` afterward without Accelerate.

Open `benchmark_output/report.html` directly from the host. The companion
`all_model_outputs.csv` contains one aligned translation column per candidate.
The complete benchmark configuration and artifact contract are documented in
`docs/TRANSLATION_BENCHMARK.md`. The complete four-candidate production command
sequence is in `docs/TRANSLATION_BENCHMARK_RUNBOOK.md`.

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

### 5.5 Merge the adapter into the base model (for vLLM)

`scripts/merge_lora_adapter.py` folds a trained LoRA adapter into the base
weights and writes an ordinary Hugging Face model directory. vLLM can also serve
the adapter unmerged with `--enable-lora`, so this is a choice, not a
requirement; merging removes the per-request LoRA path and the `--max-lora-rank`
plumbing, at the cost of one full-size checkpoint on disk per adapter.

It runs in the training image, needs no GPU, and takes two positional arguments —
the adapter directory and the output directory:

```bash
docker compose run --rm trainer python scripts/merge_lora_adapter.py \
    translategemma-farsi-science/sft_final \
    merged/translategemma-farsi-sft
```

Write the output inside the project tree (or another bind mount) so it survives
the container. Budget the full base-model size for it: ~24 GB for
`translategemma-12b-it` in bf16, on top of the staged copy under `/models`.

The base model id comes from the adapter's `adapter_config.json`; pass
`--base-model` to override it. Everything else has a working default:

| Flag | Default | When to change it |
| --- | --- | --- |
| `--device` | `cpu` | `cuda:0` when host RAM is tighter than VRAM; `auto` only when no single GPU can hold the model. |
| `--dtype` | `bfloat16` | Rarely. Must be a full-precision dtype — see below. |
| `--attn-implementation` | `eager` | Rarely. Irrelevant to the merge; the default avoids needing a FlashAttention build on the merging host. |
| `--max-shard-size` | `5GB` | To match a serving host's preferred shard size. |
| `--overwrite` | off | To rewrite a non-empty output directory. |

**Run it as a single process, never under `accelerate launch`.** Merging is an
elementwise weight update, so extra ranks would each repeat the whole merge and
race to write the same directory. More GPUs do not make it faster; `--device
auto` exists only to spread an oversized model across cards, which a 12B model
on an H100/H200 does not need.

**The merge is always done in full precision, never against 4-bit weights**, so
a run trained with `model.use_4bit: true` still merges into a bf16 base here.
Merging into a quantised base would dequantise, add, and re-quantise, losing part
of the adapter delta. Quantise the merged checkpoint afterwards if the serving
host needs it.

Besides the weights, the output directory receives two things vLLM depends on:

- the **processor** — tokenizer, chat template and preprocessor config — because
  vLLM loads the tokenizer from the model directory and does not know the base
  repository it came from;
- a **`generation_config.json`** built by
  `model_loading.make_deterministic_generation_config`, whose stop set includes
  `<end_of_turn>` (106). `config.json` alone publishes only `<eos>` (1), and a
  decoder missing 106 does not stop a fine-tuned model at all
  (`docs/2026-08-10_adapter_degeneration_analysis.md`).

It also copies the adapter's `adapter_config.json` and `run_metadata.json` under
an `adapter_` prefix and writes `merge_metadata.json` recording the source
adapter, base model, dtype and LoRA hyperparameters — keep those with the
checkpoint, since a merged model is otherwise indistinguishable from the base.

Sanity-check the result before shipping it. The merged directory is loadable by
the same evaluation path as the base model, so a quick benchmark candidate with
no `adapter` set is the cheapest end-to-end proof that the deltas actually
landed.

### 5.6 Quantise the merged checkpoint to FP8 (optional)

Only needed when the serving GPU has to hold something else as well. A 12B merge
is ~24 GB in bf16; FP8 halves that, which is the difference between fitting
beside a second model on a 32 GB card and not fitting at all.
`scripts/quantize_fp8.py` writes a compressed-tensors checkpoint vLLM detects
from its `config.json` — no serving flag to set.

**Order matters: merge first, quantise second.** Merging a LoRA adapter into
already-quantised weights rounds the delta away — a LoRA update is small enough
per channel to disappear into the quantisation step — so it would cost exactly
the fine-tune the merge exists to preserve. The script refuses an adapter
directory or an already-quantised input for this reason.

The scheme is `FP8_DYNAMIC`, which needs **no calibration corpus**. That is the
main reason to prefer it over 4-bit AWQ/GPTQ here: those pick their scales from
sample activations, and a general-purpose calibration set biases a
domain-fine-tuned model away from its domain.

#### The quantiser is a separate image

It cannot go in the training image. Measured against this repository's lock
(`torch==2.8.0`, `transformers==4.57.6`), no llm-compressor release installs:

| llmcompressor | requires |
| --- | --- |
| 0.8.x | `transformers <=4.56.2` |
| 0.9.x | `transformers <=4.57.3` |
| 0.10.x | `torch >=2.9.0` |
| 0.11.0 | `torch >=2.10.0` |
| 0.12+ | `transformers >=5.9.0` |

That is not bad luck on either side: llm-compressor tracks vLLM's release
cadence, while the training stack is frozen on the versions the evaluation
harness was validated against and pinned below transformers v5 (§6.9). Forcing
them together would mean moving `torch` or `transformers` under the trainer,
which is what `uv.lock` exists to prevent.

`scripts/quantize.Dockerfile` builds from the vLLM image instead, which already
carries the torch llm-compressor 0.10.x wants — it is the torch vLLM itself was
built against.

#### Staging it (online machine)

`docker compose build quantizer` needs network for the `pip install`, so it
belongs in Phase A alongside §3.2, and the image travels with the others:

```bash
docker compose build quantizer
docker save translategemma-quantizer:vllm0.13.0 | zstd -T0 -19 -o quantizer-image.tar.zst
```

Load it on the offline host exactly as §5.2 loads the trainer image. Nothing
resolves at run time.

#### Running it (offline GPU host)

```bash
docker compose run --rm quantizer --device cuda:0 \
    /models/translategemma-12b-merged \
    /models/translategemma-12b-merged-fp8
```

`--device` defaults to `cpu` so the quantiser never competes with a GPU that is
serving; pass `cuda:0` when the card is free. The transform is a weight
operation, not a forward pass, so CPU is a real option rather than a fallback.

Two environment settings the compose file already supplies, both worth knowing
because the failures are opaque:

- **`USER` / `LOGNAME`.** compressed-tensors decorates a class *body* with
  `@torch.compile`, so torch's inductor initialises during import and calls
  `getpass.getuser()`. The container runs as a bare numeric uid (§5.2) and the
  vLLM image has no `/etc/passwd` entry for it, so the `pwd` lookup raises
  `KeyError: getpwuid(): uid not found: 1000` before the script reaches its
  first line of work. `getpass` reads these two variables before consulting
  `pwd`.
- **`TORCHINDUCTOR_CACHE_DIR` / `TRITON_CACHE_DIR`.** The same failure from the
  other side — `getuser()` is only called to *build* a default cache path — and
  they keep the compile caches on the `/hf_cache` scratch mount.

If the compose file on the offline host predates those settings, override them
for one run without editing anything:

```bash
docker compose run --rm \
    -e USER=quantizer -e LOGNAME=quantizer \
    -e TORCHINDUCTOR_CACHE_DIR=/hf_cache/torchinductor \
    -e TRITON_CACHE_DIR=/hf_cache/triton \
    quantizer --device cuda:0 \
    /models/translategemma-12b-merged /models/translategemma-12b-merged-fp8
```

`/hf_cache` now gets written where it may not have been before, so re-check the
ownership rule from §5.2: both bind targets must be owned by `HOST_UID`.

One more failure worth naming, because the message points at the wrong thing.
Saving a model that llm-compressor dispatched across devices with CPU offload
raises a bare

```
KeyError: 'vision_tower.vision_model.embeddings.patch_embedding.weight'
```

from inside `transformers.save_pretrained`. llm-compressor's save wrapper runs
`to_accelerate(model)` — "for optimal saving with transformers" — immediately
before delegating, which converts the model to accelerate's offloaded form.
transformers then takes its offloaded save path: it builds a module map so it
can re-gather offloaded weights, and looks every state-dict key up in it. Only
the modules the conversion touched are in that map.

The failing weight is a **Conv2d**, and `QuantizationModifier` targets Linear,
so the vision tower's patch embedding is skipped no matter what `--ignore` says.
Neither `--device cpu` nor quantising the vision tower avoids this: the offload
is applied inside the save call, not inherited from how the model was loaded.

`disable_offload_conversion()` neutralises that step. It is a memory
optimisation, not a correctness requirement, and it buys nothing here because
the model is already materialised in host RAM.

It patches **both** halves of the conversion. `to_accelerate` and
`from_accelerate` are a matched pair, and skipping only the first leaves the
second trying to offload modules that compression already offloaded:

```
ValueError: Attempted to offload a module twice.
```

The restoring half only matters to a caller that keeps using the model in
memory after saving, which this script does not. Verified against llmcompressor
0.10.0.3 with transformers 4.57.3 — the versions this image resolves to — and
any hook a newer release renames is simply left unpatched.

#### What the script guarantees

The stop-token contract from `docs/2026-08-10_adapter_degeneration_analysis.md`
survives quantisation, and the script fails rather than let it not:

- the tokenizer, chat template and preprocessor configs are **copied byte for
  byte** from the merged directory, never re-saved. This image runs a different
  `transformers` than the API and the harness do, and a chat template
  re-serialised by another version — one space different in the assistant-turn
  prefix — still returns fluent Farsi;
- `generation_config.json` is copied if the save did not write one, and the run
  **fails** if its `eos_token_id` does not contain `<end_of_turn>` (106);
- the saved `config.json` is re-read and the run fails if it carries no
  `quantization_config`, since vLLM would then load the weights as dense;
- `lm_head`, the vision tower and the multimodal projector are left in full
  precision (`--ignore` to change), and `quantization_metadata.json` records the
  scheme, the ignore list and the source directory.

**Score it before serving it.** FP8 is near-lossless in general, which says
nothing certain about this adapter's terminology gains. Run §5.4's evaluation
against the FP8 directory and diff COMET/chrF against the bf16 merge on the same
test set.

#### Status: the FP8 serving path is unvalidated

As of 2026-08-17, no FP8 checkpoint has been served successfully. What is known:

- The FP8 checkpoint hits the **same vLLM 0.13.0 RoPE bug** as the bf16 merge and
  fails to load with `rope_parameters should have a 'rope_type' key` (see
  `docs/DEPLOYMENT_BACKLOG.md`). It was reproduced on both checkpoints, whose
  configs differ only in `transformers_version`.
- The fix is presumably the same overlay — `scripts/vllm_rope_shim.py` — but that
  has **not** been run against an FP8 checkpoint. The shim refuses inputs whose
  `rope_parameters` mixes layer types with other keys, and an FP8 config carries a
  `quantization_config` block that has not been checked against it.
- The bf16 merge is the only configuration measured
  (`docs/2026-08-17_serving_ab_vllm_vs_transformers.md`), and it was measured
  alone on a free GPU — not in the shared-GPU arrangement FP8 exists for.

So the shared-GPU deployment in `docker-compose.spadana.yml` is sized for a
checkpoint nobody has started. Before relying on it: shim the FP8 directory,
confirm `get_config()` accepts it, start it beside the second model, and check
both fit — then score the output, since FP8 is a different numerical path and the
A/B says nothing about it.

---

### 5.7 The serving stack (`docker-compose.spadana.yml`)

MySQL, a Spring backend, phpMyAdmin, two vLLM servers and the two API gateways
in front of them. Images are built elsewhere; that file only wires them
together. It carries no comments by design — everything explaining it is here.

#### Required environment

Copy `.env.spadana.example` to `.env` beside the compose file; it is the
annotated form of this table and of the gateway settings below it.

| Variable | Meaning |
| --- | --- |
| `TG_MERGED_MODEL_DIR` | **A rope overlay built by `scripts/vllm_rope_shim.py`**, not the merge itself. See below. |
| `GPU_DEVICE_ID` | Which physical device both vLLM servers divide. Default `0`. |
| `TG_GPU_MEMORY_UTILIZATION` / `DOTS_GPU_MEMORY_UTILIZATION` | Fractions of **total** VRAM. Default `0.55` / `0.30`. |
| `TG_MAX_MODEL_LEN` | KV-cache budget. Default `8192`. |
| `VLLM_DTYPE` | vLLM's `--dtype`. Default `bfloat16`. Not a gateway setting — the gateway loads no weights. |
| `SERVED_TG_MODEL_NAME` / `SERVED_DOT_MODEL_NAME` | vLLM's `--served-model-name`; the gateway's `TG_VLLM_MODEL` must match. |
| `MYSQL_ROOT_PASSWORD` / `MYSQL_DATABASE` | Default `root` / `fundamental`. |
| `HUGGING_FACE_HUB_TOKEN` | Optional; both vLLM servers run with `HF_HUB_OFFLINE=1` anyway. |

#### `TG_MERGED_MODEL_DIR` must be a rope overlay

vLLM 0.13.0 cannot load a Gemma 3 config written by Transformers 4.57.
`patch_rope_parameters` injects `rope_theta` into the top level of the nested
per-layer rope block, which then fails its own `ALLOWED_LAYER_TYPES` test, and
the server exits before loading weights with `rope_parameters should have a
'rope_type' key`. Build an overlay — it hardlinks the weights and rewrites
`config.json` only:

```bash
python scripts/vllm_rope_shim.py <merge> <merge>-vllm
```

Both TranslateGemma services mount the same variable. That is correct: the
overlay changes `config.json` only, so its tokenizer is the merge's tokenizer,
and the gateway must read the *same* checkpoint vLLM serves or prompts are
rendered with one tokenizer and generated by another. The gateway loads no
weights — it reads the tokenizer and chat template, nothing else.

Full background in `docs/DEPLOYMENT_BACKLOG.md`; re-check whether the overlay is
still needed after any vLLM upgrade.

#### Sharing one GPU between two models

Both vLLM servers reserve the same device, pinned by `device_ids` rather than
`count: all` so the two agree on which card they are dividing instead of leaving
it implicit. Their `--gpu-memory-utilization` fractions are of **total** VRAM,
not of what is free, which is what stops two servers starting in any order from
claiming the same bytes. They must sum to comfortably below 1.0.

Sized for a 32 GB 5090 serving an FP8 TranslateGemma beside dots.ocr:

| Service | Fraction | Budget | Contents |
| --- | --- | --- | --- |
| translategemma | 0.55 | 17.6 GB | ~13 GB weights, ~4 GB KV cache |
| dots.ocr | 0.30 | 9.6 GB | ~4 GB weights, ~5 GB KV cache |
| unclaimed | 0.15 | 4.8 GB | CUDA contexts, activations, headroom |

**These are estimates, never measured — this stack has not been started
successfully.** The bf16 merge needs ~24 GB of weights and does not fit beside a
second model at all, and the FP8 path is unvalidated (§5.6). Re-measure both
numbers against `nvidia-smi` after the first successful start.

#### vLLM flags that are deliberate

- **No `--generation-config vllm`.** The merged directory carries its own
  `generation_config.json`, whose stop set includes `<end_of_turn>` (106). vLLM
  reads it because `--generation-config` defaults to `auto`; passing `vllm`
  discards the file and the model does not stop.
- **`--max-model-len 8192`** is a KV-cache budget decision, not a request-size
  one. Left unset, vLLM takes Gemma 3's 131072 from `config.json`; at ~64 KB per
  token across the 8 global-attention layers (the 40 sliding-window layers cap at
  1024 tokens each) that reserves ~9 GB of cache for a single sequence, and vLLM
  refuses to start when the budget is smaller than one sequence. 8192 is already
  generous for a translation segment, and the cache it frees becomes concurrency.
  Raise it only for long unsplit documents, and raise
  `--gpu-memory-utilization` with it.
- **No `--chat-template-content-format`.** Nothing here uses
  `/v1/chat/completions`. The gateway renders prompts itself and posts token ids
  to `/v1/completions`, which is the only way to reproduce the SFT rendering —
  and TranslateGemma's message content (per-part
  `source_lang_code`/`target_lang_code`) is not a shape vLLM's chat parser can
  accept anyway.
- **`entrypoint: ["vllm", "serve"]` is restated** even though the image's own
  ENTRYPOINT already is that. A command starting with `serve` would be appended
  to it and ask vLLM to load a model named `serve`.

#### Ports

| Port | Service |
| --- | --- |
| `8080` | Spring backend |
| `7070` | phpMyAdmin |
| `7073` | TranslateGemma vLLM — for humans and benchmarks only; the gateway reaches it over the compose network on port 8000 |
| `8888` | TranslateGemma gateway |
| `7071` / `7072` | dots.ocr vLLM / API |
| `127.0.0.1:3306` | MySQL, bound to loopback — the backend and phpMyAdmin reach it over the compose network, so nothing needs it published to the LAN |

#### Healthchecks and start-up order

- MySQL's healthcheck gates the backend. Without it the backend races the very
  first boot of MySQL, which runs `initdb` before it accepts connections.
- The vLLM services allow a 900 s (600 s for dots.ocr) `start_period`: loading a
  12B checkpoint and capturing CUDA graphs takes minutes.
- The gateway's healthcheck runs a real translation through vLLM, so it proves
  the whole path rather than just the process. Its `start_period` is 120 s — it
  loads a tokenizer, not a checkpoint; waiting for the weights is `depends_on`'s
  job.
- The gateway has no GPU reservation and no `PYTORCH_CUDA_ALLOC_CONF`: there is
  no torch in that image.

#### Configuration lives in one file

Everything — host paths, GPU fractions, gateway wiring, translation defaults —
is in the `.env` **beside the compose file**. Copy `.env.spadana.example` to
`.env` in the directory you run `docker compose` from. That single file does two
jobs: Compose substitutes every `${NAME}` in the compose file from it, and the
gateway service declares `env_file: .env`, so the same keys reach that process
as its environment. Names it does not recognise are ignored, which is why the
MySQL and vLLM keys can share the file.

There is no second `.env` under `./translategemma/`. An earlier layout had one,
with the wiring restated under `environment:` so the compose file won on
precedence. That is exactly the arrangement that hides a mistake: a duplicate
key drifts unnoticed for as long as the compose line exists, and takes over
silently the moment it is removed. One key, one home.

The single exception is `TG_VLLM_MODEL`, which stays under `environment:` as
`${SERVED_TG_MODEL_NAME:-translategemma}`. It must equal vLLM's
`--served-model-name`, which is built from that same variable; as two literals
in a flat file the pair could drift, and a mismatch is a 404 from the upstream
on every request.

Note that container-internal paths (`TG_BASE_MODEL_ID=/merged`,
`HF_HOME=/models`) and host paths (`TG_MERGED_MODEL_DIR`) both live in this
file. They are not interchangeable: the first two must match the compose file's
mount **targets**, the last is the mount **source**.

`dots-ocr-api` keeps an `extra_hosts` entry for whatever in its own `.env` still
points at the host. The service name `http://dots-ocr-vllm:8000/v1` is the better
address: it does not depend on a published port and survives a port change.

#### First-run database bootstrap

```bash
sudo docker compose exec mysql bash
mysql -u root -p                       # then: create database fundamental;
mysql -u root -p fundamental < fundamental.sql
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

Quick evaluation loads the base model, then the adapter, then MetricX, then
COMET. On a 32 GB card, run evaluation as a separate step rather than via
`evaluation.run_after_training: true`, so training memory is fully released.
For the multi-model benchmark, prefer separate `generate`, `score`, and
`report` Compose commands. Each generated candidate is loaded and released in
turn; a separate scoring container then loads COMET and MetricX without a
translation model occupying VRAM.

### 6.2 MetricX-24 integration

Both `evaluate_metricx()` and the benchmark MetricX scorer are aligned with the reference implementation
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

A fifth aspect fails loudly rather than silently, and only against
`transformers>=4.53`: the model must be called with **`use_cache=False`**.
MetricX builds its decoder with `is_encoder_decoder=False`, so `MT5Stack`
allocates a plain `DynamicCache` instead of an `EncoderDecoderCache`;
`T5Attention` then appends the cross-attention keys to the same cache that
already holds the single decoder self-attention key, making the key length one
longer than the encoder mask:

```
RuntimeError: The size of tensor a (354) must match the size of tensor b (353)
at non-singleton dimension 3
```

The decoder runs one dummy step, so nothing is lost by disabling the cache.

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
`evaluation.metricx_enabled: false` plus `metrics.metricx.enabled: false` in
`benchmark_config.yaml`, and rely on COMET.

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
§3.4 — enable the relevant benchmark candidate, or add it with `--repo <id>`,
and re-stage.

**One path does not raise: tokenizer files.** `transformers` resolves them with
missing-entry exceptions suppressed, so an unstaged tokenizer repository yields
`None` rather than an error, and the fast-tokenizer constructor then falls
through to its tiktoken conversion branch and dies far from the cause:

```
File "transformers/convert_slow_tokenizer.py", line 1857, in convert_slow_tokenizer
    elif transformer_tokenizer.vocab_file.endswith("tekken.json"):
AttributeError: 'NoneType' object has no attribute 'endswith'
```

Read that as *this tokenizer repository is not in `/models`*, not as a
transformers bug. The usual culprit is COMET's indirect encoder repository
after `comet_model_id` was changed (§3.4); the traceback names the encoder
class (`comet/encoders/xlmr_xl.py` → `facebook/xlm-roberta-xl`), which is the
only clue to which repository is missing.

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
scripts/fetch_flash_attn3_wheel.sh                                     # preferred, ~440 MB
# Or compile the pinned official source (once, 1-3 h):
# scripts/build_flash_attn3_wheel.sh
IMAGE_TAG=cu128-fa3-py312 INSTALL_FLASH_ATTN3=1 docker compose build trainer
```

Three deliberate choices there, each of which has a failure mode attached:

**The wheel is staged outside the image.** The preferred
`scripts/fetch_flash_attn3_wheel.sh` path downloads one exact community wheel
for CUDA 12.8 / Torch 2.8.0 / Linux x86-64, resumes interrupted transfers, and
verifies its pinned SHA-256. `scripts/build_flash_attn3_wheel.sh` remains the
official-source fallback: it compiles tag `v2.8.3.post1` inside
`nvidia/cuda:12.8.1-devel-ubuntu22.04`, with persistent dependency and Ninja
object caches. Both paths drop exactly one wheel into `wheels/`; the Dockerfile
rejects zero or multiple candidates, installs the explicit file without
dependency resolution, and runs the tier-1 import/ABI verifier during the image
build.

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

The image is a superset of the default one and is the required runtime for the
checked-in `flash_attention_3` + packing configuration. For an SDPA comparison,
disable packing as well as changing the attention implementation; measure
non-padding tokens/second rather than samples/second.

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

Two model layouts work there. The unmerged one is what `README.md` shows: the
staged base repository plus `--enable-lora --lora-modules <alias>=<adapter dir>`,
which keeps adapters swappable and costs one LoRA application per request. The
merged one comes from §5.5 and is a self-contained directory that needs no LoRA
flags:

```bash
vllm serve /models/merged/translategemma-farsi-sft \
    --served-model-name farsi-science \
    --max-model-len 2048 \
    --limit-mm-per-prompt '{"image": 0}'
```

The merged directory needs no conversion step, but three things about this model
are worth setting deliberately:

- **Leave the generation config alone.** vLLM reads `generation_config.json` from
  the model directory by default, which is how `<end_of_turn>` (106) reaches the
  stop set. Passing `--generation-config vllm` discards it and the model will not
  stop generating. If a client bypasses the defaults, have it send
  `stop_token_ids: [1, 106]` explicitly.
- **It is still a multimodal Gemma 3 checkpoint.** LoRA excluded `vision_tower`
  (§`README.md`), and merging leaves the image encoder in place, so the merged
  model carries it too. This pipeline sends text only, so
  `--limit-mm-per-prompt '{"image": 0}'` avoids reserving memory for image
  inputs that never arrive.
- **Prompts must keep the trained format.** The `<<<source>>>…<<<target>>>…
  <<<text>>>…` string and the chat template are what the adapter was trained on;
  the template travels with the merged directory, so use the chat completions
  endpoint rather than hand-assembling prompts.

`--max-model-len` is worth pinning to the trained sequence length
(`training.max_length`, 2048). The base config advertises a much longer context,
and vLLM sizes its KV cache from that.

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
