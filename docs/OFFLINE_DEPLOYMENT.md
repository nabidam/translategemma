# Offline deployment guide

How to run the TranslateGemma fine-tuning pipeline on an air-gapped GPU machine
using a shipped Docker image, with the source, models and outputs bind-mounted
so everything stays visible and debuggable from the host.

Target GPUs: **RTX 6000 Ada (sm_89)**, **RTX 5090 (sm_120)** and **RTX PRO 6000
Blackwell (sm_120)**. The default `cu128` image covers all three with an NVIDIA
r570+ driver. The retained, named `cu130` image covers the same GPUs but needs
an r580+ driver.

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
| `translategemma:cu130-py312` | CUDA 13.0 / PyTorch 2.13.0 | r580.65.06 | Set `IMAGE_TAG=cu130-py312` and `PYTORCH_CUDA=cu130` when building. |

Do not build the `cu130` tag for an r570-only host: an image can build
successfully but cannot initialise CUDA at runtime on that driver.

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

This builds the default `translategemma:cu128-py312` image. To deliberately
build the retained CUDA 13.0 image for an r580+ host:

```bash
IMAGE_TAG=cu130-py312 PYTORCH_CUDA=cu130 docker compose build trainer
```

Expect roughly **9–11 GB** uncompressed. The build context contains only
`pyproject.toml` and `uv.lock`, so rebuilds after a dependency change are the
only slow ones. The default `cu128` build uses `uv sync --locked`; it fails if
`uv.lock` does not match `pyproject.toml` rather than silently resolving newer
packages.

Sanity-check the resolved environment (no GPU needed):

```bash
docker run --rm translategemma:cu128-py312 \
  python -c "import torch, peft, trl, bitsandbytes; print(torch.__version__, torch.version.cuda)"
docker run --rm translategemma:cu128-py312 cat /opt/resolved-requirements.txt
```

`torch.version.cuda` must print `12.8`. For the named CUDA 13.0 image, replace
the tag in both commands with `cu130-py312`; it must print `13.0`.

### 3.3 Export the image

```bash
docker save translategemma:cu128-py312 | zstd -19 -T0 -o translategemma-cu128-image.tar.zst
```

For the CUDA 13.0 image, save `translategemma:cu130-py312` as
`translategemma-cu130-image.tar.zst`. Transfer only the image matching the GPU
host's driver.

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
matplotlib's font cache — and those kernels are compiled for the *building*
GPU's architecture, so they are useless or actively wrong on a different card.
The container rebuilds them on first use, which is why the image carries `gcc`
(§6.3). Nothing is ever downloaded into it: every network-sourced artefact goes
through `HF_HOME=/models`. Create it empty on the target and leave it alone.

If you want to confirm that on the staging machine before wiping it:

```bash
du -sh offline_assets/hf_cache
find offline_assets/hf_cache -type f | head -20
```

Expect only `.triton/` and similar compiled caches. Anything resembling a model
snapshot there means a library bypassed `HF_HOME` and the transfer list needs
revisiting.

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

Then, in order — each step is cheap and fails fast:

```bash
# 1. Config, paths and chat template only. No weights loaded.
docker compose run --rm trainer python train.py --config config.yaml --dry-run

# 2. Every enabled stage, <=10 rows per split, one optimiser step, temp outputs.
docker compose run --rm trainer python train.py --config config.yaml --smoke-test
```

A successful `--smoke-test` proves the model loads from the offline cache, 4-bit
quantisation works on this GPU, and the evaluation path is importable.

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

QLoRA on a 12B model at `max_length: 2048` with 4-bit weights and gradient
checkpointing needs roughly 20–26 GB.

| GPU | VRAM | Guidance |
| --- | --- | --- |
| RTX PRO 6000 Blackwell | 96 GB | Defaults fine. `batch_size` can go to 8–16. |
| RTX 6000 Ada | 48 GB | Defaults fine. |
| RTX 5090 | 32 GB | Defaults usually fit but are close. On OOM, set `batch_size: 2` and `gradient_accumulation_steps: 8` in `config.yaml` — the effective batch is unchanged. |

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

Triton (pulled in by `torch`) JIT-compiles GPU kernels **while training runs**,
not at build time, and shells out to a C compiler to build the launcher stubs.
Without it:

```
RuntimeError: Failed to find C compiler. Please specify via CC environment
variable or set triton.knobs.build.impl.
```

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

### 6.7 vLLM serving

The vLLM command in `README.md` is not part of this image — `vllm` is not in
`pyproject.toml`. Serving offline needs its own image, staged the same way.

### 6.8 Transformers version boundary

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
