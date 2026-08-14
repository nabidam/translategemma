# TranslateGemma API

FastAPI translation service for TranslateGemma. Serves the base model, a
LoRA-adapted model, or both at once. Endpoint shapes follow the existing NLLB
service, so callers move over with a URL change.

## Standalone

**This directory is the deployment unit.** Copy it to the target machine, `cd`
into it, and build — every command below runs from inside `api/`, and nothing
here reads a path outside it. It imports nothing from the parent repository and
pins its own dependencies in `requirements.txt`.

```
api/                    # ← copy this, and only this, to the server
├── Dockerfile          # build context is this directory
├── requirements.txt
├── main.py             # FastAPI app
├── config.py           # TG_* settings
├── schemas.py
├── translator.py       # model loading + generation
├── prompting.py        # copy, see below
├── model_loading.py    # copy, see below
├── docker-compose.yml  # server + benchmark profile
├── scripts/            # benchmark_speed.py, run inside the image
└── .env.example
```

That independence has one cost, handled explicitly: `prompting.py` and
`model_loading.py` are **copies** of the modules of the same name at the root of
the source repository. They encode two facts that repository paid for in a
failed adapter (`docs/2026-08-10_adapter_degeneration_analysis.md`):

1. The prompt an SFT adapter was trained to continue is **not**
   `apply_chat_template(..., add_generation_prompt=True)`.
2. The stop set must include `<end_of_turn>` (106), which `config.json` alone
   does not provide.

Get either wrong and the model still returns fluent Farsi — it just never stops.
No smoke test catches that. So the copies are machine-checked **in the source
repository**, before this directory is ever copied anywhere:

```bash
# From the repository root, not from api/. Not needed on the deployment host.
uv run python scripts/sync_api_vendored.py          # re-copy after editing a root module
uv run python scripts/sync_api_vendored.py --check  # exit 1 on drift
uv run pytest tests/test_api_vendored_modules.py    # same check, in CI
```

Edit the root module, sync, commit both. `api/translator.py` is also in
`tests/test_generation_chat_template.py`'s entry-point list, so it cannot start
rendering prompts or resolving stop tokens on its own.

## Generation parity with the harness

`api/translator.py` reproduces `evaluate_translations.generate_translations`
step for step: the same generation-safe model config, the same stop set resolved
twice (once into the generation config, once passed on every `generate()` call),
the same per-system prompt rendering, the same generation kwargs, and the same
`batch_decode(..., skip_special_tokens=True)` **without a trailing strip** —
trailing whitespace is the visible signature of an unstopped decoder, so
trimming it in the server would hide a regression.

The one deliberate difference is multi-GPU. The harness shards a fixed test set
across ranks; that has no meaning for a request/response server. One process,
one model, one device.

## Model modes

`TG_MODEL_MODE` decides what is loaded:

| Mode      | Loads                          | `system` selectable per request  |
| --------- | ------------------------------ | -------------------------------- |
| `base`    | untouched checkpoint           | no — always `base`               |
| `adapter` | base + LoRA adapter (default)  | no — always `adapter`            |
| `both`    | base weights + attached adapter | yes — `"system": "base"\|"adapter"` |

`both` loads **one** copy of the 12B weights and switches the LoRA layers off
per request via PEFT's `disable_adapter()`, so the second system costs the
adapter's few hundred MB rather than another full model. It serves the exact
baseline-vs-adapter comparison the evaluation harness makes, live.

Each system is prompted the way it was trained — the adapter after the SFT
rendering, the base model after the generation prompt. Override only when
serving **merged** weights as the base system:
`TG_BASE_USE_TRAINING_RENDERING=true`.

## Endpoints

| Method | Path               | Purpose                                                   |
| ------ | ------------------ | --------------------------------------------------------- |
| GET    | `/health-check`    | Runs a real translation. `{"translator": "OK"\|"FAIL"}`.   |
| GET    | `/model-info`      | Checkpoint, adapter, mode, device, resolved stop tokens.   |
| POST   | `/translate`       | One text in, one translation out.                          |
| POST   | `/translate/batch` | Many texts, batched into shared `generate()` calls.        |

Interactive docs at `/docs`.

```bash
curl -s localhost:8000/translate -H 'content-type: application/json' -d '{
  "text": "The model relies on multi-query attention to process the genome sequence."
}'
# {"translation":"...","system":"adapter","source_lang":"en","target_lang":"fa"}

# Every option is optional; "system" requires TG_MODEL_MODE=both.
curl -s localhost:8000/translate/batch -H 'content-type: application/json' -d '{
  "texts": ["First segment.", "Second segment."],
  "system": "base",
  "source_lang": "en",
  "target_lang": "fa",
  "max_new_tokens": 256,
  "split_sentences": false
}'
```

Asking for a system that is not loaded returns 400 with the available list,
rather than silently answering as the other one.

## Configuration

Every setting is an environment variable prefixed `TG_`; `.env.example` has the
annotated list. The ones that decide what is served:

| Variable            | Default                        | Notes                                              |
| ------------------- | ------------------------------ | -------------------------------------------------- |
| `TG_BASE_MODEL_ID`  | `google/translategemma-12b-it` | Hub id or a path inside the container.              |
| `TG_MODEL_MODE`     | `adapter`                      | `base` / `adapter` / `both`.                        |
| `TG_ADAPTER_PATH`   | unset                          | Required unless mode is `base`; validated at start. |
| `TG_DEFAULT_SYSTEM` | the loaded one, else `adapter`  | Answers requests that do not name a system.         |
| `TG_LOAD_IN_4BIT`   | `false`                        | Lower VRAM, slower per token.                       |
| `TG_BATCH_SIZE`     | `8`                            | Segments per `generate()` call. Lower this on OOM.  |
| `TG_SPLIT_SENTENCES`| `false`                        | pysbd splitting; also settable per request.         |

A bad adapter path fails at startup rather than on the first request.

## Docker

Run from inside this directory; the build context is this directory:

```bash
cd api
docker build -t translategemma-api .
```

The source is baked into the image, so **a code change needs a rebuild** — but
only the last layer. Dependencies install in a layer that mentions only
`requirements.txt`, so editing the source never re-resolves them, and the uv
cache lives in a BuildKit cache mount, so even a changed `requirements.txt`
reuses every wheel already downloaded. The ~3 GB torch download happens once per
machine; a code-only rebuild takes seconds. BuildKit is required (default in
Docker 23+; export `DOCKER_BUILDKIT=1` on older hosts).

Note that a rebuild is not the expensive part of shipping a change: the model
loads once at startup, so any change costs a container restart and a few minutes
of reloading weights regardless.

Service block for the compose file this is deployed under. `context` is written
for a compose file sitting **next to** this directory (`../docker-compose.yml`);
use `context: .` if the compose file lives inside it.

```yaml
services:
  translategemma-api:
    image: ${IMAGE_NAME:-translategemma-api}
    build:
      context: ${BUILD_CONTEXT:-.}
    ports:
      - "${HOST_PORT:-8000}:${CONTAINER_PORT:-8000}"
    volumes:
      # Staged model tree plus the adapter directory. Read-only is fine: unlike
      # the trainer, this service takes no huggingface_hub download locks when
      # the model id is a local path.
      - ${MODELS_DIR:-./offline_assets/models}:/models:ro
    environment:
      # Model & System Selection
      TG_BASE_MODEL_ID: ${TG_BASE_MODEL_ID:-/models/translategemma-12b-it}
      TG_MODEL_MODE: ${TG_MODEL_MODE:-adapter}
      TG_ADAPTER_PATH: ${TG_ADAPTER_PATH:-/models/adapters/sft_final}

      # Loading Precision & Attention
      TG_DTYPE: ${TG_DTYPE:-bfloat16}
      TG_ATTN_IMPLEMENTATION: ${TG_ATTN_IMPLEMENTATION:-sdpa}
      TG_LOAD_IN_4BIT: ${TG_LOAD_IN_4BIT:-false}


      # Translation Defaults
      TG_SOURCE_LANG: ${TG_SOURCE_LANG:-en}
      TG_TARGET_LANG: ${TG_TARGET_LANG:-fa}
      TG_MAX_NEW_TOKENS: ${TG_MAX_NEW_TOKENS:-512}
      TG_DO_SAMPLE: ${TG_DO_SAMPLE:-false}
      TG_TEMPERATURE: ${TG_TEMPERATURE:-1.0}
      TG_TOP_P: ${TG_TOP_P:-1.0}
      TG_NUM_BEAMS: ${TG_NUM_BEAMS:-1}

      # Batching & Performance Settings
      TG_BATCH_SIZE: ${TG_BATCH_SIZE:-8}
      TG_MAX_BATCH_ITEMS: ${TG_MAX_BATCH_ITEMS:-128}
      TG_SPLIT_SENTENCES: ${TG_SPLIT_SENTENCES:-false}

      # Offline Cache Settings
      HF_HOME: ${HF_HOME:-/models}
      HUGGINGFACE_HUB_CACHE: ${HUGGINGFACE_HUB_CACHE:-/models/hub}
      HF_HUB_OFFLINE: ${HF_HUB_OFFLINE:-1}
      TRANSFORMERS_OFFLINE: ${TRANSFORMERS_OFFLINE:-1}
      PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
    healthcheck:
      # Generous start_period: loading a 12B checkpoint takes minutes, and the
      # check itself runs a real translation.
      test: ["CMD", "python", "-c", "import urllib.request,sys; sys.exit(0 if b'OK' in urllib.request.urlopen('http://localhost:8000/health-check').read() else 1)"]
      interval: 30s
      timeout: 30s
      retries: 3
      start_period: 600s
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: ${GPU_COUNT:-1}
              capabilities: [gpu]
```

`docker-compose.yml` in this directory ships that service, plus a `benchmark`
service behind a profile (below) that reuses the same environment block.

## Speed benchmark

`scripts/benchmark_speed.py` measures what this GPU actually delivers: decode
tokens/second across batch sizes, and how long one page of a document takes.

It measures the **served** path — `TranslationEngine.translate` for the engine
transport, `POST /translate/batch` for the HTTP one — so the numbers are numbers
the API can deliver, and the gap between the two transports is the serving
overhead. Which systems are benchmarked follows `TG_MODEL_MODE`: `base`
benchmarks the base model, `adapter` the adapted one, `both` runs both and
therefore also prices the `disable_adapter()` toggle.

```bash
# Engine transport: loads its own copy of the weights, so stop the server first
# on a single-GPU host.
docker compose run --rm benchmark

# HTTP transport: needs the server up, loads nothing itself.
docker compose up -d translategemma-api
docker compose run --rm benchmark --mode http --api-url http://translategemma-api:8000

# A real page instead of the synthetic one. PDFs are extracted with PyMuPDF in
# reading order and reflowed, then segmented with pysbd — the server's splitter.
docker compose run --rm benchmark --page-source both --page-file scripts/test.pdf --page-number 1

docker compose run --rm benchmark --help
```

Everything after the service name is passed to the script. Reports are printed
and also written to `./benchmarks` on the host as `.md` and `.json`.

| Option | Default | Purpose |
| ------ | ------- | ------- |
| `--mode` | `engine` | `engine`, `http`, or `both` (needs VRAM for two copies). |
| `--batch-sizes` | `1,2,4,8,16` | Batch sizes to sweep. |
| `--repeats` / `--warmup` | `3` / `1` | Timed runs per point; discarded runs before them. |
| `--page-source` | `synthetic` | `synthetic`, `file`, or `both` (`file`/`both` need `--page-file`). |
| `--page-words` | `250` | Words in the synthetic page — the usual page for translation pricing. |
| `--page-modes` | `whole,sentences` | One long generation, and/or pysbd-split segments. |
| `--systems` | every loaded system | Restrict to `base` / `adapter`. |

Reading the report:

* **`decode tok/s`** is output tokens per second of decode time, counted per row
  up to the first stop token. Padding columns are excluded — counting them is
  how a batched benchmark ends up claiming throughput the caller never receives.
* **`ms/step`** is the per-decode-step cost of the batch: hardware speed,
  independent of how much padding the batch carried. `decode tok/s` well below
  `batch x steps / decode s` means the batch was dominated by its slowest row.
* **`prefill s`** comes from a separate `max_new_tokens=1` run; `decode s` is
  the remainder.
* A **truncation warning** means rows hit `max_new_tokens` without emitting a
  stop token. Those rows measure the cap, not the model — raise
  `--max-new-tokens` and rerun before quoting anything.
* **`predicted s`** on the page table extrapolates from the sweep. A large gap
  against the measured time means the sweep's sentence mix does not represent
  that page.

## Local run (no Docker)

From inside this directory:

```bash
cp .env.example .env   # then edit the model paths
uv run --with-requirements requirements.txt fastapi dev main.py
```

`.env` is read relative to the working directory, so run the server from here.

## Concurrency

One model on one GPU. Requests serialize behind a lock — required, not merely
prudent, because `both` mode toggles the adapter in place and two concurrent
requests would race on it. Each `generate()` runs in a worker thread, so
`/health-check` and `/model-info` stay responsive during a long translation.
Scale out with more containers, each pinned to its own GPU via
`NVIDIA_VISIBLE_DEVICES`.
