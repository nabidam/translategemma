# TranslateGemma API

FastAPI translation service for TranslateGemma. Serves the base model, a
LoRA-adapted model, or both at once. Endpoint shapes follow the existing NLLB
service, so callers move over with a URL change.

## Standalone

`api/` runs on its own. It imports nothing from the parent repository, its
dependencies are its own `requirements.txt`, and its Docker build context is
this directory. Copy `api/` anywhere and it works.

That independence has one cost, handled explicitly: `prompting.py` and
`model_loading.py` here are **copies** of the project-root modules of the same
name. They encode two facts this repository paid for in a failed adapter (see
`docs/2026-08-10_adapter_degeneration_analysis.md`):

1. The prompt an SFT adapter was trained to continue is **not**
   `apply_chat_template(..., add_generation_prompt=True)`.
2. The stop set must include `<end_of_turn>` (106), which `config.json` alone
   does not provide.

Get either wrong and the model still returns fluent Farsi — it just never stops.
No smoke test catches that. So the copies are machine-checked:

```bash
uv run python scripts/sync_api_vendored.py          # re-copy after editing the root module
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

Build context is `api/`, not the repository root:

```bash
docker build -t translategemma-api api/
```

Rebuilds are cheap. Dependencies install in a layer that mentions only
`requirements.txt`, so editing the source never re-resolves them, and the uv
cache lives in a BuildKit cache mount — even a changed `requirements.txt` reuses
every wheel already downloaded, so the ~3 GB torch download happens once per
machine. BuildKit is required (default in Docker 23+; export
`DOCKER_BUILDKIT=1` on older hosts).

Service block for the compose file this is deployed under:

```yaml
services:
  translategemma-api:
    image: translategemma-api
    build:
      context: ./api
    ports:
      - "8000:8000"
    volumes:
      # Staged model tree plus the adapter directory. Read-only is fine: unlike
      # the trainer, this service takes no huggingface_hub download locks when
      # the model id is a local path.
      - ${MODELS_DIR:-./offline_assets/models}:/models:ro
    environment:
      TG_BASE_MODEL_ID: /models/translategemma-12b-it
      TG_MODEL_MODE: adapter
      TG_ADAPTER_PATH: /models/adapters/sft_final
      TG_SOURCE_LANG: en
      TG_TARGET_LANG: fa
      TG_BATCH_SIZE: "8"
      HF_HOME: /models
      HUGGINGFACE_HUB_CACHE: /models/hub
      HF_HUB_OFFLINE: "1"
      TRANSFORMERS_OFFLINE: "1"
      PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
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
              count: 1
              capabilities: [gpu]
```

## Local run

```bash
cp api/.env.example api/.env   # then edit the model paths
cd api && uv run --with-requirements requirements.txt fastapi dev main.py
```

## Concurrency

One model on one GPU. Requests serialize behind a lock — required, not merely
prudent, because `both` mode toggles the adapter in place and two concurrent
requests would race on it. Each `generate()` runs in a worker thread, so
`/health-check` and `/model-info` stay responsive during a long translation.
Scale out with more containers, each pinned to its own GPU via
`NVIDIA_VISIBLE_DEVICES`.
