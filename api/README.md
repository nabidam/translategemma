# TranslateGemma API

FastAPI translation gateway for TranslateGemma. The weights are served by a
vLLM container; this service owns the prompt rendering, the stop set, sentence
splitting and the request/response contract, and forwards generation to vLLM
over its OpenAI-compatible API. Endpoint shapes follow the existing NLLB
service, so callers move over with a URL change.

No torch, no CUDA, no GPU reservation on this container: it holds a tokenizer
and an HTTP client.

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
├── translator.py       # prompt rendering + vLLM client
├── prompting.py        # copy, see below
├── docker-compose.yml  # vLLM + gateway + benchmark profile
├── scripts/            # benchmark_speed.py, run inside the image
└── .env.example
```

That independence has one cost, handled explicitly: `prompting.py` is a **copy**
of the module of the same name at the root of the source repository. It encodes
two facts that repository paid for in a
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

The generation *contract* still matches `evaluate_translations`, even though the
generation itself now happens in vLLM:

* **Prompts are rendered here**, by `prompting.py`, with the checkpoint's own
  tokenizer, and sent to `POST /v1/completions` as **token ids**. Not
  `/v1/chat/completions`: that renders with `add_generation_prompt=True`, which
  is precisely the prefix the SFT adapter was never conditioned on. Not prompt
  strings either: the completions endpoint tokenizes with
  `add_special_tokens=True`, which would prepend a second `<bos>` to a rendering
  that already carries one.
* **The stop set is resolved here** and passed on every request as
  `stop_token_ids`, rather than trusting whatever the upstream inherited from a
  `config.json`.
* **Greedy by default**, expressed to vLLM as `temperature=0` — the same
  distribution `do_sample=False` produced.
* **No trailing strip** on the returned text. Trailing whitespace is the visible
  signature of an unstopped decoder, so trimming it in the gateway would hide a
  regression.

Beam search is gone with the in-process engine: the vLLM completions API has no
equivalent, so there is no `TG_NUM_BEAMS` to set. Greedy and sampling are the
two decodings this gateway can ask for.

### Where the old `model_loading.py` configs went

That module (still at the root of the source repository, no longer copied here)
built three things. Under vLLM they are owned as follows:

| What it built | Who owns it now |
| ------------- | --------------- |
| The stop set including `<end_of_turn>` (106) | Both sides. `scripts/merge_lora_adapter.py` bakes it into the merged checkpoint's `generation_config.json`, and the gateway resolves it again and sends `stop_token_ids` on every request. |
| Sampling defaults, replacing TranslateGemma's invalid `config.json` values | The gateway, explicitly, per request — `temperature`, `top_p` and `top_k` are always sent. |
| `dtype` | vLLM, via `--dtype` in `docker-compose.yml`. |

The middle row is the one that needs care. vLLM's `--generation-config` defaults
to `auto`, which reads `generation_config.json` from the model directory and
uses it as the **default sampling parameters** — so any knob the gateway left
unset would be supplied by that file. For a checkpoint merged by this
repository those defaults are the right ones, because the same function wrote
them; for a checkpoint merged elsewhere they are whatever it shipped. Sending
the whole set makes the served decoding independent of that. Do not pass
`--generation-config vllm`: it discards the file, and with it the stop set.

## Which system is served

One vLLM holds one set of weights, so one gateway serves exactly one system.
`TG_SERVED_SYSTEM` states which it is:

| Value               | The upstream holds                            |
| ------------------- | --------------------------------------------- |
| `adapter` (default) | a merged or adapted checkpoint                |
| `base`              | an untouched upstream checkpoint              |

This is not cosmetic. It selects the prompt rendering: the adapter is queried
after the SFT rendering, the base model after the generation prompt, and using
one for the other is silent — generation still returns fluent Farsi from a
prefix the model never saw. Override the pairing only when serving **merged**
weights labelled as the base system: `TG_BASE_USE_TRAINING_RENDERING=true`.

To compare base against adapter, run a gateway and a vLLM per system. There is
no in-process adapter toggle: the weights are not in this process.

## Endpoints

| Method | Path               | Purpose                                                   |
| ------ | ------------------ | --------------------------------------------------------- |
| GET    | `/health-check`    | Runs a real translation. `{"translator": "OK"\|"FAIL"}`.   |
| GET    | `/model-info`      | Checkpoint, adapter, served system, upstream, stop tokens. |
| POST   | `/translate`       | One text in, one translation out.                          |
| POST   | `/translate/batch` | Many texts, dispatched concurrently to vLLM.               |

Interactive docs at `/docs`.

```bash
curl -s localhost:8000/translate -H 'content-type: application/json' -d '{
  "text": "The model relies on multi-query attention to process the genome sequence."
}'
# {"translation":"...","system":"adapter","source_lang":"en","target_lang":"fa"}

# Every option is optional. "system" is an assertion, not a selector: it 400s
# when it disagrees with TG_SERVED_SYSTEM instead of answering as the other one.
curl -s localhost:8000/translate/batch -H 'content-type: application/json' -d '{
  "texts": ["First segment.", "Second segment."],
  "system": "adapter",
  "source_lang": "en",
  "target_lang": "fa",
  "max_new_tokens": 256,
  "split_sentences": false
}'
```

Naming a system this upstream does not serve returns 400 naming the one it
does, rather than silently answering as the other one.

## Configuration

Every setting is an environment variable prefixed `TG_`; `.env.example` has the
annotated list. The ones that decide what is served:

| Variable                     | Default                                    | Notes                                                                 |
| ---------------------------- | ------------------------------------------ | --------------------------------------------------------------------- |
| `TG_VLLM_BASE_URL`           | `http://translategemma-vllm:8000/v1`       | OpenAI-compatible base URL of the vLLM server.                         |
| `TG_VLLM_MODEL`              | `model`                                    | Must match vLLM's `--served-model-name`.                               |
| `TG_VLLM_TIMEOUT`            | `300`                                      | Seconds per upstream request.                                          |
| `TG_VLLM_MAX_RETRIES`        | `2`                                        | Retries on connection errors and 5xx; 4xx is never retried.            |
| `TG_MAX_CONCURRENT_REQUESTS` | `32`                                       | In-flight upstream requests across all callers.                        |
| `TG_BASE_MODEL_ID`           | `google/translategemma-12b-it`             | The checkpoint vLLM serves. Read locally for the tokenizer only.       |
| `TG_TOKENIZER_PATH`          | `TG_BASE_MODEL_ID`                         | Override when the tokenizer does not live with the checkpoint.         |
| `TG_SERVED_SYSTEM`           | `adapter`                                  | `base` or `adapter`; which system the upstream weights are.            |
| `TG_ADAPTER_PATH`            | unset                                      | Provenance only — merged weights already contain the adapter.          |
| `TG_BATCH_SIZE`              | `8`                                        | Prompts per upstream request. vLLM does the GPU batching.              |
| `TG_SPLIT_SENTENCES`         | `false`                                    | pysbd splitting; also settable per request.                            |

How the upstream was loaded — dtype, attention kernel, quantization — is
configured on the vLLM service (`--dtype`, and the flags beside it in
`docker-compose.yml`) and is deliberately **not** mirrored here. A gateway-side
copy of those flags would be a claim nothing verifies, and `/model-info` would
report it even after the upstream was relaunched differently.

A `TG_BASE_MODEL_ID` that does not resolve to a checkpoint directory fails at
startup rather than on the first request.

## Docker

Run from inside this directory; the build context is this directory:

```bash
cd api
docker build -t translategemma-api .
```

The source is baked into the image, so **a code change needs a rebuild** — but
only the last layer. Dependencies install in a layer that mentions only
`requirements.txt`, so editing the source never re-resolves them, and the uv
cache lives in a BuildKit cache mount. There is no torch in the image any more,
so both the build and the restart are fast: the gateway starts in seconds
because it loads a tokenizer, not a 12B checkpoint. BuildKit is required
(default in Docker 23+; export `DOCKER_BUILDKIT=1` on older hosts).

Restarting the gateway does **not** reload the weights: vLLM keeps them, and
`depends_on: service_healthy` holds the gateway back until vLLM answers
`/health`. That is the point of the split — shipping a change to the serving
contract no longer costs a model load.

`docker-compose.yml` in this directory ships both services:

```bash
docker compose up -d              # vLLM, then the gateway once vLLM is healthy
docker compose logs -f translategemma-vllm
```

* `translategemma-vllm` serves `${MODEL_PATH:-/models/translategemma-12b-merged}`
  — the checkpoint `scripts/merge_lora_adapter.py` writes — with the GPU
  reservation, and publishes `${VLLM_PORT:-8001}` for humans and for direct
  benchmarking. The gateway reaches it over the compose network, not that port.
  The context length comes from the checkpoint's `config.json`; add
  `--max-model-len` only if vLLM refuses to start because the KV cache for the
  full context does not fit the GPU.
* `translategemma-api` publishes `${HOST_PORT:-8000}` and reserves no GPU.

`TG_BASE_MODEL_ID` and `MODEL_PATH` must name the **same** checkpoint: the
gateway renders prompts with the tokenizer at one and vLLM generates from the
other. Both default to `/models/translategemma-12b-merged` from the shared
read-only `${MODELS_DIR}` mount, so they agree unless one is overridden alone.

A `benchmark` service behind a profile (below) reuses the same environment
block.

## Speed benchmark

`scripts/benchmark_speed.py` measures what this GPU actually delivers: decode
tokens/second across batch sizes, and how long one page of a document takes.

It measures the **served** path end to end — `POST /translate/batch` on the
running API, which renders prompts and forwards them to vLLM — so the numbers
are numbers a caller can receive. The in-process engine transport is gone: the
API loads no weights, so there is nothing left to measure without the server.

```bash
# Needs both services up; the benchmark loads no weights itself.
docker compose up -d
docker compose run --rm benchmark --api-url http://translategemma-api:8000

# A real page instead of the synthetic one. PDFs are extracted with PyMuPDF in
# reading order and reflowed, then segmented with pysbd — the server's splitter.
docker compose run --rm benchmark --page-source both --page-file scripts/test.pdf --page-number 1

docker compose run --rm benchmark --help
```

Everything after the service name is passed to the script. Reports are printed
and also written to `./benchmarks` on the host as `.md` and `.json`.

| Option | Default | Purpose |
| ------ | ------- | ------- |
| `--mode` | `http` | Only `http` exists; kept so existing invocations keep working. |
| `--batch-sizes` | `1,2,4,8,16` | Batch sizes to sweep. |
| `--repeats` / `--warmup` | `3` / `1` | Timed runs per point; discarded runs before them. |
| `--page-source` | `synthetic` | `synthetic`, `file`, or `both` (`file`/`both` need `--page-file`). |
| `--page-words` | `250` | Words in the synthetic page — the usual page for translation pricing. |
| `--page-modes` | `whole,sentences` | One long generation, and/or pysbd-split segments. |
| `--page-batch-size` | `1` | Accepted for compatibility only: the server batches the page by its own `TG_BATCH_SIZE`, and the report notes the discrepancy. |
| `--systems` | every loaded system | Restrict to `base` / `adapter`. |

Reading the report:

* **Token counts are approximate.** The response carries text, not token ids, so
  output tokens are recovered by re-encoding with the same tokenizer — within a
  token or two. Wall time, words/minute and pages/hour are exact, and they are
  what this benchmark exists for.
* **`decode tok/s`** is therefore approximate output tokens per second of decode
  time; `ms/step` has no batch-internal view from outside the server and should
  be read as a rough per-token cost.
* **`prefill s`** comes from a separate `max_new_tokens=1` run; `decode s` is
  the remainder. Over HTTP it also carries the gateway's rendering time.
* **`peak VRAM`** is blank: the process measuring has no GPU.
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

Requests are concurrent end to end. There is no GPU lock any more — nothing in
this process touches a GPU — so callers queue only on
`TG_MAX_CONCURRENT_REQUESTS` and then inside vLLM's scheduler, which merges
every in-flight request into its own running batch. A sentence-split request
dispatches its segments concurrently rather than one chunk at a time, so a long
document is limited by the slowest segment plus scheduling, not by the sum of
its chunks.

Prompt rendering and tokenization run in a worker thread, so the event loop
keeps serving `/health-check` and `/model-info` under load.

Scale the gateway out with more replicas — it is stateless and cheap. Scale
generation by giving vLLM more GPUs (`TENSOR_PARALLEL_SIZE`), which is the
resource that actually binds.
