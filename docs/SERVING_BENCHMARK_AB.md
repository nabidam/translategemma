# Serving A/B: vLLM + merged checkpoint vs FastAPI + transformers + LoRA

Measures generation speed of the two serving strategies on the same GPU, with
the same decoding settings, through the same benchmark client.

| | Arm A | Arm B |
|---|---|---|
| Runtime | vLLM `v0.13.0` | transformers `generate()` |
| Weights | merged checkpoint, bf16 | base bf16 + LoRA applied at run time |
| Batching | continuous, scheduler-owned | fixed chunks behind a GPU lock |
| API process | CPU-only gateway | holds the weights itself |
| Image | `translategemma-api-gw` | `translategemma-api-hf` |

Arm B is the pre-`21e5ab0` deployment. That commit
(`refactor(api): serve through vllm instead of in-process transformers`) deleted
`api/model_loading.py` and dropped torch/peft from `api/requirements.txt`, so the
arm is not reachable from `HEAD`.

**Both images are called `translategemma-api` by their own Dockerfile headers**,
which is how a host ends up with one image under that name and no way to tell
which arm it is. Check before doing anything else:

```bash
docker run --rm translategemma-api:latest cat /opt/resolved-requirements.txt | grep -iE '^(torch|peft)'
```

Output means it is arm B (the transformers server); silence means it is arm A
(the gateway). Retag it explicitly, and keep the explicit names from then on:

```bash
docker tag translategemma-api:latest translategemma-api-hf:latest   # if arm B
docker tag translategemma-api:latest translategemma-api-gw:latest   # if arm A
```

Nothing about the request contract changed in that commit: `/translate/batch`
and `/model-info` are identical on both sides, which is why one benchmark binary
(`api/scripts/benchmark_speed.py` at `HEAD`) drives both arms unmodified. Do not
use each image's own copy of the script; two different measurement programs
produce two incomparable reports.

The client runs in the **gateway** image specifically, and cannot be run in the
transformers one: at `21e5ab0^` the `api/` tree has no `Settings.vllm_base_url`,
no `Settings.resolved_tokenizer_path` and no `translator.load_processor`, all of
which `HEAD`'s benchmark imports.

## Prerequisites on the GPU host

- `vllm/vllm-openai:v0.13.0`
- The merged checkpoint, **and** the unmerged base plus the LoRA adapter it was
  merged from. Arm A and arm B must be the same fine-tune, or the benchmark
  compares two models rather than two runtimes.
- The production stack (`docker-compose.spadana.yml`) stopped. `dots-ocr-vllm`
  in particular holds 0.30 of total VRAM; leaving it up caps arm A's KV cache
  and serialises arm B against another process's kernels.

### Build whichever image is missing

Both builds need network for their pip layer; do them before the host goes
offline. `api/` is self-contained, so either build can be run from a copy of
that one directory.

**Arm A, the gateway** — CPU-only, no torch, a few hundred MB and a few minutes:

```bash
cd /path/to/translategemma/api
DOCKER_BUILDKIT=1 docker build -t translategemma-api-gw:latest .
```

**Arm B, the transformers server** — pulls torch cu128, ~3 GB:

```bash
cd /path/to/translategemma
git worktree add /tmp/tg-transformers 21e5ab0^
cd /tmp/tg-transformers/api
DOCKER_BUILDKIT=1 docker build -t translategemma-api-hf:latest .
git worktree remove /tmp/tg-transformers      # afterwards
```

The worktree keeps the working branch untouched. Tag both explicitly rather than
taking the `-t translategemma-api` in either Dockerfile header, which is what
makes the two indistinguishable on a host that has built both.

### Configure

```bash
cp .env.bench.example .env.bench
$EDITOR .env.bench          # four paths, one GPU id
```

## Run

```bash
./scripts/bench_ab.sh
```

That brings up arm A, waits for its healthcheck, benchmarks it, tears it down,
does the same for arm B, and prints the comparison. Roughly 40-70 minutes for
the default sweep, most of it model loading and CUDA-graph capture.

Options:

```bash
./scripts/bench_ab.sh --arm vllm                     # one arm only
./scripts/bench_ab.sh --batch-sizes 1,4,16 --repeats 3
./scripts/bench_ab.sh --page-file api/scripts/test.pdf   # add a real page
./scripts/bench_ab.sh -- --no-prefill-probe          # anything after -- goes to the client
```

Reports land in `benchmark_ab/` as `benchmark_http_<stamp>_<tag>.{json,md}`,
tagged `vllm-bf16` and `hf-lora`. Re-comparison without re-running:

```bash
python scripts/compare_bench_reports.py benchmark_ab/
```

## What makes the comparison valid

These are enforced by `docker-compose.bench.yml`, and are the things to re-check
if a number looks wrong:

- **One arm at a time.** They share a GPU. `bench_ab.sh` tears each arm fully
  down before the next comes up.
- **Identical decoding.** Greedy, `num_beams=1`, same `TG_MAX_NEW_TOKENS`, same
  language pair, splitting off. Greedy means both arms emit the same text, which
  is what makes their tokens/second commensurable — `compare_bench_reports.py`
  flags any configuration where the two output-token counts drift more than 5%.
- **`TG_BATCH_SIZE` ≥ the largest swept batch.** The server re-chunks a
  `/translate/batch` request by its own `TG_BATCH_SIZE`; at the default 8, the
  batch-16 and batch-32 rows would silently be 2 and 4 sequential chunks. The
  compose file derives it from the sweep.
- **bf16 on both sides.** Serving the fp8 quant on arm A would fold a
  quantisation effect into a runtime question. Run fp8 afterwards as a third arm
  with its own tag if you want that number.

## Reading the result

Report three numbers per arm, not one:

1. **Latency at batch 1** — what a single interactive caller feels. The closest
   fight; arm B has no scheduler overhead, but pays a per-layer `B@A` LoRA
   matmul on every generated token.
2. **Throughput at the largest batch** — what a document queue gets. Arm A
   should win by a wide margin: arm B serialises behind a GPU lock, so its
   "batch 32" is really sequential chunks. That is the honest result of the old
   architecture, not a misconfiguration.
3. **Page wall-clock / words-per-minute** — the figure a non-engineer can act
   on.

### Confounds to state in any writeup

- Merged-vs-adapter is not purely a runtime difference: applying LoRA at run
  time is genuinely extra compute. To isolate the runtime alone, add a third arm
  running vLLM with `--enable-lora` over the *unmerged* base, which is what
  `vllm_api.py` at the repo root already targets (`--lora-modules`).
- Arm B renders prompts in the same process that generates; arm A splits that
  across a container hop. Small, but it is inside the end-to-end number.
- vLLM's advantage is concurrency, and the sweep drives it through one client.
  A concurrency sweep (many simultaneous single-item callers) would widen the
  gap further; this benchmark does not measure it.
