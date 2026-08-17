#!/usr/bin/env bash
#
# Run the vLLM-vs-transformers serving benchmark, one arm at a time.
#
# The sequencing is the point of this script: the two arms share one GPU, so
# running them together would measure contention rather than either strategy.
# Each arm is brought up, waited for, benchmarked, and torn down before the next
# one starts.
#
#   ./scripts/bench_ab.sh                 # both arms, then the comparison
#   ./scripts/bench_ab.sh --arm vllm      # arm A only
#   ./scripts/bench_ab.sh --arm hf        # arm B only
#   ./scripts/bench_ab.sh --batch-sizes 1,4,16 --repeats 3
#
# Requires the environment described in docs/SERVING_BENCHMARK_AB.md, most of it
# from an .env.bench file next to docker-compose.bench.yml.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

COMPOSE_FILE="docker-compose.bench.yml"
ENV_FILE="${BENCH_ENV_FILE:-.env.bench}"

# Read into this shell as well as handed to compose. Compose gets the file via
# --env-file, but the script itself needs BENCH_OUTPUT_DIR (where reports land)
# and TG_BASE_MODEL_PATH (where the client finds arm B's tokenizer); without
# this they would silently fall back to defaults that disagree with what the
# containers were given.
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

OUTPUT_DIR="${BENCH_OUTPUT_DIR:-./benchmark_ab}"
# Container-side path of arm B's base checkpoint. Must match the compose
# default for TG_BASE_MODEL_PATH.
BASE_MODEL_PATH="${TG_BASE_MODEL_PATH:-/base}"

ARMS="vllm hf"
BATCH_SIZES="1,2,4,8,16,32"
# Above the script's defaults of 3/1. vLLM captures CUDA graphs on its first
# real request and the transformers arm grows its allocator pools on its first
# few; one warm-up run is not enough to get either past that.
REPEATS=5
WARMUP=2
PAGE_ARGS=(--page-source synthetic)
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arm)          ARMS="$2"; shift 2 ;;
    --batch-sizes)  BATCH_SIZES="$2"; shift 2 ;;
    --repeats)      REPEATS="$2"; shift 2 ;;
    --warmup)       WARMUP="$2"; shift 2 ;;
    --page-file)    PAGE_ARGS=(--page-source both --page-file "$2"); shift 2 ;;
    --)             shift; EXTRA_ARGS=("$@"); break ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ -f "$ENV_FILE" ]]; then
  COMPOSE=(docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE")
else
  echo "No $ENV_FILE found; relying on the exported environment." >&2
  COMPOSE=(docker compose -f "$COMPOSE_FILE")
fi

# Fail here, naming the variable, rather than several minutes in with a compose
# interpolation error or a container that mounted the wrong tree.
for required in TG_MERGED_MODEL_DIR TG_BASE_MODEL_DIR TG_ADAPTER_DIR HF_CACHE_DIR; do
  path="${!required:-}"
  if [[ -z "$path" ]]; then
    echo "$required is not set (see .env.bench.example)." >&2; exit 2
  fi
  if [[ "$path" != /* ]]; then
    echo "$required must be an absolute path, got '$path'. A relative source is parsed as a named volume." >&2; exit 2
  fi
  if [[ ! -d "$path" ]]; then
    echo "$required points at '$path', which is not a directory on this host." >&2; exit 2
  fi
done

# A directory that exists but holds no config.json is not a checkpoint. Caught
# here because the failure it otherwise produces is a traceback from inside
# vLLM or transformers, minutes later, that does not name the mount.
if [[ ! -f "$TG_MERGED_MODEL_DIR/config.json" ]]; then
  echo "TG_MERGED_MODEL_DIR=$TG_MERGED_MODEL_DIR has no config.json; not a checkpoint directory." >&2
  exit 2
fi
if [[ ! -f "$TG_ADAPTER_DIR/adapter_config.json" ]]; then
  echo "TG_ADAPTER_DIR=$TG_ADAPTER_DIR has no adapter_config.json; not a PEFT adapter directory." >&2
  exit 2
fi

# TG_BASE_MODEL_PATH is a CONTAINER path under /base. Map it back to the host to
# check it before mounting: a mistyped HuggingFace cache directory (the repo
# separator is a DOUBLE dash, models--google--name) otherwise fails only when
# arm B starts, long after arm A has finished.
if [[ "$BASE_MODEL_PATH" == /base ]]; then
  _host_base="$TG_BASE_MODEL_DIR"
elif [[ "$BASE_MODEL_PATH" == /base/* ]]; then
  _host_base="$TG_BASE_MODEL_DIR/${BASE_MODEL_PATH#/base/}"
else
  echo "TG_BASE_MODEL_PATH=$BASE_MODEL_PATH must be /base or a path under it." >&2
  exit 2
fi
# -e, which follows symlinks: cache snapshots are symlinks into ../../blobs/,
# and a snapshot whose blobs are outside the mount is exactly the failure mode
# this is guarding.
if [[ ! -e "$_host_base/config.json" ]]; then
  echo "TG_BASE_MODEL_PATH=$BASE_MODEL_PATH resolves to $_host_base on this host, which has no readable config.json." >&2
  echo "For a cached model the repo directory is models--<org>--<name> (double dashes)." >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"

# The sweep sends up to this many texts in one /translate/batch call. The server
# re-chunks by TG_BATCH_SIZE, so that setting must be at least this large or the
# large-batch rows quietly become several sequential chunks on both arms.
MAX_BATCH="${BATCH_SIZES##*,}"
export TG_BENCH_MAX_BATCH="$MAX_BATCH"

log() { printf '\n=== %s ===\n' "$*"; }

# Waits for compose's own healthcheck rather than polling the port: a vLLM
# server accepts connections well before it has finished capturing CUDA graphs,
# and benchmarking it in that window measures the capture.
wait_healthy() {
  local service="$1" deadline=$((SECONDS + ${2:-1200}))
  log "waiting for $service to become healthy"
  while (( SECONDS < deadline )); do
    local cid status
    cid="$("${COMPOSE[@]}" ps -q "$service" 2>/dev/null || true)"
    if [[ -n "$cid" ]]; then
      status="$(docker inspect -f '{{.State.Health.Status}}' "$cid" 2>/dev/null || echo unknown)"
      case "$status" in
        healthy)   echo "$service healthy after ${SECONDS}s"; return 0 ;;
        unhealthy) echo "$service reported unhealthy" >&2; "${COMPOSE[@]}" logs --tail 50 "$service" >&2; return 1 ;;
      esac
    fi
    sleep 10
  done
  echo "Timed out waiting for $service" >&2
  "${COMPOSE[@]}" logs --tail 50 "$service" >&2
  return 1
}

# Everything down, both profiles, before and after every arm. Leaving one arm's
# weights resident would shrink the other's usable VRAM.
teardown_all() {
  "${COMPOSE[@]}" --profile vllm --profile hf --profile bench down --remove-orphans >/dev/null 2>&1 || true
}

dump_logs() {
  echo "--- logs ---" >&2
  "${COMPOSE[@]}" --profile vllm --profile hf logs --tail "${BENCH_LOG_TAIL:-200}" >&2 2>/dev/null || true
}

# Teardown destroys the containers, and with them the logs of whatever just
# failed. Anything that exits nonzero therefore dumps them first. Set
# BENCH_KEEP_ON_FAILURE=1 to leave the containers up for `docker compose exec`
# and manual poking -- remember they hold the GPU until torn down by hand.
on_exit() {
  local status=$?
  if (( status != 0 )); then
    echo "bench_ab.sh failed (exit $status)." >&2
    dump_logs
    if [[ "${BENCH_KEEP_ON_FAILURE:-0}" == "1" ]]; then
      echo "BENCH_KEEP_ON_FAILURE=1, leaving containers up. Tear down with:" >&2
      echo "  docker compose --env-file $ENV_FILE -f $COMPOSE_FILE --profile vllm --profile hf --profile bench down" >&2
      return
    fi
  fi
  teardown_all
}
trap on_exit EXIT

# `up -d` failing leaves an exited container whose logs are the only useful
# diagnostic, so the failure is caught here rather than by set -e.
start_profile() {
  local profile="$1"
  if ! "${COMPOSE[@]}" --profile "$profile" up -d; then
    echo "Failed to start the '$profile' profile." >&2
    return 1
  fi
}

run_bench() {
  local tag="$1" api_url="$2" tokenizer_dir="$3"
  log "benchmarking $tag"
  # TG_BENCH_TOKENIZER_DIR points the client at the checkpoint the arm under
  # test actually serves. Token counts are recovered by re-encoding the
  # returned text, so a mismatched tokenizer would silently skew them.
  TG_BENCH_TOKENIZER_DIR="$tokenizer_dir" \
    "${COMPOSE[@]}" --profile bench run --rm bench \
      --api-url "$api_url" \
      --batch-sizes "$BATCH_SIZES" \
      --repeats "$REPEATS" \
      --warmup "$WARMUP" \
      "${PAGE_ARGS[@]}" \
      --output-dir /out \
      --tag "$tag" \
      "${EXTRA_ARGS[@]}"
}

for arm in $ARMS; do
  teardown_all
  case "$arm" in
    vllm)
      log "arm A: vLLM serving the merged checkpoint"
      start_profile vllm
      wait_healthy translategemma-vllm 1800
      wait_healthy translategemma-api 300
      run_bench vllm-bf16 http://translategemma-api:8000 /merged
      ;;
    hf)
      log "arm B: FastAPI + transformers, base weights + LoRA adapter"
      start_profile hf
      wait_healthy translategemma-api-hf 1800
      run_bench hf-lora http://translategemma-api-hf:8000 "$BASE_MODEL_PATH"
      ;;
    *) echo "Unknown arm: $arm" >&2; exit 2 ;;
  esac
done

teardown_all

log "comparison"
# The deployment hosts do not necessarily have a Python interpreter -- the whole
# stack ships as images. Prefer the host's, fall back to the gateway image,
# which is already pulled and carries a 3.12.
compare_reports() {
  if command -v python3 >/dev/null 2>&1; then
    python3 scripts/compare_bench_reports.py "$OUTPUT_DIR"
    return
  fi
  echo "No host python3; running the comparison in ${TG_API_IMAGE:-translategemma-api-gw:latest}." >&2
  docker run --rm \
    --user "$(id -u):$(id -g)" \
    -v "$REPO_ROOT/scripts/compare_bench_reports.py:/compare.py:ro" \
    -v "$(cd "$OUTPUT_DIR" && pwd):/out:ro" \
    --entrypoint python3 "${TG_API_IMAGE:-translategemma-api-gw:latest}" \
    /compare.py /out
}
compare_reports || {
  echo "Comparison skipped or failed; the per-arm reports are in $OUTPUT_DIR." >&2
}
