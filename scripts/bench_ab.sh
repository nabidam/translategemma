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
OUTPUT_DIR="${BENCH_OUTPUT_DIR:-./benchmark_ab}"

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
trap teardown_all EXIT

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
      "${COMPOSE[@]}" --profile vllm up -d
      wait_healthy translategemma-vllm 1800
      wait_healthy translategemma-api 300
      run_bench vllm-bf16 http://translategemma-api:8000 /merged
      ;;
    hf)
      log "arm B: FastAPI + transformers, base weights + LoRA adapter"
      "${COMPOSE[@]}" --profile hf up -d
      wait_healthy translategemma-api-hf 1800
      run_bench hf-lora http://translategemma-api-hf:8000 /base
      ;;
    *) echo "Unknown arm: $arm" >&2; exit 2 ;;
  esac
done

teardown_all

log "comparison"
python3 scripts/compare_bench_reports.py "$OUTPUT_DIR" || {
  echo "Comparison skipped or failed; the per-arm reports are in $OUTPUT_DIR." >&2
}
