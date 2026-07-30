#!/usr/bin/env bash
# Run a fixed workload against a model-profiled TokenSpeed server.
# Usage: e2e_arnorm_bench.sh <label> <input_len> <output_len> <prompts> <concurrency> [seed] [extra args...]
set -euo pipefail

LBL="${1:?label}"
IL="${2:?input length}"
OL="${3:?output length}"
NP="${4:?number of prompts}"
CC="${5:?concurrency}"
SEED="${6:-0}"
shift 6 || true

MODEL_PATH="${MODEL_PATH:?source an AR+RMSNorm model profile or set MODEL_PATH}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "$MODEL_PATH")}"
MODEL_LABEL="${MODEL_LABEL:-$SERVED_MODEL_NAME}"
HARDWARE_LABEL="${HARDWARE_LABEL:-unknown-hardware}"
PORT="${PORT:-8100}"
CONTAINER="${CONTAINER:-jeremwan-tokenspeed-profiler}"
RESULT_ROOT="${AR_RMSNORM_RESULT_ROOT:-/home/jeremwan/tokenspeed/tokenspeed-kernel/benchmark/results/ar_rmsnorm/raw}"
RUN_DATE="${RUN_DATE:-$(date -u +%F)}"
RUN_ROOT="${RUN_ROOT:-${RESULT_ROOT}/runs/${MODEL_LABEL}/${HARDWARE_LABEL}/${RUN_DATE}}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"
SAFE_LBL="${LBL//[^a-zA-Z0-9_.-]/_}"
LOG="${LOG_DIR}/bench-${SAFE_LBL}-seed${SEED}.log"

mkdir -p "$LOG_DIR"
echo "===== [${LBL}] model=${MODEL_LABEL} in=${IL} out=${OL} n=${NP} conc=${CC} seed=${SEED} ====="
docker exec "$CONTAINER" bash -lc "cd /home/jeremwan/tokenspeed && \
  PYTHONPATH=/home/jeremwan/tokenspeed/tokenspeed-kernel-amd/python:/home/jeremwan/tokenspeed/tokenspeed-kernel/python:/home/jeremwan/tokenspeed/python:${PYTHONPATH:-} \
  python -m tokenspeed.cli bench serve --backend openai --host 127.0.0.1 --port '${PORT}' \
    --model '${MODEL_PATH}' --served-model-name '${SERVED_MODEL_NAME}' \
    --dataset-name random --random-input-len '${IL}' --random-output-len '${OL}' \
    --num-prompts '${NP}' --request-rate inf --max-concurrency '${CC}' --seed '${SEED}' \
    --ignore-eos --extra-body '{\"temperature\":0}' $* 2>&1" | tee "$LOG"

echo "===== summary: ${LOG} ====="
rg 'Output token throughput|Total token throughput|Median TPOT|Mean TPOT|Median TTFT|Mean TTFT|Benchmark duration|Request throughput' "$LOG"

