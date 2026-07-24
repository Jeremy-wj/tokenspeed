#!/usr/bin/env bash
# Run a fixed workload, retain the full output, and print key metrics.
# Usage: bench.sh <label> <input_len> <output_len> <num_prompts> <concurrency> [seed] [extra bench args...]
set -euo pipefail
LBL="${1:?label}"; IL="${2:?in}"; OL="${3:?out}"; NP="${4:?nprompts}"; CC="${5:?conc}"; SEED="${6:-0}"; shift 6 || true
PORT="${PORT:-8100}"
CONTAINER="${CONTAINER:-jeremwan-tokenspeed}"
LOG_DIR="${LOG_DIR:-/home/jeremwan/ar_rmsnorm_e2e/logs}"
SAFE_LBL="${LBL//[^a-zA-Z0-9_.-]/_}"
LOG="${LOG_DIR}/bench_${SAFE_LBL}_seed${SEED}.log"
mkdir -p "$LOG_DIR"
echo "===== [${LBL}] in=${IL} out=${OL} n=${NP} conc=${CC} seed=${SEED} ====="
docker exec "$CONTAINER" bash -lc "cd /home/jeremwan/tokenspeed && \
  python -m tokenspeed.cli bench serve --backend openai --host 127.0.0.1 --port ${PORT} \
    --model /data/models/openai/gpt-oss-120b --served-model-name gpt-oss-120b \
    --dataset-name random --random-input-len ${IL} --random-output-len ${OL} \
    --num-prompts ${NP} --request-rate inf --max-concurrency ${CC} --seed ${SEED} \
    --ignore-eos --extra-body '{\"temperature\":0}' $* 2>&1" | tee "$LOG"

echo "===== summary: ${LOG} ====="
grep -E 'Output token throughput|Total token throughput|Median TPOT|Mean TPOT|Median TTFT|Mean TTFT|Benchmark duration|Request throughput' "$LOG"
