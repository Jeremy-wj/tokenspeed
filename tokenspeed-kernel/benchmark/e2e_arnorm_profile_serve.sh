#!/usr/bin/env bash
# Launch a model with a qualified torch or Proton profiling lifecycle.
# Usage:
#   e2e_arnorm_profile_serve.sh <torch|proton-roctracer-graph|proton-rocprofiler-graph> \
#     <world_size> <visible_devices> <fusion_cap> <backend> [serve args...]
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

MODE="${1:?profile mode}"
WS="${2:?world size}"
HVD="${3:?visible devices}"
CAP="${4:?fusion cap}"
BACKEND="${5:-auto}"
shift 5

MODEL_LABEL="${MODEL_LABEL:?source an AR+RMSNorm model profile}"
HARDWARE_LABEL="${HARDWARE_LABEL:-unknown-hardware}"
export CONTAINER="${CONTAINER:-${TOKENSPEED_CONTAINER:-jeremwan-tokenspeed-profiler}}"
RESULT_ROOT="${AR_RMSNORM_RESULT_ROOT:-${SCRIPT_DIR}/results/ar_rmsnorm/raw}"
RUN_DATE="${RUN_DATE:-$(date -u +%F)}"
export RUN_ROOT="${RUN_ROOT:-${RESULT_ROOT}/runs/${MODEL_LABEL}/${HARDWARE_LABEL}/${RUN_DATE}}"
PROFILE_ROOT="${PROFILE_ROOT:-${RUN_ROOT}/profiles/${MODE}}"
export TOKENSPEED_PROFILER_DIR="${TOKENSPEED_PROFILER_DIR:-${PROFILE_ROOT}}"

case "$MODE" in
  torch)
    export RUN_LABEL="${RUN_LABEL:-profile_torch_${MODEL_LABEL}_tp${WS}}"
    export TOKENSPEED_PROFILE_FORWARD_MARKERS=1
    ;;
  proton-roctracer-graph)
    export RUN_LABEL="${RUN_LABEL:-profile_proton_roctracer_${MODEL_LABEL}_tp${WS}}"
    export TOKENSPEED_KERNEL_PROFILE=0
    export TOKENSPEED_KERNEL_PROFILE_EARLY=0
    export TOKENSPEED_KERNEL_PROFILE_BEFORE_GRAPHS=1
    export TOKENSPEED_KERNEL_PROFILE_GRAPH_SCOPES=1
    export TOKENSPEED_KERNEL_PROFILE_OUTPUT="${PROFILE_ROOT}/rank-{pid}"
    export TOKENSPEED_KERNEL_PROFILE_DATA=tree
    export TOKENSPEED_KERNEL_PROFILE_BACKEND=roctracer
    export TOKENSPEED_KERNEL_PROFILE_HOOK=triton
    export TOKENSPEED_KERNEL_PROFILE_OUTPUT_FORMAT=hatchet
    ;;
  proton-rocprofiler-graph)
    export RUN_LABEL="${RUN_LABEL:-profile_proton_rocprofiler_${MODEL_LABEL}_tp${WS}}"
    export TOKENSPEED_KERNEL_PROFILE=1
    export TOKENSPEED_KERNEL_PROFILE_EARLY=1
    export TOKENSPEED_KERNEL_PROFILE_EARLY_KEEP_ACTIVE=1
    export TOKENSPEED_KERNEL_PROFILE_GRAPH_SCOPES=1
    export TOKENSPEED_KERNEL_PROFILE_OUTPUT="${PROFILE_ROOT}/rank-{pid}"
    export TOKENSPEED_KERNEL_PROFILE_DATA=tree
    export TOKENSPEED_KERNEL_PROFILE_BACKEND=rocprofiler
    export TOKENSPEED_KERNEL_PROFILE_HOOK=triton
    export TOKENSPEED_KERNEL_PROFILE_OUTPUT_FORMAT=hatchet
    ;;
  *)
    echo "unsupported profile mode: ${MODE}" >&2
    exit 2
    ;;
esac

exec "$(dirname "$0")/e2e_arnorm_serve.sh" \
  "$WS" "$HVD" "$CAP" "$BACKEND" "$@"

