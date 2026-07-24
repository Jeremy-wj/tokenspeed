#!/usr/bin/env bash
# Launch gpt-oss-120b with a qualified profiling lifecycle.
# Usage:
#   profile_serve.sh <torch|proton-roctracer-graph|proton-rocprofiler-graph> \
#     <world_size> <visible_devices> <fusion_cap> <arnorm_backend> [serve args...]
#
# Torch mode: run e2e_gptoss_bench.sh with --profile-activities CPU GPU.
# Proton graph modes: run the benchmark WITHOUT --profile, then finalize with:
#   docker exec "$CONTAINER" curl -sS -X POST http://127.0.0.1:8101/stop_profile \
#     -H 'Content-Type: application/json' -d '{}'
set -euo pipefail

MODE="${1:?profile mode}"
WS="${2:?world size}"
HVD="${3:?visible devices}"
CAP="${4:?fusion cap}"
BACKEND="${5:-auto}"
shift 5

export CONTAINER="${CONTAINER:-jeremwan-tokenspeed-profiler}"
PROFILE_ROOT="${PROFILE_ROOT:-/home/jeremwan/ar_rmsnorm_profiles/qualified}"
export TOKENSPEED_PROFILER_DIR="${TOKENSPEED_PROFILER_DIR:-${PROFILE_ROOT}/${MODE}}"

case "${MODE}" in
  torch)
    export RUN_LABEL="${RUN_LABEL:-profile_torch_ws${WS}}"
    ;;
  proton-roctracer-graph)
    export RUN_LABEL="${RUN_LABEL:-profile_proton_roctracer_graph_ws${WS}}"
    export TOKENSPEED_KERNEL_PROFILE=0
    export TOKENSPEED_KERNEL_PROFILE_EARLY=0
    export TOKENSPEED_KERNEL_PROFILE_BEFORE_GRAPHS=1
    export TOKENSPEED_KERNEL_PROFILE_GRAPH_SCOPES=1
    export TOKENSPEED_KERNEL_PROFILE_OUTPUT="${PROFILE_ROOT}/${MODE}/rank-{pid}"
    export TOKENSPEED_KERNEL_PROFILE_DATA=tree
    export TOKENSPEED_KERNEL_PROFILE_BACKEND=roctracer
    export TOKENSPEED_KERNEL_PROFILE_HOOK=triton
    export TOKENSPEED_KERNEL_PROFILE_OUTPUT_FORMAT=hatchet
    ;;
  proton-rocprofiler-graph)
    export RUN_LABEL="${RUN_LABEL:-profile_proton_rocprofiler_graph_ws${WS}}"
    export TOKENSPEED_KERNEL_PROFILE=1
    export TOKENSPEED_KERNEL_PROFILE_EARLY=1
    export TOKENSPEED_KERNEL_PROFILE_EARLY_KEEP_ACTIVE=1
    export TOKENSPEED_KERNEL_PROFILE_GRAPH_SCOPES=1
    export TOKENSPEED_KERNEL_PROFILE_OUTPUT="${PROFILE_ROOT}/${MODE}/rank-{pid}"
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

exec "$(dirname "$0")/e2e_gptoss_serve.sh" \
  "${WS}" "${HVD}" "${CAP}" "${BACKEND}" "$@"
