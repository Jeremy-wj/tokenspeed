#!/usr/bin/env bash
# Make a torch ROCm wheel use the container's SYSTEM ROCm runtime instead of
# its own bundled copy. Required to fix the ROCm 7.2.0 graph-capture bug
# (see benchmark/results/ar_rmsnorm/docs/profiling-workflow.md).
#
# Why: torch's ROCm wheels (e.g. `torch==2.11.0+rocm7.2`, the only torch-2.11
# ROCm build download.pytorch.org publishes) bundle libamdhip64/librccl/... in
# torch/lib and link them via DT_RPATH=$ORIGIN. RPATH wins over LD_LIBRARY_PATH,
# so the bundled (buggy 7.2.26015) HIP is used no matter what base image you run
# on. Relocating those bundled libs makes the loader fall back to the system
# ROCm in /opt/rocm (which must be >= 7.2.1; 7.2.4 preferred) via ldconfig.
#
# Idempotent. Run AFTER every torch (re)install, including after
# `install_deps_rocm.sh` (which force-reinstalls torch==2.11.0+rocm7.2 and thus
# re-bundles the buggy libs).
set -euo pipefail

ROCM_LIB="${ROCM_LIB:-/opt/rocm/lib}"
# Core ROCm *runtime* and tracing libs that must come from one system release.
# Keeping the wheel's ROCm 7.2.0 libroctracer beside system HIP/HSA 7.2.4 causes
# native Kineto/Proton activity-buffer faults. Compute libs
# (rocblas/hipblaslt/etc.) are intentionally left bundled.
LIBS=(libamdhip64 librccl libhsa-runtime64 libamd_comgr librocm-core \
      librocprofiler-register libroctx64 libroctracer64)

TL="$(python3 -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))')"
BK="${TL}/_bundled_rocm_backup"
mkdir -p "${BK}"

# Note: a plain ctypes load always resolves the SYSTEM libamdhip64 via ldconfig,
# so this reports the system runtime, not torch's RPATH-bundled one. The
# authoritative before/after check is the graph-capture probe at the end.
sys_hip_runtime() {
  python3 - <<'PY'
import ctypes
v = ctypes.c_int()
ctypes.CDLL("libamdhip64.so").hipRuntimeGetVersion(ctypes.byref(v))
print(v.value)
PY
}

echo "torch/lib          : ${TL}"
echo "system rocm        : $(cat /opt/rocm/.info/version 2>/dev/null || echo '?')"
echo "system HIP runtime : $(sys_hip_runtime 2>/dev/null || echo '?') (target for fallback)"

moved=0
for base in "${LIBS[@]}"; do
  # Only relocate if the system provides a replacement, else we'd break torch.
  if ! ls "${ROCM_LIB}/${base}.so"* >/dev/null 2>&1; then
    echo "skip ${base}: no system copy in ${ROCM_LIB}"
    continue
  fi
  for f in "${TL}/${base}.so"*; do
    [ -e "${f}" ] || continue
    mv "${f}" "${BK}/"
    echo "moved $(basename "${f}") -> _bundled_rocm_backup/"
    moved=$((moved + 1))
  done
done

echo "relocated ${moved} file(s)"
python3 -c 'import torch;print("torch", torch.__version__, "| cuda.is_available", torch.cuda.is_available())'

PROBE="$(dirname "$0")/probe_hip_event_query_capture.py"
if [ -f "${PROBE}" ] && [ "${RUN_PROBE:-1}" = "1" ]; then
  echo "=== runtime graph-capture probe (expect PASS on >=7.2.1) ==="
  python3 "${PROBE}"
fi
