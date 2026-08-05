#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SOURCE_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"

CONTAINER="${CONTAINER:-jeremwan-ar-rmsnorm-profiler-mi355x-${SOURCE_COMMIT:0:8}}"
MODEL_PATH="${MODEL_PATH:-/data/models/glm-5.2-fp8}"
WS2_DEVICES="${WS2_DEVICES:-1,2}"
WS4_DEVICES="${WS4_DEVICES:-1,2,4,5}"
WS8_DEVICES="${WS8_DEVICES:-0,1,2,3,4,5,6,7}"
SINGLE_DEVICE="${SINGLE_DEVICE:-1}"
STAMP="${STAMP:-$(date -u +%Y-%m-%dT%H%M%SZ)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${REPO_ROOT}/tokenspeed-kernel/benchmark/results/ar_rmsnorm/raw/current/glm-5.2-fp8/mi355x/${STAMP}/environment-qualification}"

CONTAINER_REPO_ROOT=/home/jeremwan/tokenspeed
KERNEL_ROOT="${CONTAINER_REPO_ROOT}/tokenspeed-kernel"
PYTHONPATH_VALUE="${CONTAINER_REPO_ROOT}/tokenspeed-kernel-amd/python:${CONTAINER_REPO_ROOT}/tokenspeed-kernel/python:${CONTAINER_REPO_ROOT}/python:${KERNEL_ROOT}:${KERNEL_ROOT}/benchmark"

mkdir -p "${ARTIFACT_ROOT}"/{audit,logs,profiles,smoke}

if ! docker container inspect "${CONTAINER}" >/dev/null 2>&1; then
  echo "container does not exist: ${CONTAINER}" >&2
  exit 1
fi
if [[ "$(docker container inspect --format '{{.State.Running}}' "${CONTAINER}")" != "true" ]]; then
  echo "container is not running: ${CONTAINER}" >&2
  exit 1
fi
if [[ "$(docker container inspect --format '{{ index .Config.Labels "io.tokenspeed.owner" }}' "${CONTAINER}")" != "jeremwan" ]]; then
  echo "refusing container without jeremwan ownership label: ${CONTAINER}" >&2
  exit 1
fi

run_logged() {
  local name="$1"
  shift
  echo "=== ${name} ==="
  timeout --signal=TERM --kill-after=30s 15m "$@" \
    > >(tee "${ARTIFACT_ROOT}/logs/${name}.log") \
    2> >(tee "${ARTIFACT_ROOT}/logs/${name}.err" >&2)
}

container_bash() {
  local name="$1"
  local command="$2"
  run_logged "${name}" docker exec \
    -e "MODEL_PATH=${MODEL_PATH}" \
    -e "PYTHONPATH=${PYTHONPATH_VALUE}" \
    "${CONTAINER}" bash -lc "${command}"
}

guard_gpus() {
  local name="$1"
  local snapshot="${ARTIFACT_ROOT}/audit/gpu-processes-${name}.json"
  amd-smi process --json >"${snapshot}"
  python3 - "${snapshot}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
busy = []
for entry in payload:
    gpu = entry.get("gpu")
    for process in entry.get("process_list", []):
        if process.get("process_info") != "No running processes detected":
            busy.append({"gpu": gpu, "process": process})
if busy:
    raise SystemExit(f"foreign GPU processes detected: {busy}")
if len(payload) != 8:
    raise SystemExit(f"expected 8 GPUs, found {len(payload)}")
print("PASS: all eight GPUs idle")
PY
}

IMAGE="$(
  docker container inspect --format '{{.Image}}' "${CONTAINER}"
)"
docker image inspect "${IMAGE}" >"${ARTIFACT_ROOT}/audit/image-inspect.json"
docker container inspect "${CONTAINER}" >"${ARTIFACT_ROOT}/audit/container-inspect.json"
git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=all \
  >"${ARTIFACT_ROOT}/audit/git-status.txt"
git -C "${REPO_ROOT}" rev-parse HEAD \
  >"${ARTIFACT_ROOT}/audit/git-head.txt"
amd-smi list >"${ARTIFACT_ROOT}/audit/amd-smi-list.txt"
amd-smi static --json >"${ARTIFACT_ROOT}/audit/amd-smi-static.json"
amd-smi topology --json >"${ARTIFACT_ROOT}/audit/amd-smi-topology.json"

python3 - "${MODEL_PATH}" >"${ARTIFACT_ROOT}/audit/model-identity.json" <<'PY'
import glob
import json
import os
import sys

root = os.path.realpath(sys.argv[1])
with open(os.path.join(root, "config.json"), encoding="utf-8") as handle:
    config = json.load(handle)
with open(os.path.join(root, "model.safetensors.index.json"), encoding="utf-8") as handle:
    index = json.load(handle)
shards = sorted(glob.glob(os.path.join(root, "model-*-of-*.safetensors")))
payload = {
    "path": root,
    "architectures": config.get("architectures"),
    "model_type": config.get("model_type"),
    "hidden_size": config.get("hidden_size"),
    "num_hidden_layers": config.get("num_hidden_layers"),
    "quantization_config": config.get("quantization_config"),
    "index_total_size": index.get("metadata", {}).get("total_size"),
    "shard_count": len(shards),
    "all_shards_readable": all(os.access(path, os.R_OK) for path in shards),
}
expected = {
    "hidden_size": 6144,
    "num_hidden_layers": 78,
    "index_total_size": 755617140416,
    "shard_count": 141,
    "all_shards_readable": True,
}
bad = {key: (payload.get(key), value) for key, value in expected.items() if payload.get(key) != value}
if bad:
    raise SystemExit(f"model identity mismatch: {bad}")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

container_bash runtime-audit "
  set -euo pipefail
  python3 - <<'PY'
import ctypes
import json
from importlib.metadata import PackageNotFoundError, version
import torch

hip = ctypes.CDLL('libamdhip64.so')
runtime = ctypes.c_int()
if hip.hipRuntimeGetVersion(ctypes.byref(runtime)) != 0:
    raise SystemExit('hipRuntimeGetVersion failed')
packages = {}
for name in (
    'apache-tvm-ffi', 'tokenspeed-iris', 'tokenspeed-kernel',
    'tokenspeed-kernel-amd', 'tokenspeed-mooncake', 'tokenspeed-proton',
    'tokenspeed-scheduler', 'tokenspeed-smg', 'tokenspeed-smg-grpc-proto',
    'tokenspeed-smg-grpc-servicer', 'tokenspeed-triton',
    'tokenspeed-triton-kernels', 'torch-memory-saver', 'transformers',
    'xgrammar',
):
    packages[name] = version(name)
try:
    packages['torchaudio'] = version('torchaudio')
except PackageNotFoundError:
    packages['torchaudio'] = None
payload = {
    'python_torch': torch.__version__,
    'torch_hip_compile_label': torch.version.hip,
    'hip_runtime': runtime.value,
    'cuda_available': torch.cuda.is_available(),
    'device_count': torch.cuda.device_count(),
    'packages': packages,
}
if payload['python_torch'] != '2.11.0+rocm7.2':
    raise SystemExit(payload)
if payload['hip_runtime'] != 70253211:
    raise SystemExit(payload)
if payload['device_count'] != 8 or not payload['cuda_available']:
    raise SystemExit(payload)
if packages['torchaudio'] is not None:
    raise SystemExit(payload)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
  python3 -m pip freeze --all
  /opt/rocm/bin/rocprofv3 --version
  cat /opt/rocm/.info/version
"

container_bash loader-audit "
  set -euo pipefail
  TORCH_LIB=\$(python3 -c 'import os,torch; print(os.path.join(os.path.dirname(torch.__file__), \"lib\", \"libtorch_hip.so\"))')
  ldd \"\${TORCH_LIB}\" | awk '/amdhip|hsa|rccl|roctx|roctracer/'
  python3 - \"\${TORCH_LIB}\" <<'PY'
import subprocess
import sys

targets = ('libamdhip64', 'libhsa-runtime64', 'librccl', 'libroctx64', 'libroctracer64')
lines = subprocess.run(['ldd', sys.argv[1]], check=True, capture_output=True, text=True).stdout.splitlines()
resolved = {target: next((line.strip() for line in lines if target in line), None) for target in targets}
bad = {key: value for key, value in resolved.items() if value is not None and '/opt/rocm/' not in value}
missing = [key for key, value in resolved.items() if value is None]
if bad or missing:
    raise SystemExit(f'loader mismatch bad={bad} missing={missing}')
print('PASS:', resolved)
PY
"

container_bash gpu-mapping "
  python3 - <<'PY'
import ctypes
import json
import torch

hip = ctypes.CDLL('libamdhip64.so')
mapping = []
for ordinal in range(torch.cuda.device_count()):
    bus_id = ctypes.create_string_buffer(32)
    rc = hip.hipDeviceGetPCIBusId(bus_id, len(bus_id), ordinal)
    if rc != 0:
        raise SystemExit(f'hipDeviceGetPCIBusId failed ordinal={ordinal} rc={rc}')
    props = torch.cuda.get_device_properties(ordinal)
    mapping.append({
        'hip_ordinal': ordinal,
        'pci_bus_id': bus_id.value.decode(),
        'name': props.name,
        'total_memory': props.total_memory,
    })
print(json.dumps(mapping, indent=2))
PY
"

guard_gpus hip-event-query
container_bash hip-event-query "
  cd '${KERNEL_ROOT}'
  HIP_VISIBLE_DEVICES='${SINGLE_DEVICE}' \
    python3 benchmark/probe_hip_event_query_capture.py
"

guard_gpus torch-kineto-graph
container_bash torch-kineto-graph "
  cd '${KERNEL_ROOT}'
  HIP_VISIBLE_DEVICES='${SINGLE_DEVICE}' \
    python3 -m benchmark.probe_rocm_profiler_stack \
      --profiler torch --graph --repeats 8 --record-shapes \
      --output '${ARTIFACT_ROOT}/profiles/torch-kineto-graph.json'
"

guard_gpus proton-roctracer-eager
container_bash proton-roctracer-eager "
  cd '${KERNEL_ROOT}'
  ROCR_VISIBLE_DEVICES='${SINGLE_DEVICE}' \
    python3 -m benchmark.probe_rocm_profiler_stack \
      --profiler proton --proton-backend roctracer \
      --proton-data trace --proton-output-format chrome_trace \
      --repeats 8 \
      --output '${ARTIFACT_ROOT}/profiles/proton-roctracer-eager'
"

guard_gpus proton-rocprofiler-graph
container_bash proton-rocprofiler-graph "
  cd '${KERNEL_ROOT}'
  ROCR_VISIBLE_DEVICES='${SINGLE_DEVICE}' \
    python3 -m benchmark.probe_rocm_profiler_stack \
      --profiler proton --proton-backend rocprofiler \
      --proton-data tree --proton-output-format hatchet \
      --graph --profile-before-runtime --profile-before-graph \
      --graph-scopes --replay-scope --repeats 8 \
      --output '${ARTIFACT_ROOT}/profiles/proton-rocprofiler-graph'
"

container_bash profiler-artifact-validation "
  python3 - <<'PY'
import glob
import json
import os

root = '${ARTIFACT_ROOT}/profiles'
torch_path = os.path.join(root, 'torch-kineto-graph.json')
with open(torch_path, encoding='utf-8') as handle:
    trace = json.load(handle)
if not trace:
    raise SystemExit('empty Kineto trace')
patterns = {
    'proton_roctracer': os.path.join(root, 'proton-roctracer-eager*'),
    'proton_rocprofiler': os.path.join(root, 'proton-rocprofiler-graph*'),
}
resolved = {}
for name, pattern in patterns.items():
    files = [path for path in glob.glob(pattern) if os.path.isfile(path) and os.path.getsize(path) > 0]
    if not files:
        raise SystemExit(f'missing nonempty {name} output for {pattern}')
    resolved[name] = [{'path': path, 'bytes': os.path.getsize(path)} for path in files]
print(json.dumps(resolved, indent=2))
PY
"

guard_gpus communication-tests
container_bash communication-tests "
  cd '${KERNEL_ROOT}'
  HIP_VISIBLE_DEVICES='${WS8_DEVICES}' \
    pytest -q test/ops/test_triton_shmem_communication.py
"

container_bash campaign-unit-tests "
  cd '${KERNEL_ROOT}'
  HIP_VISIBLE_DEVICES='${SINGLE_DEVICE}' \
    pytest -q test/test_ar_rmsnorm_graph_sweep.py \
      test/test_ar_rmsnorm_eager_sweep.py
"

run_operator_arm() {
  local arm="$1"
  local impl="$2"
  local overrides="$3"
  guard_gpus "operator-${arm}"
  container_bash "operator-${arm}" "
    cd '${KERNEL_ROOT}'
    source benchmark/profiles/ar_rmsnorm/glm_5_2_fp8_mi350x.env
    export MODEL_PATH='${MODEL_PATH}'
    export HIP_VISIBLE_DEVICES='${WS8_DEVICES}'
    export TS_TRITON_SHMEM_VISIBLE_DEVICES='${WS8_DEVICES}'
    export BENCH_WS=8 BENCH_N=6144 BENCH_M=42 BENCH_MAX_TOKEN_NUM=42
    export BENCH_CALLS_PER_GRAPH=156 BENCH_N_WARMUP=50 BENCH_N_REPEAT=1000
    export BENCH_IMPL='${impl}'
    export BENCH_JSON='${ARTIFACT_ROOT}/smoke/operator-${arm}.json'
    ${overrides}
    python3 -m benchmark.probe_ar_rmsnorm_graph_perf
  "
}

run_operator_arm upstream-unfused production_unfused ":"
run_operator_arm iris-fused auto ":"
run_operator_arm triton-forced triton_shmem "
  export AR_NORM_PROFILE_ID=unqualified-manual
  export TS_TRITON_SHMEM_FUSION_MIN_M=1
"

guard_gpus glm-profile-transition
container_bash glm-profile-transition "
  cd '${KERNEL_ROOT}'
  source benchmark/profiles/ar_rmsnorm/glm_5_2_fp8_mi350x.env
  export MODEL_PATH='${MODEL_PATH}'
  export HIP_VISIBLE_DEVICES='${WS8_DEVICES}'
  export TS_TRITON_SHMEM_VISIBLE_DEVICES='${WS8_DEVICES}'
  export BENCH_IMPL=triton_shmem_profile BENCH_WS=8 BENCH_N=6144
  export PROBE_MS='1,2,16,32,33,40,42'
  export PROBE_REPLAYS=1000 PROBE_MAX_REPLAYS=1000 PROBE_TIMEOUT_S=840
  export PROBE_JSON='${ARTIFACT_ROOT}/smoke/glm-profile-transition.json'
  python3 -m benchmark.probe_ar_rmsnorm_transitions
"

container_bash definitive-campaign-dry-run "
  cd '${KERNEL_ROOT}'
  source benchmark/profiles/ar_rmsnorm/glm_5_2_fp8_mi350x.env
  export MODEL_PATH='${MODEL_PATH}'
  python3 benchmark/run_ar_rmsnorm_graph_sweep.py \
    --spec benchmark/results/ar_rmsnorm/studies/mi350x/2026-08-glm-5.2-fp8-definitive-sweep/campaign.json \
    --devices '2=${WS2_DEVICES}' \
    --devices '4=${WS4_DEVICES}' \
    --devices '8=${WS8_DEVICES}' \
    --dry-run | tee '${ARTIFACT_ROOT}/smoke/definitive-campaign-dry-run.json'
"

python3 - "${ARTIFACT_ROOT}/smoke/definitive-campaign-dry-run.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("processes") != 315:
    raise SystemExit(f"expected 315 processes, got {payload.get('processes')}")
print("PASS: definitive dry run contains 315 processes")
PY

python3 - "${ARTIFACT_ROOT}" "${CONTAINER}" "${IMAGE}" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

root, container, image = sys.argv[1:]
files = []
for directory, _, names in os.walk(root):
    for name in sorted(names):
        path = os.path.join(directory, name)
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        files.append({
            "path": os.path.relpath(path, root),
            "bytes": os.path.getsize(path),
            "sha256": digest.hexdigest(),
        })
manifest = {
    "schema": "tokenspeed.ar-rmsnorm.environment-qualification.v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "container": container,
    "image_id": image,
    "status": "pass",
    "files": files,
}
with open(os.path.join(root, "qualification-manifest.json"), "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

echo "PASS: qualification artifacts: ${ARTIFACT_ROOT}"
