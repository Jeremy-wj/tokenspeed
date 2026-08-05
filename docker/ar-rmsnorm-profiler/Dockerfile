ARG BASE_IMAGE=lightseekorg/tokenspeed-runner-amd@sha256:cc64fe47dc2254677c2ad8ab7e66b851e003ce423d87f2f884f6ba385948f110
FROM ${BASE_IMAGE}

ARG MAX_JOBS=32
ARG SOURCE_COMMIT=unknown
ARG SOURCE_BRANCH=unknown
ARG SOURCE_DATE_EPOCH=0
ARG BASE_IMAGE_DIGEST=sha256:cc64fe47dc2254677c2ad8ab7e66b851e003ce423d87f2f884f6ba385948f110

LABEL org.opencontainers.image.title="TokenSpeed GLM AR+RMSNorm ROCm profiler" \
      org.opencontainers.image.description="Behaviorally faithful ROCm 7.2.4 / torch 2.11 profiler environment for the GLM-5.2-FP8 definitive operator campaign" \
      org.opencontainers.image.source="https://github.com/lightseekorg/tokenspeed" \
      org.opencontainers.image.revision="${SOURCE_COMMIT}" \
      io.tokenspeed.source.branch="${SOURCE_BRANCH}" \
      io.tokenspeed.base.digest="${BASE_IMAGE_DIGEST}" \
      io.tokenspeed.scheduler.flat_kvcache="OFF" \
      io.tokenspeed.environment.contract="ar-rmsnorm-glm-5.2-fp8-rocm7.2.4-v1" \
      io.tokenspeed.owner="jeremwan"

USER root

ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/rocm7.2 \
    PIP_CONSTRAINT=/opt/tokenspeed/constraints.txt \
    GFX_ARCH=gfx950 \
    ROCM_VERSION=7.2 \
    ROCM_PATH=/opt/rocm \
    PYTORCH_ROCM_ARCH=gfx950 \
    AITER_ROCM_ARCH=gfx950 \
    HSA_NO_SCRATCH_RECLAIM=1 \
    GIT_COMMIT=${SOURCE_COMMIT} \
    TOKENSPEED_KERNEL_GIT_BRANCH=${SOURCE_BRANCH} \
    SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH}

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        openmpi-bin \
        libopenmpi-dev \
        libssl-dev \
        libnuma1 \
        libnuma-dev \
        libibverbs1 \
        libibverbs-dev \
        rdma-core \
        infiniband-diags \
        ibverbs-providers \
        pkg-config \
        lsof \
        jq && \
    rm -rf /var/lib/apt/lists/*

COPY docker/ar-rmsnorm-profiler/constraints.txt /opt/tokenspeed/constraints.txt

RUN python3 -m pip install --upgrade pip "setuptools<82" wheel && \
    python3 -m pip install cmake ninja

COPY . /workspace

# Preserve the qualified install order. The GLM audit explicitly requires the
# scheduler to be rebuilt from current source with Flat KV disabled.
RUN MAX_JOBS=${MAX_JOBS} \
    python3 -m pip install --force-reinstall --no-deps \
        ./tokenspeed-kernel-amd --no-build-isolation && \
    MAX_JOBS=${MAX_JOBS} TOKENSPEED_KERNEL_BACKEND=rocm \
    python3 -m pip install ./tokenspeed-kernel/python --no-build-isolation && \
    MAX_JOBS=${MAX_JOBS} \
    python3 -m pip install ./tokenspeed-scheduler \
        --config-settings=cmake.define.TOKENSPEED_FLAT_KVCACHE=OFF && \
    python3 -m pip install ./python --no-build-isolation && \
    (python3 -m pip uninstall -y torchaudio || true)

# Docker builds have no GPU. Relocate the bundled runtime/tracing libraries now
# and run the graph-capture probe later in the qualified runtime container.
RUN cd /workspace/tokenspeed-kernel && \
    RUN_PROBE=0 bash benchmark/fix_torch_hip_bundling.sh

RUN python3 - <<'PY'
from importlib.metadata import PackageNotFoundError, version
import torch

expected = {
    "apache-tvm-ffi": "0.1.12",
    "tokenspeed-iris": "0.0.0.post20260511",
    "tokenspeed-proton": "3.8.10.post20260721",
    "tokenspeed-smg": "1.8.0.post20260728",
    "tokenspeed-smg-grpc-proto": "0.4.14.post20260728",
    "tokenspeed-smg-grpc-servicer": "0.7.0.post20260728",
    "tokenspeed-triton": "3.8.10.post20260721",
    "tokenspeed-triton-kernels": "1.0.0.post20260510",
    "torch-memory-saver": "0.0.9.post1",
    "transformers": "5.12.0",
    "xgrammar": "0.2.2",
}
actual = {name: version(name) for name in expected}
if actual != expected:
    raise SystemExit(f"package contract mismatch: expected={expected} actual={actual}")
if torch.__version__ != "2.11.0+rocm7.2":
    raise SystemExit(f"unexpected torch version: {torch.__version__}")
try:
    version("torchaudio")
except PackageNotFoundError:
    pass
else:
    raise SystemExit("torchaudio must be absent")
print("qualified package contract:", actual)
PY

RUN mkdir -p /opt/tokenspeed/audit && \
    python3 -m pip freeze --all | LC_ALL=C sort \
        > /opt/tokenspeed/audit/pip-freeze.txt && \
    sha256sum /opt/tokenspeed/constraints.txt \
        > /opt/tokenspeed/audit/constraints.sha256

CMD ["sleep", "infinity"]
