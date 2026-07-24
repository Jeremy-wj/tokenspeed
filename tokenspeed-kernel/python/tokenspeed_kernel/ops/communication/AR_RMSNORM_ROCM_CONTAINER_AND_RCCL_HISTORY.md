# AR+RMSNorm ROCm container and RCCL incident history

This document preserves the environment provenance and incident evidence that
informed the current fused AR+RMSNorm deployment. It is historical reference, not
the benchmark decision record. Current performance results live in
`AR_RMSNORM_MI350X_E2E_BENCHMARKS.md`; the backend design lives in
`AR_RMSNORM_SYMM_MEM_MIGRATION.md`.

## 1. Current verified artifact

- Container: `jeremwan-tokenspeed`
- Image: `jeremwan/tokenspeed:rocm7.2.4-torch2.11`
- Image ID:
  `sha256:96f8b38d54c7da59f7888def76be81e99bf7512117bb2769609fadc7f19d230f`
- Parent working environment: container `ts-migrate-mi350x`, image
  `diprajap-tokenspeed:serve-base`
  (`sha256:817320835887...`)
- Created: 2026-07-22 UTC; local size about 77.2 GB
- Registry state: local tag only (`RepoDigests=[]`). This is not a portable
  registry-pinned artifact and was not produced with `docker save`/`docker load`.
- Runtime: torch `2.11.0+rocm7.2`, compiled-HIP label `7.2.26015`, but
  `libtorch_hip.so` resolves `libamdhip64.so` and `librccl.so` from
  `/opt/rocm/lib`; `/opt/rocm/.info/version` is `7.2.4`.

The image is a committed derivative of the known-working serve container. That
choice retained its Ubuntu 22.04, Python 3.10, torch 2.11, model-serving packages,
and editable-install layout. A fresh `rocm/pytorch` base would be cleaner but would
not recreate the same serving environment.

“Verified” means the library, deterministic runtime, captured RCCL, communication,
and final ws=2/4/8 serving gates pass. The image fixes the HIP runtime defect; the
kernel barrier fixes in §6 are additionally required for stable serving.

## 2. Canonical recreation

The investigation used several temporary images to isolate the runtime failure.
They are not required to recreate the final artifact. The shortest faithful path
is one snapshot, one throwaway upgrade container, one final commit:

```bash
# Preserve the source container; do not modify diprajap-tokenspeed:serve-base.
docker commit ts-migrate-mi350x jeremwan/tokenspeed:_pre724

docker run -d --name jeremwan-ts-upgrade \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  --ipc=host --shm-size=16g --network=host \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /home/jeremwan:/home/jeremwan -v /data:/data \
  -w /home/jeremwan/tokenspeed -e HSA_ENABLE_IPC_MODE_LEGACY=1 \
  jeremwan/tokenspeed:_pre724 sleep infinity
```

Inside `jeremwan-ts-upgrade`, repoint only the ROCm repository and upgrade exactly
the installed ROCm userspace family. Keep `amdgpu` and `libdrm` packages on their
existing repository because they interface with the host driver.

```bash
sed -i 's#rocm/apt/7.2 #rocm/apt/7.2.4 #' \
  /etc/apt/sources.list.d/rocm.list
apt-get update
apt-get install -y \
  amd-smi-lib comgr composablekernel-dev half hip-dev hip-doc hip-runtime-amd \
  hip-samples hipblas hipblas-common-dev hipblas-dev hipblaslt hipblaslt-dev \
  hipcc hipcub-dev hipfft hipfft-dev hipfort-dev hipify-clang hiprand \
  hiprand-dev hipsolver hipsolver-dev hipsparse hipsparse-dev hipsparselt \
  hipsparselt-dev hiptensor hiptensor-dev hsa-amd-aqlprofile hsa-rocr \
  hsa-rocr-dev miopen-hip miopen-hip-dev openmp-extras-dev \
  openmp-extras-runtime rccl rccl-dev rocblas rocblas-dev rocfft rocfft-dev \
  rocm rocm-cmake rocm-core rocm-dbgapi rocm-debug-agent \
  rocm-developer-tools rocm-device-libs rocm-gdb rocm-hip rocm-llvm \
  rocm-opencl rocm-opencl-dev rocm-opencl-sdk rocm-openmp rocm-smi-lib \
  rocminfo rocprim-dev rocprofiler rocprofiler-compute rocprofiler-dev \
  rocprofiler-plugins rocprofiler-register rocprofiler-sdk \
  rocprofiler-sdk-rocpd rocprofiler-sdk-roctx rocprofiler-systems rocrand \
  rocrand-dev rocsolver rocsolver-dev rocsparse rocsparse-dev rocthrust-dev \
  roctracer roctracer-dev rocwmma-dev

cd /home/jeremwan/tokenspeed/tokenspeed-kernel
bash benchmark/fix_torch_hip_bundling.sh
```

The apt transaction upgrades ROCm `7.2.0.70200-43~22.04` packages to
`7.2.4.70204-93~22.04`; the key runtime transition is
`hip-runtime-amd 7.2.26015.70200-43~22.04` to
`7.2.53211.70204-93~22.04`. RCCL remains 2.27.7 with the matching ROCm package
revision.

Run the gates in §3 before committing:

```bash
docker commit \
  --change 'CMD ["sleep","infinity"]' \
  --message "ts-migrate-mi350x env + ROCm userspace 7.2.4 (HIP 7.2.53211) + torch bundled-HIP relocation (RCCL graph-capture hang fixed)" \
  jeremwan-ts-upgrade jeremwan/tokenspeed:rocm7.2.4-torch2.11

docker run -d --name jeremwan-tokenspeed \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  --ipc=host --shm-size=16g --network=host \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /home/jeremwan:/home/jeremwan -v /data:/data \
  -w /home/jeremwan/tokenspeed -e HSA_ENABLE_IPC_MODE_LEGACY=1 \
  jeremwan/tokenspeed:rocm7.2.4-torch2.11 sleep infinity
```

`fix_torch_hip_bundling.sh` is idempotent. Re-run it after every torch
installation because the torch 2.11 wheel restores its RPATH-preferred bundled
ROCm libraries.

## 3. Verification gates

Run from the final `jeremwan-tokenspeed` container, not only the temporary build
container:

```bash
cat /opt/rocm/.info/version
python3 -c 'import torch; print(torch.__version__, torch.version.hip)'
ldd /opt/venv/lib/python3.10/site-packages/torch/lib/libtorch_hip.so \
  | grep -E 'libamdhip64|librccl'

cd /home/jeremwan/tokenspeed/tokenspeed-kernel
HIP_VISIBLE_DEVICES=5 python3 benchmark/probe_hip_event_query_capture.py
HIP_VISIBLE_DEVICES=5,7 TORCH_NCCL_BLOCKING_WAIT=0 \
  torchrun --standalone --nproc_per_node=2 benchmark/rccl_graph_abi_check.py
pytest -q test/ops/test_triton_shmem_communication.py
```

Observed state: system HIP runtime `70253211`; graph-capture probe PASS; ws=2/4/8
RCCL all-reduce capture/replay PASS without blocking wait; communication tests
10/10 PASS, including folded ws=4 and true two-shot ws=8 graphs. On the shared
MI350X host, check KFD processes and utilization before every GPU run and do not
launch on occupied GPUs.

## 4. Incident summary

The original symptom was described as an intermittent large unfused RCCL
all-reduce stall during serve, near `6656×2880` bf16. Later failures also showed
`HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION`, rank loss, TCPStore errors, and apparent
collective timeout. Fusion-off failures established that the fused kernel was not
required for the failure.

The concrete defect found during investigation was in HIP 7.2.26015 graph capture:
while one thread held a GLOBAL capture, a watchdog-like thread performing
`hipEventQuery` under THREAD_LOCAL capture semantics received
`hipErrorStreamCaptureUnsupported` and invalidated the graph. The minimal
`benchmark/probe_hip_event_query_capture.py` reproduces this without TokenSpeed,
Triton, symmetric memory, or RCCL. It fails with runtime 70226015 and passes with
70253211. This is consistent with:

- `pytorch/pytorch#177309`
- PyTorch workaround PR `pytorch/pytorch#176251`
- runtime fix `ROCm/rocm-systems#3176`, released in ROCm 7.2.1

PyTorch's `ProcessGroupNCCL` watchdog queries outstanding collective events.
`TORCH_NCCL_BLOCKING_WAIT=1` avoids creating that watchdog thread, which explains
why it was an effective compatibility workaround on the old image. It also removes
asynchronous timeout detection. The latest ws=4 refresh still required it, so it
remains a compatibility fallback rather than proof of root-cause closure.

## 5. Evidence and its boundary

Evidence retained:

1. `probe_rccl_hang.py` completed 300 iterations at suspect sizes, with and
   without rank jitter. This argues against a pure size-dependent RCCL defect.
2. `serve_ws4_cap0_auto.log` records fusion-off illegal-instruction queue aborts
   followed by rank/store failures.
3. `probe_hip_event_query_capture.py` deterministically isolates the runtime
   failure and the 7.2.4 fix.
4. `rccl_graph_abi_check.py` validates RCCL all-reduce capture/replay against the
   system 7.2.4 libraries.
5. With blocking wait on the old image, uncontended ws=4 serve completed direct
   6656-token prefills, a `416×16=6656` batch, and repeated 64-request stress.

Not retained: no archived log contains the originally reported
`NumelIn=19169280` watchdog timeout or the 2-of-4 rank-spin snapshot. Therefore:

- the HIP graph-capture defect and its fix are proven;
- a pure large-message RCCL failure is unlikely;
- attribution of every historical “RCCL hang,” including the original report, is
  not proven.

If a similar stall recurs on ROCm 7.2.1 or later, treat it as a new issue. Test
`--disable-overlap-schedule`, `--enforce-eager`, and RCCL-only serving before
investigating mixed HIP-IPC/RCCL or cross-stream ordering.

## 6. July 2026 serving refresh and resolution

The first refresh found three apparent serving failures:

1. ws=4 unfused without blocking wait aborted with an illegal instruction.
2. ws=4 fused with multi-wave folded copy-in aborted with the same fault class.
3. ws=8 unfused faulted immediately after warmup.

The initial logs were asynchronous and did not identify a faulting kernel. Follow-up
single-variable controls separated the causes.

### 6.1 RCCL was not the refreshed failure

`rccl_graph_abi_check.py` was corrected to capture `dist.all_reduce` itself.
Eager all-reduce, captured all-reduce replay, and a compute graph pass at ws=2/4/8
with `TORCH_NCCL_BLOCKING_WAIT=0`. Final ws=4/8 serving also completes without
blocking wait.

The historical HIP 7.2.26015 event-query defect remains proven and is fixed by
the loaded 7.2.4 runtime. It does not explain the refreshed faults.

### 6.2 Unfused Triton AR barrier defect

The small-message AMD Triton AR used multiple wavefronts around a scalar
system-scope signal barrier. The scalar release/acquire did not synchronize
sibling wavefront peer loads or completion before persistent-buffer reuse.

`symm_mem_workgroup_barrier` now brackets scalar cross-rank barriers in AMD
all-reduce, native fused AR+RMSNorm, and RS/AG kernels.

- `TS_TRITON_AR_WORKGROUP_SYNC=0`: ws=8 readiness is followed by memory faults
  on four GPU nodes (`serve_resolve_ws8_unfused_nosync.log`).
- synchronization enabled: the complete ws=8 unfused campaign passes.

This resolves the previous ws=8 unfused fault and removes a likely contributor
to the transient ws=4 unfused abort.

### 6.3 Folded copy-in semantics

Folded copy-in writes coarse symmetric input, signals peers, pulls peer data, and
signals buffer reuse inside one Triton program. Workgroup synchronization alone
cannot make sibling wavefront memory effects part of a scalar wavefront's
system-scope release/acquire.

The folded specialization therefore uses one wavefront:

- `TS_TRITON_SHMEM_FOLD_NUM_WARPS=4`: deterministic ws=8 post-readiness memory
  fault (`serve_resolve_ws8_fused_fold4warp.log`).
- `TS_TRITON_SHMEM_FOLD_NUM_WARPS=1`: full ws=4/8 campaigns pass.

Folded copy-in is default ON with one wave. Other paths retain their tuned
multi-wave settings.

### 6.4 Final gate

The final ws=2/4/8 fused and unfused matrix completed with blocking wait disabled,
two seeds at concurrency 8/16/32, zero failed requests, and no HIP/HSA faults.
Exact results and deployment policy are in
`AR_RMSNORM_MI350X_E2E_BENCHMARKS.md`.

The image's historical commit message says “RCCL graph-capture hang fixed.” That
string remains provenance; current evidence additionally requires the kernel
barrier fixes above.

## 7. Profiling compatibility boundary

The original profiler failure was an environment mismatch, not a fundamental
torch 2.11 / ROCm 7.2 incompatibility. `fix_torch_hip_bundling.sh` moved the
wheel's HIP/HSA/ROCTX/RCCL libraries but omitted `libroctracer64.so`.
Consequently `libtorch_hip.so` loaded:

- system ROCm 7.2.4 `libamdhip64`, `libhsa-runtime64`, `libroctx64`, and RCCL;
- the torch wheel's bundled ROCm 7.2.0 `libroctracer64`.

That mixed activity stack segfaulted in
`roctracer::MemoryPool::Write` from `THCPEvent_wait`. Relocating the bundled
`libroctracer64.so` so the loader selects `/opt/rocm/lib/libroctracer64.so`
fixes both torch/Kineto and eager Proton roctracer. The fix script now includes
`libroctracer64` in the required system-runtime family.

### 7.1 Profiler-qualified artifact

```text
image:     jeremwan/tokenspeed:rocm7.2.4-torch2.11-profiler
image ID:  sha256:ad3ea3f8cae8ca38cf12824b15c606d0630118c6e04b4087e191b04619a6c135
container: jeremwan-tokenspeed-profiler
torch:     2.11.0+rocm7.2 (compiled HIP label 7.2.26015)
runtime:   system HIP 7.2.53211 / ROCm 7.2.4
tracing:   system roctracer 4.1.70204 / rocprofiler-sdk 1.1.0
```

The image differs from `jeremwan/tokenspeed:rocm7.2.4-torch2.11` only by
relocating torch's bundled roctracer. Compute libraries remain wheel-bundled.
`ldd libtorch_hip.so` resolves HIP, HSA, ROCTX, RCCL, and roctracer from
`/opt/rocm/lib`.

Passed in the committed image:

- standalone torch CPU+GPU eager and graph Chrome traces;
- TP=2 normal graph-serving torch CPU+GPU profile, both rank files complete;
- deterministic HIP event-query/capture probe with runtime 70253211;
- ws=2 eager and captured RCCL plus compute-graph replay with blocking wait off;
- standalone eager Proton roctracer Chrome trace;
- standalone graph Proton tree/Hatchet with both roctracer and rocprofiler.

The TP=2 torch serving capture produced 1.7/1.8 MiB rank traces, about 92,000
events and 29,139 GPU kernels per rank, including 1,152
`fused_ar_rmsnorm_oneshot_blocked_kernel` events.

Recreation from the serving image:

```bash
cd /home/jeremwan/tokenspeed/tokenspeed-kernel
bash benchmark/fix_torch_hip_bundling.sh
ldd /opt/venv/lib/python3.10/site-packages/torch/lib/libtorch_hip.so \
  | grep -E 'libamdhip64|libhsa-runtime|libroctx|libroctracer|librccl'
```

All five libraries in that check must resolve under `/opt/rocm/lib`.

### 7.2 Why the all-bundled alternative was rejected

A separate container restored all torch-wheel ROCm 7.2.0 runtime libraries.
That coherent stack also passed the standalone torch graph-profiler probe, but
reproduced the deterministic HIP 70226015 defect:

```text
hipEventQuery -> hipErrorStreamCaptureUnsupported
capture       -> hipErrorStreamCaptureInvalidated
```

It could be used only with watchdog avoidance such as
`TORCH_NCCL_BLOCKING_WAIT=1`. The all-system 7.2.4 runtime/tracing family works
without that workaround and is the promoted configuration.

### 7.3 Proton graph support

Proton's AMD graph behavior depends on backend, lifecycle, and data format:

- `data=trace` / `chrome_trace` works for eager kernels, but graph replay
  finalization fails with missing CPU-scope correlations or emits incomplete
  metric-only output.
- `data=tree` / `hatchet` is the supported graph path in the installed
  tokenspeed-proton 3.8.10.post20260709 build.
- roctracer must start after HIP/model initialization but before graph capture.
  The new `TOKENSPEED_KERNEL_PROFILE_BEFORE_GRAPHS=1` lifecycle hook starts at
  that boundary.
- rocprofiler must configure before HIP/HSA registration. Use
  `TOKENSPEED_KERNEL_PROFILE=1` plus
  `TOKENSPEED_KERNEL_PROFILE_EARLY=1`.
- both graph modes require the session to remain active through graph capture
  and replay. `TOKENSPEED_KERNEL_PROFILE_EARLY_KEEP_ACTIVE=1` prevents the
  early session from being used only as a backend-prime.
- `TOKENSPEED_KERNEL_PROFILE_GRAPH_SCOPES=1` adds explicit decode and prefill
  replay scopes so Hatchet attributes replayed kernels.

The rocprofiler TokenSpeed run finalized two valid per-rank Hatchet files. Each
contained decode and prefill replay scopes, 353 timed nodes, and timed fused
AR+RMSNorm kernels. It still emitted many `Cannot find graph` warnings for graph
objects outside the attributed paths, so this mode is usable but noisy.

Legacy roctracer is the cleaner torch/Kineto path and is stable for eager Proton.
Its full multi-rank TokenSpeed graph/Hatchet path remains shaky: one experiment
produced a complete rank file and a zero-byte peer file. Profiler control now
checks every rank result instead of accepting rank zero alone.

Do not use ROCm 7.14/TheRock preview packages for this setup. All conclusions
above use torch 2.11 with released ROCm 7.2.4 userspace.
