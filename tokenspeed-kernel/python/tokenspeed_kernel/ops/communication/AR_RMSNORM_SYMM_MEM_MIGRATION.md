# Fused AllReduce + Residual + RMSNorm: PyTorch symmetric-memory migration

## Current state

The migration from rocSHMEM to PyTorch symmetric memory is complete. The AMD
backend is named `triton_shmem` and is the `TS_ARNORM_BACKEND=auto` default when
the fused operation is eligible. It adds no runtime dependency beyond torch.

Validated coverage:

- MI300X: correctness at ws=1/2/4/8, graph capture/replay at ws=2/8, and
  noise-controlled microbenchmarks.
- MI350X: correctness, graph capture/replay, op decomposition, and complete
  ws=2/4/8 gpt-oss-120b serving. The barrier-related memory faults found during
  the first refresh are resolved. Current data and deployment guidance are in
  `AR_RMSNORM_MI350X_E2E_BENCHMARKS.md`.
- ROCm serving environment: use container `jeremwan-tokenspeed` from image
  `jeremwan/tokenspeed:rocm7.2.4-torch2.11`. Exact provenance and the bounded
  HIP/RCCL investigation are in
  `AR_RMSNORM_ROCM_CONTAINER_AND_RCCL_HISTORY.md`.

Shipping defaults:

```text
TS_ARNORM_BACKEND=auto                  # auto -> triton_shmem
TS_TRITON_SHMEM_COARSE=1                # coarse HBM data buffers over HIP IPC
TS_TRITON_SHMEM_INKERNEL_BARRIER=1      # one-shot barriers in the fused kernel
TS_TRITON_SHMEM_FOLD_COPYIN=1           # fold one-shot copy-in
TS_TRITON_SHMEM_FOLD_NUM_WARPS=1        # system barrier covers the whole program
TS_TRITON_SHMEM_WORKGROUP_SYNC=1        # bracket scalar cross-rank barriers
TS_TRITON_SHMEM_GRID_CAP=-1             # auto: gfx950 ws4 cap 128 at M>=256
TS_TRITON_SHMEM_GRID_CAP_MIN_M=-1       # use validated auto crossover
TS_TRITON_SHMEM_ONESHOT_MAX_M=256       # one-shot overlay at ws>=4
TS_TRITON_SHMEM_BARRIER_GRID=0           # M-dependent grid; fixed grid is opt-in
TS_TRITON_AR_WORKGROUP_SYNC=1           # safe unfused Triton AR barriers
```

For pure TP (`dp=1`, overlap depth 1, no speculative decode), keep these defaults.
If TP ranks can replay different-M graphs concurrently, set a nonzero
`TS_TRITON_SHMEM_BARRIER_GRID`; the fixed participant set is divergence-safe but
slower. Do not infer the fusion threshold from the historical MI300X RCCL proxy:
use the current serving results in the companion report.

Folded copy-in is deliberately single-wave. A four-wave folded specialization
reproduces the ws=8 memory fault because a scalar system release/acquire cannot
order sibling wavefront memory effects. Non-folded and two-shot kernels retain
their tuned multi-wave settings.

## 1. Goal and scope

The original fused Triton kernels came from the external `triton-shmem` project
and used `rocshmem4py`. TokenSpeed needed the same fused all-reduce + residual-add
+ RMSNorm operation without shipping rocSHMEM.

PyTorch symmetric memory was selected because:

1. TokenSpeed already uses `torch.distributed._symmetric_memory` for AMD
   collectives.
2. It ships with torch and adds no build or deployment dependency.
3. Its device pointer table supports the same translation used by rocSHMEM.
4. Its signal-pad barrier is device-side and graph-capture-safe.

The supported contract remains AMD ROCm, bf16, contiguous 2-D
`(num_tokens, hidden)` tensors, a one-dimensional RMSNorm weight, and a process
group larger than one.

The upstream repository is a reference only. Production kernels are vendored in
TokenSpeed and must import `triton`/`tl` from `tokenspeed_kernel._triton`.

## 2. Implementation map and dispatch

Core files:

- `ops/communication/triton_shmem.py`: state allocation, path selection, barriers,
  and launches.
- `ops/communication/_triton_shmem_kernels.py`: vendored kernels, architecture
  profiles, and signal-pad barriers.
- `ops/communication/_coarse_shmem.py`: coarse-grained HIP-IPC data buffers.
- `ops/communication/triton.py`: public dispatcher and state caches.
- `test/ops/test_triton_shmem_communication.py`: correctness and graph tests.
- `benchmark/bench_triton_shmem_ar_rmsnorm.py`: crossover benchmark.
- `benchmark/probe_ar_rmsnorm_decomp.py`: phase and serving-baseline probe.

Dispatch:

```text
runtime/layers/layernorm.py
  -> communication/triton.py::allreduce_residual_rmsnorm
     -> TS_ARNORM_BACKEND=auto|triton_shmem
        -> communication/triton_shmem.py
           -> communication/_triton_shmem_kernels.py
```

The state-level architecture recommendation is one-shot at ws<=2 and two-shot at
ws>=4. The call-level `ONESHOT_MAX_M=256` overlay routes small M through one-shot
even when the state is two-shot. The variants are:

- `oneshot_wholerow`: power-of-two N, one symmetric input buffer.
- `oneshot_blocked`: arbitrary N, one symmetric input buffer.
- `twoshot_blocked`: arbitrary N, symmetric input/output/residual-output buffers.

## 3. Pointer translation

rocSHMEM translated a local allocation to a peer address as:

```text
peer_ptr = heap_bases[peer] + (local_ptr - heap_bases[rank])
```

PyTorch symmetric memory exposes a device table of peer pointers for each
allocation. Passing that table to the same translation gives:

```text
buffer_ptrs[peer] + (local_ptr - buffer_ptrs[rank])
```

The arithmetic is equivalent. The important difference is allocation scope:
rocSHMEM has one heap-base table, while symmetric memory has one pointer table per
allocation. Therefore:

- one-shot uses the input table;
- two-shot uses distinct input, output, and residual-output tables;
- offsets must never be assumed to transfer between allocations.

This is the primary migration invariant.

## 4. Synchronization and buffer lifetime

Every variant requires a leading and trailing cross-rank barrier:

- leading: all peer inputs are visible before a one-shot pull or two-shot push;
- trailing: all peers are finished with the persistent symmetric input before the
  next invocation overwrites it; for two-shot it also orders peer writes.

The one-shot default performs both barriers inside the fused kernel. This removes
two launch boundaries but the barrier work remains. The number of participating
blocks is M-dependent, which is safe when all TP ranks execute the same M.
Each scalar cross-rank barrier is bracketed by workgroup barriers; folded copy-in
uses one wavefront so the scalar system fence orders every phase-0 store and peer
read.

`TS_TRITON_SHMEM_BARRIER_GRID=G` launches a fixed participant set, including
zero-row blocks. It prevents different-M graph replays from using different signal
slots. Measurements found every useful fixed G slower than the M-dependent default,
so this is a robustness control, not a performance optimization.

Two-shot uses separate one-block barriers by default. An M-dependent in-kernel
two-shot barrier regressed performance; it is enabled only with a fixed barrier
grid.

## 5. Memory substrate

On ROCm, torch symmetric-memory data allocations were fine-grained. Measured bulk
local bandwidth was about 107 GB/s versus about 3255 GB/s for coarse-grained HBM,
making the initial port uncompetitive.

The production backend therefore:

1. allocates data buffers with ordinary coarse-grained `torch.empty`;
2. exports and opens them with HIP IPC;
3. builds the same uint64 peer-pointer tables consumed by the kernels;
4. leaves only the signal pad in fine-grained symmetric memory.

The kernels and pointer arithmetic are unchanged by this substrate. Peer read/write
coherence was validated across the signal-pad barrier.

HIP IPC export requires the torch caching allocator not to use expandable segments.
If coarse allocation or IPC setup fails, state creation declines and the dispatcher
falls back rather than running a partially configured backend.

## 6. Environment

### MI350X

Use the verified local image and container:

```text
serving image:   jeremwan/tokenspeed:rocm7.2.4-torch2.11
profiling image: jeremwan/tokenspeed:rocm7.2.4-torch2.11-profiler
profiler ID:     sha256:ad3ea3f8cae8ca38cf12824b15c606d0630118c6e04b4087e191b04619a6c135
container:       jeremwan-tokenspeed-profiler
model:           /data/models/openai/gpt-oss-120b
```

The container uses torch `2.11.0+rocm7.2` with system ROCm 7.2.4 libraries after
relocating torch's bundled ROCm runtime. Re-run
`benchmark/fix_torch_hip_bundling.sh` after any torch reinstall; it must relocate
the bundled roctracer as well as HIP/HSA/ROCTX/RCCL. The deterministic HIP probe,
eager and captured RCCL, final ws=2/4/8 serving, and TP=2/4 torch graph profiling
pass without blocking wait. See the historical environment document for exact
recreation, verification, and evidence boundaries.

The host is shared. Before every GPU run, inspect KFD processes and utilization.
For ws<8 avoid physical GPU 3/HIP index 0; use all eight only from an idle snapshot.

### MI300X reference environment

The migration was originally validated in:

```text
rocm/pytorch:rocm7.2.2_ubuntu22.04_py3.10_pytorch_release_2.7.1
```

The required comm-op install subset was:

```bash
apt-get update
apt-get install -y openmpi-bin libopenmpi-dev libssl-dev pkg-config
python3 -m pip install --upgrade pip "setuptools<82" wheel
pip install --force-reinstall --no-deps ./tokenspeed-kernel-amd \
  --no-build-isolation
TOKENSPEED_KERNEL_BACKEND=rocm \
PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/rocm7.2 \
  pip install tokenspeed-kernel/python/ --no-build-isolation
```

The authoritative full environment setup remains
`test/ci_system/install_deps_rocm.sh`.

## 7. MI300X validation record

### Correctness and capture

- Dedicated backend tests pass ws=1/2/4/8 across all three variants.
- Dispatcher correctness passes against an fp32 reference with 2e-2 tolerance.
- HIP graph capture/replay passes at ws=2 one-shot and ws=8 two-shot.

### Noise protocol

`benchmark/run_ar_rmsnorm_noise_controlled.sh` isolates each world size in a new
process, uses 30 warmups and 150 repeats, cools down between runs, and performs two
passes. The container could not pin clocks. Large configurations (at least 0.5 ms)
were stable within 5%; sub-0.5 ms margins were treated as trend data.

### Representative large-tensor result

The table reports max-rank p50 latency in ms. The baseline is RCCL all-reduce plus
eager residual/RMSNorm; it is not the serving-faithful unfused stack.

| ws | M×N | RCCL proxy | triton_shmem | iris |
|---:|---:|---:|---:|---:|
| 2 | 16384×4096 | 3.11 | 2.95 | 2.94 |
| 2 | 16384×16384 | 12.30 | 12.00 | 12.30 |
| 4 | 16384×2880 | 1.22 | 1.89 | 3.14 |
| 4 | 16384×16384 | 6.75 | 10.4 | 18.2 |
| 8 | 16384×2880 | 0.73 | 1.12 | 5.17 |
| 8 | 16384×16384 | 3.99 | 6.56 | 35.3 |

At large M, `triton_shmem` was parity with RCCL at ws=2 and about 0.6–0.7x the
RCCL proxy at ws=4/8. It was 1.7–5.4x faster than Iris at ws>=4. This established
the coarse substrate and two-shot scaling; it does not determine the current
MI350X serving threshold.

Arbitrary M/N support was also validated at ws=8 with non-power-of-two N
`{3584,5120}` and non-divisible M `{2000,32768}`.

Historical data:

- `benchmark/results/mi300x_results/ar_rmsnorm_noise_controlled/`
- `benchmark/results/mi300x_results/ar_rmsnorm_model_targeted/`
- `benchmark/results/mi300x_results/ar_rmsnorm_extended_range.csv`

## 8. Invariants and next work

Preserve:

1. per-allocation peer-pointer tables;
2. leading and trailing ordering around persistent-buffer reuse;
3. coarse data buffers with only the signal pad fine-grained;
4. graph-capture-safe device synchronization;
5. graceful fallback when eligibility or IPC setup fails.

Next work should be driven by current MI350X data:

- scoped traces found 93.8% `M=32`, 6.2% `M=64/128`, and 100% one-shot blocked
  calls at TP=2/4; graph-serving traces further show the TP=4 M=32 fused kernel
  about 16% above the max-rank unfused AR+RMSNorm median sum, so target that
  narrow graph path before changing the conservative one-shot threshold;
- remove two-shot copies only after tracing proves an explicit caller-owned
  symmetric-buffer lifetime contract; two-shot did not occur in the profiled
  decode windows, and paired-copy, borrowed-output, and barrier-folding
  shortcuts were slower or unsafe;
- extend subgroup TP graph coverage beyond the validated disjoint/interleaved
  eager cases when a subgroup deployment is planned;
- validate scheduler collective-order invariants under DP/speculative decode
  before using a fixed grid;
- revisit allocator independence only on a ROCm build that supports expandable
  segments or requires a pluggable allocator;
- re-run runtime, correctness, and capture gates after torch/ROCm changes.

The current crossover, serving baseline, e2e A/B results, and future measurement
matrix live in `AR_RMSNORM_MI350X_E2E_BENCHMARKS.md`.
