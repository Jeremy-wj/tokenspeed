# AR+RMSNorm serving integration analysis

Updated: 2026-07-29

## Purpose and evidence boundary

This report analyzes how the Triton symmetric-memory all-reduce + residual-add +
RMSNorm kernel is integrated into TokenSpeed serving. It does not replace the
deployment decision or optimization queue in
[GPT-OSS-120B status](gpt-oss-120b-status.md). It also does not reopen the
external kernel-landscape sweep; ideas from that sweep are prior art or closed
work unless a new integration mechanism changes their preconditions.

The conclusions use four evidence classes:

- **Measured:** current curated TokenSpeed traces, graph probes, or end-to-end
  results.
- **Implemented:** behavior confirmed in the current source but not necessarily
  qualified for deployment.
- **Reference pattern:** behavior in PyTorch, Kraken, vLLM, SGLang, AITER, Iris,
  or upstream triton-shmem that may generalize.
- **Hypothesis:** a new TokenSpeed integration direction requiring measurement.

Current campaign outcomes and candidate dispositions are intentionally omitted
here; the status page is the sole deployment source.

## Executive judgment

TokenSpeed already has the essential structure of an effective serving
integration:

1. The compiler defers row-parallel reductions and places fusion exactly at the
   consuming norm.
2. The runtime retains a complete unfused fallback when the fused backend
   declines.
3. Persistent graph-stable communication and per-site output state is explicit.
4. Coarse HIP-IPC data buffers avoid the severe ROCm fine-grained-memory
   bandwidth penalty while PyTorch symmetric memory supplies a small coherent
   signal pad.
5. Small decode shapes route to one-shot pull while larger shapes can use
   two-shot.
6. Correctness and performance are tested at graph-transition and serving
   levels, not only in eager microbenchmarks.

The remaining gap is performance, not unexplained stability. The hot path still
pays for a persistent-buffer lifecycle: publish input, rendezvous, pull peers,
use fp32 scratch for N=2880, materialize two outputs, and protect input reuse.
The no-exit input ring still needs graph-stable slot/epoch identity, while any
output optimization must retain the profile-v4 per-site ownership contract.

PyTorch synchronization channels remain useful prior art for future concurrent
streams, but they are not the smallest ring fix. Capture-frozen positional slots
plus an explicit graph/forward epoch directly address the measured host-phase
failure and can later coexist with separate signal channels.

## Current serving data flow

### Compiler placement

`_should_fuse_allreduce_norm` requires an all-reduce mode, actual parallelism,
the server integration flag, a positive token count, and the configured token
cap:

- `python/tokenspeed/runtime/models/base/comm_ops.py:121-135`

The compiler marks the row-parallel result as `Partial`, inserts a
`DeferredReduceOp`, and places `FusedReduceNormOp` at the immediately consuming
norm. At runtime `DeferredReduceOp` is a marker/no-op; the fused norm performs
the reduction or restores the explicit all-reduce:

- `python/tokenspeed/runtime/models/base/comm_ops.py:261-323`
- `python/tokenspeed/runtime/models/base/compiler.py:248-285`
- `python/tokenspeed/runtime/models/base/compiler.py:420-422`

This placement is important. Both GPT-OSS sites are immediate dependencies:
post-attention norm feeds the MoE block, and post-MoE norm feeds the next layer.
There is little unrelated layer work that can hide the completed collective.
Integration should reduce the lifecycle inside a site rather than assume broad
cross-layer overlap.

### Runtime and fallback

On AMD, `RMSNorm.forward_with_allreduce_fusion` calls the Triton communication
entry point. A declined backend sets `needs_unfused_allreduce`, then executes the
full all-reduce before ordinary residual-add/RMSNorm:

- `python/tokenspeed/runtime/layers/layernorm.py:159-217`

The kernel-package dispatch checks device, contiguity, bf16 dtype, shape,
weight, process-group size, and workspace cap. State is keyed by:

```text
(process-group identity, max token count, hidden size, dtype)
```

An optional `TS_TRITON_SHMEM_FUSION_MAX_M` performance gate is independent of
the workspace cap. A forced triton-shmem backend or a performance-gate decline
returns to the caller's complete unfused path rather than silently selecting a
different fused implementation:

- `tokenspeed-kernel/python/tokenspeed_kernel/ops/communication/triton.py:1578-1676`

This is production-quality behavior and should remain invariant. Frameworks
that add more dispatch dimensions must still prove the exact executed path
rather than infer it from configuration.

### Persistent state and shape routing

State construction chooses a base kernel once from world size and hidden size,
sizes the signal pad before the first symmetric allocation, and allocates
persistent input, optional second input, two-shot output buffers, optional
per-site returned-output rings, and local fp32 scratch:

- `tokenspeed-kernel/python/tokenspeed_kernel/ops/communication/triton_shmem.py:248-478`

At TP=4/N=2880 the state is normally `twoshot_blocked`, but calls with
`M <= 256` use the one-shot blocked overlay. Qualified profile
`gpt-oss-120b-mi350x-qualified-v4` selects persistent per-site local outputs
for that captured range, retains the one-slot exit-barrier contract, uses
explicit copy-in, and selects two-shot only for larger eager M:

- `tokenspeed-kernel/python/tokenspeed_kernel/ops/communication/triton_shmem.py:612-720`

The historical 2026-07-24 profile traces (not the qualified-v4 profile) show:

- decode: 1,080 one-shot calls per rank;
- prefill: 144 two-shot calls per rank;
- 29,967 kernels/rank fused versus 31,333 unfused;
- profiled GPU window about 7.8% shorter.

Source:
`studies/mi350x/2026-07-profile-guided-followup/corrected_profile_comparison.json`
(relative to `ar_rmsnorm/`).

### Memory substrate

PyTorch symmetric memory on ROCm uses fine-grained VMM memory. TokenSpeed
measured about 105 GB/s for that memory versus about 3,200 GB/s for coarse HBM,
so bulk data uses ordinary `torch.empty` allocations exported through HIP IPC.
Only the signal pad remains in fine-grained symmetric memory:

- `tokenspeed-kernel/python/tokenspeed_kernel/ops/communication/_coarse_shmem.py:21-47`
- `tokenspeed-kernel/python/tokenspeed_kernel/ops/communication/triton_shmem.py:305-397`

Each input, output, and residual-output allocation owns a separate peer-pointer
table. The translated pointer is the peer base plus the local allocation
offset; tables cannot be reused across independent allocations.

This coarse-data/fine-signal split is the right ROCm adaptation of PyTorch's
model. Replacing it with a generic symmetric-memory pool for bulk data would
reintroduce the measured local-bandwidth failure. PyTorch MemPool ideas are
relevant to lifecycle and signal allocation, not as a hot-path bulk-data
replacement.

### Synchronization and graph replay

Shipping one-shot has two ordering points:

1. a leading rendezvous publishes each rank's input before peer pulls;
2. a trailing rendezvous prevents input reuse before all peer pulls complete.

Folded copy-in is deliberately one wavefront. Workgroup barriers surround the
scalar system-scope CAS so it represents all wavefront memory access:

- `tokenspeed-kernel/python/tokenspeed_kernel/ops/communication/triton_shmem.py:102-175`
- `tokenspeed-kernel/python/tokenspeed_kernel/ops/communication/_triton_shmem_kernels.py:240-294`

The diagnostic GPT-OSS two-slot ring advances the input slot every fused call and omits the
one-shot exit barrier. It relies on the next call's leading rendezvous to prove
completion before the old slot is reused two calls later. Two-shot retains its
completion barrier because peer output pushes must be visible before copy-out:

- [Producer lifetime contract](producer-lifetime-contract.md)
- `tokenspeed-kernel/python/tokenspeed_kernel/ops/communication/triton_shmem.py:540-610`

The current proof requires an even number of ring advances in a captured graph.
GPT-OSS has 72 eligible sites, but the mechanism is not safe as a generic
framework default for odd-site graphs without an explicit graph epoch.

An M-dependent in-kernel grid is fast for pure TP but unsafe when TP ranks can
execute different M concurrently. Fixed participants stabilized decode but
deadlocked prefill. The profile instead disables overlap scheduling so ranks
transition M/graphs in lockstep while retaining the fast grid:

- `tokenspeed-kernel/python/tokenspeed_kernel/ops/communication/triton_shmem.py:205-226`
- `studies/mi350x/2026-07-repeatability/repeatability-summary.json`
  (relative to `ar_rmsnorm/`)

## Evidence boundary

This document explains architecture and transferable integration mechanisms.
It does not maintain campaign results. Use:

- [GPT-OSS-120B status](gpt-oss-120b-status.md) for the current decision;
- [MI350X upper bound](mi350x-upper-bound.md) for measured costs and ceilings;
- `studies/mi350x/2026-07-repeatability/` (relative to `ar_rmsnorm/`) for
  serving-failure evidence.

## Cross-framework patterns that transfer

### PyTorch symmetric memory

PyTorch's core model is allocate, collectively rendezvous once, then retain
peer buffer and signal-pad pointers. Rendezvous mappings are cached by
allocation and group:

- PyTorch checkout: `torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp:9-38`
- PyTorch checkout: `torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory.cu:888-936`

Three concepts are directly relevant:

1. **Synchronization channels.** PyTorch explicitly isolates barriers on
   different streams so one barrier cannot consume another's signal:
   `torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp:31-38`.
2. **Zero-after-success invariant.** Every successful synchronization must
   return its signal slots to zero:
   `torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory-inl.cuh:133-173`.
3. **Persistent allocation identity.** `alloc_id` provides deterministic
   addresses for graph/memory planners and rejects unsafe simultaneous reuse:
   `torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp:156-185`.

PyTorch also defines three fence patterns for prior writes, in-kernel writes,
and safe post-read reuse:

- PyTorch checkout: `torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory-inl.cuh:176-219`

TokenSpeed implements equivalent release/acquire behavior, but it does not yet
make channel/epoch identity a first-class serving concept.

MemPool uses `use_on_oom=False` to avoid rank allocation desynchronization and
`no_split=True` because sharing one signal pad between concurrent tensors is
undefined without stream tracking:

- PyTorch checkout: `torch/distributed/_symmetric_memory/__init__.py:2278-2322`

This is strong evidence against pooling unrelated TokenSpeed communication
states onto one signal pad without channel isolation.

CUDA/NCCL-only PyTorch features—multimem, NCCL copy-engine collectives,
higher-precision NCCL symmetric reductions, and NVSHMEM collective launch—do
not have a current ROCm serving analogue and should not enter the near-term
TokenSpeed queue.

### Kraken

Kraken's Triton barrier exposes explicit previous/subsequent-memory flags and
the same zero-reset graph-friendly CAS protocol:

- Kraken checkout: `kraken/_ptx_utils/symm_mem_barrier.py:97-158`

Its most transferable new pattern is producer/consumer progress signaling:
communication chunks publish readiness and a persistent consumer processes
tiles in ready order. The concept is portable, but Kraken's TMA,
`cuStreamWaitValue32`, PTX, and copy-engine details are CUDA-specific:

- Kraken checkout: `kraken/fused/all_gather_matmul.py:77-123`

For AR+RMSNorm, the immediate consumer dependency limits generic overlap.
Progress signaling becomes relevant only after producer-direct output or a
producer epilogue can publish row/tile readiness. It is a long-horizon
architecture, not a drop-in side-stream optimization.

### vLLM, SGLang, and AITER

The common serving lesson is a dispatch policy indexed by message bytes, world
size, topology, graph mode, and backend capability:

- vLLM custom AR uses one-stage for ws=2 or small fully connected payloads and
  two-stage for larger payloads:
  `csrc/custom_all_reduce.cuh:589-598`.
- vLLM registers graph buffer IPC addresses after capture:
  `vllm/distributed/device_communicators/custom_all_reduce.py:196-228`.
- SGLang's AMD fusion selects one-stage below 128 KiB and handles piecewise
  graph capture explicitly:
  `python/sglang/srt/distributed/parallel_state.py:735-794`.

TokenSpeed already has the kernel-family crossover (`M <= 256` one-shot) and a
full fallback, but its policy is distributed across server flags, an M gate,
state construction, and call-level routing. A unified profile should resolve
`(architecture, topology, world size, hidden, M/bytes, graph mode,
divergence capability)` to backend, kernel family, barrier mode, and eligibility
with one logged decision.

vLLM's post-capture registration is useful prior art for future caller-owned
graph buffers. It does not justify replacing TokenSpeed's persistent decode
pool now; the current pool avoids per-call registration and is already
graph-stable.

### Iris

Iris deliberately transforms persistent program IDs across AMD XCDs:

- Iris checkout: `iris/ccl/triton/all_reduce.py:171-181`

This remains a genuinely untested AMD integration/launch hypothesis, but it is
not a priority during paired qualification. It should later be evaluated as
a topology-profile dimension rather than hard-coded into a generic kernel.

## Transferable constraints

The cross-framework evidence supports graph/stream synchronization channels,
explicit graph epochs, control-plane isolation, pre-capture state warmup,
persistent allocation identity, topology-qualified dispatch, and
producer-direct readiness publication as design dimensions. They are not a
second optimization queue. Candidate ordering and dependencies live in the
[integration roadmap](integration-optimization-roadmap-2026-07.md), while
accepted and rejected deployment choices live only in
[GPT-OSS-120B status](gpt-oss-120b-status.md). Any candidate must preserve the
[producer and buffer lifetime contract](producer-lifetime-contract.md).

## Resulting integration principles

1. Treat communication state as a graph resource with identity, ownership,
   synchronization channel, and epoch—not just a cached allocation.
2. Keep coarse HIP-IPC data and fine-grained signal memory separate.
3. Keep state construction and all rendezvous outside capture and measured
   steady state.
4. Dispatch from a logged model/topology profile and retain complete unfused
   semantics on every decline.
5. Isolate health/control traffic from performance traffic while testing the
   transition deliberately in a safety suite.
6. Require marker-aligned max-rank evidence and full restart-randomized serving
   before promotion.
7. Optimize buffer lifetime before broadening producer/consumer fusion.
8. Do not infer end-to-end value from pooled kernel medians or from CUDA-only
   mechanisms without a ROCm feasibility proof.

## Primary sources

TokenSpeed:

- [Project index](../README.md)
- [Current status](gpt-oss-120b-status.md)
- [Backend design and safety](backend-design-and-safety.md)
- [Producer and buffer lifetime contract](producer-lifetime-contract.md)
- [Profiling workflow](profiling-workflow.md)
- [MI350X upper bound](mi350x-upper-bound.md)
- `studies/mi350x/2026-07-repeatability/` (relative to `ar_rmsnorm/`)

External checkout-relative references:

- PyTorch: `torch/csrc/distributed/c10d/symm_mem/` and
  `torch/distributed/_symmetric_memory/`
- Kraken: `kraken/`
- vLLM: `csrc/` and `vllm/`
- SGLang: `python/sglang/`
- AITER: repository root
- Iris: `iris/`
- triton-shmem: repository root
- AR fusion landscape: external analysis root
