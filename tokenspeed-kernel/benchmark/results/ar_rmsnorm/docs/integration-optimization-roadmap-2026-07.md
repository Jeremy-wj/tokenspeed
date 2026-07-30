# AR+RMSNorm integration optimization roadmap

Updated: 2026-07-30

## Scope and ownership

This document describes mechanisms and their technical dependencies for the
TokenSpeed `triton_shmem` AR+residual+RMSNorm integration. It does not own
deployment policy, execution order, or rejected-candidate priorities; those
remain in [GPT-OSS-120B status](gpt-oss-120b-status.md).

After the upstream-main rebase, `triton_shmem` remains an explicit experimental
backend. The 2026-07-30 reset now supersedes the legacy opportunity estimates:

- upstream-unfused is the deployment control;
- Iris fused regressed TPOT +2.55% and throughput -2.29%;
- `triton_shmem` regressed TPOT +10.47% and throughput -10.58%;
- both fused campaigns completed 15/15 safe pairs and failed promotion.

The current evidence is in the
[post-rebase baseline study](../studies/mi350x/2026-07-post-rebase-baseline/README.md).

The profile-v4 graph-lifetime investigation is closed. Its evidence and
chronology live in the
[serving root-cause record](gpt-oss-120b-serving-root-cause.md), not here.
Future work must preserve the reserved-sink padding and persistent captured
output lifetime contracts established there.

## Mechanism dependency map

```text
closed graph-lifetime contracts
  -> explicit upstream-unfused deployment control
     -> ordinary Iris/RCCL graph-transition investigation
     -> Iris fused N=2880 critical-path investigation
     -> deterministic state initialization and observable identity
        -> only then reconsider a materially faster fused candidate
```

Local input-ring, producer-direct, and barrier-removal work is deprioritized.
The completed serving campaign shows that improving an isolated
`triton_shmem` phase is not the current bottleneck. Any reconsideration still
depends on graph-stable identity and exact caller-owned output APIs, and must
first project a graph critical-path win large enough to clear the campaign
thresholds.

## Post-rebase optimization order

1. **Preserve the control.** Keep `--disable-allreduce-fusion` and require
   resolved false-state proof. Do not benchmark against an auto-enabled arm
   mislabeled unfused.
2. **Resolve mixed transport transitions.** The synthetic transition matrix
   passes Iris-only graphs but times out when ordinary Iris and RCCL-fallback
   graphs share the captured sequence. Determine whether graph-pool sharing,
   communicator epochs, or capture ordering is responsible.
3. **Explain Iris fused cost.** At WS=4/N=2880, Iris fused is slower than
   ordinary Iris plus RMSNorm in eager, fixed-graph, marker-aligned, and
   end-to-end evidence. Profile staging, device-barrier, grid, and persistent
   kernel choices before changing integration policy.
4. **Keep `triton_shmem` diagnostic.** Its eager advantage ends at M=256; the
   M=257 two-shot transition, graph replay, and serving campaign are
   unfavorable. Do not tune grid caps or rings next.
5. **Require a material pre-campaign projection.** A candidate must beat the
   unfused M=32 graph path and aligned max-rank target stage before another
   three-block campaign.

## Deterministic state initialization and identity

### Warm every planned state before capture

Before model graph capture:

1. enumerate graph buckets and model profiles;
2. construct each `(group, cap, hidden, dtype)` communication state;
3. open HIP-IPC mappings and complete rendezvous;
4. compile every selected one-/two-shot specialization;
5. execute a correctness-checked dry call per path;
6. verify assigned signal slots return to zero.

Relevant TokenSpeed touchpoints, relative to the repository root:

- state construction:
  `tokenspeed-kernel/python/tokenspeed_kernel/ops/communication/triton.py:1626-1655`;
- state implementation:
  `tokenspeed-kernel/python/tokenspeed_kernel/ops/communication/triton_shmem.py:264-478`;
- graph bucket selection:
  `python/tokenspeed/runtime/execution/cuda_graph_wrapper.py:106-124`.

Useful upstream patterns:

- PyTorch rejects workspace growth during capture:
  `torch/distributed/_symmetric_memory/__init__.py:100-138`;
- vLLM warms kernels before graph capture:
  `vllm/v1/worker/gpu_worker.py:677-716`.

This targets startup determinism and early failure detection, not steady-state
TPOT. If it does not change startup variance, retain the dry correctness check
that prevents lazy initialization during capture.

### Give each state a stable logical identity

The identity should include:

```text
group generation
model profile
world size
hidden size and dtype
workspace cap
input/output allocation generation
graph owner or graph set
signal channel range
```

Reject destruction or recreation while a captured graph references the state's
addresses. Treat opened HIP-IPC handles as resources owned by the identity.
TokenSpeed can retain coarse allocation; it needs persistent lifecycle
semantics, not necessarily PyTorch's allocator.

References:

- PyTorch `alloc_id` semantics:
  `torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp:156-185`;
- TokenSpeed ownership/cache:
  `tokenspeed-kernel/python/tokenspeed_kernel/ops/communication/triton_shmem.py:248-250,317-397,480-489`.

### Make each fused call self-identifying

Extend `kernel_scope` with:

- state ID, graph ID, and graph epoch;
- barrier channel/range and barrier/compute grids;
- ring slot before and after;
- actual and executed M;
- one-/two-shot and folded-copy decisions.

Existing scope and forward marker:

- `tokenspeed-kernel/python/tokenspeed_kernel/ops/communication/triton_shmem.py:752-793`;
- `python/tokenspeed/runtime/execution/model_executor.py:750-791`.

The analyzer must align ranks by forward ID and reject state/path/channel
disagreement rather than aggregate mismatched forwards.

Completion means no lazy allocation or rendezvous during capture/measurement,
all planned states pass dry correctness and zero-pad checks, and every rank
reports one consistent decision for each aligned forward.

## Graph-stable ring ownership

### Measured opportunity and failed precondition

The host-alternated two-slot implementation remains disabled, but its graph
measurement identifies a reusable mechanism:

- baseline graph replay: 35.35 us/site;
- two-slot replay: 29.98 us/site;
- difference: 5.37 us/site / 15.19%;
- no eager improvement.

Source: `../studies/mi350x/2026-07-repeatability/input-ring-summary.json`.
Across 72 sites, 5.37 us/site sums to about 0.387 ms; this is not a TPOT
forecast.

The failed precondition is mutable host phase. The implementation advances
`_input_ring_index` on each call:
`tokenspeed-kernel/python/tokenspeed_kernel/ops/communication/triton_shmem.py:650-655`.
Incidental even-site parity does not cover interleaved graphs, odd-site models,
fallback, exceptions, or future side streams.

### Bind slot selection to graph position and epoch

The smallest robust design binds each captured call to:

```text
(forward_epoch + call_index) mod 2
```

Replay uses the frozen buffer pointer instead of consulting mutable host phase.
A one-bit boundary epoch provides explicit handoff among graph, eager,
fallback, odd-site, and interleaved paths.

Each graph resource descriptor records:

- state and graph ID;
- starting epoch and advances per replay;
- allowed one-/two-shot transitions;
- signal channel/slot range;
- stream role.

Replay validates the expected starting epoch and publishes the resulting end
epoch. Interleaving graphs must have independent ring/channel ownership or
explicit serialized ownership. Keep the one-shot exit barrier for every path
without this complete descriptor.

### Partition signal ranges only where concurrency requires it

Signal channels solve cross-stream barrier isolation; they are not a substitute
for ring-slot identity. If concurrent roles share a namespace, reserve separate
ranges for one-shot, two-shot, whole-grid barriers, control probes, and future
progress signaling. Size the pad before allocation and reject profiles whose
ranges do not fit.

References:

- PyTorch channel semantics and sizing:
  `torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp:31-38,203-211`;
- PyTorch signal allocation:
  `torch/distributed/_symmetric_memory/__init__.py:2230-2271`;
- TokenSpeed sizing:
  `tokenspeed-kernel/python/tokenspeed_kernel/ops/communication/triton_shmem.py:305-311`.

PyTorch's `no_split=True` MemPool rule also warns that tensors sharing one
segment and signal pad can race without stream tracking:
`torch/distributed/_symmetric_memory/__init__.py:2302-2320`. If MemPool is
evaluated, limit it to small signal-only allocations; do not move hot coarse
data buffers into fine-grained pooled memory without a new ownership proof.

### Requalification boundary

Before reconsidering two-slot reuse, require:

- odd/even call chains and replays ending on both slots;
- interleaved graph variants sharing one state;
- graph/eager and one-/two-shot transitions in both directions;
- fallback, cancel, exception, health, and warmup paths;
- zero signal slots and valid next epoch after each completed path;
- authoritative traces proving ring and exit-barrier decisions;
- the full serving promotion campaign.

Any slot/epoch disagreement is a no-go. Exact thresholds and current ordering
remain in the status and methodology documents.

## Unified model, topology, and shape dispatch

Current policy is distributed across server enablement, token/workspace caps,
backend selection, kernel recommendation, call-level one-shot overlays, barrier
controls, and architecture-specific grid caps. A new model must not inherit
GPT-OSS-120B thresholds implicitly.

### Resolve one inspectable profile

```text
(architecture, direct-peer topology, rank set, world size,
 hidden, dtype, actual/executed M, graph mode, divergence capability)
  -> (fuse or decline, backend, kernel family, grid policy,
      barrier channel/mode, workspace cap, output/lifetime policy)
```

Log the profile once at startup and the selected row/path per forward.
Environment variables can remain diagnostic overrides; production behavior
should resolve through one inspectable object.

References:

- TokenSpeed current routing:
  `tokenspeed-kernel/python/tokenspeed_kernel/ops/communication/triton_shmem.py:637-648`;
- vLLM one-/two-stage thresholds:
  `csrc/custom_all_reduce.cuh:589-598`;
- SGLang fused one-stage gate:
  `python/sglang/srt/distributed/parallel_state.py:760-793`.

### Validate direct-access topology

At state creation, record logical rank, HIP index, physical GPU, NUMA node, and
the peer-access matrix. Decline one-shot pull unless full direct access is
established, and qualify thresholds separately for each supported rank set.

PyTorch references:

- `torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp:92-96`;
- `torch/csrc/distributed/c10d/symm_mem/intra_node_comm.cpp:25-150`.

### Keep prefill and decode decisions independent

At TP=4/N=2880, the measured operator crossover lies between M=256 and M=512;
at M>=512, fused two-shot is 26.8-49.8% slower than the production unfused
baseline. See [MI350X upper bound](mi350x-upper-bound.md).

Keep these concepts distinct:

- workspace compatibility cap;
- graph-capture bucket cap;
- performance eligibility threshold;
- one-/two-shot transport crossover.

Operator crossover is dispatch evidence, not end-to-end promotion evidence.
A resolved profile is complete when all ranks choose the same path, supported
thresholds have serving-level evidence, and unsupported configurations decline
to complete unfused semantics.

Tiny-message push/pull policies and XCD-aware mapping are useful prior art but
remain kernel/launch alternatives, not substitutes for lifecycle design:

- SGLang dispatch:
  `python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py:239-280`;
- Iris XCD mapping: `iris/ccl/triton/all_reduce.py:171-181`.

## Producer-direct input and progress publication

### Add exact caller-owned producer output

Required API changes:

- dense row-parallel GEMM accepts an exact caller-owned output view;
- active MXFP4 MoE finalize accepts or forms the reserved `[M,N]` output;
- compiler/runtime reserves graph-stable symmetric storage before launch;
- fallback retains a valid rank-local partial tensor.

The normative ownership constraints are in the
[producer/buffer lifetime contract](producer-lifetime-contract.md).

Separately measured copy work is 13.54 us/site, but overlaps other phases and
is not a 13.54-us forecast. The audited 15-25 us/site architectural target
requires copy, synchronization, scratch, and lifetime changes to work together;
see [MI350X upper bound](mi350x-upper-bound.md).

### Publish readiness from the producer

After direct writes, the producer epilogue can release row/tile progress only
after its stores. The AR+RMSNorm consumer acquires ready units in an order
compatible with peer communication.

Kraken provides conceptual prior art:

- progress-waited persistent consumer:
  `kraken/fused/all_gather_matmul.py:77-123`;
- barrier semantics:
  `kraken/_ptx_utils/symm_mem_barrier.py:97-158`.

ROCm must use Triton/HIP-supported system-scope atomics and workgroup ordering;
PTX, TMA, and `cuStreamWaitValue32` are not portable implementations.

### Preserve consumer lifetimes

`residual_out` remains live until the next fused norm, while `norm_out` feeds
the next producer. Borrowed or ping-pong storage therefore needs independent
epoch ownership. Persistent per-site caller-owned outputs remain the safe graph
default until these consumer lifetimes are proven.

Prototype producer-direct input only after graph epochs exist, exact output
pointers are guaranteed, fallback remains valid, and a combined isolated path
projects below 25 us/site. Stop if direct write merely replaces one staging copy
with another or progress waiting adds more synchronization than it removes.

## Design invariants

- Preserve complete unfused semantics on every decline or failure.
- Treat graph savings as screening evidence, not TPOT forecasts.
- Require explicit ownership for graph-stable addresses, epochs, and channels.
- Qualify every model/topology/crossover independently.
- Require integration complexity to purchase a material serving effect.
