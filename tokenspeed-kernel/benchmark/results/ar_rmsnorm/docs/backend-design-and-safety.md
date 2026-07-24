# `triton_shmem` backend design and safety

## Contract

The AMD backend fuses:

```text
all-reduce sum
  + residual
  -> residual_out
  -> RMSNorm(weight, epsilon)
  -> norm_out
```

Supported inputs are contiguous bf16 2-D tensors `(tokens, hidden)`, a
one-dimensional RMSNorm weight, and a process group larger than one. Hidden size
is taken from the tensor at runtime; production dispatch does not assume 2880.

## Dispatch

```text
runtime/layers/layernorm.py
  -> communication/triton.py::allreduce_residual_rmsnorm
  -> communication/triton_shmem.py
  -> communication/_triton_shmem_kernels.py
```

State cache key:

```text
(process_group identity, max_token_num, hidden_dim, dtype)
```

A different model hidden size creates a distinct state and symmetric allocation.

Kernel variants:

- `oneshot_wholerow`: power-of-two hidden size, pull reduction, local output;
- `oneshot_blocked`: arbitrary hidden size, blocked reduction with fp32 scratch;
- `twoshot_blocked`: row-sharded reduction with symmetric output pushes.

At world size 4 or 8, the state is normally two-shot, while the call-level
one-shot overlay handles small token counts.

## Symmetric pointer translation

Each allocation has its own peer-pointer table:

```text
peer_ptr = buffer_ptrs[peer] + (local_ptr - buffer_ptrs[rank])
```

The input, output, and residual-output allocations are independent. Offsets or
pointer tables must never be reused across allocations.

## Memory substrate

Fine-grained symmetric memory on ROCm provides the signal pad but is too slow for
bulk tensor traffic. Data buffers therefore use coarse-grained `torch.empty`
allocations exported and opened through HIP IPC.

Required structure:

- coarse HBM for data;
- fine-grained symmetric memory only for the signal pad;
- graph-capture-safe device synchronization;
- graceful decline to the production unfused path if allocation or IPC setup
  fails.

HIP IPC requires expandable allocator segments to remain disabled on the
validated ROCm build.

## Synchronization

Every operation requires:

1. a leading cross-rank barrier so peer input writes are visible;
2. a trailing barrier so peers finish reading persistent input before reuse.

The scalar signal-pad CAS uses system-scope release/acquire semantics.
Multi-wave programs require workgroup barriers around that scalar operation.

Folded one-shot copy-in is deliberately single-wave:

```text
TS_TRITON_SHMEM_FOLD_NUM_WARPS=1
```

A workgroup barrier alone cannot make sibling-wave stores part of another
wavefront's system release. Four-wave folded copy-in reproduced memory faults.

## Grid policies

The normal one-shot grid is M-dependent and never exceeds the CU count. This is
safe for pure TP when all ranks execute the same M.

`TS_TRITON_SHMEM_BARRIER_GRID=G` fixes the participant set and can prevent
different-M graph deadlocks, but measured fixed grids regress pure-TP latency.
It is a robustness control for DP/speculative/overlapped execution, not a
default optimization.

The gfx950/ws4 selective compute cap activates at M>=256. It was validated in
the gpt-oss campaign and remains environment-overridable. New model profiles
must sweep their own widths and token ranges before treating it as optimal.

## Preserved invariants

1. Per-allocation peer-pointer tables.
2. Leading and trailing ordering around persistent-buffer reuse.
3. Coarse data buffers and fine-grained signal pad.
4. Single-wave folded copy-in.
5. Workgroup synchronization for multi-wave barrier participants.
6. Grid residency below the deadlock limit.
7. Full unfused fallback if fused state creation or eligibility declines.
8. Caller ownership of returned outputs unless an explicit graph-stable lifetime
   contract is introduced.

## Validation evidence

- MI300X correctness: ws=1/2/4/8; graph capture at ws=2/8.
- MI350X correctness and serving: ws=2/4/8.
- Captured RCCL: ws=2/4/8 with blocking wait disabled.
- TP=4 default shared-state multigraph transitions: pass.
- Current studies:
  - `../studies/mi350x/2026-07-serving-baseline/`
  - `../studies/mi350x/2026-07-grid-and-two-shot/`
  - `../studies/mi350x/2026-07-path-and-width-sweeps/`
  - `../studies/mi300x/migration-baseline/`

Current model policy is documented in
[gpt-oss-120B status](gpt-oss-120b-status.md), not in this implementation
reference.

