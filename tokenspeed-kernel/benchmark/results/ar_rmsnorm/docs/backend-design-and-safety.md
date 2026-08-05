# `triton_shmem` backend design and safety

Updated: 2026-08-03

## Scope

This is the implementation and safety reference for the explicit local
`triton_shmem` backend. It is not the upstream `auto` backend and does not own
deployment policy. Selection requires `TS_ARNORM_BACKEND=triton_shmem`; every
decline returns to the caller's complete unfused path.

Current model policy is in the
[GPT-OSS-120B](gpt-oss-120b-status.md) and
[GLM-5.2-FP8](glm-5.2-fp8-status.md) status pages.

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
  -> TS_ARNORM_BACKEND=triton_shmem
  -> communication/triton_shmem.py
  -> communication/_triton_shmem_kernels.py
```

State cache key:

```text
(process_group identity, max_token_num, hidden_dim, dtype, device,
 profile and allocation/synchronization policy)
```

A different model hidden size or any environment-owned lifetime policy creates
a distinct state and symmetric allocation. Diagnostic overrides cannot reuse an
incompatible state created earlier in the process.

Known profiles are validated before any communication allocation. GPT-OSS
core-v3 requires gfx950, TP=4, hidden 2880, bf16, max-token cap 2048, HIP
`1,2,5,6`, pure TP, and its exact ring/kernel/grid policy. GLM profile v2
requires gfx950, TP=8, hidden 6144, bf16, max-token cap 42, all eight visible
devices, 156 input/output sites, a four-warp padded core through M42, and an M2
lower performance gate. GLM is operator-qualified only; captured model serving
remains unqualified. Unknown profiles and known profile mismatches decline
collectively to the complete unfused path.

`TS_TRITON_SHMEM_FUSION_MAX_M` is a diagnostic performance eligibility gate
independent of `max_token_num`; zero disables it. The former M=256 deployment
gate is rejected and no qualified profile enables it.
`TS_TRITON_SHMEM_FUSION_MIN_M` is the matching lower gate; profile v2 uses it
to capture complete ordinary fallback at M1.

Kernel variants:

- `oneshot_wholerow`: power-of-two hidden size, pull reduction, local output;
- `oneshot_wholerow_padded`: arbitrary hidden size padded to a power-of-two
  register row; scratch-free, masked, decode-specialized;
- `oneshot_blocked`: arbitrary hidden size, blocked reduction with fp32 scratch;
- `twoshot_blocked`: row-sharded reduction with symmetric output pushes.

The padded kernel keeps runtime M non-specialized. Scalar M1 specialization
expanded its persistent row loop into a much larger branch-heavy gfx950
program; `do_not_specialize=["M"]` restores the compact M2 code shape.
Explicit diagnostic `padded`/`blocked` variant selection applies when WS<=2 is
already one-shot as well as to the small-M overlay at WS>=4.

At TP=4 or TP=8, the state is normally two-shot, while the call-level
one-shot overlay handles small token counts. GPT-OSS core-v3 uses padded
whole-row for M<=64, blocked one-shot through M384, and eager/standalone
two-shot above M384. Captured production calls above M384 decline.

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
validated ROCm build. Export/open success is agreed across every rank. If any
rank cannot use a coarse HIP-IPC allocation, all ranks use fine-grained
symmetric memory for that and subsequent data buffers instead of mixing rank
outcomes or hanging in rendezvous.

## Synchronization

Every operation requires:

1. a leading cross-rank barrier so peer input writes are visible;
2. a trailing barrier so peers finish reading persistent input before reuse.

Generic behavior retains both. The rejected two-slot input ring delays reuse by
one call but depends on mutable host phase and remains disabled.

GPT-OSS core-v3 reserves 72 input sites; GLM profile v2 reserves 156. Capture
freezes one distinct symmetric view per unconditional fused call, and reuse is
delayed for a complete model-profile forward so intervening leading rendezvous
prove peer reads complete. Only exact validated site-ring profiles omit the
one-shot exit barrier. Unknown site counts retain it. Two-shot always retains
its output-completion barrier. The GPT-OSS ownership proof is detailed in the
[lifetime contract](producer-lifetime-contract.md); GLM currently has synthetic
156-site graph and transition evidence only.

Eager two-shot may return state-owned outputs only when two symmetric
norm/residual pairs are available. Calls alternate pairs to prevent the current
residual input aliasing the next residual output. Captured and caller-owned
output paths retain copy-out at the direct operator layer. Production
triton-shmem dispatch is stricter: captured calls above core-v3's persistent
M384 output-ring cap decline and capture the complete ordinary fallback. This
prevents two-shot prefill graphs from retaining transient custom-kernel output
pointers.

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

The generic backend has no selective compute cap and defaults to separate
barrier kernels. Core-v3 explicitly enables in-kernel barriers and a gfx950/TP=4
compute cap of 128 at M>=256. New model profiles must sweep their own widths,
token ranges, and divergence behavior before enabling either performance
policy.

## Preserved invariants

1. Per-allocation peer-pointer tables.
2. Leading and trailing ordering around persistent-buffer reuse, or a
   graph-stable site/epoch contract that replaces one-shot trailing ordering.
3. Coarse data buffers and fine-grained signal pad.
4. Explicit stream-ordered copy-in; folded copy-in remains diagnostic because
   serving invalidated the narrower single-wave synthetic proof.
5. Workgroup synchronization for multi-wave barrier participants.
6. Grid residency below the deadlock limit.
7. Full unfused fallback if fused state creation or eligibility declines.
8. Persistent caller-owned storage for outputs referenced by captured custom
   kernels; transient capture-time allocations are not a lifetime contract.

## Validation evidence

- Legacy base-path correctness: MI300X ws=1/2/4/8 with graph capture at ws=2/8;
  MI350X ws=2/4/8.
- Captured RCCL: ws=2/4/8 with blocking wait disabled.
- TP=4 default shared-state multigraph transitions: pass.
- The model-specific persistent-output lifetime contract was validated for
  GPT-OSS-120B's 72 captured fused sites; see the
  [lifetime contract](producer-lifetime-contract.md).
- Two-slot/no-exit ring: even-call graph and shared multigraph tests pass, but
  later canonical serving faults; keep disabled pending graph-stable slot phase.
- Profile-v2 site ring: eager wraparound, two interleaved 72-call graphs,
  1000 transition replays, bounded serving, and 15/15 safety pairs pass.
- Eager two-shot borrowed output: 72-site chained correctness plus
  M512/1024/2048 ping-pong and caller-output fallback pass.
- Core-v3 padded whole-row: random correctness, two interleaved 72-call graphs,
  20-test non-WS8 suite, 1000 transition replays, bounded serving, marker
  profiles, and 15/15 restricted-campaign safety pairs pass. WS=8 is not
  core-v3-qualified.
- Default-compatible serving: normal prefill graphs (40 buckets through M2048),
  normal decode capture, 0.95 HBM utilization, overlap scheduling, and generated
  health probes complete the M128-M4096 prefill ladder plus long decode on the
  qualified rank set. The dated compatibility study owns current measurements.
- GLM-5.2 profile v2: the padded extension establishes M2-M42 as profitable and
  M43 as the RCCL crossover. Its seven-M transition matrix proves ordinary
  fallback at M1 and padded triton at M2-M42 over 1,000 replays and 1,649
  checked operations/rank. Production graph serving and restart-randomized
  performance qualification remain open.
- Dynamic-M padded M1: WS4 156-site replay improved 21.8% versus the specialized
  kernel and passed a bounded 70-replay M1/M2/M42 eager/graph integration probe.
  It remains 3.9% behind unfused at WS4, so profile v2 retains M1 fallback
  pending qualified WS8 timing.
- GLM definitive MI355X sweep: all 105 forced candidate artifacts resolved the
  four-warp padded path across WS2/4/8. The complete 315-result matrix confirms
  the WS8 raw M1-M42 window and M43 RCCL loss border. This is cross-machine
  operator evidence and does not requalify the MI350X profile.

Producer-direct inputs and any genericization of borrowed outputs or
trailing-barrier removal require the separate
[producer and buffer lifetime contract](producer-lifetime-contract.md).

