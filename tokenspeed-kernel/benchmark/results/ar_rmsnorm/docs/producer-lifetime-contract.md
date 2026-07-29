# GPT-OSS-120B AR+RMSNorm producer and buffer lifetime contract

Updated: 2026-07-29

This contract was derived from the pre-rebase `triton_shmem` implementation.
It remains a safety requirement for explicit `triton_shmem` and useful prior
art for any captured backend; its performance assumptions are legacy after
upstream `3f88dcc2`.

## Scope

This document establishes the ownership requirements that must be satisfied
before a row-parallel producer writes directly into `triton_shmem` symmetric
storage or the fused backend returns borrowed persistent outputs. It applies to
GPT-OSS-120B, TP=4, N=2880, and the compiler-managed all-reduce path.

It is a safety contract, not a deployment decision. The canonical optimization
queue remains in [GPT-OSS-120B status](gpt-oss-120b-status.md).

## Current execution and ownership

The compiler marks row-parallel attention and MoE outputs as `Partial`, defers
their reduction, and inserts `FusedReduceNormOp` at the immediately following
norm:

```text
attention o_proj / MoE finalize
  -> newly allocated rank-local Partial tensor
  -> DeferredReduceOp (no-op)
  -> FusedReduceNormOp
  -> norm_out consumed by the next compute module
  -> residual_out retained until the next fused norm
```

The attention producer is `GptOssAttention.o_proj`, a
`RowParallelLinear(reduce_results=False)`. Its quantization method calls
`tokenspeed_kernel.mm`, whose current API allocates and returns its output; it
does not accept caller-owned output storage.

The MoE producer calls `tokenspeed_kernel.moe_apply`. The active MXFP4 path
allocates the second GEMM output and, for top-k greater than one, creates the
final `[M,N]` tensor with a separate reduction. It also has no general
caller-owned output argument.

`CompiledDecoderLayer.forward` replaces `state.hidden_states` immediately after
each step. The rank-local producer output has one semantic consumer:
`FusedReduceNormOp`. It is not cloned or retained by the compiler. This makes
producer-direct input semantically possible, but only after the producer APIs
can honor an explicit output pointer.

## Required producer-direct input contract

A producer may write directly into symmetric input storage only when all of the
following hold:

1. The backend reserves the buffer before the producer launch and returns a
   correctly shaped contiguous bf16 view.
2. The selected GEMM/MoE implementation accepts that exact view as its output;
   copying an already-produced tensor into the view is not producer-direct.
3. Every TP rank chooses the same logical ring slot for the same model site.
4. The pointer is graph-stable for capture and every replay.
5. No producer, fallback path, or graph transition overwrites a slot until all
   peers have completed their pulls from its previous generation.
6. If fusion declines, the partial tensor remains valid for the complete
   unfused all-reduce plus norm fallback.
7. The contract is explicit in the producer API; pointer borrowing must not be
   inferred from allocator reuse or tensor object lifetime.

The current `tokenspeed_kernel.mm` and `moe_apply` APIs fail requirement 2.
Producer-direct input therefore must not be enabled yet.

## Output lifetime contract

The two fused outputs have different lifetimes:

- `norm_out` is consumed by the immediately following attention or MoE
  producer. Stream ordering normally completes that consumer before the next
  fused site.
- `residual_out` remains live until the next fused norm and is read while that
  norm writes the next residual.

A single borrowed output allocation is therefore invalid. At minimum,
`residual_out` requires ping-pong storage so the current residual input and next
residual output never alias. Any borrowed `norm_out` must also prove that the
next producer has completed before its slot is reused. Returning a mutable
state-owned tensor without this contract is unsafe under graph replay.

The earlier assumption that `torch.empty_like` inside capture gave each fused
site stable storage was disproven in full serving. Those tensors were transient
allocations in the HIP graph private pool; captured custom kernels retained raw
output pointers after Python tensor lifetimes and graph-pool reuse/layout had
diverged.

Qualified profile `gpt-oss-120b-mi350x-qualified-v4` preallocates 72 `norm_out`
and 72 `residual_out` slots before capture for `M <= 256`. Each GPT-OSS fused
site captures one persistent pair, and a complete 72-site forward does not
revisit it until all prior consumers are dead. Larger eager-prefill outputs may
remain dynamically allocated because no graph retains their pointers. This
contract is model-specific and must not be inferred for a model with a
different site count or conditional execution.

## One-barrier input-ring proof

The backend-only lifetime prototype is a two-slot symmetric input ring, used by
every one-shot and two-shot call:

```text
call k writes/pulls slot k mod 2
call k+1 writes/pulls the other slot
the leading rendezvous of call k+1 proves every rank completed call k
call k+2 may then reuse call k's slot
```

For one-shot pull kernels, this permits the trailing reuse barrier to be
omitted: the next call's leading rendezvous supplies the required cross-rank
completion proof before the old slot is reused. Two-shot must retain its
completion barrier because peer output pushes must be visible before local
copy-out, but it must still participate in input-slot alternation so a
one-shot-to-two-shot transition cannot overwrite the preceding slot.

This proof depends on:

- one ordered stream per TP rank for these calls;
- every rank executing the same call sequence and ring slot;
- no out-of-band writer to either input slot;
- a graph replay containing an even number of ring advances, or an explicit
  graph-level ring epoch. GPT-OSS-120B has 72 eligible sites, so its qualified
  decode graph has even parity; profile-v4 prefill is eager and does not extend
  that graph proof;
- retaining the existing fixed-participant safeguards for M-divergent
  execution.

An odd-site captured graph that always restarts from the same captured slot can
reuse the previous replay's final slot immediately and is not covered by this
proof. Even GPT-OSS's even call count was insufficient because mutable
host-side phase during capture is not explicit graph/call-site slot identity.
The no-exit ring remains disabled.

## Prototype backend stage

The diagnostic implementation:

1. Allocate two symmetric input buffers and two independent peer-pointer
   tables.
2. Alternate the selected input slot for all fused paths.
3. Adds a one-shot kernel constexpr that retains the leading barrier but omits
   the trailing barrier only when the double-buffer ring is active.
4. Keep two-shot's trailing completion barrier.
5. Exposes one opt-in environment flag.
6. Leaves the ring disabled by default.

Required validation:

- eager correctness with changing inputs;
- 100+ repeated graph replays;
- even and odd synthetic call chains;
- one-shot/two-shot and M-shape graph transitions;
- shared-state interleaved multigraph execution;
- the complete GPT-OSS multi-arm serve;
- max-rank M=32 graph latency and per-forward production traces.

The ring saved 5.37 us/site in graph replay but later faulted in canonical
serving. Before producer APIs are extended, replace host alternation with
capture-frozen positional slots `(forward_epoch + call_index) mod 2` plus an
explicit graph/forward epoch. Retain the exit barrier on any path without that
complete identity contract:

- add caller-owned output support to the active dense GEMM;
- add a fused/caller-owned MoE final output path;
- reserve symmetric ring slots through an explicit compiler/runtime interface;
- add ping-pong caller-owned fused outputs.

## Decision boundary

The input ring cleared the 5 us/site performance threshold but failed the
serving safety gate. Producer-direct plumbing remains blocked until graph-stable
slot identity makes the no-exit lifetime proof valid in serving.
