# GPT-OSS-120B AR+RMSNorm producer and buffer lifetime contract

Updated: 2026-08-01

This is the normative ownership contract for explicit `triton_shmem` and any
captured backend that borrows persistent inputs or outputs.

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

Profile `gpt-oss-120b-mi350x-triton-realigned-v2` now implements the eager
two-shot case with two coarse symmetric `norm_out`/`residual_out` pairs and
independent peer-pointer tables. Consecutive two-shot sites alternate pairs;
the completion barrier remains, so peer pushes are visible before return.
Direct captured two-shot calls and calls with explicit caller outputs retain
the copied compatibility path for operator testing. Production dispatch now
declines captured calls above M384 to avoid transient returned outputs. A
72-site chained correctness test covers M257 and repeated M512/1024/2048 eager
calls with residual outputs fed into the next site.

Core-v3 inherits these lifetime contracts unchanged; it replaces only the
M<=64 decode core and has independently passed graph, transition, bounded-serve,
and 15/15 safety pairs under both restricted and default-compatible campaigns.

The earlier assumption that `torch.empty_like` inside capture gave each fused
site stable storage was disproven in full serving. Those tensors were transient
allocations in the HIP graph private pool; captured custom kernels retained raw
output pointers after Python tensor lifetimes and graph-pool reuse/layout had
diverged.

Core-v3 preallocates 72 `norm_out` and 72 `residual_out` slots through its
one-shot M384 cap. Each GPT-OSS fused site captures one persistent pair, and a
complete 72-site forward does not revisit it until all prior consumers are
dead. Eager two-shot calls use the separate two-pair borrowed-output path.
Captured calls above M384 decline triton-shmem before launch and capture the
complete ordinary all-reduce + RMSNorm fallback; they never return transient
custom-kernel outputs. Direct operator calls with explicit caller outputs retain
the copied two-shot compatibility path. This contract is model-specific and
must not be inferred for a model with a different site count or conditional
execution.

## Rejected two-slot prototype

The former host-alternated two-slot input ring saved 5.37 us/site but faulted in
serving. Even site-count parity was insufficient because mutable capture-time
host phase did not establish graph/call-site slot identity. It remains disabled.
Any future generic ring requires capture-frozen position plus an explicit
graph/forward epoch and must retain the exit barrier on every unmatched path.

## Qualified per-site input ownership

The realigned profile does not enable the rejected two-slot ring. It reserves
72 one-shot input sites with
`TS_TRITON_SHMEM_INPUT_SITE_RING=72`. During capture, each unconditional
GPT-OSS fused call selects a distinct view; the graph freezes that pointer.
A site is not reused until a complete 72-call forward has executed.

Because every later site has a leading rendezvous before the old site can be
reused, peers have completed the prior pull. The one-shot exit barrier is
therefore omitted only for this explicit model-profile ring. Generic states,
unknown site counts, two-shot completion, and profiles without the ring retain
their original barriers.

Validation includes eager wraparound, two interleaved 72-call graph variants,
100 replays in the focused test, 1000 M-boundary transition replays, bounded
serving, and 15/15 safety pairs. The 72-call graph improved by 19.8%.

## Decision boundary

The original two-slot input ring cleared the 5 us/site performance threshold
but failed the serving safety gate. The per-site replacement now has
graph-stable identity and passed serving. Producer-direct plumbing remains
blocked independently: active `tokenspeed_kernel.mm` and MXFP4 `moe_apply`
still cannot honor exact caller-owned output views, and fallback must preserve a
rank-local partial tensor.
