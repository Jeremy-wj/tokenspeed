# Upstream-main rebase and AR+RMSNorm baseline reset

Updated: 2026-07-30

## Executive decision

The local six-commit AR+RMSNorm series was rebased from `f35ea4ef` onto
upstream `main` at `3f88dcc2`. The old branch head was `a031a98b`; the rebased
head is `7751b072`.

This is a performance-baseline boundary. Upstream changed both sides of the
comparison:

- fused AMD AR+residual+RMSNorm now selects Iris by default;
- ordinary small AMD all-reduce also selects Iris;
- AMD can reduce two independent tensors in one Iris launch;
- Kimi K3 adds a separate NVIDIA/TRT-LLM one-shot lane and latent-norm fusion;
- graph, profiling, serving, and communication runtime code moved substantially.

Consequently, every pre-rebase latency, throughput, kernel, graph, and
operator result in this project is **legacy performance evidence**. Profile v4
remains valuable proof of the old `triton_shmem` lifetime and transition
contracts, but it does not qualify the new upstream default or provide a
current performance baseline.

The 2026-07-30 reset now provides that baseline. Upstream-unfused with explicit
fusion disablement is the GPT-OSS-120B TP=4 control; Iris and `triton_shmem`
completed safety qualification but failed performance promotion.

The rebased policy is:

```text
TS_ARNORM_BACKEND unset or auto -> upstream Iris -> native symm_mem fallback
TS_ARNORM_BACKEND=iris          -> Iris only -> complete unfused caller fallback
TS_ARNORM_BACKEND=symm_mem      -> native symm_mem -> complete unfused fallback
TS_ARNORM_BACKEND=triton_shmem  -> local experimental backend -> complete
                                   unfused caller fallback
```

`triton_shmem` remains available for controlled comparison, but no longer
replaces upstream behavior implicitly.

## Mechanical rebase record

### Safety and provenance

- Original local-only head: `a031a98b1bb1a6fecf3a5ac5a6059d2e5b53d77b`
- Fresh upstream target: `3f88dcc2575af30eb44ef02448ee880ebcc40f25`
- Rebased local head: `7751b072460193d78f638026a113ca1fdca7d84e`
- Original series: six commits
- Rebased series: six commits
- Backup ref:
  `refs/backup/jeremwan-triton-shmem-experiments-pre-rebase-20260729`
- External Git bundle and raw-evidence archive:
  `/home/jeremwan/tokenspeed-pre-rebase-backups/20260729T2047Z/`
- Ignored raw evidence verified after promotion: 5,473 files

Commit mapping:

```text
cd109677 -> b4a8e58d  First working version, very slow performance
1a263041 -> 5c1458ab  Perf tuning: fix memory allocation mode and implement barriers
42290463 -> b185e674  Barrier fixes and complete gptoss120B results
12c35bf0 -> 91427b46  Profiling and trace refactor
a1b46833 -> 5cfc190b  Project reorganization
a031a98b -> 7751b072  Serving/integration fixes and documentation
```

### Significant conflict-resolution changes

1. **Generic AMD all-reduce:** kept upstream Iris `all_reduce` and
   `all_reduce_two`. Removed the local native symmetric-buffer replacement and
   its `TS_TRITON_AR_DISABLE`, `TS_TRITON_AR_MAX_BYTES`, and
   `TS_TRITON_AR_WORKGROUP_SYNC` controls.
2. **Fused AR+RMSNorm routing:** restored upstream Iris-first behavior for
   `auto`; retained local `triton_shmem` only as an explicit experimental
   selector. The local complete unfused fallback remains.
3. **Profiling:** kept upstream VizTracer-to-Proton flow events, metric
   filtering, finalize recovery, AMD visibility checks, and TP finalize
   barrier. Retained local pre-graph startup, graph scopes, import-time session
   shutdown, and forward markers.
4. **Graph replay:** kept upstream valid-row replay and stale-tail clearing,
   wrapped by the local optional profiling scope. Preserved the local universal
   reserved sink and persistent per-site fused outputs because they solve
   distinct captured-address lifetime faults.
5. **Runtime/serving auto-merges:** retained upstream EPD/SIGTERM lifecycle,
   communication topology behavior, and current server policy. Preserved only
   orthogonal local serving controls and safety fixes.

## Changed default AR+RMSNorm behavior

### Call path

The compiler-level decision is unchanged:

```text
models/base/comm_ops.py
  FusedReduceNormOp or FinalNormOp
    -> RMSNorm.forward_with_allreduce_fusion
       -> tokenspeed_kernel.ops.communication.triton
          .allreduce_residual_rmsnorm
```

Fusion is still selected when all-reduce fusion is enabled and the token count
does not exceed `comm_fusion_max_num_tokens`. What changed is the AMD kernel
selected after that decision.

### Upstream Iris-first path

For eligible contiguous 2-D bf16 tensors, upstream creates or reuses
`IrisAllReduceResidualRMSNorm` keyed by process group, workspace cap, hidden
width, and dtype. Iris owns a symmetric heap, stages rank-local partials, and
runs its fused all-reduce + residual-add + RMSNorm kernel. If Iris eligibility
does not hold under `auto`, the native PyTorch symmetric-memory implementation
is attempted.

This differs from the old branch in several important ways:

- no `TS_ARNORM_BACKEND` override is required to reach the upstream path;
- `auto` no longer enters `triton_shmem`;
- `TS_TRITON_SHMEM_FUSION_MAX_M` affects only explicit `triton_shmem` runs;
- the ordinary unfused all-reduce baseline may use Iris too, so old “unfused”
  numbers are not a stable control for the new code;
- upstream auto-enables fusion for supported single-node AMD TP mappings,
  rather than applying the old TP<=2 restriction derived from `triton_shmem`
  results.

### Preserved complete fallback

The local `RMSNorm.forward_with_allreduce_fusion` safeguard remains. If a fused
backend returns no result after the compiler deferred reduction, the runtime
performs the missing all-reduce before ordinary residual-add + RMSNorm. Without
this, a backend decline could normalize rank-local partials.

This fallback is especially important now that four explicit dispatch modes
exist and each has different eligibility. It is correctness behavior, not a
performance policy.

### Explicit local `triton_shmem`

The project implementation remains structurally distinct:

- coarse HIP-IPC allocations carry bulk data;
- PyTorch symmetric memory supplies the signal pad;
- one-shot whole-row, one-shot blocked, and two-shot blocked kernels are
  selected by world size, hidden width, and token count;
- optional per-site persistent outputs satisfy captured-graph address lifetime;
- leading/trailing barriers or a separately proven epoch/ring contract protect
  peer-buffer reuse;
- a model-specific M gate can decline to the complete unfused path.

These features are still useful research mechanisms. They are not evidence
that the implementation should override Iris by default.

### Immediate consequences

1. Profile v4 no longer describes the default backend.
2. Prior fused-versus-unfused pairs changed treatment and control
   simultaneously.
3. Old dispatch crossovers, grid caps, barrier costs, and output-ring overhead
   cannot be transferred numerically to Iris.
4. Safety findings about captured pointers, padding rows, and transition
   identity remain applicable engineering constraints for any backend using
   persistent or captured buffers.
5. A new campaign must record the resolved backend and observed kernel
   signatures; `enable_allreduce_fusion=True` is no longer sufficient identity.

## New AR fusion primitives in `comm_ops`

Upstream added four public runtime operations. They are not one replacement
for this project's fused AR+RMSNorm; they cover two different model patterns.

### `all_reduce_two`

`all_reduce_two(first, second, group)` extends `CommBackend` with a paired
collective. Unsupported backends fall back to two ordinary all-reduces. On
node-local AMD, `TritonAllReduceBackend` can dispatch one Iris kernel when both
tensors are contiguous bf16, share a device, are non-empty, and fit the common
workspace.

Kimi K3 uses this for routed and shared expert outputs. It saves a launch and
shares synchronization, but it does **not** add a residual or normalize either
tensor.

Comparison with local `triton_shmem`:

- same broad goal: amortize collective launch/synchronization overhead;
- different payload: two independent reductions versus one reduction with
  residual and RMSNorm epilogue;
- different memory system: Iris symmetric heap versus coarse HIP-IPC data plus
  a fine-grained signal pad;
- different result contract: two reduced tensors versus `(norm_out,
  residual_out)`.

### `prepare_all_reduce_lane`

This backend capability prepares a wider one-shot Lamport lane through the
TRT-LLM all-reduce backend. Its base implementation returns false; the auto
backend delegates to TRT-LLM. Because it is collective, real failures are not
swallowed—rank disagreement during preparation would be unsafe.

This is workspace/topology preparation, not an operator. It has no AMD
`triton_shmem` equivalent today.

### `prepare_all_reduce_fusion`

This kernel-level preparation initializes TRT-LLM fused-all-reduce workspace
before graph capture. It returns false on non-NVIDIA platforms. The runtime
wrapper converts initialization failures to an unavailable result so callers
can choose an ordinary path.

The analogous local requirement is eager creation/rendezvous of
`triton_shmem` state before capture. The local implementation currently
exposes that through backend state creation and profile warmup rather than one
platform-neutral preparation API.

### `all_reduce_latent_norm`

This is a separate NVIDIA/TRT-LLM fusion:

```text
[routed latent partial | shared hidden partial]
  -> one-shot all-reduce in a persistent lane
  -> RMSNorm only on the latent prefix
  -> reduced lane written in place
```

The freshly fetched upstream commit `3f88dcc2` sets
`trigger_completion_at_end=True` from `comm_ops`. This completes the one-shot
Lamport AR at the end of latent normalization instead of leaving completion
for a later operation.

It differs from this project's operator in every key contract:

- Kimi lane fusion is currently NVIDIA/TRT-LLM; local `triton_shmem` is AMD;
- Kimi uses a one-row persistent concatenated lane; local code supports
  arbitrary `(M, N)` within its workspace;
- Kimi normalizes only a latent prefix in place; local code adds a separate
  residual and emits distinct norm/residual outputs;
- Kimi's completion flag is part of one-shot Lamport sequencing; local code
  uses explicit barriers and experimental ring/epoch mechanisms.

The reusable idea is not the kernel itself but the preparation/lifetime API:
reserve graph-stable storage, prepare collectively before capture, make
completion explicit, and expose a complete fallback.

## Legacy result overview

All numbers below identify the old code/runtime and must not be used as a
post-rebase baseline.

### Final profile-v4 campaign

- GPT-OSS-120B, MI350X, TP=4, 3 restart blocks and 15 pairs
- safety: 15/15 completed without a safety failure
- fused median TPOT: **+1.44%** regression, 95% CI +1.20% to +1.65%
- output throughput: **-1.46%**, 95% CI -1.91% to -1.02%
- disposition: old `triton_shmem` candidate rejected for promotion

This remains the strongest old-backend safety result, not a current backend
decision.

### Historical 2026-07-24 matched pair

- median TPOT: 12.560 ms fused versus 12.755 ms unfused (**-1.5%**)
- throughput: **+0.09%**
- kernels per rank: 29,967 fused versus 31,333 unfused
- profiled GPU window: about **-7.8%**
- fused one-shot medians: roughly 34–36.5 us

This pair used an older serving profile and is legacy mechanism evidence.

### Operator and graph screens

- old TP=4/N=2880 crossover: fused faster through M=256 and **26.8% slower**
  at M=512; degradation reached **49.8%** at M=8192
- M=128: 43.78 us fused versus 65.45 us old unfused
- M=256: 61.08 us versus 66.84 us
- two-slot input graph replay: 35.35 to 29.98 us/site (**-15.19%**), later
  rejected in serving because host-phase slot identity was not graph-stable
- block-N=2048 graph override: about **-11.4%** in the isolated replay, not
  promoted
- the M=256 performance gate, fixed-grid variants, folded copy-in, and
  host-alternated input ring remain rejected old-backend candidates

### Profiling, stability, and migration evidence

- 2026-07-23 Proton summaries contain strong rank skew and pre-profile-v4
  kernel timings; they are diagnostic only
- graph-padding and captured-output lifetime root causes are closed historical
  incidents whose invariants remain relevant
- the 2026-07-27 contaminated repeatability root and intermediate HIP-graph
  diagnosis remain incident records, not baselines
- MI300X migration sweeps are legacy across both hardware generation and
  backend/runtime revision

Raw artifacts remain immutable. Legacy labels are applied in indexes and
curated summaries rather than rewriting captured logs and traces.

## Completed baseline reset

The 2026-07-30 GPT-OSS-120B TP=4 campaign completed the evidence ladder on
WS=2/4; WS=8 remains deferred because physical GPU 3 is occupied.

The reset also exposed a policy gap: leaving `--enable-allreduce-fusion` absent
does not pin fusion off because upstream auto-enables it on supported AMD TP
topologies. The project added `--disable-allreduce-fusion`, and the canonical
control now proves resolved `enable_allreduce_fusion=False`.

Completed actions:

1. **Upstream unfused control:** ordinary eligible decode AR resolves to Iris,
   followed by standalone RMSNorm; larger payloads fall back to RCCL.
2. **Upstream default fused control:** Iris `auto` state, signatures,
   correctness, fixed graph, transitions, and serving were captured.
3. **Explicit local candidate:** `TS_ARNORM_BACKEND=triton_shmem` was run on
   identical code, runtime, topology, and workloads with rank-side selected
   backend proof.
4. **Operator matrix:** WS=2/4, model widths, M=1/256/257/2048 boundaries,
   one-/two-shot, fallback, and separate `all_reduce_two` evidence were
   collected with two noise-opposed passes.
5. **Graph/transition matrix:** 1,000-replay fixed graphs and bounded
   interleaved graph/eager transitions passed for both fused candidates. A
   synthetic mixed ordinary-Iris/RCCL captured transition timed out and remains
   open; bounded real serving passed.
6. **Authoritative profiling:** forward markers identified 73 ordinary
   Iris+RMSNorm sites, 72 fused Iris sites, and 72 fused `triton_shmem` sites
   per decode forward.
7. **End-to-end qualification:** both candidates completed three restart
   blocks and fifteen randomized pairs versus upstream-unfused.

Final result:

- Iris fused: **+2.55%** median TPOT (95% CI +1.00% to +5.43%) and
  **-2.29%** output throughput (95% CI -4.78% to -0.89%);
- explicit `triton_shmem`: **+10.47%** median TPOT (95% CI +6.71% to
  +19.23%) and **-10.58%** output throughput (95% CI -23.21% to -4.49%);
- both campaigns: 15/15 pairs completed without a safety failure;
- deployment: upstream-unfused with explicit fusion disablement;
- promotion: both fused candidates rejected.

Sources:
[post-rebase baseline study](../studies/mi350x/2026-07-post-rebase-baseline/README.md)
and the live [GPT-OSS-120B status](gpt-oss-120b-status.md).

## Verification performed during rebase

- `git range-diff` preserved all six logical commits
- promoted branch merge-base equals `3f88dcc2`
- Python compilation passed for the changed runtime, profiling, graph, and
  communication modules
- kernel profiling tests: 20 passed
- graph-analysis and repeatability-runner tests: 24 passed
- runtime test collection was blocked by the existing container's stale
  `tokenspeed_scheduler` binary, which lacks the new upstream
  `PagedCacheTransferPolicy` symbol
- raw evidence checksum verification passed for all 5,473 ignored files

