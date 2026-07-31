# GPT-OSS-120B fused AR+RMSNorm status

Updated: 2026-07-31

This is the sole live deployment decision and priority page for GPT-OSS-120B,
TP=4, on MI350X (gfx950). Dated study conclusions remain historical evidence
and do not override this page.

## Deployment decision

Promote explicit `triton_shmem` fusion for the **capacity objective** only on
the qualified rank set:

```text
HIP_VISIBLE_DEVICES=1,2,5,6
physical GPUs=0,2,4,6
AR_NORM_PROFILE_ID=gpt-oss-120b-mi350x-triton-core-v3
TS_ARNORM_BACKEND=triton_shmem
--enable-allreduce-fusion
```

The 3-block x 5-seed campaign completed 15/15 safe pairs:

- output throughput: **+1.29%** (95% CI +0.52% to +2.08%);
- median TPOT: **-0.80%** (95% CI -1.42% to -0.25%);
- mean TPOT: **-1.21%** (95% CI -1.93% to -0.50%);
- median absolute TPOT: 12.716 ms upstream-unfused, 12.642 ms triton;
- median absolute throughput: 2381.2 versus 2402.1 output tokens/s.

This clears the predeclared +1% capacity gate with no latency regression. It
does not clear the -1.5% latency objective.

Keep explicit upstream-unfused as the control and fallback for unqualified rank
sets, profile mismatch, or backend decline:

```text
TS_ARNORM_BACKEND=auto
ENABLE_ALLREDUCE_FUSION=0
--disable-allreduce-fusion
--comm-fusion-max-num-tokens 2048
```

`ENABLE_ALLREDUCE_FUSION` is the launcher input; the emitted `RUN_ENV` must show
`ENABLE_FUSION=0`. A valid control must also prove resolved
`enable_allreduce_fusion=False`; upstream otherwise auto-enables fusion on
supported AMD TP mappings.

## Qualified profile

The source of truth for profile defaults is
`benchmark/profiles/ar_rmsnorm/gpt_oss_120b_mi350x.env`. Core dispatch is:

```text
M<=64:    padded scratch-free whole-row one-shot, four warps
M=65-384: blocked one-shot
M>384:    two-shot
```

Required lifetime behavior:

- 72 graph-stable input sites remove the one-shot exit barrier only for the
  qualified model profile;
- 72 persistent output sites preserve captured output addresses;
- eager two-shot calls ping-pong two symmetric output pairs;
- captured two-shot and explicit caller-output paths retain copy-out;
- folded copy-in and the mutable two-slot input ring remain disabled;
- unknown profiles retain generic barriers and complete unfused fallback.

Graph and serving evidence:

- 72-call M32 graph: **16.35 us/site**, down 35.2% from profile v2;
- scratch-free core: **9.93 us/site**, within 1.13 us/site of Iris;
- marker target stage: **1.492 ms** versus 1.866 ms upstream-unfused;
- max-rank GPU period: **13.071 ms** versus 13.108 ms upstream-unfused.

Promotion is limited to HIP `1,2,5,6`. Requalify every other TP=4 rank set
independently. WS=8 remains deferred and unqualified.

## Evidence

- [Core-v3 tuning and campaign](../studies/mi350x/2026-07-triton-shmem-core-tuning/README.md)
- [Three-backend decomposition](../studies/mi350x/2026-07-triton-shmem-decomposition/README.md)
- [Profile-v2 realignment](../studies/mi350x/2026-07-triton-shmem-realignment/README.md)
- [Post-rebase baseline reset](../studies/mi350x/2026-07-post-rebase-baseline/README.md)
- [Benchmark and promotion methodology](benchmark-methodology-recommendations-2026-07.md)

## Historical boundaries

- The initial post-rebase Iris and explicit-triton campaigns were safe but
  slower than upstream-unfused.
- Realigned profile v2 removed copy-out and the one-shot exit barrier but still
  failed performance promotion; it remains the lifetime-integration baseline.
- Pre-rebase profile v4 is legacy performance evidence. Its reserved-sink
  padding and persistent captured-output requirements remain safety invariants.
- The M=256 performance gate and mutable two-slot/no-exit input ring remain
  rejected.
- Legacy marker-free trace grouping is heuristic only.

Detailed historical percentages belong in their study summaries, not in this
live decision record.

## Priorities

1. Deploy core-v3 only on qualified HIP `1,2,5,6`, retaining immutable profile
   and per-rank padded-kernel proof.
2. Keep explicit fusion-off as the matched control and fallback.
3. Requalify other TP=4 rank sets; qualify WS=8 only when the full device set is
   available.
4. Do not repeat blocked-core cap/grid/block/XCD/fast-path sweeps.
5. Treat producer-direct output and progress publication as one systems
   project spanning dense GEMM, active MXFP4 MoE, graph ownership, and fallback.
6. Investigate the mixed ordinary-Iris/RCCL captured-transition timeout without
   weakening production fallback or graph isolation.
7. Keep `all_reduce_two` separate; retain its initial correctness incident
   despite later clean reproductions.
8. Preserve graph-stable lifetime, universal sink padding, complete fallback,
   and explicit synchronization contracts in every candidate.

Technical contracts:
[backend design](backend-design-and-safety.md),
[producer lifetime](producer-lifetime-contract.md), and
[profiling workflow](profiling-workflow.md).
