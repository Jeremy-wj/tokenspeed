# GPT-OSS-120B fused AR+RMSNorm status

Updated: 2026-08-04

This is the live deployment decision and priority page for GPT-OSS-120B,
TP=4, on MI350X (gfx950). Dated study conclusions remain historical evidence
and do not override this page.

## Deployment decision

Keep explicit upstream-unfused as the deployment default. Core-v3 is safe with
base TokenSpeed defaults on the qualified rank set. The final clean 15-pair
campaign applied `--disable-overlap-schedule` symmetrically as a reversible
performance policy and measured:

- output throughput: **+0.45%** (95% CI -0.05% to +0.92%);
- median TPOT: **-0.51%** (95% CI -0.90% to -0.06%);
- mean TPOT: **-0.48%** (95% CI -0.93% to +0.01%).

Median TPOT improves, but throughput does not clear the +1% capacity threshold
and latency does not reach -1.5%. Base overlap scheduling remains supported; it
showed repeatable fresh-server mode variance and is not used for performance
qualification. Current deployment therefore uses:

```text
HIP_VISIBLE_DEVICES=1,2,5,6
physical GPUs=0,2,4,6
AR_NORM_PROFILE_ID=gpt-oss-120b-mi350x-triton-core-v3
TS_ARNORM_BACKEND=auto
ENABLE_ALLREDUCE_FUSION=0
--disable-allreduce-fusion
--comm-fusion-max-num-tokens 2048
```

Explicit triton-shmem remains a safety-qualified candidate and diagnostic:

```text
TS_ARNORM_BACKEND=triton_shmem
--enable-allreduce-fusion
--disable-overlap-schedule  # matched performance policy, not correctness
```

For deployment/control, `ENABLE_ALLREDUCE_FUSION` is the launcher input and the
emitted `RUN_ENV` must show `ENABLE_FUSION=0`. A valid control must also prove
resolved
`enable_allreduce_fusion=False`; upstream otherwise auto-enables fusion on
supported AMD TP mappings.

## Safety-qualified profile

The source of truth for profile defaults is
`benchmark/profiles/ar_rmsnorm/gpt_oss_120b_mi350x.env`. Core dispatch is:

```text
M<=64:    padded scratch-free whole-row one-shot, four warps
M=65-384: blocked one-shot
M>384:    eager/standalone two-shot; captured production calls decline
```

Required lifetime behavior:

- 72 graph-stable input sites remove the one-shot exit barrier only for the
  qualified model profile;
- 72 persistent output sites preserve captured output addresses;
- eager two-shot calls ping-pong two symmetric output pairs;
- captured calls above M384 decline to complete ordinary fallback;
- explicit caller-output two-shot calls retain copy-out;
- folded copy-in and the mutable two-slot input ring remain disabled;
- unknown profiles retain generic barriers and complete unfused fallback.

Base serving behavior is now part of the qualified profile:

```text
gpu_memory_utilization=0.95
prefill graphs enabled through M2048
automatic decode capture sizes
overlap scheduling enabled
generated health probes enabled
```

Known profile mismatches decline collectively before state creation. Generic
triton-shmem defaults to separate barriers and no architecture-specific grid
cap; core-v3 explicitly owns its faster gfx950/TP=4 policy.

Historical restricted-configuration graph and serving evidence:

- 72-call M32 graph: **16.35 us/site**, down 35.2% from profile v2;
- scratch-free core: **9.93 us/site**, within 1.13 us/site of Iris;
- marker target stage: **1.492 ms** versus 1.866 ms upstream-unfused;
- max-rank GPU period: **13.071 ms** versus 13.108 ms upstream-unfused.

Safety qualification is limited to HIP `1,2,5,6`. Requalify every other TP=4
rank set independently. WS=8 remains deferred and unqualified.

## Definitive campaign result

The [current-machine definitive campaign](../studies/mi350x/2026-08-gpt-oss-120b-definitive-sweep/README.md)
completed its five-triplet extension on 2026-08-04. The initial M92 timeout was
a benchmark teardown-lifetime bug; releasing captured collective graphs before
destroying `ProcessGroupNCCL` fixed the graph and transition paths. The complete
micro matrix supported no positive cap, so cap `0` was frozen for every WS.

The current machine used model path `/data/models/openai/gpt-oss-120b`
(realpath `/data/dev/morhuang/models/gpt-oss-120b`) and nested topology-balanced
HIP sets `1,5`, `1,2,5,6`, and all eight devices. Triton versus
upstream-unfused measured:

- WS2: throughput **+1.10%** (95% CI -0.89% to +3.11%), median TPOT
  **-0.48%** (95% CI -2.03% to +1.07%): inconclusive;
- WS4: throughput **+1.27%** (95% CI +0.49% to +2.01%), median TPOT
  **-1.30%** (95% CI -1.91% to -0.63%): promising capacity screen;
- WS8: throughput **-2.48%** (95% CI -6.93% to +0.22%), median TPOT
  **+0.63%** (95% CI +0.03% to +1.25%): loss.

All 45 server lifecycles and authoritative marker analyses completed. Five
pairs cannot promote deployment; existing WS4 policy remains live, explicit
upstream-unfused remains default, and WS2/8 receive no deployment qualification.

## Evidence

- [Default compatibility and requalification](../studies/mi350x/2026-07-default-compatibility/README.md)
- [Core-v3 tuning and campaign](../studies/mi350x/2026-07-triton-shmem-core-tuning/README.md)
- [Three-backend decomposition](../studies/mi350x/2026-07-triton-shmem-decomposition/README.md)
- [Profile-v2 realignment](../studies/mi350x/2026-07-triton-shmem-realignment/README.md)
- [Post-rebase baseline reset](../studies/mi350x/2026-07-post-rebase-baseline/README.md)
- [Definitive current-machine campaign](../studies/mi350x/2026-08-gpt-oss-120b-definitive-sweep/README.md)
- [Benchmark and promotion methodology](benchmark-methodology-recommendations-2026-07.md)

## Engineering highlights

Most important optimizations:

- scratch-free masked 4096-lane decode core for hidden 2880;
- 72 graph-stable input sites, removing the qualified one-shot exit barrier;
- coarse HIP-IPC data buffers with a fine-grained signal pad;
- two-pair eager two-shot output borrowing;
- profile-owned four-warp and gfx950/TP=4 grid policies.

Critical bugs closed:

- padded graph rows aliasing live request slot 0 and underflowing KV pages;
- transient capture-time custom-kernel outputs;
- folded-copy publication without a valid system-release contract;
- scalar barriers racing sibling wavefront memory operations;
- fused decline omitting the required all-reduce;
- incompatible states sharing a cache key after policy changes.

Captured calls above M384 now fail closed to ordinary AR+RMSNorm. Exact
mechanisms and validation are in the default-compatibility study and technical
contracts below.

## Historical boundaries

- The initial post-rebase Iris and explicit-triton campaigns were safe but
  slower than upstream-unfused.
- The restricted core-v3 campaign measured +1.29% throughput and cleared the
  capacity gate with eager prefill, C32-only decode capture, 0.90 HBM,
  disabled overlap, and passive health. That promotion is historical after
  restoring base defaults.
- Realigned profile v2 removed copy-out and the one-shot exit barrier but still
  failed performance promotion; it remains the lifetime-integration baseline.
- Pre-rebase profile v4 is legacy performance evidence. Its reserved-sink
  padding and persistent captured-output requirements remain safety invariants.
- The M=256 performance gate and mutable two-slot/no-exit input ring remain
  rejected.
- Legacy marker-free trace grouping is heuristic only.

Detailed historical percentages belong in their study summaries, not in this
live decision record.

## Next steps / restart points

1. Keep explicit fusion-off as deployment default, matched control, and
   fallback.
2. Treat producer-direct input/copy removal and progress publication as one
   systems project spanning dense GEMM, active MXFP4 MoE, graph ownership, and
   fallback.
3. Qualify WS=8 when all devices are available; independently requalify every
   additional TP=4 rank set.
4. Isolate overlap-scheduler fresh-server mode variance before using overlap
   for performance comparisons.
5. Do not repeat blocked-core cap/grid/block/XCD/fast-path sweeps.
6. Investigate mixed ordinary-Iris/RCCL captured transitions without weakening
   production fallback or graph isolation.
7. Keep `all_reduce_two` separate and retain its initial correctness incident.
8. Preserve graph-stable lifetime, sink padding, complete fallback, and
   explicit synchronization contracts in every candidate.
9. Run the separate 15-pair promotion stage only if the WS4 promising screen
   justifies its 16–22 hour commitment; do not reopen closed local kernel
   searches or transfer this result to WS2/8.

Technical contracts:
[backend design](backend-design-and-safety.md),
[producer lifetime](producer-lifetime-contract.md), and
[profiling workflow](profiling-workflow.md).
