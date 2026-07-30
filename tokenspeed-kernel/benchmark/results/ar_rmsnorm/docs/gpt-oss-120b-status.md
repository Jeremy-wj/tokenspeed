# GPT-OSS-120B fused AR+RMSNorm status

Updated: 2026-07-30

This is the sole live deployment decision and priority page for GPT-OSS-120B,
TP=4, on MI350X (gfx950).

## Deployment decision

- **Deploy with all-reduce fusion explicitly disabled.** Upstream auto-enables
  fusion on this topology, so the canonical control requires
  `--disable-allreduce-fusion`.
- The post-rebase upstream-unfused control is the current GPT-OSS-120B TP=4
  policy on MI350X.
- Iris `auto` completed 15/15 safe pairs but regressed median TPOT by **+2.55%**
  (95% CI +1.00% to +5.43%) and output throughput by **-2.29%** (95% CI
  -4.78% to -0.89%). It is rejected for GPT-OSS performance.
- Explicit `triton_shmem` completed 15/15 safe pairs but regressed median TPOT
  by **+10.47%** (95% CI +6.71% to +19.23%) and output throughput by
  **-10.58%** (95% CI -23.21% to -4.49%). It remains experimental and rejected.
- Profile `gpt-oss-120b-mi350x-qualified-v4` is legacy performance evidence.
  It qualified the old `triton_shmem` integration, not Iris or the rebased
  runtime.
- The graph-padding and captured-output lifetime faults remain closed
  historical incidents. Their invariants still apply to captured buffers.

Canonical post-rebase raw campaigns:

- `../raw/current/gpt-oss-120b/mi350x/2026-07-30/2026-07-30-post-rebase-iris-vs-unfused-v4/`
- `../raw/current/gpt-oss-120b/mi350x/2026-07-30/2026-07-30-post-rebase-triton-shmem-vs-unfused-v3/`

Curated study:
[`2026-07-post-rebase-baseline`](../studies/mi350x/2026-07-post-rebase-baseline/README.md).

See [upstream-main rebase impact](upstream-main-rebase-impact-2026-07.md) for
the backend analysis and baseline reset.

## Post-rebase qualified TP=4 profile

Scope:

- model: `/data/models/openai/gpt-oss-120b`, hidden size 2880 bf16 elements;
- hardware: AMD Instinct MI350X (gfx950), canonical HIP set `1,2,3,5`;
- code head: `eb69cf15`, upstream merge base `3f88dcc2`;
- profile ID: `gpt-oss-120b-mi350x-post-rebase-v1`;
- base image ID:
  `sha256:ad3ea3f8cae8ca38cf12824b15c606d0630118c6e04b4087e191b04619a6c135`;
- runtime corrections: repository-pinned Transformers/SMG/XGrammar, current
  source-tree PYTHONPATH, and `tokenspeed-scheduler` rebuilt from current source.

Required control:

```text
TS_ARNORM_BACKEND=auto
--disable-allreduce-fusion
--comm-fusion-max-num-tokens 2048
```

The server proof must contain both `ENABLE_FUSION=0` and resolved
`enable_allreduce_fusion=False`. A missing enable flag is not an unfused
control because upstream otherwise auto-enables fusion.

Lower-level findings:

- WS=2/4 correctness passed for Iris and `triton_shmem`; WS=8 is deferred while
  physical GPU 3 remains occupied.
- fixed M=32 graph p50: 33.12 us upstream-unfused, 46.06 us Iris fused,
  44.52 us `triton_shmem`;
- marker-aligned target-stage median: 4.779 ms upstream-unfused, 6.146 ms Iris,
  5.943 ms `triton_shmem`;
- `triton_shmem` retains an eager M<=256 screen advantage but crosses sharply
  at M=257 and loses under graph replay and end-to-end serving;
- synthetic mixed ordinary-Iris/RCCL captured transitions timed out; bounded
  real serving passed and the hazard remains open.

## Legacy qualified TP=4 profile

Scope:

- model: `/data/models/openai/gpt-oss-120b`, hidden size 2880 bf16 elements;
- hardware: AMD Instinct MI350X (gfx950), canonical HIP set `1,2,3,5`;
- runtime: torch 2.11 with released ROCm 7.2.4 userspace;
- shared-host rule: exclude physical GPU 3 / HIP index 0 for TP<8.

The pre-rebase profile-v4 reproduction requires:

```text
AR_NORM_PROFILE_ID=gpt-oss-120b-mi350x-qualified-v4
TS_TRITON_SHMEM_FOLD_COPYIN=0
TS_TRITON_SHMEM_DOUBLE_BUFFER_INPUT=0
TS_TRITON_SHMEM_OUTPUT_RING=72
TS_TRITON_SHMEM_BARRIER_GRID=0
TS_SERVE_ENGINE_MODULE=tokenspeed.runtime.entrypoints.safe_smg_server
TOKENSPEED_DEEP_HEALTH_MODE=passive
--gpu-memory-utilization 0.90
--disable-overlap-schedule
--disable-prefill-graph
--cudagraph-capture-sizes 32
```

TP=4 opt-in additionally requires:

```text
TS_ARNORM_BACKEND=triton_shmem
--enable-allreduce-fusion
--comm-fusion-max-num-tokens 2048
```

A valid fused run must show `enable_allreduce_fusion=True` in server arguments
and fused kernel signatures in every rank trace.

## Historical 2026-07-24 evidence

The favorable 2026-07-24 matched pair used an older serving profile. It is
historical mechanism evidence and was superseded first by profile v4 and then
by the upstream baseline reset; it must not be described as current.

- generic fused decode median: 34.1–36.5 µs across ranks;
- maximum same-rank unfused AR + RMSNorm median sum: 36.919 µs;
- GPU kernels per rank: 29,967 fused versus 31,333 unfused;
- profiled GPU-window span: about 313.8 ms fused versus 340.5 ms unfused
  (-7.8%);
- mean median TPOT: 12.560 ms fused versus 12.755 ms unfused (-1.5%).

Sources:
`../studies/mi350x/2026-07-profile-guided-followup/corrected_profile_comparison.json`,
`../studies/mi350x/2026-07-profile-guided-followup/e2e_summary.json`, and
`../raw/current/gpt-oss-120b/mi350x/2026-07-24/`.

## Legacy dispositions

- Profile-v4's `triton_shmem` performance rejection remains the final decision
  for the pre-rebase implementation only.
- Keep the M=256 performance gate rejected.
- Keep the host-alternated two-slot/no-exit input ring rejected until slot
  identity is graph-stable across captured variants and request waves.
- Preserve reserved-sink graph padding, persistent per-site output storage,
  explicit copy-in, the original one-shot exit barrier, passive health, eager
  prefill, and disabled overlap.
- Treat legacy trace grouping as heuristic; future traces must use
  `tokenspeed.model_forward.v1` markers.

## Priorities

1. Keep explicit fusion-off as the GPT-OSS-120B TP=4 deployment control.
2. Investigate the mixed ordinary-Iris/RCCL captured-transition timeout without
   weakening the production fallback or graph isolation.
3. Explain the N=2880 Iris fused cost relative to ordinary Iris plus RMSNorm.
   Require a clear graph critical-path win before another serving campaign.
4. Deprioritize local `triton_shmem` tuning. Its eager small-M win did not
   survive graph or serving qualification.
5. Keep `all_reduce_two` separate. One initial WS=4 two-ordinary control
   correctness event remains retained despite five clean reproductions.
6. Requalify WS=8 only after physical GPU 3 is idle; do not transfer TP=4
   thresholds or percentages.
7. Preserve graph-stable output lifetime, universal sink padding, complete
   fallback, and explicit completion/barrier contracts in every candidate.

Technical ownership:
[serving root cause](gpt-oss-120b-serving-root-cause.md),
[backend design](backend-design-and-safety.md),
[producer lifetime contract](producer-lifetime-contract.md), and
[profiling workflow](profiling-workflow.md).

