# AR+RMSNorm study index

Studies are dated evidence. The
[GPT-OSS-120B status](../docs/gpt-oss-120b-status.md) is the only live
deployment and priority record.

## MI350X: current checkpoint chain

Read these newest-to-oldest:

1. [Default compatibility](mi350x/2026-07-default-compatibility/README.md) —
   restored base TokenSpeed defaults, final Perfetto traces, and the clean
   matched no-overlap non-promotion decision on HIP `1,2,5,6`.
2. [Core-v3 tuning](mi350x/2026-07-triton-shmem-core-tuning/README.md) —
   scratch-free padded decode core and historical restricted-configuration
   capacity promotion.
3. [Backend decomposition](mi350x/2026-07-triton-shmem-decomposition/README.md)
   — captured/eager stage accounting and closed optimization paths.
4. [Profile-v2 realignment](mi350x/2026-07-triton-shmem-realignment/README.md) —
   graph-stable input sites, borrowed two-shot outputs, and the lifetime
   baseline inherited by core-v3.
5. [Post-rebase baseline](mi350x/2026-07-post-rebase-baseline/README.md) —
   explicit upstream-unfused control and initial Iris/triton reset.

## MI350X: legacy performance and durable safety evidence

Everything below predates upstream `3f88dcc2`; numeric performance conclusions
are legacy. Safety, incident, and methodology findings remain engineering
evidence.

- [Repeatability and incidents](mi350x/2026-07-repeatability/README.md) —
  graph-padding and captured-output root causes, chronology, and profile-v4
  qualification.
- [Upper bound and token cap](mi350x/2026-07-upper-bound/README.md) —
  old-runtime crossover and planning arithmetic.
- [Rejected M=256 gate](mi350x/2026-07-cap-gate/README.md)
- [Profile-guided follow-up](mi350x/2026-07-profile-guided-followup/README.md) —
  corrected 2026-07-24 trace and end-to-end comparison.
- [Serving baseline](mi350x/2026-07-serving-baseline/README.md)
- [Grid and two-shot screens](mi350x/2026-07-grid-and-two-shot/README.md)
- [Path and width sweeps](mi350x/2026-07-path-and-width-sweeps/README.md)
- [Profiling summaries](mi350x/2026-07-profiling-summary/README.md)

## MI300X

- [Migration baseline](mi300x/migration-baseline/README.md) — migration-era regression
  evidence, legacy across hardware and runtime generations.

Prefer each study's machine-readable summary for exact values. Raw logs, traces,
and campaign trees under `raw/` are local and Git-ignored.

## MI350X: GLM-5.2-FP8

- [Initial WS=8 baseline](mi350x/2026-08-glm-5.2-fp8-baseline/README.md) —
  N=6144 operator characterization, 156-site profile-v1 validation, model
  bring-up incidents, and non-promotion serving screen.

The [GLM-5.2-FP8 status](../docs/glm-5.2-fp8-status.md) is the live decision
record for this model. GPT-OSS profile policy does not transfer across model,
world size, hidden width, site count, or serving configuration.
