# AR+RMSNorm project index

This project integrates fused all-reduce + residual-add + RMSNorm from
triton-shmem into TokenSpeed and optimizes it against end-to-end model serving.

## Current checkpoint

GPT-OSS-120B TP=4 on MI350X is safe with base TokenSpeed defaults, but the
default-compatible profile is not performance-promoted:

- the final clean 15-pair no-overlap campaign measured +0.45% throughput and
  -0.51% median TPOT; it did not clear capacity or latency promotion;
- base overlap scheduling remains safe but is excluded from performance
  qualification because fresh servers occupy distinct performance modes;
- retain explicit upstream-unfused with `--disable-allreduce-fusion` as
  deployment default, control, and fallback;
- triton-shmem remains safety-qualified only on HIP `1,2,5,6` (physical GPUs
  `0,2,4,6`);
- WS=8 remains unqualified.

GLM-5.2-FP8 TP=8 has a validated operator-level candidate but no promotable
serving baseline:

- profile v1 improves the synthetic 156-call M32 graph by 9.61%;
- its shared-state 1000-replay transition matrix passes on all eight MI350X;
- the current AMD FP8 model stack measures only 0.212 output tokens/s in the
  bounded concurrency-16 control;
- retain explicit upstream-unfused until the model baseline, captured serving,
  and full paired campaign are qualified.

The [GPT-OSS-120B status](docs/gpt-oss-120b-status.md) is the sole live
deployment and priority record. Earlier campaign decisions are dated evidence,
not current policy. All pre-rebase performance results are legacy; their safety
and incident findings remain valid engineering evidence.

## Evidence map

- [Study index](studies/README.md) — all curated studies and their status.
- [Default compatibility](studies/mi350x/2026-07-default-compatibility/README.md)
  — restored base defaults, safety evidence, and current 15-pair decision.
- [Core-v3 tuning](studies/mi350x/2026-07-triton-shmem-core-tuning/README.md) —
  historical restricted-configuration promotion evidence.
- [Backend decomposition](studies/mi350x/2026-07-triton-shmem-decomposition/README.md)
  — stage accounting and closed optimization paths.
- [Realignment](studies/mi350x/2026-07-triton-shmem-realignment/README.md) —
  profile-v2 lifetime and integration baseline.
- [Post-rebase baseline](studies/mi350x/2026-07-post-rebase-baseline/README.md) —
  upstream-unfused, Iris, and initial explicit-triton reset.
- [Repeatability and incident study](studies/mi350x/2026-07-repeatability/README.md)
  — pre-rebase graph-lifetime root causes and qualification.
- [GLM-5.2-FP8 baseline](studies/mi350x/2026-08-glm-5.2-fp8-baseline/README.md)
  — WS=8/N=6144 characterization, bring-up incidents, and profile-v1 screen.

## Durable references

- [Backend design and safety](docs/backend-design-and-safety.md)
- [Producer and buffer lifetime contract](docs/producer-lifetime-contract.md)
- [Benchmarking and promotion methodology](docs/benchmark-methodology-recommendations-2026-07.md)
- [Profiling and campaign workflow](docs/profiling-workflow.md)
- [Remaining integration roadmap](docs/integration-optimization-roadmap-2026-07.md)
- [Serving root-cause record](docs/gpt-oss-120b-serving-root-cause.md)
- [Upstream-main rebase record](docs/upstream-main-rebase-impact-2026-07.md)
- [ROCm 7.2 migration and incidents](docs/history/rocm-7.2-migration-and-incidents.md)
- [GLM-5.2-FP8 status](docs/glm-5.2-fp8-status.md)

## Artifact policy

Study READMEs and compact JSON/CSV summaries are tracked. Raw logs, traces, and
campaign trees under either `raw/` or `studies/**/raw/` are local and ignored.
The CSV manifests under `manifests/` describe only the 2026-07-24 consolidation;
later campaigns keep generated provenance in their ignored raw roots.

