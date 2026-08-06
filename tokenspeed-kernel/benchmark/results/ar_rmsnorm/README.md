# AR+RMSNorm project index

This project integrates fused all-reduce + residual-add + RMSNorm from
triton-shmem into TokenSpeed and optimizes it against end-to-end model serving.

## Final synthesis

The [final results and public handoff](docs/final-results-and-handoff-2026-08.md)
synthesizes the completed GLM-5.2-FP8 and GPT-OSS-120B campaigns, the four
headline figures, the engineering changes that produced the current backend,
and bounded restart points for future contributors.

No further development or benchmarking is planned in this branch.
GPT-OSS-120B's 15-pair promotion stage remains intentionally incomplete.
Explicit upstream-unfused remains the deployment default for both models; the
results identify promising model- and world-size-specific fusion opportunities,
not a universal promotion.

## Current checkpoints

Each model has one live decision page. Dated studies supply evidence but do not
override those pages.

### GPT-OSS-120B

The [GPT-OSS-120B status](docs/gpt-oss-120b-status.md) owns TP=4 policy on
MI350X. Core-v3 is safe on HIP `1,2,5,6`, but its default-compatible campaign
did not clear the latency or capacity promotion gates. Explicit
upstream-unfused remains the deployment default. A
[current-machine definitive campaign](studies/mi350x/2026-08-gpt-oss-120b-definitive-sweep/README.md)
completed its five-triplet screen: WS4 is promising at +1.27% throughput,
WS2 is inconclusive, and WS8 loses. Promotion was not run, so deployment policy
is unchanged.

### GLM-5.2-FP8

The [GLM-5.2-FP8 status](docs/glm-5.2-fp8-status.md) owns TP=8 policy on
MI350X. Captured 156-site graphs establish a diagnostic profile-v2 opportunity
at M=2-42, with ordinary fallback at M=1 and M>=43. This is operator evidence,
not a serving claim; explicit upstream-unfused remains the deployment default.
A [definitive cross-world-size sweep](studies/mi350x/2026-08-glm-5.2-fp8-definitive-sweep/README.md)
completed the predeclared matrix on 8x MI355X. Its model-faithful WS=8 rows
favor padded Triton through M42 by raw replay timing and through M40 after
reset-copy adjustment, then lose to RCCL at M43. The hardware change prevents
that result from overriding the MI350X profile or deployment decision.

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
- [GPT-OSS-120B definitive sweep](studies/mi350x/2026-08-gpt-oss-120b-definitive-sweep/README.md)
  — complete current-machine WS=2/4/8 eager, graph, transition, marker, and
  five-triplet serving screen.
- [GLM-5.2-FP8 baseline](studies/mi350x/2026-08-glm-5.2-fp8-baseline/README.md)
  — WS=8/N=6144 profile-v2 characterization and validation.
- [GLM-5.2-FP8 definitive sweep](studies/mi350x/2026-08-glm-5.2-fp8-definitive-sweep/README.md)
  — completed MI355X execution of the predeclared WS=2/4/8 operator campaign.

## Durable references

- [Backend design and safety](docs/backend-design-and-safety.md)
- [GPT-OSS producer and buffer lifetime contract](docs/producer-lifetime-contract.md)
- [Benchmarking and promotion methodology](docs/benchmark-methodology-recommendations-2026-07.md)
- [Profiling and campaign workflow](docs/profiling-workflow.md)
- [Remaining integration roadmap](docs/integration-optimization-roadmap-2026-07.md)
- [GPT-OSS serving root-cause record](docs/gpt-oss-120b-serving-root-cause.md)
- [GPT-OSS upstream-main rebase record](docs/upstream-main-rebase-impact-2026-07.md)
- [ROCm 7.2 migration and incidents](docs/history/rocm-7.2-migration-and-incidents.md)
- [GLM-5.2-FP8 status](docs/glm-5.2-fp8-status.md)

## Artifact policy

Study READMEs, synthesis figures, and compact JSON/CSV summaries are tracked.
Raw logs, traces, and campaign trees under either `raw/` or `studies/**/raw/`
are local and ignored. The CSV manifests under `manifests/` describe only the
2026-07-24 consolidation; later campaigns keep generated provenance in their
ignored raw roots.

