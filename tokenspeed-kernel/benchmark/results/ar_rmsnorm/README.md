# AR+RMSNorm project index

This project integrates fused all-reduce + residual-add + RMSNorm from
triton-shmem into TokenSpeed and optimizes it against end-to-end model serving.

## Current checkpoints

Each model has one live decision page. Dated studies supply evidence but do not
override those pages.

### GPT-OSS-120B

The [GPT-OSS-120B status](docs/gpt-oss-120b-status.md) owns TP=4 policy on
MI350X. Core-v3 is safe on HIP `1,2,5,6`, but its default-compatible campaign
did not clear the latency or capacity promotion gates. Explicit
upstream-unfused remains the deployment default. A
[current-machine definitive campaign](studies/mi350x/2026-08-gpt-oss-120b-definitive-sweep/README.md)
is planned for WS=2/4/8; it has not been run and WS=8 remains unqualified.

### GLM-5.2-FP8

The [GLM-5.2-FP8 status](docs/glm-5.2-fp8-status.md) owns TP=8 policy on
MI350X. Captured 156-site graphs establish a diagnostic profile-v2 opportunity
at M=2-42, with ordinary fallback at M=1 and M>=43. This is operator evidence,
not a serving claim; explicit upstream-unfused remains the deployment default.
A [definitive cross-world-size sweep](studies/mi350x/2026-08-glm-5.2-fp8-definitive-sweep/README.md)
is specified but has not been run.

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
  — unrun current-machine WS=2/4/8 eager, graph, and serving contract.
- [GLM-5.2-FP8 baseline](studies/mi350x/2026-08-glm-5.2-fp8-baseline/README.md)
  — WS=8/N=6144 profile-v2 characterization and validation.
- [GLM-5.2-FP8 definitive sweep](studies/mi350x/2026-08-glm-5.2-fp8-definitive-sweep/README.md)
  — unrun WS=2/4/8 campaign contract and reporting layout.

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

Study READMEs and compact JSON/CSV summaries are tracked. Raw logs, traces, and
campaign trees under either `raw/` or `studies/**/raw/` are local and ignored.
The CSV manifests under `manifests/` describe only the 2026-07-24 consolidation;
later campaigns keep generated provenance in their ignored raw roots.

