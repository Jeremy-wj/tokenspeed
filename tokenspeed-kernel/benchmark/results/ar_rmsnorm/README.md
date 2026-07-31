# AR+RMSNorm project index

This project integrates fused all-reduce + residual-add + RMSNorm from
triton-shmem into TokenSpeed and optimizes it against end-to-end model serving.

## Current checkpoint

GPT-OSS-120B TP=4 on MI350X has passed parity for the capacity objective:

- promote profile `gpt-oss-120b-mi350x-triton-core-v3` only on qualified HIP
  `1,2,5,6` (physical GPUs `0,2,4,6`);
- the 15-pair campaign measured **+1.29% output throughput** (95% CI +0.52% to
  +2.08%) and **-0.80% median TPOT** (95% CI -1.42% to -0.25%);
- retain explicit upstream-unfused with `--disable-allreduce-fusion` as the
  control and fallback for every unqualified rank set or profile mismatch;
- WS=8 remains unqualified.

The [GPT-OSS-120B status](docs/gpt-oss-120b-status.md) is the sole live
deployment and priority record. Earlier campaign decisions are dated evidence,
not current policy. All pre-rebase performance results are legacy; their safety
and incident findings remain valid engineering evidence.

## Evidence map

- [Study index](studies/README.md) — all curated studies and their status.
- [Core-v3 tuning](studies/mi350x/2026-07-triton-shmem-core-tuning/README.md) —
  current qualification and promotion evidence.
- [Backend decomposition](studies/mi350x/2026-07-triton-shmem-decomposition/README.md)
  — stage accounting and closed optimization paths.
- [Realignment](studies/mi350x/2026-07-triton-shmem-realignment/README.md) —
  profile-v2 lifetime and integration baseline.
- [Post-rebase baseline](studies/mi350x/2026-07-post-rebase-baseline/README.md) —
  upstream-unfused, Iris, and initial explicit-triton reset.
- [Repeatability and incident study](studies/mi350x/2026-07-repeatability/README.md)
  — pre-rebase graph-lifetime root causes and qualification.

## Durable references

- [Backend design and safety](docs/backend-design-and-safety.md)
- [Producer and buffer lifetime contract](docs/producer-lifetime-contract.md)
- [Benchmarking and promotion methodology](docs/benchmark-methodology-recommendations-2026-07.md)
- [Profiling and campaign workflow](docs/profiling-workflow.md)
- [Remaining integration roadmap](docs/integration-optimization-roadmap-2026-07.md)
- [Serving root-cause record](docs/gpt-oss-120b-serving-root-cause.md)
- [Upstream-main rebase record](docs/upstream-main-rebase-impact-2026-07.md)
- [ROCm 7.2 migration and incidents](docs/history/rocm-7.2-migration-and-incidents.md)

## Artifact policy

Study READMEs and compact JSON/CSV summaries are tracked. Raw logs, traces, and
campaign trees under either `raw/` or `studies/**/raw/` are local and ignored.
The CSV manifests under `manifests/` describe only the 2026-07-24 consolidation;
later campaigns keep generated provenance in their ignored raw roots.

