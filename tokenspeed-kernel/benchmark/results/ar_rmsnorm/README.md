# AR+RMSNorm project index

This directory contains the documentation and evidence for fused all-reduce +
residual-add + RMSNorm.

## Current outcome

The 2026-07-29 rebase onto upstream `main` at `3f88dcc2` changed the default
AMD fused and unfused communication backends. The 2026-07-30 GPT-OSS-120B
TP=4 campaign re-established the baseline: upstream-unfused with
`--disable-allreduce-fusion` is the supported policy. Iris fused regressed TPOT
by +2.55% and throughput by -2.29%; explicit `triton_shmem` regressed TPOT by
+10.47% and throughput by -10.58%. Both completed 15/15 safe pairs and both
failed performance promotion.

All pre-rebase results, including profile v4, remain legacy performance
evidence. The old safety and incident findings remain historical technical
evidence.

The sole live deployment decision and priority list is the
[GPT-OSS-120B status](docs/gpt-oss-120b-status.md). The durable technical
incident record is [GPT-OSS-120B serving root cause](docs/gpt-oss-120b-serving-root-cause.md).
The rebase, backend analysis, conflict log, and baseline-reset plan are in
[upstream-main rebase impact](docs/upstream-main-rebase-impact-2026-07.md).

## Evidence

- [Study index](studies/README.md) — dated curated evidence.
- [Repeatability study](studies/mi350x/2026-07-repeatability/README.md) —
  chronology, artifact index, and final dispositions.
- Final 2026-07-29 raw campaign:
  `raw/current/gpt-oss-120b/mi350x/2026-07-29/2026-07-29-output-ring-v4-fused-vs-unfused/`.
- Canonical post-rebase campaigns:
  `raw/current/gpt-oss-120b/mi350x/2026-07-30/2026-07-30-post-rebase-iris-vs-unfused-v4/`
  and
  `raw/current/gpt-oss-120b/mi350x/2026-07-30/2026-07-30-post-rebase-triton-shmem-vs-unfused-v3/`.
- [Post-rebase baseline study](studies/mi350x/2026-07-post-rebase-baseline/README.md).
- Large logs and traces under `raw/` are intentionally Git-ignored.

## Reference documents

- [Backend design and safety](docs/backend-design-and-safety.md)
- [Producer and buffer lifetime contract](docs/producer-lifetime-contract.md)
- [Profiling workflow](docs/profiling-workflow.md)
- [Benchmarking and promotion methodology](docs/benchmark-methodology-recommendations-2026-07.md)
- [MI350X upper bound](docs/mi350x-upper-bound.md)
- [Integration optimization roadmap](docs/integration-optimization-roadmap-2026-07.md)
- [Serving integration analysis](docs/serving-integration-analysis-2026-07.md)
- [ROCm 7.2 migration and incident history](docs/history/rocm-7.2-migration-and-incidents.md)
- [Upstream-main rebase and baseline reset](docs/upstream-main-rebase-impact-2026-07.md)

The CSV manifests under `manifests/` describe the 2026-07-24 consolidation, not
the later campaign inventory. Later raw campaigns retain their own generated
manifests, checksums, and summary provenance within their campaign roots.
The current quarantine inventory is
[manifests/deletion-review.md](manifests/deletion-review.md).

