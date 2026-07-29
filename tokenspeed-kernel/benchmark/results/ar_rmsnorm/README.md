# AR+RMSNorm project index

This directory contains the documentation and evidence for fused all-reduce +
residual-add + RMSNorm.

## Current outcome

GPT-OSS-120B, TP=4, on MI350X (gfx950) completed the profile-v4 qualification
campaign: three restart blocks and fifteen fused/unfused pairs passed without a
safety failure. Fusion remains opt-in because decode median TPOT regressed
1.44% and output throughput regressed 1.46%.

The sole live deployment decision and priority list is the
[GPT-OSS-120B status](docs/gpt-oss-120b-status.md). The durable technical
incident record is [GPT-OSS-120B serving root cause](docs/gpt-oss-120b-serving-root-cause.md).

## Evidence

- [Study index](studies/README.md) — dated curated evidence.
- [Repeatability study](studies/mi350x/2026-07-repeatability/README.md) —
  chronology, artifact index, and final dispositions.
- Final 2026-07-29 raw campaign:
  `raw/current/gpt-oss-120b/mi350x/2026-07-29/2026-07-29-output-ring-v4-fused-vs-unfused/`.
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

The CSV manifests under `manifests/` describe the 2026-07-24 consolidation, not
the later campaign inventory. Later raw campaigns retain their own generated
manifests, checksums, and summary provenance within their campaign roots.
The current quarantine inventory is
[manifests/deletion-review.md](manifests/deletion-review.md).

