# MI350X grid and two-shot study — legacy

These CSVs record pre-rebase `triton_shmem` grid, one-shot, two-shot, cap, and
barrier experiments. Their timings and dispatch conclusions apply only to the
old local backend and runtime.

Upstream `main` at `3f88dcc2` uses Iris by default for fused AR+RMSNorm and
ordinary AMD all-reduce. New grid or transport recommendations require a fresh
operator and graph matrix. The raw CSVs remain immutable historical evidence.

See
[upstream-main rebase impact](../../../docs/upstream-main-rebase-impact-2026-07.md).
