# MI350X path and width sweeps — legacy

These CSVs contain pre-rebase `triton_shmem` path, width, world-size, and
workspace-cap screens. They remain useful for reconstructing old dispatch
choices, not for selecting a post-rebase backend or threshold.

The upstream `3f88dcc2` baseline changed both the fused default and ordinary
AMD all-reduce. Repeat the relevant shapes against upstream-unfused, Iris
`auto`, and explicit `triton_shmem` before making a new recommendation.

See
[upstream-main rebase impact](../../../docs/upstream-main-rebase-impact-2026-07.md).
