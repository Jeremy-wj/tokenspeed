# MI350X serving baseline — legacy

These CSV and text artifacts are pre-rebase GPT-OSS-120B serving screens for
the local `triton_shmem` implementation. They include early TP=2/4/8 crossover,
one-shot-overlay, and fused/unfused observations.

They are not a baseline for upstream `main` at `3f88dcc2`: the default fused
AR+RMSNorm backend and ordinary AMD all-reduce both changed. Preserve the files
as historical mechanism evidence and do not infer a current auto-enable policy
from them.

See
[upstream-main rebase impact](../../../docs/upstream-main-rebase-impact-2026-07.md).
