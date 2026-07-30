# AR+RMSNorm study index

Curated studies answer dated questions; they do not override the current
deployment decision in the
[GPT-OSS-120B status](../docs/gpt-oss-120b-status.md).

Except for the explicitly marked post-rebase study below, studies predate the
upstream-main rebase at `3f88dcc2` and are **legacy performance evidence**.
This includes profile v4. Safety incidents and lifetime findings remain
historical engineering evidence.

## MI350X (gfx950)

- [`2026-07-post-rebase-baseline/`](mi350x/2026-07-post-rebase-baseline/README.md)
  — canonical GPT-OSS-120B TP=4 upstream-unfused, Iris, and `triton_shmem`
  reset; current performance and deployment evidence.
- `2026-07-serving-baseline/` — initial TP=2, TP=4, and TP=8 serving crossover.
- `2026-07-grid-and-two-shot/` — grid and two-shot integration screens.
- `2026-07-path-and-width-sweeps/` — operator path and width sweeps.
- `2026-07-profile-guided-followup/` — corrected 2026-07-24 trace and e2e
  comparison; historical TP=4 evidence from the older serving profile.
- `2026-07-profiling-summary/` — Proton summaries.
- `2026-07-upper-bound/` — token-cap crossover and opportunity bounds.
- `2026-07-cap-gate/` — rejected M=256 performance-gate implementation.
- [`2026-07-repeatability/`](mi350x/2026-07-repeatability/README.md) —
  GPT-OSS-120B TP=4 incident chronology, root-cause evidence, and final
  profile-v4 qualification.

## MI300X

- `mi300x/migration-baseline/` — migration-era regression evidence.

Prefer machine-readable JSON summaries inside each study. Raw logs and traces
remain under `../raw/` and are intentionally Git-ignored.
