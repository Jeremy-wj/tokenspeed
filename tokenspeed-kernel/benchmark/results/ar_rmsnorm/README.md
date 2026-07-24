# AR+RMSNorm project index

This directory is the canonical home for fused all-reduce + residual-add +
RMSNorm project documentation, curated studies, manifests, and local raw
artifacts.

## Current state

- Backend: AMD `triton_shmem` over PyTorch symmetric memory and coarse HIP-IPC
  data buffers.
- Validated hardware: MI300X and MI350X.
- Current model campaign: gpt-oss-120B, hidden size 2880.
- Deployment: TP=2 auto-enables fusion; TP=4/8 remain explicit opt-in.
- Correct TP=4 opt-in requires `TS_ARNORM_BACKEND=triton_shmem`,
  `--enable-allreduce-fusion`, and a positive fusion token cap.
- Corrected TP=4 traces show generic fused decode near the unfused AR + RMSNorm
  median sum, 1,366 fewer GPU kernels per rank, and a 7.8% shorter profiled GPU
  window.
- Width-specific tile policies remain diagnostic only. A broad small-M policy
  faulted during a real graph transition and was removed.

Current conclusions and next steps:
[docs/gpt-oss-120b-status.md](docs/gpt-oss-120b-status.md).

## Documentation

- [gpt-oss-120B status](docs/gpt-oss-120b-status.md) — current evidence,
  deployment decision, and next work.
- [Backend design and safety](docs/backend-design-and-safety.md) — dispatch,
  pointer translation, memory substrate, synchronization, and invariants.
- [Profiling workflow](docs/profiling-workflow.md) — environment, matched A/B
  proof requirements, commands, analyzers, and artifact naming.
- [ROCm 7.2 migration and incidents](docs/history/rocm-7.2-migration-and-incidents.md)
  — solved environment, RCCL, and barrier investigations.

## Evidence layout

- `studies/mi350x/2026-07-serving-baseline/` — qualification crossover,
  decomposition, and e2e summaries.
- `studies/mi350x/2026-07-grid-and-two-shot/` — selective grid cap and
  two-shot integration experiments.
- `studies/mi350x/2026-07-path-and-width-sweeps/` — width/path sweeps.
- `studies/mi350x/2026-07-profile-guided-followup/` — corrected trace analysis,
  matched e2e, and rejected small-M candidates.
- `studies/mi350x/2026-07-profiling-summary/` — Proton summaries.
- `studies/mi300x/migration-baseline/` — migration-era MI300X evidence.

Large logs and traces live under `raw/` and are intentionally Git-ignored.
Their checksums, provenance, and dispositions are tracked in
`manifests/artifacts-pre-migration.csv` and `manifests/migration-map.csv`.
Review candidates are listed in `manifests/deletion-review.md`.

## Current visual traces

Current matched TP=4 torch traces are under:

```text
raw/current/gpt-oss-120b/mi350x/2026-07-24/traces/torch/
  tp4-fused-generic-block512/
  tp4-unfused/
```

Filenames identify the arm and rank; no artifact is named `final` or `latest`.

