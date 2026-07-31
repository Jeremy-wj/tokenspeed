# GPT-OSS-120B TP=4 repeatability on MI350X (gfx950)

Campaign period: 2026-07-27 through 2026-07-29.

The reproducible harness is `benchmark/run_ar_rmsnorm_repeatability.py`. This
page is a chronology and artifact index; technical root-cause detail belongs in
the [serving incident record](../../../docs/gpt-oss-120b-serving-root-cause.md)
and the JSON summaries below. The live deployment decision is the
[GPT-OSS-120B status](../../../docs/gpt-oss-120b-status.md).

This directory is pre-rebase history. The post-rebase reset is in
[`2026-07-post-rebase-baseline`](../2026-07-post-rebase-baseline/README.md);
it is baseline evidence, not the live deployment decision.

## Chronology

- **2026-07-27:** The first repeatability root was contaminated by a foreign GPU
  process and is retained only as an incident artifact. It must not be used for
  performance conclusions. The runner gained per-server preflight, GPU
  isolation, hard timeouts, and orphan cleanup.
- **2026-07-27:** Passive health, disabled overlap, and the original exit
  barrier formed the stable control. The M=256 gate and host-alternated
  two-slot/no-exit input ring were rejected by later serving failures.
- **2026-07-28:** Fresh unfused failures showed that earlier C32/0.90 passes
  were intermittent. KV capacity was confirmed as an incidence amplifier, not
  the cause.
- **2026-07-28:** Reserved-sink graph padding closed the base fault: C32 dummy
  rows had aliased mutable request-pool slot 0.
- **2026-07-29:** Persistent output storage for all 72 fused call sites closed
  the remaining captured-output lifetime fault and defined qualified profile
  `gpt-oss-120b-mi350x-qualified-v4`.
- **2026-07-29:** The final campaign completed three restart blocks and fifteen
  fused/unfused pairs without a safety failure. Promotion failed on performance:
  decode median TPOT was +1.44% and output throughput was -1.46%.

Final raw campaign:
`../../../raw/current/gpt-oss-120b/mi350x/2026-07-29/2026-07-29-output-ring-v4-fused-vs-unfused/`.

## Artifact index

Final decisions and incident roots:

- `repeatability-summary.json` — campaign history and final dispositions.
- `e2e-stability-resolution-summary.json` — stability evidence and
  profile-v4 supersession.
- `graph-padding-sentinel-root-cause.json` — base serving root cause.
- `fused-output-lifetime-root-cause.json` — fusion-specific root cause and
  final campaign pointer.
- `kv-cache-headroom-root-cause.json` — confirmed incidence amplifier.
- `hip-graph-runtime-root-cause.json` — retained superseded diagnosis.

Historical controls and transitions:

- `serving-stability-summary.json`
- `residual-stability-summary.json`
- `fused-prefill-transition-resolution.json`
- `input-ring-summary.json`
- `input-ring-pass1-baseline.json`
- `input-ring-pass1-double.json`
- `input-ring-pass2-baseline.json`
- `input-ring-pass2-double.json`

Trace interpretation:

- `segmented-trace-summary.json` — heuristic legacy segmentation.
- `stable-controls-forward-marker-summary.json`
- `forward-marker-smoke-summary.json`

## Final dispositions

- **Profile v4 safety:** qualified.
- **TP=4 promotion:** rejected; fusion remains opt-in.
- **M=256 performance gate:** rejected.
- **Host-alternated input ring:** rejected pending graph-stable slot identity.
- **Reserved-sink padding and persistent per-site outputs:** required.
- **Passive health, disabled overlap, eager prefill, explicit copy-in, and the
  original exit barrier:** retained.
- **2026-07-27 contaminated root:** incident evidence only.
