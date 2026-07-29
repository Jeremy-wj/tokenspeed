# GPT-OSS-120B fused AR+RMSNorm status

Updated: 2026-07-29

This is the sole live deployment decision and priority page for GPT-OSS-120B,
TP=4, on MI350X (gfx950).

## Deployment decision

- TP=2 continues to auto-enable fusion.
- TP=4 and TP=8 remain explicit opt-in.
- The qualified TP=4 profile
  `gpt-oss-120b-mi350x-qualified-v4` passed safety qualification but is not
  promoted: fused decode median TPOT regressed 1.44% (95% CI +1.20% to +1.65%)
  and output throughput regressed 1.46% (95% CI -1.91% to -1.02%).
- The base graph-padding fault and the fusion-specific captured-output lifetime
  fault are closed. This is a performance rejection, not an unresolved
  stability workaround.

The final campaign completed three independent restart blocks and fifteen
fused/unfused pairs without a safety failure. Raw campaign:
`../raw/current/gpt-oss-120b/mi350x/2026-07-29/2026-07-29-output-ring-v4-fused-vs-unfused/`.

## Qualified TP=4 profile

Scope:

- model: `/data/models/openai/gpt-oss-120b`, hidden size 2880 bf16 elements;
- hardware: AMD Instinct MI350X (gfx950), canonical HIP set `1,2,3,5`;
- runtime: torch 2.11 with released ROCm 7.2.4 userspace;
- shared-host rule: exclude physical GPU 3 / HIP index 0 for TP<8.

Profile v4 requires:

```text
AR_NORM_PROFILE_ID=gpt-oss-120b-mi350x-qualified-v4
TS_TRITON_SHMEM_FOLD_COPYIN=0
TS_TRITON_SHMEM_DOUBLE_BUFFER_INPUT=0
TS_TRITON_SHMEM_OUTPUT_RING=72
TS_TRITON_SHMEM_BARRIER_GRID=0
TS_SERVE_ENGINE_MODULE=tokenspeed.runtime.entrypoints.safe_smg_server
TOKENSPEED_DEEP_HEALTH_MODE=passive
--gpu-memory-utilization 0.90
--disable-overlap-schedule
--disable-prefill-graph
--cudagraph-capture-sizes 32
```

TP=4 opt-in additionally requires:

```text
TS_ARNORM_BACKEND=triton_shmem
--enable-allreduce-fusion
--comm-fusion-max-num-tokens 2048
```

A valid fused run must show `enable_allreduce_fusion=True` in server arguments
and fused kernel signatures in every rank trace.

## Historical 2026-07-24 evidence

The favorable 2026-07-24 matched pair used an older serving profile. It is
historical mechanism evidence and is deployment-superseded by profile v4; it
must not be described as the current TP=4 result.

- generic fused decode median: 34.1–36.5 µs across ranks;
- maximum same-rank unfused AR + RMSNorm median sum: 36.919 µs;
- GPU kernels per rank: 29,967 fused versus 31,333 unfused;
- profiled GPU-window span: about 313.8 ms fused versus 340.5 ms unfused
  (-7.8%);
- mean median TPOT: 12.560 ms fused versus 12.755 ms unfused (-1.5%).

Sources:
`../studies/mi350x/2026-07-profile-guided-followup/corrected_profile_comparison.json`,
`../studies/mi350x/2026-07-profile-guided-followup/e2e_summary.json`, and
`../raw/current/gpt-oss-120b/mi350x/2026-07-24/`.

## Current dispositions

- Keep the profile-v4 fused implementation opt-in; do not rerun the same
  candidate for promotion.
- Keep the M=256 performance gate rejected.
- Keep the host-alternated two-slot/no-exit input ring rejected until slot
  identity is graph-stable across captured variants and request waves.
- Preserve reserved-sink graph padding, persistent per-site output storage,
  explicit copy-in, the original one-shot exit barrier, passive health, eager
  prefill, and disabled overlap.
- Treat legacy trace grouping as heuristic; future traces must use
  `tokenspeed.model_forward.v1` markers.

## Priorities

1. Require a changed performance mechanism before another promotion campaign;
   target copy/ownership overhead and prefill dispatch without weakening the
   proven lifetime contract.
2. Use the repeatability runner's three-block/fifteen-pair minimum, fresh
   servers, GPU isolation, signature proof, checksums, and failure gates.
3. Replace host-phase input-ring reuse with explicit graph/call-site slot
   identity or a device-side epoch before reconsidering exit-barrier removal.
4. Defer producer-direct output until input reuse is transition-safe.
5. Keep prefill and decode separate and retain model-specific op, graph,
   transition, and end-to-end safety gates.

Technical ownership:
[serving root cause](gpt-oss-120b-serving-root-cause.md),
[backend design](backend-design-and-safety.md),
[producer lifetime contract](producer-lifetime-contract.md), and
[profiling workflow](profiling-workflow.md).

