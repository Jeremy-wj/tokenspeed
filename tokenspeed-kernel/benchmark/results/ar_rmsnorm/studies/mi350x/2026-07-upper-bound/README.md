# MI350X upper-bound and token-cap probes

Date: 2026-07-24

Purpose: validate the TP=4/N=2880 fusion crossover and determine whether the
2048-token fusion ceiling should be raised.

This is historical 2026-07-24 operator and old-profile evidence, not the
qualified profile-v4 end-to-end result.

Environment:

- container: `jeremwan-tokenspeed-profiler`
- image: `jeremwan/tokenspeed:rocm7.2.4-torch2.11-profiler`
- devices: `HIP_VISIBLE_DEVICES=1,2,3,5`
- world size: 4
- dtype: bf16
- hidden size: 2880
- warmups/repeats: 50/200 per timed arm
- metric: max-rank p50 HIP-event latency
- unfused baseline: production transport gate followed by residual RMSNorm

Artifacts:

- `token-cap-pass1-ascending.csv` — M=128 through 8192 in ascending order.
- `token-cap-pass2-descending.csv` — the same range in descending order.
- `decode-crossover-randomized.csv` — M={8,32,64,128,256,512} in a
  non-monotonic order.
- `upper-bound-summary.json` — machine-readable trace, e2e, bound, and cap
  conclusions.

Representative two-pass means:

- M=128: 43.78 us fused versus 65.45 us unfused.
- M=256: 61.08 us fused versus 66.84 us unfused.
- M=512: 87.03 us fused versus 68.68 us unfused.
- M=2048: 195.67 us fused versus 143.89 us unfused.
- M=8192: 728.21 us fused versus 486.16 us unfused.

Conclusion: the crossover is between 256 and 512 tokens. Raising 2048 is not
useful; the fused disadvantage increases above the cap. See
`../../../docs/mi350x-upper-bound.md` for the roofline and reusable e2e bounds.
The live deployment decision remains in
`../../../docs/gpt-oss-120b-status.md`.

The 2026-07-26 audit keeps all absolute calculations but corrects their labels:
the 5.0-6.7%, 9.0-13.7%, and 19.7% figures use the 15.263 ms profiled decode
period. Their same-absolute-saving equivalents against 12.755 ms unprofiled
TPOT are 6.0-8.0%, 10.8-16.4%, and 23.5%. The zero-cost sanity bound is 3.219
ms, or 21.1% profiled / 25.2% unprofiled.

The separate 256 performance-gate follow-up under `../2026-07-cap-gate/`
proved dispatch behavior but was later rejected by repeatability testing.
