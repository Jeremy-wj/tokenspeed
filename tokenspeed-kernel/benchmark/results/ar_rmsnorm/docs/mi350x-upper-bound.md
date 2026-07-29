# MI350X fused AR+RMSNorm upper bound

Updated: 2026-07-29

## Decision-relevant evidence

For GPT-OSS-120B decode at TP=4 on MI350X (gfx950), keep measured serving
results separate from arithmetic ceilings:

- **Current qualified end-to-end result:** profile v4 passed the complete safety
  campaign, but fused median TPOT changed **+1.44%** and output throughput
  changed **-1.46%**. Both intervals exclude zero in the unfavorable
  direction, so the current implementation is not promoted.
- **Historical observation only:** the small 2026-07-24 old-profile pair
  improved unprofiled median TPOT by **0.195 ms / 1.53%** while throughput moved
  **+0.09%**. It is useful for transfer arithmetic, not current deployment
  evidence.
- **Current-kernel-family ceiling:** **0.76-1.02 ms**, or **5.0-6.7%** of the
  profiled decode period and **6.0-8.0%** of the historical unprofiled TPOT
  denominator if the same absolute saving transfers.
- **Aggressive systems ceiling:** **1.38-2.10 ms**, or **9.0-13.7%** profiled
  and **10.8-16.4%** unprofiled equivalent, if each eligible site reaches
  15-25 us through combined lifecycle, synchronization, and scratch changes.
- **Physical communication roofline:** **3.00 ms**, or **19.7%** profiled /
  **23.5%** unprofiled equivalent. This still pays the 2.4 us/site peer-link
  floor but assumes nearly all software overhead disappears.
- **Zero-cost Amdahl ceiling:** **3.219 ms**, or **21.1%** profiled / **25.2%**
  unprofiled equivalent. This is a sanity bound, not an engineering target.

Use **1.74 ms** as the center of the aggressive planning range: **11.4%** of
the profiled period and **13.6%** of the historical unprofiled denominator.
These percentages are projections from one absolute saving, not measured
profile-v4 effects. The current decision remains in
[GPT-OSS-120B status](gpt-oss-120b-status.md).

## Decode-step arithmetic

GPT-OSS-120B has 36 layers and two eligible AR+RMSNorm sites per layer. A
steady concurrency-32 decode forward therefore has 72 opportunities at
`M=32, N=2880`. These are model-depth operations, not per-request operations:
saving 10 us at each site removes about 720 us from every request's token step,
not 720 us divided by 32.

Each site is on an immediate dependency:

1. attention output must be reduced before residual-add and post-attention norm
   can feed MoE;
2. MoE output must be reduced before residual-add and the next layer's norm.

Within a site, communication can overlap local reduction and norm work. Across
layers, the next consumer dependency prevents free overlap.

The historical matched unfused trace measured:

- 15.263 ms max-rank median GPU replay period;
- 3.219 ms max-rank AR+RMSNorm time across 73 unfused pairs;
- 21.1% target-stage share and 78.9% non-target share.

The 3.219 ms budget is rank 2's median of 15 per-replay communication-kernel
sums, independently reproduced from
`../raw/current/gpt-oss-120b/mi350x/2026-07-24/traces/torch/tp4-unfused/`.
It is intentionally not `73 × pooled kernel median`; per-replay sums preserve
call-level tail behavior.

```text
step_gain =
  (unfused_AR_RMSNorm_budget - residual_fused_stage_cost)
  / unfused_decode_replay_period
```

Profile percentages use the trace-consistent 15.263 ms period. Historical
unprofiled equivalents divide the same absolute saving by 12.755 ms. Never
derive one by scaling the other percentage.

The integration fuses 72 sites while one pair remains unfused in the trace.
Keeping that remainder at its average unfused cost gives:

- 33.5 us/site, best isolated graph candidate but not serving-safe:
  **0.76 ms**, **5.0% profile / 6.0% unprofiled equivalent**;
- 30 us/site, aggressive current-family target:
  **1.02 ms**, **6.7% / 8.0%**;
- 25 us/site: **1.38 ms**, **9.0% / 10.8%**;
- 20 us/site: **1.74 ms**, **11.4% / 13.6%**;
- 15 us/site: **2.10 ms**, **13.7% / 16.4%**;
- 2.4 us/site, direct-link floor: **3.00 ms**, **19.7% / 23.5%**.

The old-profile reconstruction is useful only as a caution about transfer:
0.390 ms of target-stage reduction became 0.200 ms / 1.31% of replay period,
then 0.195 ms of unprofiled TPOT. Its site-count grouping is heuristic because
the retained traces lack authoritative forward markers. Neither its 51.2%
kernel-to-period ratio nor 97.5% period-to-TPOT ratio is a universal scaling
factor. Source:
`../studies/mi350x/2026-07-repeatability/segmented-trace-summary.json`.

## Residual cost and aggressive assumptions

The arbitrary-width fused path still:

- populates persistent symmetric input;
- signals peer readiness and pulls every peer's rows;
- writes and rereads fp32 scratch for blocked `N=2880`;
- materializes residual and normalized outputs;
- orders persistent-input reuse before the next producer;
- waits for completion before the next dependent model kernel.

That explains why historical same-rank medians were close: at most 36.519 us
fused versus 36.919 us for unfused all-reduce plus RMSNorm. The fused path
removes launches and one materialization boundary, not the collective or most
ordering cost.

The 15-25 us/site aggressive range requires several compatible changes:

1. producer-direct writes into graph-stable symmetric storage;
2. caller-owned or double-buffered lifetime semantics that safely move or
   remove the trailing reuse barrier;
3. a whole-row/cooperative reduction or other design without blocked fp32
   scratch round trips;
4. producer readiness publication and, where practical, direct handoff to the
   next consumer;
5. one necessary synchronization phase rather than two;
6. a persistent or graph-specialized launch path.

These changes are interdependent. Direct buffers need an ownership contract,
barrier removal needs overwrite-safe storage, and the next-sublayer dependency
limits overlap. That is why 15-25 us/site is plausible as an aggressive range
while 2.4 us/site is only a physical floor.

## Communication roofline

The roofline uses published MI350X limits: 8 TB/s peak HBM3E bandwidth, seven
direct Infinity Fabric links, and 153.6 GB/s aggregate bidirectional bandwidth
per link, treated as 76.8 GB/s in each direction. Let `S = 2*M*N` bytes for one
rank's bf16 activation.

### Small-M one-shot

Both Triton all-reduce and fused one-shot pull one `S`-byte payload from each
peer, so fusion does not reduce Infinity Fabric traffic:

```text
T_link >= S / 76.8 GB/s
```

At `M=32, N=2880`, `S=184,320` bytes and the floor is 2.4 us. Serving-faithful
operator/graph probes measure about 38-43 us, while historical trace medians are
34.1-36.5 us. Small-M is therefore limited by launch, barriers, peer-load
latency, and implementation rather than bandwidth.

The serving-faithful M=32 decomposition in
`../studies/mi350x/2026-07-upper-bound/decode-crossover-randomized.csv` reports:

- complete folded fused operation: 42.92 us;
- separately measured input copy: 13.54 us;
- subtraction-derived synchronization: 12.62 us;
- subtraction-derived reduction/residual/norm core: 21.40 us.

These phases overlap, so subtraction values are diagnostic and cannot be added
or subtracted as exact independent costs. In particular, 34 us is a separately
measured graph-candidate range, not `42.92 - 13.54`.

### Large-M two-shot

At TP=4, each fused two-shot rank owns one quarter of the rows. Per peer and
direction it transfers approximately:

```text
0.25*S input pull + 0.50*S norm/residual-output pushes = 0.75*S
```

Its floor is `0.75*S / 76.8 GB/s`. An efficient reduce-scatter/all-gather needs
about `0.50*S`, so fused two-shot has a structural 1.5x communication floor
before barriers, scratch, and copy-out.

At `M=2048, N=2880`, the floors are about 115.2 us fused versus 76.8 us for an
ideal collective. Measurements preserve that ordering: 195.67 us fused versus
143.89 us for production RCCL plus RMSNorm.

## Measured crossover and cap interpretation

The first tensor dimension is scheduled token count, not sequence position or
context length. The 2048 cap is also the largest default breakable-prefill
graph bucket; raising only fusion workspace would not extend graph policy.

Three TP=4 probes used bf16 `N=2880`, production transport selection, max-rank
p50 timing, two order-opposed 50-warmup/200-repeat sweeps, and one randomized
crossover sweep. The sweeps agree:

- `M=128`: 43.78 us fused vs 65.45 us unfused (**33.1% faster**);
- `M=256`: 61.08 us vs 66.84 us (**8.6% faster**);
- `M=512`: 87.03 us vs 68.68 us (**26.8% slower**);
- `M=1024`: 110.13 us vs 89.00 us (**23.7% slower**);
- `M=2048`: 195.67 us vs 143.89 us (**36.0% slower**);
- `M=4096`: 369.18 us vs 257.65 us (**43.3% slower**);
- `M=8192`: 728.21 us vs 486.16 us (**49.8% slower**).

At `M>=1024`, pass-to-pass change spread is at most 2.3 percentage points and
only 0.4 points at M=2048, much smaller than the fused disadvantage.

Therefore:

- keep 2048 as a compatibility/workspace ceiling;
- treat the measured performance crossover as between M=256 and M=512;
- do not interpret 2048 as the best performance eligibility threshold;
- do not infer serving promotion from operator crossover.

The later M=256 gate trace proved dispatch mechanics—decode stayed fused,
prefill declined, and workspace remained 2048—but did not establish an e2e win.
Its current disposition is owned by
[GPT-OSS-120B status](gpt-oss-120b-status.md). Sources:

- `../studies/mi350x/2026-07-upper-bound/token-cap-pass1-ascending.csv`
- `../studies/mi350x/2026-07-upper-bound/token-cap-pass2-descending.csv`
- `../studies/mi350x/2026-07-upper-bound/decode-crossover-randomized.csv`
- `../studies/mi350x/2026-07-cap-gate/`

## Generalization

The formulas generalize; the numeric bound does not. For each model and
topology, measure hidden size, eligible sites, layer count, active-token
distribution, TP size, unfused communication budget, and the one-/two-shot
crossover:

```text
absolute_zero_cost_ceiling =
  unfused_AR_RMSNorm_budget / baseline_step_us

realistic_optimized_ceiling =
  (unfused_AR_RMSNorm_budget
   - eligible_calls_per_step * achievable_fused_site_us
   - unfused_remainder_us)
  / baseline_step_us
```

Benefit tends to grow with more eligible sites, a larger communication share,
small-M one-shot execution, scratch-free widths, and safe producer-owned input.
It shrinks with large-M two-shot execution, wider world sizes, efficient RCCL,
M-divergent execution, blocked scratch traffic, or a larger noncommunication
share.

Estimate `achievable_fused_site_us` from peer traffic plus defensible launch,
synchronization, and core floors. Do not inherit GPT-OSS-120B's 15-25 us range,
historical 0.195 ms result, or M<=256 crossover for another model.

## Sources

- [Current GPT-OSS-120B status](gpt-oss-120b-status.md)
- [Integration roadmap](integration-optimization-roadmap-2026-07.md)
- Historical trace summary:
  `../studies/mi350x/2026-07-profile-guided-followup/corrected_profile_comparison.json`
- Historical e2e summary:
  `../studies/mi350x/2026-07-profile-guided-followup/e2e_summary.json`
- Qualified profile-v4 campaign:
  `../studies/mi350x/2026-07-repeatability/repeatability-summary.json`
- AMD MI350X specifications:
  <https://www.amd.com/en/products/accelerators/instinct/mi350/mi350x.html> and
  <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-mi350x-platform-brochure.pdf>
