# GPT-OSS-120B definitive AR+RMSNorm campaign

Status: **complete through the five-triplet extension; promotion not run**

This study predeclares the current-machine WS=2/4/8 eager, captured-graph, and
end-to-end comparison. It does not change the
[live deployment decision](../../../docs/gpt-oss-120b-status.md).

The immutable matrix and stopping rules are in [campaign.json](campaign.json).
That file intentionally remains `status=planned` because the runners require the
predeclared specification to remain unchanged; [summary.json](summary.json) owns
the result.

## Result

### Executive conclusion

The retained data are internally consistent with the raw runs; the strange
values in [end-to-end-summary.json](end-to-end-summary.json) are not JSON or
aggregation corruption. They are recorded run-level stalls. They should not,
however, be averaged without qualification:

- **WS4 Triton is the only stable positive result.** It improved output
  throughput and median TPOT in all five paired blocks. The mean changes are
  +1.27% throughput and -1.30% TPOT. This clears the predeclared five-pair
  directional screen under both the reported bootstrap and a t-interval
  sensitivity check, but it is not the required 15-pair promotion evidence.
- **WS2 Triton is genuinely inconclusive.** Its +1.10% mean throughput is
  produced by two large positive blocks; the block median is -0.46%, only two
  of five blocks improve throughput, and only two of five improve TPOT.
- **WS8 Triton is directionally worse, but the reported -2.48% throughput
  magnitude is outlier-driven.** Four of five blocks lose throughput and four
  of five worsen TPOT, so it still fails the screen. Excluding the block-0
  cold-wave observation changes mean throughput from -2.48% to -0.33%.
- **Default Iris is not an alternative promotion candidate.** It is mixed at
  WS2, regresses at WS4, and loses throughput and TPOT in every WS8 block.
- **No deployment change is justified.** Explicit upstream-unfused remains the
  live default. A WS4 promotion campaign would require ten additional,
  correctly randomized pairs to reach the predeclared total of 15.

This is conclusive as a current-machine five-pair **screen**: WS2 does not
separate, WS4 is the only candidate worth promotion testing, and WS8 should not
be promoted. It is not conclusive evidence of a production speedup.

### Data integrity and scope

The full 162-process captured-graph matrix, both eager passes, all three
1,000-replay transition matrices, and all 45 fresh-server lifecycles completed.
The serving runs contain 5,760/5,760 completed requests, zero failures, and
2,949,120/2,949,120 requested output tokens. Every request produced exactly 512
tokens.

Independent reconstruction found:

- all 150 retained block-level paired metric values and their 30 reported means
  reproduce from the 45 raw result files within floating-point precision;
- all 30 retained contrast-statistic objects match the per-WS raw campaign
  summaries;
- all 45 arm summaries are complete, 600 retained checksum entries verify, and
  1,149 GPU-process guard samples report no unexpected active process;
- the initial RCCL timeout was a fixed probe teardown-lifetime bug, not a
  retained measurement failure: captured collective graphs must be released
  before destroying `ProcessGroupNCCL`;
- captured-decline preflight now follows the same ordinary fallback path as
  capture.

The statistical unit for serving is a fresh-server paired block, not an
individual request. The 128 requests within one lifecycle are correlated waves
and must not be treated as 128 independent replicates.

### End-to-end Triton result

Percentage changes below are paired against the upstream-unfused observation
in the same restart block. Positive throughput is better; negative TPOT is
better.

| WS | Throughput mean (reported 95% CI) | Block median | Better blocks | Median TPOT mean (reported 95% CI) | Better blocks | Screen |
|---:|---:|---:|---:|---:|---:|:---|
| 2 | +1.10% (-0.89%, +3.11%) | -0.46% | 2/5 | -0.48% (-2.03%, +1.07%) | 2/5 | inconclusive |
| 4 | +1.27% (+0.49%, +2.01%) | +1.75% | 5/5 | -1.30% (-1.91%, -0.63%) | 5/5 | promising |
| 8 | -2.48% (-6.93%, +0.22%) | -0.93% | 1/5 | +0.63% (+0.03%, +1.25%) | 1/5 | loss |

The complete block-level paired data show what the means hide:

| WS | Block | Actual arm order | Iris throughput | Iris median TPOT | Triton throughput | Triton median TPOT |
|---:|---:|:---|---:|---:|---:|---:|
| 2 | 0 | Iris, upstream, Triton | -9.76% | +0.04% | -0.46% | +1.03% |
| 2 | 1 | Iris, upstream, Triton | +6.58% | -4.89% | +4.22% | -2.65% |
| 2 | 2 | upstream, Iris, Triton | -1.05% | -0.69% | -0.85% | +0.82% |
| 2 | 3 | Iris, upstream, Triton | -0.25% | +0.18% | -1.16% | +1.24% |
| 2 | 4 | Iris, upstream, Triton | +5.83% | -4.77% | +3.72% | -2.84% |
| 4 | 0 | Triton, Iris, upstream | -0.54% | +0.79% | +1.75% | -1.33% |
| 4 | 1 | upstream, Triton, Iris | +0.30% | -0.19% | +1.76% | -2.34% |
| 4 | 2 | Triton, Iris, upstream | -1.68% | +1.39% | +0.02% | -0.16% |
| 4 | 3 | Triton, Iris, upstream | +0.46% | +0.12% | +2.39% | -1.44% |
| 4 | 4 | upstream, Triton, Iris | -2.16% | +2.19% | +0.45% | -1.20% |
| 8 | 0 | Iris, Triton, upstream | -14.81% | +2.18% | -11.08% | +0.34% |
| 8 | 1 | Triton, upstream, Iris | -1.28% | +1.33% | -0.35% | -0.33% |
| 8 | 2 | upstream, Iris, Triton | -0.52% | +1.82% | +1.03% | +0.22% |
| 8 | 3 | Iris, Triton, upstream | -1.37% | +1.79% | -1.06% | +1.55% |
| 8 | 4 | Triton, upstream, Iris | -2.37% | +2.45% | -0.93% | +1.38% |

WS2 has two run regimes. Blocks 1 and 4 use unusually slow upstream baselines
(2,024.44 and 2,049.25 output tok/s, with 15.146 and 15.111 ms median TPOT);
those are exactly the blocks that make both candidates look favorable. The
other upstream blocks are 2,126.57-2,163.70 tok/s with 14.409-14.586 ms median
TPOT. This is runtime/order confounding, not a stable Triton speedup.

WS4 is qualitatively different: all five Triton throughput deltas are positive
(+0.02% to +2.39%), and all five TPOT deltas are negative (-0.16% to -2.34%).
Leave-one-block-out throughput means remain positive (+0.99% to +1.59%), though
one drops just below the predeclared +1% mean threshold.

WS8 has a consistent sign but unstable magnitude. Excluding block 0 gives
-0.33% mean Triton throughput and +0.71% median TPOT. Default Iris is more
clearly bad: excluding block 0 still gives -1.39% throughput and +1.85% median
TPOT.

### End-to-end outliers

The material anomalies are whole first request waves, not malformed scalar
fields:

| WS / arm / block | Output tok/s | Median TTFT | Mean TTFT | First 32-request-wave mean TTFT |
|:---|---:|---:|---:|---:|
| WS2 Iris b0 | 1,919.12 | 439.21 ms | 1,125.81 ms | 3,175.01 ms |
| WS2 upstream b0 | 2,126.57 | 196.46 ms | 292.62 ms | 216.54 ms |
| WS2 Triton b0 | 2,116.82 | 175.59 ms | 280.09 ms | 179.37 ms |
| WS8 Iris b0 | 1,856.40 | 222.95 ms | 1,448.40 ms | 3,942.91 ms |
| WS8 Triton b0 | 1,937.53 | 500.22 ms | 1,192.08 ms | 3,268.24 ms |
| WS8 upstream b0 | 2,179.07 | 199.91 ms | 306.61 ms | 239.56 ms |

Later waves and TPOT normalize. For example, WS8 Iris wave-mean TTFT falls from
3,942.91 ms to 1,535.20, 155.13, and 160.35 ms; WS8 Triton falls from
3,268.24 ms to 1,175.89, 162.19, and 162.01 ms. The timed warmup is only a
1-input/16-output request, not the measured 128/512 shape. Cold graph/JIT/cache
work is therefore the most plausible explanation, but that is inference:
neither logs nor process guards prove a unique cause.

These observations are valid records of the specified procedure and were not
deleted post hoc. Robust summaries are supplied to show their influence:

| WS | Triton throughput mean | Block median | Mean without b0 | Throughput leave-one-out range | TPOT leave-one-out range |
|---:|---:|---:|---:|---:|---:|
| 2 | +1.10% | -0.46% | +1.48% | +0.31% to +1.66% | -0.91% to +0.11% |
| 4 | +1.27% | +1.75% | +1.15% | +0.99% to +1.59% | -1.58% to -1.03% |
| 8 | -2.48% | -0.93% | -0.33% | -3.36% to -0.33% | +0.40% to +0.87% |

### Statistical limits and order confounding

Each block contains one seed, so the reported “paired hierarchical
restart-block bootstrap” reduces to an ordinary bootstrap of only five values.
Its percentile interval is useful as the predeclared screen, but it is fragile:
there is no lower-level replication within a block, and the endpoint Monte
Carlo error is visible. For example, exact enumeration gives a +0.45% WS4
throughput lower endpoint versus the reported +0.49%.

A conventional t(4) sensitivity interval is wider:

| WS / metric | Reported bootstrap 95% CI | t(4) sensitivity 95% CI |
|:---|:---|:---|
| WS2 throughput | -0.89% to +3.11% | -2.19% to +4.38% |
| WS2 median TPOT | -2.03% to +1.07% | -3.06% to +2.10% |
| WS4 throughput | +0.49% to +2.01% | +0.04% to +2.51% |
| WS4 median TPOT | -1.91% to -0.63% | -2.26% to -0.33% |
| WS8 throughput | -6.93% to +0.22% | -8.54% to +3.58% |
| WS8 median TPOT | +0.03% to +1.25% | -0.36% to +1.63% |

Only the directional WS4 result excludes zero under both constructions. The
reported WS8 TPOT interval does not survive this sensitivity check.

The five-block schedule also has a concrete randomization defect. The extension
runner reinitialized the arm-order RNG with the original seed and applied the
block offset only to labels; it did not advance the RNG. Consequently blocks
3-4 repeat blocks 0-1 at every WS:

- WS2 uses `I,U,T; I,U,T; U,I,T; I,U,T; I,U,T`; Triton is last in all five
  blocks, so implementation and third-position effects cannot be separated.
- WS4 uses only `T,I,U` three times and `U,T,I` twice.
- WS8 uses `I,T,U; T,U,I; U,I,T; I,T,U; T,U,I`.

The existing results remain a usable paired screen, especially the all-block
WS4 sign agreement, but the campaign is not fully order-balanced. Any promotion
run must fix the schedule generator and must not simply continue the defective
sequence.

### Microbenchmark analysis

The captured-graph data are stable; the eager data are schedule-sensitive.
Graph pass reversal produces near-zero median shifts, a largest Triton p50
spread of 2.27% (WS2 M65), and a largest overall spread of 4.96% (WS8 Iris
M91). Median per-rank imbalance is below 0.15%.

At every captured M, the Triton p50 change versus upstream is:

| M | WS2 raw / reset-adjusted | WS4 raw / reset-adjusted | WS8 raw / reset-adjusted |
|---:|---:|---:|---:|
| 1 | +61.18% / +113.94% | -0.28% / +18.94% | +5.64% / +17.40% |
| 32 | +17.11% / +39.37% | -5.36% / +7.70% | +1.56% / +10.84% |
| 64 | +8.66% / +25.48% | +2.62% / +14.04% | +5.40% / +13.02% |
| 65 | +7.45% / +24.22% | +24.87% / +38.39% | +46.80% / +57.45% |
| 91 | -1.17% / +11.60% | +14.94% / +26.13% | +33.40% / +41.78% |
| 92 | -31.93% / -26.03% | -23.36% / -18.72% | -15.17% / -11.89% |
| 128 | -22.49% / -16.32% | -14.97% / -9.84% | +77.91% / +91.74% |
| 256 | -13.04% / -7.95% | +30.20% / +38.85% | +107.82% / +122.65% |
| 384 | -10.81% / -5.84% | +45.74% / +55.83% | +126.63% / +142.64% |
| 385 | +0.01% / +0.01% | +0.06% / +0.10% | -0.11% / -0.11% |
| 512 | -0.11% / -0.13% | -0.19% / -0.19% | +0.39% / +0.44% |

Positive values mean Triton is slower. “Reset-adjusted” subtracts each
fallback arm's benchmark-reset copy; it is a derived serving-shaped diagnostic,
not a directly measured kernel. Reset cost is 3.72%-24.66% of raw graph p50
across the matrix.

The boundary discontinuities explain why one cap cannot summarize the profile:

| Boundary | WS2 upstream / Iris / Triton | WS4 upstream / Iris / Triton | WS8 upstream / Iris / Triton |
|:---|---:|---:|---:|
| M64 to M65 | +1.58% / +0.23% / +0.45% | +1.66% / -0.97% / +23.71% | +0.97% / +0.61% / +40.62% |
| M91 to M92 | +44.29% / +0.97% / -0.62% | +48.91% / +0.48% / -0.71% | +58.26% / +2.48% / +0.64% |
| M384 to M385 | +0.39% / +0.19% / +12.57% | +0.16% / +0.41% / -31.23% | +39.53% / +0.70% / -38.50% |

M64-to-M65 exposes the padded-to-blocked Triton cliff at WS4/8; M91-to-M92
switches the unfused control from ordinary Iris to RCCL; M384-to-M385 switches
captured Triton to ordinary fallback. The WS8 unfused graph curve is also
non-monotonic, so interpolation across these boundaries is invalid.

Eager p50 is more variable. Median pass-2/pass-1 shifts are:

| WS | Upstream | Iris | Triton |
|---:|---:|---:|---:|
| 2 | -12.92% | -9.83% | -14.38% |
| 4 | -9.43% | -18.87% | -13.01% |
| 8 | -0.66% | -1.15% | +0.55% |

Pass number, reversed arm order, elapsed time, and reversed M traversal all
change together, so the cause cannot be uniquely assigned. At WS8 M32, the
pooled -1.30% Triton result hides a pass reversal: -5.94% in pass 1 and +3.91%
in pass 2.

Eager arithmetic means are not suitable as primary performance evidence. Of
11,700 retained iterations per arm, 109 upstream samples exceed 10x their row
median, compared with five Triton and zero Iris samples. The worst upstream
sample is 7,567.3 us at WS4 M512, 92.5x its row median. These tails can reverse
mean and p50 conclusions: at WS4 M384 Triton is 16.89% slower by p50 but 51.12%
faster by mean; at WS8 M384 it is 39.38% slower by p50 but 42.05% faster by
mean. Those “mean speedups” are baseline stalls, not credible steady-state
kernel gains.

### Cap and transition decision

No positive actual-M upper cap is supported. Every allowed positive cap
includes the unfavorable M32 point. In the cap-64 region, adjusted graph
results favor Triton at 0/3 M values for every WS; eager favors it at 0/3 for
WS2/4 and 3/3 for WS8, but WS8 reverses by pass and conflicts with captured
execution. Larger caps include additional severe WS4/8 losses.

[selected-policy.json](selected-policy.json) therefore freezes `fusion_max_m=0`
for all WS. **Here `0` means no additional actual-M upper gate; it does not
mean disable fusion or prove that unlimited fusion is optimal.** It preserves
the predeclared Triton profile for exactly one serving screen while the
upstream-unfused deployment default contains the risk.

All transition probes pass:

| WS | Ranks | Steps/rank | Checked operations | Failures | Max norm error | Max residual error |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 1,117 | 3,352 | 0 | 0.007812 | 0.03125 |
| 4 | 4 | 1,117 | 6,704 | 0 | 0.003907 | 0.06250 |
| 8 | 8 | 1,117 | 13,408 | 0 | 0.019818 | 0.62500 |

They are correctness and retained-lifetime evidence only. The artifacts
explicitly set `performance_conclusions_allowed=false`, use 1-2 calls per M,
and do not reproduce the serving-shaped 72-site captured graph.

### Forward-marker analysis

All 18 retained analyses were checked against 84 rank traces: 144 unique
forward IDs and 672 rank-forward records. Every expected rank is present, and
all IDs agree across ranks on mode, actual/executed M, execution class, batch
metadata, and primary backend path. Detailed kernel counts agree for 143/144
IDs; the final WS2 Iris direct-decode window has 71 rather than 72 fused calls
on rank 0, consistent with trace-window clipping, while preserving the correct
primary path.

The marker windows are diagnostics, not an actual-M distribution from the
timed concurrency-32 workload:

| Execution cohort | Actual to executed M | Forwards |
|:---|:---|---:|
| decode graph | 1 to 1 | 63 |
| decode graph | 64 to 64 | 53 |
| prefill graph | 1 to 16 | 1 |
| prefill graph | 64 to 64 | 1 |
| prefill graph | 128 to 128 | 4 |
| prefill graph | 256 to 256 | 3 |
| prefill graph | 320 to 320 | 1 |
| prefill graph | 512 to 512 | 9 |
| eager | 7,808 / 7,872 / 8,000 / 8,064 | 1 / 3 / 4 / 1 |

There is no M32 marker. The only actual/executed mismatch is a WS8 Iris marker
padded from actual M1 to executed M16. Upstream uses ordinary-Iris decode and
RCCL prefill; Iris uses the fused graph path; Triton uses the expected
WS-specific one-shot variants and completely falls back to RCCL at M512 and at
high eager M.

Several post-benchmark diagnostic forwards have long full-GPU periods, but
most excess time is between kernels rather than in AR+RMSNorm:

| Cohort | Full period | Summed kernels | Non-kernel gap |
|:---|---:|---:|---:|
| WS2 Iris eager | 774.17 ms | 146.17 ms | 628.01 ms |
| WS4 Iris eager | 742.92 ms | 92.25 ms | 650.67 ms |
| WS4 Triton eager | 347.01 ms | 107.31 ms | 239.70 ms |
| WS8 Iris M320 graph | 599.64 ms | 55.15 ms | 544.50 ms |
| WS8 Triton M256 graph | 346.38 ms | 56.16 ms | 290.22 ms |
| WS8 Iris eager | 971.35 ms | 121.07 ms | 850.28 ms |
| WS8 Triton eager | 376.68 ms | 114.32 ms | 262.35 ms |

The WS2 Iris and WS8 candidate gaps align with their block-0 serving anomalies,
suggesting transient scheduler/GPU waiting rather than a slow AR+RMSNorm
kernel. WS4 is a counterexample, and diagnostics ran after the timed benchmark,
so this is circumstantial rather than causal evidence. Marker periods must not
be used to rewrite the paired serving result.

### Final decision

The retained evidence supports:

1. reject all positive max-M caps;
2. retain cap `0` only as “no additional gate” for the completed screen;
3. keep explicit upstream-unfused as the deployment default;
4. treat WS2 as inconclusive and WS8 as a directionally negative,
   outlier-sensitive loss;
5. if promotion is desired, test only WS4 with ten additional pairs after
   fixing arm-order randomization and using a measured-shape warmup.

The compact retained artifacts are:

- [summary.json](summary.json);
- [graph-sweep.csv](graph-sweep.csv) and
  [graph-sweep-summary.json](graph-sweep-summary.json);
- [eager-sweep.csv](eager-sweep.csv) and
  [eager-sweep-summary.json](eager-sweep-summary.json);
- [selected-policy.json](selected-policy.json);
- [forward-cohorts.json](forward-cohorts.json);
- [end-to-end-summary.json](end-to-end-summary.json).

The ignored raw root is `raw/2026-08-04-definitive-v1`. It retains the complete
machine/model hashes, setup retries, source amendments, all micro and transition
samples, 45 serving lifecycles, traces, checksums, and GPU-process guard history.

## Purpose

Prior GPT-OSS performance data spans multiple implementations, rank sets, and
serving policies. This campaign replaces those environment-tied percentages
with one current-code comparison of:

1. explicit production unfused;
2. default Iris fused;
3. the WS-specific optimized Triton profile.

Reserved-sink graph padding, persistent outputs, the 72-site lifetime contract,
complete fallback, and closed tuning conclusions remain durable engineering
evidence.

## Current-machine identity

The plan declared `/data/models/openai-gpt-oss-120b`; this machine used
`/data/models/openai/gpt-oss-120b`, resolving to
`/data/dev/morhuang/models/gpt-oss-120b`. Before collection, record:

- kernel and parent TokenSpeed commits, dirty state, and campaign-file hashes;
- container tag, immutable image ID, mounts, imports, and package versions;
- model realpath plus config, tokenizer, index, and shard manifest hashes;
- all GPU identities, PCI/KFD/NUMA mapping, XGMI and peer topology, clocks,
  RAS/ECC state, and HIP-to-physical mapping;
- verified-idle, nested, topology-balanced WS2/4/8 device sets.

The historical image ID and old model path are evidence identities, not
qualification for this machine.

## Microbenchmark contract

All arms use N=2880 bf16 and max-rank-per-iteration statistics. Eager and graph
results are reported separately.

Captured 72-site primary pass:

```text
M=1,32,64,65,91,92,128,256,384,385,512
50 warmups, 1000 measured replays
```

Order-opposed graph confirmation:

```text
M=32,64,65,91,92,384,385
```

Two order-opposed eager passes:

```text
M=1,32,64,65,91,92,128,256,384,385,512,1024,2048
30 warmups, 150 measured iterations
```

This is 162 graph processes, 18 eager processes, and three WS-specific
transition probes. The boundaries resolve:

- M64-to-M65 padded versus blocked one-shot;
- M91-to-M92 ordinary Iris versus RCCL in the unfused control;
- M384-to-M385 Triton one-shot versus captured ordinary fallback.

The graph report preserves raw replay time and separately reports the unfused
benchmark-reset copy. Reset-adjusted values are serving-shaped diagnostics, not
directly measured kernels.

Dry-run both micro schedules from the `tokenspeed-kernel` root:

```bash
SPEC=benchmark/results/ar_rmsnorm/studies/mi350x/\
2026-08-gpt-oss-120b-definitive-sweep/campaign.json

PYTHONPATH=. python3 benchmark/run_ar_rmsnorm_graph_sweep.py \
  --spec "$SPEC" \
  --devices 2=<two-devices> --devices 4=<four-devices> \
  --devices 8=<eight-devices> --dry-run

PYTHONPATH=. python3 benchmark/run_ar_rmsnorm_eager_sweep.py \
  --spec "$SPEC" \
  --devices 2=<two-devices> --devices 4=<four-devices> \
  --devices 8=<eight-devices> --dry-run
```

The graph dry run must report 162 processes; eager must report 18. Actual
collection additionally requires an immutable `--output-root`. Run the
transition probe once per WS after freezing `selected-policy.json`.

For each frozen WS profile:

```bash
export AR_NORM_DEVICES=<frozen-devices>
export GPT_OSS_DEFINITIVE_FUSION_MAX_M=<selected-cap>
source benchmark/profiles/ar_rmsnorm/\
gpt_oss_120b_mi350x_definitive_ws<WS>.env

HIP_VISIBLE_DEVICES="$AR_NORM_DEVICES" \
TS_TRITON_SHMEM_VISIBLE_DEVICES="$AR_NORM_DEVICES" \
BENCH_IMPL=triton_shmem_profile BENCH_WS=<WS> BENCH_N=2880 \
PROBE_MS=1,32,64,65,91,92,384,385,512,2048 \
PROBE_REPLAYS=1000 PROBE_JSON=<raw-root>/transitions-ws<WS>.json \
PYTHONPATH=python:. python3 -m benchmark.probe_ar_rmsnorm_transitions
```

## Profile and cap decision

Keep `COMM_FUSION_MAX_NUM_TOKENS=2048` as workspace and capture identity.
Lowering it is not a free launcher optimization: it changes allocation sizes,
profile identity, graph topology, and fallback capture.

After microbenchmarks and before serving, freeze one actual-M eligibility
policy per WS:

```text
cap M64, cap M91, cap M384, or retain no additional gate with cap 0
```

The current schema has no separate “disable fusion” value; cap `0` is
unlimited, not disabled.

A cap requires consistent graph passes, relevant eager direction, margin beyond
pass spread, and the observed serving M distribution. Multiple caps must never
be tried against end-to-end results. Every selected profile reruns correctness
and transitions.

## End-to-end contract

The core campaign uses three matched three-arm triplets per WS: 27 fresh server
lifecycles total. Every observation uses:

```text
input/output: 128 / 512
prompts/concurrency: 128 / 32
request rate: infinite
temperature: 0
ignore EOS: true
overlap scheduling: disabled symmetrically
```

Output throughput is primary; median TPOT is the guardrail. The plan calls for
randomized arm order, one unfused observation is shared by both contrasts, and
every observation gets a fresh server because server reuse previously
reproduced scheduler/GPU stalls. The collected extension reset the order RNG,
so the realized five-block schedule is not fully balanced; see the result
analysis above.

The first triplet at each WS also collects untimed concurrency-64 and direct
sequential-M512 marker traces. `tokenspeed.model_forward.v1` must prove
actual/executed M, graph/eager mode, and one rank-consistent backend path.

Extend from three to five triplets per WS only when signs disagree, a retained
failure is ambiguous, or the result lands within the existing promotion
thresholds. A deployment promotion still requires 15 pairs; stopping earlier
produces a definitive current-machine screen, not promotion evidence.

Inspect the three-WS serving schedule without starting a server:

```bash
PYTHONPATH=. python3 benchmark/run_gpt_oss_definitive_e2e.py \
  --spec "$SPEC" \
  --selected-policy benchmark/results/ar_rmsnorm/studies/mi350x/\
2026-08-gpt-oss-120b-definitive-sweep/selected-policy.json \
  --devices 2=<two-devices> --devices 4=<four-devices> \
  --devices 8=<eight-devices> --stage core --dry-run
```

Dry-run accepts the pending policy as ungated for schedule validation. Real
serving refuses to start until that file is `frozen` with one allowed cap per
WS. Run `--stage extension` into the same output root to add blocks 3-4, and
`--stage promotion` to add blocks 5-14. Stage directories remain separate; the
orchestrator rebuilds the combined per-WS paired analysis after each stage.

## Runtime

Historical process rates imply 2-3 hours for microbenchmarks and transitions,
plus 2.25-3.25 hours for 27 server lifecycles. Reserve **4-6 hours** for the core
campaign. The five-triplet extension adds about 1.5-2 hours. A full 15-triplet
promotion campaign is a separate 16-22-hour commitment.

## Result integration

After collection, track:

- [reporting-schema.json](reporting-schema.json);
- [selected-policy.json](selected-policy.json), changed from pending to frozen;
- `summary.json`;
- `graph-sweep.csv` and `graph-sweep-summary.json`;
- `eager-sweep.csv` and `eager-sweep-summary.json`;
- `forward-cohorts.json`;
- `end-to-end-summary.json`.

The final report must join executed-M forward cohorts to matching micro rows,
then keep operator delta, target-stage delta, full max-rank forward period,
TPOT, and throughput as separate outcomes. Update the live status only after
all artifacts for the selected stopping stage are complete.

Generate compact micro summaries with:

```bash
python3 benchmark/analyze_ar_rmsnorm_graph_sweep.py <graph-raw-root> \
  --max-m 512 --output-json graph-sweep-summary.json \
  --output-csv graph-sweep.csv

python3 benchmark/analyze_ar_rmsnorm_eager_sweep.py <eager-raw-root> \
  --output-json eager-sweep-summary.json --output-csv eager-sweep.csv
```

For each arm/WS trace set, run `analyze_ar_rmsnorm_forwards.py` with
`--graph-summary`, `--eager-summary`, `--arm`, and `--expected-world-size`.
Reject missing cohorts or cross-rank path disagreement before interpreting
end-to-end metrics.

## Time-saving exclusions

Do not repeat overlap root-cause work, M256 gating, blocked-core/grid/block
searches, folded copy-in, mutable two-slot rings, old hardware-fault theories,
marker-free grouping, or `all_reduce_two`. Do not flush L2 for this warm,
repeated-serving target, and do not use profiler request latency as a
performance metric.

## Restart point

The definitive five-triplet screen is complete. If deployment promotion is
wanted for the promising WS4 result, first reserve the separate 16–22 hour
budget, fix the scheduler so `block_offset` advances rather than resets the
arm-order RNG, and recheck shared-host idleness. Only then run
`--stage promotion` into the same E2E root to add blocks 5–14. Do not
reinterpret five pairs as promotion evidence or transfer the WS4 result to
WS2/8.
