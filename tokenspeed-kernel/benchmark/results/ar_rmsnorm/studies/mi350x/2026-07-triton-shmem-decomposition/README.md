# GPT-OSS-120B backend decomposition on MI350X

Date: 2026-07-30

## Disposition

This study explains the profile-v2 bottlenecks and closes local copy/barrier
tuning:

- captured M32 was dominated by the blocked triton decode core;
- eager M512-M1024 had a competitive two-shot core, but copy plus required
  synchronization erased the advantage over RCCL;
- at M2048 both the core and integration overhead lost.

The later [core-v3 study](../2026-07-triton-shmem-core-tuning/README.md)
replaced the blocked decode core and cleared capacity under restricted controls.
[Default compatibility](../2026-07-default-compatibility/README.md) later
restored fusion-off deployment. The stage accounting and closed-path
conclusions remain valid for profile v2.

Exact results are in [summary.json](summary.json). Raw probe output is local
under `raw/` and Git-ignored.

## Method

- GPT-OSS-120B, N=2880 bf16, TP=4, MI350X;
- HIP `1,2,5,6` (physical GPUs `0,2,4,6`);
- profile `gpt-oss-120b-mi350x-triton-realigned-v2`;
- eager: two M-order-opposed passes, 30 warmups, 150 samples;
- graph: two 72-call M32 passes, 50 warmups, 1000 replays;
- each iteration reduced to max rank before p50;
- every path checked against fp32 all-reduce + residual RMSNorm.

The probes use cumulative prefixes so marginal stages remain additive.
Independent kernel timings are diagnostic because launches and synchronization
can overlap.

## Captured M32 critical path

Values are means of two passes in microseconds per site.

### Upstream-unfused

- probe-only reset copy: 2.05;
- ordinary Iris transport: 12.75;
- standalone residual RMSNorm: 2.16;
- probe total: 16.96;
- serving-faithful total excluding reset: **14.90**.

The reset exists only because the probe repeats an in-place transport; serving
already receives a producer-created rank-local partial.

### Iris fused

- symmetric-heap copy-in: 2.05;
- entry barrier: 4.18;
- fused core: 8.80;
- exit barrier: 4.03;
- total: **19.05**.

### Realigned triton-shmem

- site-ring copy-in: 2.05;
- entry rendezvous: 5.29;
- blocked core: 17.86;
- exit synchronization: 0;
- wrapper/lifetime remainder: 0.03;
- total: **25.23**.

Relative to Iris, triton's copy was equal, entry rendezvous was 1.11 us slower,
the core was 9.05 us slower, and the removed exit barrier recovered 4.03 us.
Across 72 sites, profile v2 remained 0.445 ms behind Iris and 0.743 ms behind
serving-faithful upstream-unfused.

Eager M32 misleadingly favored triton (50.23 us versus 67.01 Iris and 68.00
upstream-unfused) because graph capture amortized much more Iris launch and
barrier overhead. Candidate ranking therefore must use the 72-site captured
path.

## Eager crossover

Public max-rank p50 totals, listed as upstream-unfused / Iris fused / triton
profile v2:

- M256: 66.11 / 73.73 / 55.61 us;
- M384: 66.33 / 85.73 / 71.91 us;
- M512: 65.64 / 112.24 / 73.54 us;
- M1024: 90.32 / 167.79 / 103.59 us;
- M2048: 145.54 / 280.11 / 184.42 us.

The M384 one-shot boundary was appropriate. At M512-M1024, triton copy and
entry/exit synchronization cost about 25-30 us while the two-shot core was
competitive. At M2048 the core itself was about 14.2 us slower than the
complete preallocated RCCL + RMSNorm stack.

The ordinary transport switches from Iris at M91 to RCCL at M92 in this
configuration. Comparisons across that boundary must identify the selected
transport.

## Transfer to serving

Profile-v2 production evidence had the same direction:

- marker target sum: 2.122 ms triton versus 1.866 ms upstream-unfused;
- max-rank GPU period: 13.703 ms versus 13.108 ms;
- campaign median TPOT: +3.75%;
- campaign output throughput: -2.73%.

Aggregate prefill bucket labels are not executed M proof. A nominal M512
workload executed near M144; sequential direct 512-token requests were required
to prove the two-shot path.

## Closed and remaining mechanisms

Closed:

- decode copy-in as the main differentiator;
- decode exit barrier;
- eager two-shot copy-out;
- allocation/lifetime wrapper overhead;
- block width, grid, fixed-barrier, and folded-copy sweeps.

Remaining after profile v2:

- replace the blocked scratch-based decode core;
- combine producer-direct output and progress publication for M512-M1024;
- use a different reduction algorithm before reconsidering M2048.

Core-v3 completed the first item. The second is a cross-layer systems project,
not another local environment-knob campaign.
