# GPT-OSS-120B post-rebase AR+RMSNorm baseline

Campaign date: 2026-07-30 UTC.

This study re-establishes the GPT-OSS-120B, TP=4, MI350X baseline after the
rebase onto upstream `3f88dcc2`. The measured code head was `eb69cf15`; each
campaign manifest records the full dirty-tree hash and script hashes. Raw
artifacts are immutable under:

- `../../../raw/current/gpt-oss-120b/mi350x/2026-07-30/2026-07-30-post-rebase-operator-baseline/`
- `../../../raw/current/gpt-oss-120b/mi350x/2026-07-30/2026-07-30-post-rebase-graph-transition/`
- `../../../raw/current/gpt-oss-120b/mi350x/2026-07-30/2026-07-30-post-rebase-stability-unfused/`
- `../../../raw/current/gpt-oss-120b/mi350x/2026-07-30/2026-07-30-post-rebase-iris-vs-unfused-v4/`
- `../../../raw/current/gpt-oss-120b/mi350x/2026-07-30/2026-07-30-post-rebase-triton-shmem-vs-unfused-v3/`

The machine-readable decision record is [summary.json](summary.json); all
paired decode changes are in [paired-decode.json](paired-decode.json).

## Canonical decision

Use the upstream ordinary-all-reduce plus standalone RMSNorm path, with
all-reduce fusion **explicitly disabled**, for GPT-OSS-120B TP=4 on the
canonical HIP rank set `1,2,3,5`.

Upstream auto-enables fusion on this topology, so leaving the CLI flag absent
does not create an unfused control. This campaign added and qualified
`--disable-allreduce-fusion`; the serving proof requires the resolved
`enable_allreduce_fusion=False` argument.

Both fused candidates passed 15/15 paired workloads without a safety failure
but failed performance qualification:

- Iris `auto`: median TPOT **+2.55%** (95% CI +1.00% to +5.43%) and output
  throughput **-2.29%** (95% CI -4.78% to -0.89%).
- explicit `triton_shmem`: median TPOT **+10.47%** (95% CI +6.71% to +19.23%)
  and output throughput **-10.58%** (95% CI -23.21% to -4.49%).

Positive TPOT is worse; negative throughput is worse. Neither candidate is
eligible for latency or capacity promotion.
`triton_shmem` also showed two clean-but-severe block-1 throughput losses
(-62.56% and -30.14%); they are retained observations, not dropped outliers.

## Evidence ladder

### Environment and identity

The base profiler image remained
`sha256:ad3ea3f8cae8ca38cf12824b15c606d0630118c6e04b4087e191b04619a6c135`.
The rebased runtime required package corrections in the container writable
layer: Transformers 5.12, current SMG packages, XGrammar 0.2.2, and a scheduler
wheel rebuilt from the current source because the published 0.1.3 wheel lacked
`PagedCacheTransferPolicy`. Source-tree `PYTHONPATH` is required for the AMD
kernel package. These corrections are part of the measured environment, not
optional setup.

Physical GPU 3 remained occupied and was never selected. World-size 8 is
deferred. The canonical TP=4 rank set maps HIP `1,2,3,5` to physical GPUs
`0,2,1,4`.

### Correctness and operator screening

WS=2/4 Iris and `triton_shmem` communication suites passed. The operator
campaign used two order-opposed passes, 30 warmups, 150 measured iterations,
20-second cooldowns, and per-iteration max-rank statistics. Raw rank samples
are retained.

At WS=4/N=2880, `triton_shmem` is faster than the control through M=256 but
crosses sharply at M=257. Iris fused is slower than the control across the
serving fusion range. Large-M rows are stable; sub-millisecond rows retain the
measured variance floor and are screening evidence only.

`all_reduce_two` was evaluated separately. Its fused Iris launch was faster for
eligible pairs, but one initial two-ordinary control produced a rank-local
correctness error; five fresh-process attempts did not reproduce it. This is
not GPT-OSS AR+RMSNorm evidence and does not change the deployment decision.

### Graph and transitions

All three arms passed 1,000-replay fixed M=32 graph screens with changing
inputs and odd/even calls. At one call per graph, max-rank p50 was 33.12 us for
production-unfused, 46.06 us for Iris fused, and 44.52 us for `triton_shmem`.

Both fused candidates passed bounded cross-M graph/eager transitions.
Production-unfused passed Iris-only M=1/M=32 transitions but reproducibly timed
out when synthetic captured Iris and RCCL-fallback graphs shared one transition
matrix. Real GPT serving passed 128- and 512-token bounded workloads, so this
remains an actionable synthetic transition hazard rather than a reproduced
serving failure.

### Marker-aligned profiling

Every rank trace proved its backend:

- control: 73 ordinary Iris all-reduces plus 73 standalone RMSNorms;
- Iris candidate: 72 fused Iris kernels;
- `triton_shmem`: 72 fused one-shot kernels.

Marker-aligned max-rank target-kernel medians were 4.779 ms, 6.146 ms, and
5.943 ms per decode forward respectively. Kineto request latency is perturbed;
the path identity and target-stage direction, not profiled request latency, are
the authoritative result.

### Restart-randomized serving

Each final campaign used three restart blocks, five paired seeds per block,
fresh decode servers, separate prefill servers, randomized arm order,
nonintrusive KFD PID guards, hard timeouts, profile/signature proof, and 10,000
hierarchical bootstrap draws. Final checksums verify after teardown.

Several earlier roots are intentionally retained as incident evidence:
dependency failures, a missing explicit fusion-off policy, a shell
environment-propagation bug, signature-only trace misuse, and foreign GPU1
contamination. None contributes to the final estimates.

Checksums were regenerated after final teardown and verified for the operator,
graph/transition, stability, Iris, and `triton_shmem` evidence roots.

## Optimization realignment

Do not invest next in local `triton_shmem` launch tuning. Its M<=256 eager
advantage does not survive graph replay or serving, and the completed campaign
is decisively unfavorable.

The next useful work is:

1. keep explicit fusion-off as the GPT-OSS deployment control;
2. isolate the ordinary Iris/RCCL captured-transition hazard;
3. profile why Iris fused costs more than ordinary Iris plus RMSNorm at N=2880;
4. only revisit a fused candidate after it beats the unfused graph critical
   path by enough to clear the 1.5% TPOT or 1% throughput campaign threshold.

World-size 8 and other rank sets require independent qualification after
physical GPU 3 becomes available.
