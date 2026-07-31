# MI350X GPT-OSS-120B triton-shmem realignment

Date: 2026-07-30

## Campaign decision

The realignment produced a qualified `triton_shmem` performance improvement,
but not a deployment promotion. This conclusion was later superseded by
[core-v3 tuning](../2026-07-triton-shmem-core-tuning/README.md); it remains the
historical profile-v2 decision.

- The campaign retained upstream-unfused with `--disable-allreduce-fusion`.
- Retain profile `gpt-oss-120b-mi350x-triton-realigned-v2` as an explicit
  experimental profile.
- The full clean campaign completed 15/15 pairs on HIP `1,2,5,6`
  (physical GPUs 0,2,4,6). Median decode TPOT was **+3.75%** slower
  (95% CI +0.20% to +7.34%) and output throughput was **-2.73%**
  (95% CI -5.80% to +0.34%) versus upstream-unfused.
- This is materially better than the post-rebase `triton_shmem` baseline
  (+10.47% TPOT, -10.58% throughput), but it does not clear either promotion
  objective.

Machine-readable results are in [summary.json](summary.json). Raw artifacts are
under `raw/` in this study and are intentionally Git-ignored.

## Implemented changes

### Eager two-shot borrowed outputs

`TS_TRITON_SHMEM_BORROW_TWOSHOT_OUTPUT=1` allocates two coarse symmetric norm
and residual output pairs, each with independent peer-pointer tables. Eager
two-shot calls alternate pairs and return the selected views directly, removing
the two local copy-out launches. The trailing completion barrier remains.
Captured two-shot calls and caller-provided outputs retain the copied path.

Two pairs are required: the next fused site reads the previous `residual_out`
while writing a new one, so a single returned residual buffer would alias.

### Graph-stable input site ring

`TS_TRITON_SHMEM_INPUT_SITE_RING=72` reserves one coarse symmetric input slot
for each unconditional GPT-OSS fused site up to the one-shot cap. Capture binds
each call to a distinct pointer. Reuse is delayed for a full 72-site forward,
so intervening leading rendezvous prove peer reads complete and the one-shot
exit barrier can be omitted without the rejected mutable two-slot replay phase.

This is a model-profile contract, not a generic default. Unknown captured call
counts retain the ordinary exit barrier by leaving the site ring disabled.

### Dispatch

The measured one-shot overlay is extended from M=256 to **M=384**. M=512
remains two-shot. Diagnostic `TS_TRITON_SHMEM_TWOSHOT_BLOCK_N` was added; the
default 512-element tile remained best or tied and no override is selected.

## Operator and graph results

Two order-opposed WS=4/N=2880 passes on the post-rebase baseline HIP `1,2,3,5`
set show
the combined M=384/site-ring/borrowed-output candidate versus the current
post-rebase `triton_shmem` baseline:

| M | baseline mean (us) | candidate mean (us) | change |
|---:|---:|---:|---:|
| 257 | 117.2 | 73.6 | -37.2% |
| 384 | 111.9 | 78.7 | -29.7% |
| 512 | 117.4 | 91.4 | -22.2% |
| 1024 | 118.2 | 103.3 | -12.5% |
| 2048 | 195.1 | 185.6 | -4.9% |

The directly timed two-shot copy-out was 18-19 us at M=257-1024. Removing it
explains most of the improvement through M=1024; the kernel/transport term
dominates at M=2048.

For a 72-call M=32 graph, the site ring reduced triton-shmem from a two-pass
mean **30.35 us/site** to **24.36 us/site** (-19.8%), or about 0.43 ms per
forward. On the uncontended alternate set, the matched 72-call p50s were:

- upstream-unfused: 16.96 us/site;
- Iris fused: 18.97 us/site;
- realigned triton-shmem: 25.45 us/site.

The later 25.23 us/site decomposition value uses cumulative stage-prefix
probes; the 25.45 value above is the public matched graph p50. They answer
different accounting questions and should not be merged.

The optimization is therefore real, but triton-shmem still trails Iris and the
unfused control on the full decode graph.

## Serving and profile evidence

The clean 3-block x 5-seed campaign is
`raw/full-campaign-alt-v3/`. It used HIP `1,2,5,6` after two baseline-set
attempts were correctly rejected when foreign Python processes appeared on
physical GPU 1.

Several prefill median-TTFT cohorts improved despite decode rejection:

- aggregate M128: -4.16% (95% CI -6.10% to -2.34%);
- aggregate M256: -4.15% (95% CI -10.28% to -0.25%);
- aggregate M1024: -2.80% (95% CI -4.52% to -0.72%);
- aggregate M2048: -3.81% (95% CI -6.19% to -1.99%).

Matched marker-aligned decode traces measured a 2.122 ms triton target-kernel
sum versus 1.866 ms upstream-unfused (+13.8%), and a 13.703 ms versus
13.108 ms max-rank GPU period (+4.5%). This explains why the large improvement
over the old triton path did not become an end-to-end promotion.

The signature workload was corrected during exploration: aggregate “M512” was
scheduled as approximately M144 after the rebase. The final campaign uses
sequential direct 512-token requests and proves 72 two-shot calls per rank.

## Validation

- `test_triton_shmem_communication.py -k "not world8"`: 16 passed,
  3 WS=8 tests deselected.
- Repeatability harness tests: 21 passed.
- Borrowed-output test: 72 chained M257 sites plus M512/1024/2048, changing
  inputs, pointer alternation, residual non-aliasing, and caller-output fallback.
- Site-ring tests: eager wraparound and two interleaved 72-call graphs,
  100 replays each.
- Transition probe: 1000 replays across M=1/32/255/256/257/512/2048, passed.
- Bounded 128- and 512-token GPT-OSS serves: passed with 0 failed requests.
- Full campaign: 15/15 safe pairs and direct-M512 signature proof on all ranks.

WS=8 remains deferred because physical GPU 3 is occupied.

## Exhausted immediate paths

- One-shot caps 384/512/768/1024: M=384 is the best robust boundary; larger
  caps lose bandwidth scaling.
- Fixed in-kernel barrier grids 8-256: shape-specific or regressive; no global
  setting survived.
- One- and two-shot block widths 128-4096: flat or slower than defaults.
- Decode grid caps below 32: sharply regressive.
- Folded copy-in with the site ring: 26.3-26.7 us/site, slower than explicit
  copy-in at 24.7-25.4 us/site, and still carries the prior serving hazard.
- A separate one-block leading barrier: only ~2% screening gain, below the 5%
  retention threshold.
- Producer-direct input remains blocked because active dense GEMM and MXFP4 MoE
  APIs cannot accept exact caller-owned output views.

The remaining decode gap is in the leading publication/rendezvous and blocked
pull/reduction path. Removing it requires producer/API and system-scope progress
work, not another local environment-knob sweep.

## Follow-up decomposition

The
[three-backend decomposition study](../2026-07-triton-shmem-decomposition/README.md)
directly times eager and captured stages for upstream-unfused, Iris fused, and
realigned triton-shmem. It confirms:

- decode copy-in is at parity with Iris and triton's exit barrier is already
  eliminated;
- the blocked triton decode core is 9.05 us/site slower than Iris, partly offset
  by a 4.03 us/site exit-barrier saving;
- at M512-M1024, the two-shot core is competitive but 25-30 us of
  copy/synchronization overhead determines the loss;
- at M2048, both the core and integration overhead lose to RCCL + RMSNorm.

The follow-up closes remaining local tuning doors and makes decode-core
algorithm work, or a combined producer-direct/progress project, the only
evidence-supported continuations.

That decode-core continuation is now complete:
[core-v3 tuning](../2026-07-triton-shmem-core-tuning/README.md) replaces the
blocked M<=64 path with a scratch-free padded whole-row kernel, improves the
72-call graph by 35.2%, and clears the capacity promotion gate. Profile v2
remains the lifetime-integration baseline. Current deployment policy belongs in
[GPT-OSS-120B status](../../../docs/gpt-oss-120b-status.md).
