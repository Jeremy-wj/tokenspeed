# GPT-OSS-120B triton-shmem decode-core tuning

Date: 2026-07-31

## Decision

**Historical restricted configuration only.** This campaign used eager
prefill, C32-only decode capture, 0.90 HBM, overlap off, and passive health.
Default-compatible requalification supersedes its deployment conclusion; the
live GPT-OSS policy record is
[GPT-OSS-120B status](../../../docs/gpt-oss-120b-status.md).

Profile `gpt-oss-120b-mi350x-triton-core-v3` was promoted for the **capacity
objective under that restricted configuration** on the qualified TP=4 rank set:

```text
HIP_VISIBLE_DEVICES=1,2,5,6
physical GPUs=0,2,4,6
AR_NORM_PROFILE_ID=gpt-oss-120b-mi350x-triton-core-v3
TS_ARNORM_BACKEND=triton_shmem
--enable-allreduce-fusion
```

The full 3-block x 5-seed campaign completed 15/15 safe pairs:

- output throughput: **+1.29%** (95% CI +0.52% to +2.08%);
- median TPOT: **-0.80%** (95% CI -1.42% to -0.25%);
- mean TPOT: **-1.21%** (95% CI -1.93% to -0.50%);
- median absolute TPOT: 12.716 ms unfused, 12.642 ms triton;
- median absolute throughput: 2381.2 versus 2402.1 output tokens/s.

Percentage point estimates are arithmetic means of the 15 paired changes;
`median TPOT` names each run's TPOT statistic, not a median across pair-level
changes. Confidence intervals use the paired hierarchical bootstrap.

Under the restricted configuration, this clears the predeclared capacity gate:
at least +1% throughput with CI excluding zero and no TPOT regression. The
latency objective's -1.5% threshold is not cleared.

Keep explicit upstream-unfused as the fallback for unqualified rank sets,
profile mismatch, or backend decline. WS=8 remains deferred while physical GPU
3 is occupied.

Machine-readable results are in [summary.json](summary.json) and the 15 absolute
pairs are in [paired-decode.json](paired-decode.json). Raw logs, traces, and
campaign artifacts are under `raw/` and intentionally Git-ignored.

## Why the prior sweep missed this

The imported MI300X optimization sweep used power-of-two widths such as
N=1024/4096/16384. Those shapes dispatch to the scratch-free whole-row kernel.
GPT-OSS N=2880 is non-power-of-two and therefore used
`oneshot_blocked`, which performs:

1. six 512-element peer-reduction blocks;
2. an fp32 scratch store for the complete row;
3. a second six-block pass that reloads scratch and gamma.

The follow-up decomposition measured this blocked core at 17.86 us/site versus
Iris's 8.80 us/site scratch-free whole-row core. Earlier block/grid/peer/XCD
tuning did not change that algorithmic difference.

The landscape reports pointed to the missing specialization:

- whole-token ownership keeps RMS reduction inside one program;
- AITER's one-stage fused kernel uses one block per token;
- Iris's fused shim uses a 4096-lane masked row for N=2880;
- the prior MI300X sweep never tested a non-power-of-two padded whole row.

## Implemented kernel

`fused_ar_rmsnorm_oneshot_wholerow_padded_kernel`:

- one program owns one token row;
- `BLOCK_N=next_power_of_2(2880)=4096`;
- lanes 2880-4095 are masked and contribute zero;
- all four peer partials are accumulated in fp32 registers;
- residual add, sum-of-squares, gamma, and normalized output remain in one pass;
- no fp32 scratch write or reload;
- the existing in-kernel entry rendezvous and 72-site input ring are unchanged;
- the one-shot exit barrier remains omitted by delayed site reuse.

Dispatch is intentionally narrow:

- M<=64: padded whole-row, four warps;
- M=65..384: existing blocked one-shot;
- M>384: existing two-shot.

This protects the M256/M384 eager guardrails, where forcing padded whole-row was
7-15% slower than blocked.

## Graph-captured decode result

Two independent 72-call M32 decompositions, each with 50 warmups and 1000
replays:

- padded triton total: **16.35 us/site**;
- blocked profile-v2 triton: 25.23 us/site;
- Iris fused: 19.06 us/site;
- standard unfused probe including reset copy: 16.97 us/site;
- serving-faithful unfused operation excluding reset: 14.92 us/site.

The padded kernel improves full triton graph cost by **35.2%** and beats fused
Iris by **14.2%**. It is 3.7% faster than the standard graph probe total, while
remaining 1.43 us/site behind the serving-faithful unfused operation after the
probe-only reset copy is removed.

Stage means:

- copy-in: 2.05 us/site;
- entry rendezvous: 4.41 us/site;
- scratch-free core: **9.93 us/site**;
- exit sync: 0;
- wrapper remainder: noise-level.

The core improves **44.4%** from 17.86 to 9.93 us/site and is now within
1.13 us/site of Iris's 8.80 us/site core.

## Launch tuning

The candidate was ranked only under 72-call graph replay.

Warp sweep:

- one warp: 18.70 us/site;
- two warps: 17.31 us/site;
- **four warps: 16.33 us/site**;
- eight warps: 17.65 us/site.

Grid sweep with four warps:

- 16 programs: 17.96 us/site;
- 24 programs: 18.57 us/site;
- **32 programs: 16.33 us/site**.

The winning launch is one program per decode row and four warps. Iris's
eight-warp choice does not transfer directly because pointer/barrier codegen and
register pressure differ.

## Marker-aligned serving trace

The uncontaminated bounded profile completed 128/128 requests and resolved the
padded kernel on every rank.

- padded target-kernel sum: **1.492 ms**;
- blocked profile-v2 target-kernel sum: 2.122 ms;
- matched upstream-unfused target-kernel sum: 1.866 ms;
- padded max-rank GPU period: **13.071 ms**;
- upstream-unfused GPU period: 13.108 ms.

The target stage improves 29.7% versus blocked triton and is 20.0% below
unfused. Total GPU period reaches parity, satisfying the pre-campaign projection
gate.

## Validation

- padded random correctness with M1/4/32 and blocked fallback M128/256: pass;
- two interleaved 72-call padded graphs, 100 replays: pass;
- full communication suite: 18 passed, 3 WS8 tests deselected;
- transition matrix: 1000 replays, 1649 operations, zero failed steps;
- path proof: padded M1/M32, blocked M255/256/257, two-shot M512/2048;
- bounded 128-token serve: 128 completed, 0 failed;
- bounded 512-token profiled serve: 128 completed, 0 failed;
- full campaign: 15/15 safe pairs and all direct-M512 signature proofs passed.

One profiled attempt and the final initial campaign arm were rejected when a
transient external Python process opened contexts across the selected GPUs.
Both were rerun cleanly; no contaminated result is included.

## End-to-end interpretation

Profile v2 was still +3.75% median TPOT and -2.73% throughput. The padded kernel
changes that to -0.80% TPOT and +1.29% throughput.

The improvement transfers because it targets the exact captured M32 path:

- same 72 sites;
- same site/output rings;
- same copy and entry rendezvous;
- same max-rank graph execution;
- only the scratch-based core is replaced.

Large-M prefill dispatch is unchanged. Prefill results remain noisy, but the
campaign finds no promotion-blocking guardrail regression. M512 median TTFT
improves 1.86% with CI excluding zero; M2048 throughput improves 4.74% with CI
excluding zero.

## Remaining headroom

The dominant 9.05 us/site decode-core problem is resolved. Remaining local gap:

- padded core is 1.13 us/site slower than Iris;
- entry rendezvous is about 0.23 us/site slower than Iris;
- copy-in remains 2.05 us/site and requires producer-direct plumbing to remove;
- total is about 1.43 us/site slower than serving-faithful unfused.

Do not reopen the blocked-kernel block/grid/XCD/fast-path sweeps. Further work
would be a much smaller kernel-codegen investigation or the already-documented
producer-direct/progress-publication systems project. The current profile has
cleared the capacity objective under restricted controls and remains the kernel
baseline, not current deployment policy.
