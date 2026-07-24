# gpt-oss-120B fused AR+RMSNorm status

Updated: 2026-07-24

## Scope

- Model: `/data/models/openai/gpt-oss-120b`
- Residual hidden size: 2880 bf16 elements
- Hardware: 8× AMD Instinct MI350X (`gfx950`)
- Runtime: torch 2.11 + released ROCm 7.2.4 userspace
- Serving/profiling container: `jeremwan-tokenspeed-profiler`
- Shared-host rule: exclude physical GPU 3 / HIP index 0 for TP<8.

## Deployment decision

- TP=2: fusion auto-enables.
- TP=4/8: fusion is available but remains explicit opt-in.
- TP=4/8 opt-in requires:

```text
TS_ARNORM_BACKEND=triton_shmem
--enable-allreduce-fusion
--comm-fusion-max-num-tokens 2048
```

Backend selection or a positive cap alone does not prove fusion. A valid fused
run must also show `enable_allreduce_fusion=True` in server arguments and fused
kernel signatures in every rank trace.

## Corrected current evidence

Current matched TP=4 traces:

```text
../raw/current/gpt-oss-120b/mi350x/2026-07-24/traces/torch/
  tp4-fused-generic-block512/
  tp4-unfused/
```

Current matched e2e logs:

```text
../raw/current/gpt-oss-120b/mi350x/2026-07-24/e2e/
  tp4-fused-generic-block512/
  tp4-unfused/
```

The corrected profile window shows:

- fused: 1,080 decode one-shot calls and 144 prefill two-shot calls per rank;
- unfused: 1,095 Triton all-reduces and 1,241 separate RMSNorm calls per rank;
- generic fused decode median: 34.1–36.5 µs across ranks;
- max same-rank unfused AR + RMSNorm median sum: 36.92 µs;
- GPU kernels per rank: 29,967 fused vs 31,333 unfused;
- profiled GPU-window span: about 313.8 ms fused vs 340.5 ms unfused (-7.8%).

Machine-readable source:
`../studies/mi350x/2026-07-profile-guided-followup/corrected_profile_comparison.json`.

## End-to-end result

Workload: random input 128, output 512, 128 prompts, concurrency 32, temperature
0, ignore EOS, seeds 0 and 1.

- unfused mean median TPOT: 12.755 ms;
- generic fused mean median TPOT: 12.560 ms;
- fused change: -1.5%;
- diagnostic BLOCK_N=2048 fused TPOT: 12.550 ms, only 0.1% beyond generic and
  below the noise floor.

Source:
`../studies/mi350x/2026-07-profile-guided-followup/e2e_summary.json`.

The current pair favors fusion, but earlier validated campaigns regressed at
TP=4 and the margin is comparable to run variability. TP=4 therefore remains
opt-in.

## Active kernel paths

Decode is dominated by small-M `oneshot_blocked` with folded copy-in. Prefill
can enter `twoshot_blocked`; it must not be described as absent merely because
a decode-only scope did not observe it.

Shipping defaults remain generic:

```text
TS_TRITON_SHMEM_COARSE=1
TS_TRITON_SHMEM_INKERNEL_BARRIER=1
TS_TRITON_SHMEM_FOLD_COPYIN=1
TS_TRITON_SHMEM_FOLD_NUM_WARPS=1
TS_TRITON_SHMEM_WORKGROUP_SYNC=1
TS_TRITON_SHMEM_ONESHOT_MAX_M=256
TS_TRITON_SHMEM_ONESHOT_BLOCK_N=0
TS_TRITON_SHMEM_GRID_CAP=-1
TS_TRITON_SHMEM_GRID_CAP_MIN_M=-1
TS_TRITON_SHMEM_BARRIER_GRID=0
TS_TRITON_AR_WORKGROUP_SYNC=1
TORCH_NCCL_BLOCKING_WAIT=0
```

`TS_TRITON_SHMEM_ONESHOT_BLOCK_N` is diagnostic only.

## Rejected and closed directions

- Width-aware small-M tiles improved isolated graph replay by 10–23% across
  N={1536,2880,5120,7168}, but faulted during a real serving graph transition.
- A diagnostic N=2880 BLOCK_N=2048 specialization did not move TPOT beyond
  noise.
- Reducing the M=32 grid increased latency.
- Fixed barrier grids are correctness controls for M-divergent execution, not
  pure-TP optimizations.
- Multi-wave folded copy-in is unsafe.
- Paired copy kernels, `foreach_copy_`, and borrowed mutable outputs were slower,
  shape-regressing, or unsafe without a lifetime contract.

Rejected-candidate data:
`../studies/mi350x/2026-07-profile-guided-followup/small_m_width_sweep.json`.

## Next work

1. Trace producer/consumer lifetimes for producer-direct symmetric input or
   caller-owned output buffers.
2. Separate prefill/TTFT profiling from decode TPOT before modifying two-shot.
3. Require shared-state interleaved graph transitions and a full multi-arm serve
   before promoting any kernel policy.
4. Validate every new model/hidden size with its own op, graph, and e2e matrix;
   do not inherit gpt-oss tuning as universal policy.

Implementation invariants and the profiling runbook are maintained separately:
[backend design](backend-design-and-safety.md) and
[profiling workflow](profiling-workflow.md).

