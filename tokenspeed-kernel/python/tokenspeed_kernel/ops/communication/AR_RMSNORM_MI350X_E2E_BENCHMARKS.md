# Fused AR+RMSNorm on MI350X: current benchmark and fault-resolution record

This is the canonical MI350X record for the `triton_shmem` fused all-reduce +
residual-add + RMSNorm backend. Backend design and MI300X evidence are in
`AR_RMSNORM_SYMM_MEM_MIGRATION.md`. Container provenance and the historical
HIP/RCCL investigation are in
`AR_RMSNORM_ROCM_CONTAINER_AND_RCCL_HISTORY.md`.

## Current decision

All previously failing serving configurations now complete with
`TORCH_NCCL_BLOCKING_WAIT=0`:

- ws=2/4/8 fused and unfused serving;
- random input 128/output 512 at concurrency 8/16/32;
- two seeds per point, zero failed requests;
- ws=8 includes the physical GPU that faulted previously.

Two independent AMD barrier defects were fixed:

1. **Unfused Triton AR:** a scalar system-scope signal barrier did not represent
   sibling wavefront loads/stores. Disabling the new workgroup synchronization
   reproduces the ws=8 post-warmup memory faults; enabling it completes the full
   campaign.
2. **Folded fused copy-in:** phase-0 stores and peer pulls used a scalar
   cross-rank barrier in a multi-wave program. A workgroup barrier is necessary
   but not sufficient because the system release/acquire is wave-scoped. The
   safe folded specialization uses one wavefront. Four warps reproduce the
   post-readiness memory fault; one warp completes ws=4/8 serving.

Folded copy-in is again default ON, with one warp. In-kernel barriers, coarse
HIP-IPC buffers, and the small-M one-shot overlay remain enabled.

Performance policy remains conservative:

- **ws=2:** auto-enable fusion; e2e ranges from parity to a 2.2% TPOT win.
- **ws=4/8:** do not auto-enable fusion. The isolated fused operator wins at the
  smallest M, but complete serving is slower over the measured workload. Users
  can still opt in explicitly for a validated workload.

## 1. Environment and defaults

- Hardware: 8× AMD Instinct MI350X (`gfx950`), shared development host.
- Container: `jeremwan-tokenspeed`.
- Image: `jeremwan/tokenspeed:rocm7.2.4-torch2.11`,
  `sha256:96f8b38d54c7da59f7888def76be81e99bf7512117bb2769609fadc7f19d230f`.
- Runtime: torch `2.11.0+rocm7.2`, loaded HIP 7.2.53211 and RCCL 2.27.7.
- Model: `/data/models/openai/gpt-oss-120b`; bf16 residual width N=2880.

Resolved fused defaults:

```text
TS_ARNORM_BACKEND=auto
TS_TRITON_SHMEM_COARSE=1
TS_TRITON_SHMEM_INKERNEL_BARRIER=1
TS_TRITON_SHMEM_FOLD_COPYIN=1
TS_TRITON_SHMEM_FOLD_NUM_WARPS=1
TS_TRITON_SHMEM_WORKGROUP_SYNC=1
TS_TRITON_SHMEM_ONESHOT_MAX_M=256
TS_TRITON_SHMEM_BARRIER_GRID=0
TS_TRITON_AR_WORKGROUP_SYNC=1
TORCH_NCCL_BLOCKING_WAIT=0
```

`TS_TRITON_SHMEM_WORKGROUP_SYNC=0`,
`TS_TRITON_SHMEM_FOLD_NUM_WARPS=4`, and
`TS_TRITON_AR_WORKGROUP_SYNC=0` are diagnostic controls only.

All final runs began from an idle KFD snapshot. For ws<8 physical GPU 3/HIP
index 0 was excluded. The final ws=8 campaigns were also checked for foreign KFD
processes during execution.

## 2. Method

Op-level results use HIP events, per-rank medians, and the maximum rank median.
Each row has 30 warmups, 100–150 timed repetitions, a correctness gate, and two
passes. Tables report the mean of pass p50 values.

The baselines answer different questions:

1. Section 3 retains RCCL + eager residual/`F.rms_norm` for historical crossover
   continuity. It is deliberately pessimistic.
2. Section 3 custom-AR rows are standalone small-message context.
3. Section 4 uses the serving-faithful stack: Triton AR through 512 KiB, RCCL
   above it, then TokenSpeed Triton residual-add RMSNorm. At N=2880/bf16, the
   transport switches between M=64 and M=128.

End-to-end runs use random input length 128, output length 512, temperature 0,
ignore EOS, and prompt count four times concurrency. Values are means of two
seed medians. Curated data:
`benchmark/results/ar_rmsnorm_mi350x_e2e/mi350x_latest_e2e_summary.csv`.

## 3. Op-level crossover: RCCL + eager norm proxy

Speedup is RCCL proxy latency divided by fused latency. Above one favors fusion.
Raw data: `mi350x_resolved_cross_pass{1,2}_ws{2,4,8}.csv`.

| M | ws=2 | ws=4 | ws=8 |
|---:|---:|---:|---:|
| 8 | 1.07 | 0.77 | 0.73 |
| 64 | 0.87 | 1.00 | 0.83 |
| 128 | 0.86 | 1.04 | 0.67 |
| 256 | 1.08 | 0.70 | 0.47 |
| 384 | 1.26 | 0.61 | 0.60 |
| 512 | 1.21 | 0.67 | 0.64 |
| 768 | 1.13 | 0.81 | 0.76 |
| 1024 | 1.07 | 0.86 | 0.79 |
| 2048 | 0.99 | 0.77 | 0.89 |

ws=2 is favorable from M=256 through M=1024. ws=4 has only narrow parity near
M=64–128. ws=8 never beats this proxy.

### Custom Triton-AR extras

Only M=8/64 are eligible. `fused/custom >1` means fused is slower. These rows do
not inform the crossover conclusion.

| ws | M | custom unfused ms | fused/custom |
|---:|---:|---:|---:|
| 2 | 8 | 0.0491 | 1.16 |
| 2 | 64 | 0.0493 | 1.10 |
| 4 | 8 | 0.0524 | 1.11 |
| 4 | 64 | 0.0530 | 1.06 |
| 8 | 8 | 0.0518 | 1.14 |
| 8 | 64 | 0.0808 | 0.79 |

## 4. Decomposition against the serving baseline

Speedup is serving-unfused divided by fused-default latency. Above one favors
fusion. Raw data:
`mi350x_resolved_decomp_pass{1,2}_ws{2,4,8}.csv`.

| ws | M | transport | fused ms | unfused ms | speedup |
|---:|---:|:---|---:|---:|---:|
| 2 | 8 | Triton AR | 0.0463 | 0.0577 | 1.25 |
| 2 | 32 | Triton AR | 0.0449 | 0.0596 | 1.33 |
| 2 | 64 | Triton AR | 0.0447 | 0.0598 | 1.34 |
| 2 | 128 | RCCL | 0.0451 | 0.0610 | 1.35 |
| 2 | 256 | RCCL | 0.0437 | 0.0650 | 1.49 |
| 2 | 512 | RCCL | 0.0692 | 0.0786 | 1.14 |
| 2 | 1024 | RCCL | 0.1268 | 0.1303 | 1.03 |
| 4 | 8 | Triton AR | 0.0479 | 0.0615 | 1.29 |
| 4 | 32 | Triton AR | 0.0465 | 0.0619 | 1.33 |
| 4 | 64 | Triton AR | 0.0463 | 0.0629 | 1.36 |
| 4 | 128 | RCCL | 0.0464 | 0.0651 | 1.40 |
| 4 | 256 | RCCL | 0.0767 | 0.0663 | 0.87 |
| 4 | 512 | RCCL | 0.0877 | 0.0683 | 0.78 |
| 4 | 1024 | RCCL | 0.1098 | 0.0893 | 0.81 |
| 8 | 8 | Triton AR | 0.0555 | 0.0609 | 1.10 |
| 8 | 32 | Triton AR | 0.0576 | 0.0604 | 1.05 |
| 8 | 64 | Triton AR | 0.0637 | 0.0760 | 1.19 |
| 8 | 128 | RCCL | 0.0774 | 0.0627 | 0.81 |
| 8 | 256 | RCCL | 0.1112 | 0.0659 | 0.59 |
| 8 | 512 | RCCL | 0.0943 | 0.0656 | 0.70 |
| 8 | 1024 | RCCL | 0.1079 | 0.0786 | 0.73 |

The production baseline reverses the earlier optimistic framing:

- ws=2 wins throughout the sampled range;
- ws=4 wins only through M=128;
- ws=8 wins only while the unfused transport is Triton AR (M<=64).

## 5. End-to-end gpt-oss-120b

TPOT delta is `(fused/unfused)-1`; lower is better. Throughput delta is
`(fused/unfused)-1`; higher is better.

| ws | concurrency | unfused TPOT ms | fused TPOT ms | TPOT delta | unfused tok/s | fused tok/s | throughput delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 8 | 11.57 | 11.60 | +0.3% | 679 | 679 | -0.1% |
| 2 | 16 | 12.89 | 12.78 | -0.9% | 1226 | 1232 | +0.5% |
| 2 | 32 | 15.12 | 14.78 | -2.2% | 2078 | 2137 | +2.8% |
| 4 | 8 | 10.31 | 11.18 | +8.4% | 764 | 706 | -7.6% |
| 4 | 16 | 11.40 | 12.21 | +7.1% | 1384 | 1274 | -7.9% |
| 4 | 32 | 12.40 | 13.06 | +5.4% | 2528 | 2168 | -14.2% |
| 8 | 8 | 10.97 | 12.04 | +9.8% | 716 | 652 | -8.9% |
| 8 | 16 | 12.34 | 13.06 | +5.8% | 1038 | 1208 | +16.4%* |
| 8 | 32 | 13.41 | 13.72 | +2.3% | 2347 | 2286 | -2.6% |

`*` The ws=8 concurrency-16 unfused throughput pair has 15.9% seed spread; TPOT
is stable and shows a regression, so no throughput win is claimed.

All 36 final logs completed with zero request failures and no HIP/HSA fault:

- `bench_final_ws2_{fused,unfused}_...`
- `bench_final_ws4_{fused,unfused}_...`
- `bench_resolve_ws8_{fused_clean,unfused}_...`

## 6. Exact fault causes and controls

### 6.1 ws=8 unfused memory-access fault

The serving path uses TokenSpeed Triton AR at M<=91. Its kernel used four
wavefronts but called a scalar `symm_mem_barrier` before peer loads and before
buffer reuse. The scalar release/acquire did not synchronize sibling wavefronts.

Fix: `symm_mem_workgroup_barrier` brackets the scalar barrier with workgroup
barriers in AMD all-reduce, native fused AR+RMSNorm, and RS/AG kernels.

Single-variable serve control:

- `serve_resolve_ws8_unfused_nosync.log`,
  `TS_TRITON_AR_WORKGROUP_SYNC=0`: readiness followed by memory faults on four
  GPU nodes and fatal abort.
- synchronized Triton AR: full ws=8 unfused campaign completed.

### 6.2 Folded copy-in fault

Folded copy-in writes the coarse symmetric input in phase 0, signals peers, pulls
peer data, and signals that the persistent buffer may be reused. With multiple
wavefronts, a workgroup barrier does not promote every wavefront's memory effects
into the scalar wave's system-scope release/acquire. This caused peer reads or
reuse before all data operations were globally ordered.

Fix: the folded specialization runs with
`TS_TRITON_SHMEM_FOLD_NUM_WARPS=1`. The scalar system release/acquire then orders
the complete program. The non-folded and two-shot paths retain their tuned
multi-wave settings.

Single-variable serve control:

- `serve_resolve_ws8_fused_fold4warp.log`: four-wave fold, immediate memory
  fault after readiness.
- `serve_resolve_ws8_fused_clean.log`: one-wave fold, full campaign completed.

### 6.3 RCCL/watchdog attribution

`rccl_graph_abi_check.py` now captures RCCL itself rather than a compute-only
graph. Eager all-reduce, captured all-reduce replay, and compute graph replay pass
at ws=2/4/8 with blocking wait disabled. Final ws=4/8 serving also passes without
blocking wait.

Therefore the refreshed faults were not RCCL failures and do not require
`TORCH_NCCL_BLOCKING_WAIT=1`. The old HIP 7.2.26015 event-query defect remains a
valid historical issue, fixed by the loaded 7.2.4 runtime.

### 6.4 Host fallback correctness

If fused state creation or eligibility declined, `RMSNorm` previously normalized
rank-local partials because the caller had already deferred its all-reduce.
The fallback now explicitly executes the production `AutoBackend` all-reduce
before residual-add RMSNorm. A spawned two-rank regression forces the decline and
checks the full result.

### 6.5 Hardware and environment exclusion

The final qualification recorded no foreign KFD processes. MI350X ECC counters
were zero on all eight GPUs, and captured RCCL passed every world size. Disabled
barrier controls faulted on different GPU-node subsets, while the synchronized
controls passed on the same devices. This rules against a fixed bad GPU or a
size-dependent RCCL failure as the cause of the reproduced faults.

The host still uses `amdgpu.noretry=1` and does not expose useful XGMI error
counters through `amd-smi`; those are platform limitations, not required
workarounds for the resolved configurations.

## 7. Validation and deployment

Passed:

- 10 communication tests across ws=1/2/4/8, including folded ws=4 graph stress
  and genuine ws=8 two-shot graph capture;
- forced fused-decline fallback regression;
- captured RCCL at ws=2/4/8 without blocking wait;
- full final ws=2/4/8 fused and unfused e2e matrix;
- two-pass crossover and production-baseline decomposition.

Deployment defaults:

- AMD TP=2: fusion auto-enables.
- AMD TP=4/8: fusion remains explicitly available but no longer auto-enables,
  because current e2e data show regressions.
- DP, overlap depth greater than one, or speculative decode still require a
  fixed `TS_TRITON_SHMEM_BARRIER_GRID` and separate validation.

Next optimization work should target the M=256 barrier expansion and two-shot
cost before broadening fusion at ws=4/8.
