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
For ws=4 blocked one-shot calls, the integration now caps the grid at 128 CTAs
from M>=256; this removes the M=256 barrier-participant expansion without
changing smaller-M launches.

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
TS_TRITON_SHMEM_GRID_CAP=-1             # auto: ws4 cap=128, other ws uncapped
TS_TRITON_SHMEM_GRID_CAP_MIN_M=-1       # auto: activate at M=256
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
The ws4 M=256 cell incorporates the later `ar_rmsnorm_opt2` selective-cap result.

| M | ws=2 | ws=4 | ws=8 |
|---:|---:|---:|---:|
| 8 | 1.07 | 0.77 | 0.73 |
| 64 | 0.87 | 1.00 | 0.83 |
| 128 | 0.86 | 1.04 | 0.67 |
| 256 | 1.08 | 0.88 | 0.47 |
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
The ws4 M=256 fused value incorporates the direct two-pass selective-cap result.

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
| 4 | 256 | RCCL | 0.0612 | 0.0663 | 1.08 |
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
- ws=4 wins through M=256 after the selective grid cap;
- ws=8 wins only while the unfused transport is Triton AR (M<=64).

### 4.1 M=256 integration optimization

The M=256 cliff was not reduction math. On gfx950 ws=4 the blocked one-shot grid
grew from 128 to 256 CTAs, and every CTA executed two system-scope cross-rank
barriers. Two-pass focused results:

| ws4 width N | uncapped M=256 ms | cap=128 ms | change |
|---:|---:|---:|---:|
| 1536 | 0.0591 | 0.0547 | -7.5% |
| 2880 | 0.0760 | 0.0612 | **-19.5%** |
| 5120 | 0.0965 | 0.0868 | -10.0% |
| 7168 | 0.1232 | 0.1147 | -6.9% |

The shipping cap activates only at M>=256. A blanket cap regressed M=192 and
larger widths below M=256, while prior fixed-grid attempts charged every small-M
call for idle barrier participants. This selective dynamic cap preserves
M-dependent participation and pure-TP graph semantics. N=512 already uses the
whole-row kernel's 64-CTA cap and is unchanged.

Raw data: `benchmark/results/ar_rmsnorm_opt2/pass{1,2}_ws4_{prod,cap128}.csv`.

The final ws8 sweep covered N={512,1536,2880,5120,7168}. No common cap was safe:
cap=128 strongly helped N=1536 at M=224–256, was only a borderline ~3% win for
N=2880 at M=256, and regressed N=5120/7168. N=512 cap results were order-sensitive.
Therefore ws8 remains uncapped rather than adding width-specific defaults without
model-level serving gates.

### 4.2 Two-shot integration follow-up

The current two-shot wrapper pays copy-in, two one-block barriers, the tuned
kernel, and two copy-outs. Several integration-only alternatives were measured:

- a custom paired copy-out kernel was slower than two `copy_` calls;
- `torch._foreach_copy_` saved about 3 µs at M=512 but regressed M=1024 by
  roughly 10 µs;
- returning ping-pong symmetric outputs removed about 13 µs in microbenchmarks
  but faulted under captured serving because the borrowed residual lifetime
  crossed later fused calls;
- in-kernel/fixed-grid two-shot barriers remain slower, as in the prior sweep.

Forced-path data found large isolated width-specific crossovers:

| ws | N=512 | N=1536 | N=2880 | N=5120 | N=7168 |
|---:|---:|---:|---:|---:|---:|
| 4 | >=2048 | 896 | 384 | 256 | 192 |
| 8 | >=2048 | 384 | 192 | 160 | 160 |

These are operator bounds, not shipping settings. Raising gpt-oss N=2880 to
M=384 again completed the early loads but faulted during the concurrency-32
serving arm. The conservative global threshold therefore remains 256. No unsafe
or shape-regressing two-shot change was enabled.

The remaining safe opportunities require an explicit runtime buffer-lifetime
contract (producer-direct symmetric input or caller-owned symmetric outputs),
not another barrier/grid retry.

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

The final conservative ws4 cap was rechecked at concurrency 32:

- fused TPOT: 12.60 ms mean (12.81/12.39);
- fused output: 2482 tok/s mean (2444/2520);
- 3.5% lower TPOT and 14.5% higher output throughput than the prior fused arm;
- still 1.7% higher TPOT than unfused, so ws4 remains opt-in.

Sources: `bench_last_ws4_conservative_c32_seed{0,1}.log`.

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
- disjoint and interleaved TP subgroups on world size 4;
- width/grid/path sweeps across N={512,1536,2880,5120,7168}.

Allocator independence was also probed. Torch 2.11 reports expandable segments
unsupported on this ROCm platform, so a dedicated pluggable allocator would add
packaging and correctness risk without changing current steady-state behavior.
It remains a future portability item rather than a performance optimization.

Fixed-grid DP/speculative work was not promoted: fixed participants avoid one
deadlock mode but cannot make mismatched TP collective sequences semantically
correct, and every measured fixed grid regressed latency.

Deployment defaults:

- AMD TP=2: fusion auto-enables.
- AMD TP=4/8: fusion remains explicitly available but no longer auto-enables,
  because current e2e data show regressions.
- DP, overlap depth greater than one, or speculative decode still require a
  fixed `TS_TRITON_SHMEM_BARRIER_GRID` and separate validation.

This completes the pre-profiling optimization sweep. Further work should start
from traces: measure real `(M,N,path)` frequency and producer/consumer lifetimes
before introducing a caller-owned symmetric-buffer contract. Barrier folding,
generic paired-copy retries, mutable borrowed outputs, and isolated
width-specific thresholds should not be revisited without new trace evidence.

## 8. July 2026 end-to-end profiling follow-up

Initial profiling attempts failed because torch loaded its bundled ROCm 7.2.0
`libroctracer64.so` beside system ROCm 7.2.4 HIP/HSA/ROCTX. Relocating that last
bundled tracing library fixes torch/Kineto and eager Proton. The profiler-qualified
artifact and exact linkage are documented in the environment history:

```text
jeremwan/tokenspeed:rocm7.2.4-torch2.11-profiler
sha256:ad3ea3f8cae8ca38cf12824b15c606d0630118c6e04b4087e191b04619a6c135
```

The usable evidence now includes:

- matched eager Proton controls retained from the first pass;
- normal graph-serving torch CPU+GPU Chrome traces at TP=2 and TP=4;
- Proton rocprofiler graph tree/Hatchet output with explicit replay scopes.

Physical GPU 3 remained excluded. The torch traces are the primary production
timeline. Proton graph support is complementary: tree/Hatchet works, while
trace/Chrome graph output remains unsupported or incomplete in the installed
Proton build.

### 8.1 Matched eager results

At concurrency 32:

- TP=2, input 128/output 64: fused TPOT 96.03 ms versus unfused 166.13 ms
  (-42.2%); output throughput 304.22 versus 183.57 tok/s.
- TP=4, input 128/output 32: fused TPOT 156.18 ms versus unfused 118.78 ms
  (+31.5%); output throughput 180.10 versus 225.63 tok/s.

The direction matches the production policy: TP=2 benefits, while TP=4 remains
slower end to end. The eager magnitudes must not be projected onto graph serving.

### 8.2 Shape, path, and launch frequency

A narrow communication Proton scope now records `M`, `N`, world size, folded
copy-in, and the selected one-shot/two-shot path. Per rank:

- TP=2: 1,080 calls at `M=32,N=2880` (93.8%) and 72 at
  `M=64,N=2880` (6.2%);
- TP=4: 1,080 calls at `M=32,N=2880` (93.8%) and 72 at
  `M=128,N=2880` (6.2%);
- every observed fused call used `oneshot_blocked` with folded copy-in;
- no observed call used two-shot.

The longer TP=2 window contained 2,304 fused AR+RMSNorm launches per rank. Its
matched unfused window contained about 2,263 custom Triton AR launches and 2,409
separate RMSNorm launches. TP=4 similarly reduced roughly 2,336 separate AR/norm
launches to 1,152 fused launches in the scoped window.

GPU kernel residency is rank-skewed because the scalar/system barriers wait for
peers. In the TP=4 fused trace, median one-shot kernel duration was 32.6 us on
rank 0 and 836-874 us on ranks 1-3. The unfused custom AR showed the same class
of skew (23.8 us to 1,206 us across ranks). Therefore the TP=4 e2e regression is
not evidence that reduction arithmetic or a particular fused barrier retry is
the next optimization. These eager residency values are superseded for production
graph diagnosis by §8.3.

### 8.3 Production graph-serving traces

The corrected torch/Kineto environment produced complete traces on every rank:

- TP=2 fused: 1,152 fused kernels per rank; median duration 20.3-21.3 us and
  p95 43.5-45.7 us.
- TP=4 fused: 1,080 fused kernels per rank; median duration 34.2-44.3 us and
  p95 49.0-78.7 us.
- TP=4 unfused: 1,095 custom AR kernels and 1,241 RMSNorm kernels per rank.
  Median custom AR was 24.8-31.9 us and RMSNorm 5.8-6.2 us.

At TP=4, the max-rank fused median (44.3 us) is about 16% above the max-rank
sum of unfused AR plus RMSNorm medians (about 38.1 us). The max-rank accumulated
window is similarly 55.9 ms fused versus about 47.3 ms unfused. This directly
supports the small production e2e regression in §5 and corrects the eager-only
interpretation: production graph replay does not show the extreme multi-rank
residency skew seen in eager profiling.

The next kernel optimization target is therefore narrow: TP=4, `N=2880`,
decode `M=32`, blocked one-shot under graph replay. Changes must beat the
serving-faithful unfused AR+RMSNorm sum without regressing TP=2 or M=64/128.
The trace still provides no reason to alter two-shot behavior.

### 8.4 Decision

No kernel policy changes are promoted from this profiling pass:

- keep TP=2 auto-enable and TP=4/8 opt-in;
- keep the M<=256 one-shot overlay, single-wave folded copy-in, and selective
  TP=4 grid cap;
- do not raise the two-shot threshold or build a caller-owned two-shot output
  contract for this workload: two-shot did not occur in the measured windows;
- do not revisit multi-wave folding, generic paired copies, or broad fixed-grid
  policy; any new experiment should target the traced TP=4 M=32 graph path.

Additional Proton eager runs are not needed for the current decision. The
corrected torch environment now preserves production graph and overlap behavior.
A producer-direct symmetric-input contract remains a future option only after a
separate lifetime trace establishes caller-owned input/output lifetimes.

Artifacts:

- `/home/jeremwan/ar_rmsnorm_profiles/ws2_{fused,unfused}_proton_eager/`
- `/home/jeremwan/ar_rmsnorm_profiles/ws2_fused_scoped/`
- `/home/jeremwan/ar_rmsnorm_profiles/ws4_{fused,unfused}_proton_eager/`
- `/home/jeremwan/ar_rmsnorm_profiles/final_profiler_image/ws4_{fused,unfused}_torch/`
- `/home/jeremwan/ar_rmsnorm_profiles/env_rocprofiler/proton-graph-*.hatchet`
- `/home/jeremwan/ar_rmsnorm_e2e/logs/bench_profile_ws{2,4}_*_proton_eager_seed0.log`
