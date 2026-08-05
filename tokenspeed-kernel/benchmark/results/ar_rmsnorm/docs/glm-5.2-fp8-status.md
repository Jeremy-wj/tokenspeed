# GLM-5.2-FP8 fused AR+RMSNorm status

Updated: 2026-08-05

This is the live decision page for GLM-5.2-FP8, TP=8, on MI350X.

## Deployment decision

Keep explicit upstream-unfused as the default:

```text
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
TS_ARNORM_BACKEND=auto
ENABLE_ALLREDUCE_FUSION=0
--disable-allreduce-fusion
```

Profile v2 is a diagnostic operator candidate, not a deployment profile.
Captured production serving and end-to-end performance remain unqualified.

## Current MI350X operator result

At WS=8, N=6144, bf16, and 156 calls per captured graph:

- four-warp padded Triton is 3.4%-20.9% faster than upstream-unfused for
  M=2-42;
- M=1 loses and takes ordinary fallback;
- M=33 remains 9.5% faster after removing the old profile-v1 policy cliff;
- at M=43, unfused switches from ordinary Iris to RCCL and padded Triton is
  25.6% slower;
- the M=2-42 synthetic-forward saving is 0.22-0.84 ms.

The exact-profile M42 measurements are noisier than M=2-33: three clean runs
favor Triton, while one retained run entered a high-tail mode. This does not
change the current gate, but it requires confirmation.

## Candidate profile v2

Source of truth:
`benchmark/profiles/ar_rmsnorm/glm_5_2_fp8_mi350x.env`.

```text
profile: glm-5.2-fp8-mi350x-triton-v2
world size / hidden size: 8 / 6144
input / output sites: 156 / 156
M=1: ordinary fallback
M=2-42: padded 8192-lane whole-row, four warps
M>=43: ordinary fallback
```

The 1,000-replay shared-state transition matrix covers M
`1,2,16,32,33,40,42`, odd/even graphs, and changing inputs. All eight ranks
agree on dispatch; 1,649 checked operations/rank passed with zero failed steps.

Keeping M dynamic removes the pathological specialized-M1 code shape. A bounded
WS=4 check improved Triton by 21.8%, but it remained 3.9% behind unfused, so M1
continues to fall back.

## Completed cross-machine definitive sweep

The [definitive sweep contract](../studies/mi350x/2026-08-glm-5.2-fp8-definitive-sweep/README.md)
compares upstream-unfused, default Iris fused, and forced padded Triton at
WS=2/4/8. It densely samples M=1-48 around both profitability borders and
retains larger loss points through M=256.

The immutable 315-process matrix completed on 8x MI355X/gfx950, not MI350X.
At model-faithful WS=8 with 156 sites:

- raw replay favors padded Triton at every measured M from 1 through 42;
- reset-copy-adjusted timing favors M1-M40, is effectively tied at M41
  (+0.17%), and loses at M42 (+2.19%);
- M43 switches upstream to RCCL and makes padded Triton 33.6% slower;
- forced padded Triton beats fused Iris at every measured M, including
  post-cap diagnostic rows.

WS2/WS4 remain scaling evidence only. Three failed attempts were retained and
passed on bounded resume: two RCCL teardown timeouts fixed by releasing captured
graphs before communicator destruction, and one transient WS8/M40 ordinary-Iris
validation failure. The final matrix is complete, and no foreign GPU process
was observed.

This result does not alter the MI350X profile. In particular, MI355X's M1 win
does not overturn MI350X's measured M1 loss, and adjusted MI355X M41/M42 results
do not rewrite the pre-existing MI350X M42 cap. It also remains fixed-shape
operator evidence rather than serving qualification.

## Remaining promotion gates

1. Repeat the matrix on MI350X or independently define and qualify an
   MI355X-specific profile before changing hardware-scoped operator policy.
2. Restore bounded GLM graph serving and capture authoritative
   `tokenspeed.model_forward.v1` markers.
3. Compare executed decode M with the M=2-42 operator window.
4. Run restart-randomized end-to-end serving after the FP8 GEMM/MoE baseline is
   representative.

## Evidence

- [Representative WS=8 baseline](../studies/mi350x/2026-08-glm-5.2-fp8-baseline/README.md)
- [Completed definitive sweep](../studies/mi350x/2026-08-glm-5.2-fp8-definitive-sweep/README.md)
- [Benchmark and promotion methodology](benchmark-methodology-recommendations-2026-07.md)
- [Profiling workflow](profiling-workflow.md)
