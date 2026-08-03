# GLM-5.2-FP8 fused AR+RMSNorm status

Updated: 2026-08-03

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

## Current operator result

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

## Pending definitive sweep

The [definitive sweep contract](../studies/mi350x/2026-08-glm-5.2-fp8-definitive-sweep/README.md)
compares upstream-unfused, default Iris fused, and forced padded Triton at
WS=2/4/8. It densely samples M=1-48 around both profitability borders and
retains larger loss points through M=256.

The campaign has **not been run**. Do not replace this decision with planned
results. After collection, WS=8 is the model-faithful result; WS=2/4 are scaling
evidence only.

## Remaining promotion gates

1. Complete the predeclared definitive operator sweep and retained-failure
   analysis.
2. Restore bounded GLM graph serving and capture authoritative
   `tokenspeed.model_forward.v1` markers.
3. Compare executed decode M with the M=2-42 operator window.
4. Run restart-randomized end-to-end serving after the FP8 GEMM/MoE baseline is
   representative.

## Evidence

- [Representative WS=8 baseline](../studies/mi350x/2026-08-glm-5.2-fp8-baseline/README.md)
- [Definitive sweep contract](../studies/mi350x/2026-08-glm-5.2-fp8-definitive-sweep/README.md)
- [Benchmark and promotion methodology](benchmark-methodology-recommendations-2026-07.md)
- [Profiling workflow](profiling-workflow.md)
