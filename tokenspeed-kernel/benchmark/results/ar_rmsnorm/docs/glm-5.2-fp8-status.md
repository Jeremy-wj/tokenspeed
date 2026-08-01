# GLM-5.2-FP8 fused AR+RMSNorm status

Updated: 2026-08-01

This is the live decision page for GLM-5.2-FP8, TP=8, on 8x MI350X.

## Deployment decision

Keep explicit upstream-unfused as the default:

```text
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
TS_ARNORM_BACKEND=auto
ENABLE_ALLREDUCE_FUSION=0
--disable-allreduce-fusion
```

The `glm-5.2-fp8-mi350x-triton-v1` profile is a diagnostic candidate, not a
deployment profile. It improves the 156-call M32 operator graph by 9.61%, but a
five-pair one-restart serving canary measured +0.59% mean TPOT and did not clear
the evidence or performance gates.

The current GLM FP8 model path is itself the dominant blocker. The bounded
fusion-off concurrency-16 screen measured 0.212 output tokens/s and 84.551 s
median TPOT. Do not run a fifteen-pair AR campaign until the AMD FP8 GEMM/MoE
baseline is made representative.

## Candidate profile

Source of truth:
`benchmark/profiles/ar_rmsnorm/glm_5_2_fp8_mi350x.env`.

```text
M<=32: padded 8192-lane whole-row one-shot, four warps
input sites: 156
output sites: 156
world size: 8
hidden size: 6144
serving fusion cap: 32
```

Synthetic safety evidence:

- full 156-call graph, 1,000 replays, changing inputs: pass;
- shared-state transition matrix, 1,000 replays: pass;
- M `1,8,16,31,32`, odd/even graphs: pass;
- 1,690 checked operations/rank, zero failed steps;
- exact profile resolution on all eight ranks in bounded eager serving.

Captured production serving is not qualified. The working server profile uses
eager execution because normal GLM graph startup did not reach bounded
readiness.

## Required serving controls

The only bounded configuration on the current image required:

```text
tokenspeed-scheduler rebuilt with TOKENSPEED_FLAT_KVCACHE=OFF
--moe-backend triton
--enforce-eager
--max-model-len 80000
--max-num-seqs 16
--disable-kvstore
--kvstore-ratio 0
TOKENSPEED_DEEP_HEALTH_MODE=passive
TOKENSPEED_PASSIVE_HEALTH_STUCK_SEC=3600
```

These controls establish a diagnostic baseline; they are not production
recommendations.

## Evidence

- [GLM-5.2-FP8 baseline study](../studies/mi350x/2026-08-glm-5.2-fp8-baseline/README.md)
- [Benchmark and promotion methodology](benchmark-methodology-recommendations-2026-07.md)
- [Profiling workflow](profiling-workflow.md)

## Next steps

1. Optimize or replace the AMD block-FP8 GEMM and Triton FP8 MoE path.
2. Restore bounded graph serving and capture authoritative forward markers.
3. Requalify the 156-site lifetime contract in real captured model execution.
4. Only then run three restart blocks and fifteen paired observations.
