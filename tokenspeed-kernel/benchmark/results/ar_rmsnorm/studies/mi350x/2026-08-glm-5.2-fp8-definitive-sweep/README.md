# GLM-5.2-FP8 definitive AR+RMSNorm sweep

Status: **complete on 8x MI355X; operator evidence only**

This is the completed captured-operator comparison from the immutable
[campaign](campaign.json). The matrix was predeclared for MI350X/gfx950 and was
executed without shape or arm changes on MI355X/gfx950. The hardware change is
part of the result identity: these numbers do not requalify the MI350X profile.

Exact values and provenance are in [summary.json](summary.json),
[graph-sweep-summary.json](graph-sweep-summary.json), and
[graph-sweep.csv](graph-sweep.csv).

## Decision

At model-faithful WS=8/N=6144 with 156 captured sites:

- padded Triton is 2.9%-23.0% faster than raw upstream-unfused at every
  measured M from 1 through 42;
- after removing the benchmark-only reset copy, the profitable values are
  M=`1,2,4,8,16,24,32,36,40`; M41 is effectively tied (+0.17%) and M42 loses
  by 2.19%;
- M43 changes upstream transport from ordinary Iris to RCCL and padded Triton
  becomes 33.6% slower; losses grow through +415.4% at M256;
- forced padded Triton is faster than default fused Iris at every measured M,
  including forced post-cap rows, but that is not a deployment claim.

Keep explicit upstream-unfused as the deployment default. The measured raw
border supports the existing M42 upper bound as operator screening, while the
reset-copy-adjusted border and 7.49% M40 pass spread argue against widening or
promoting a profile. MI355X makes M1 profitable, unlike the prior MI350X
baseline, but cross-machine evidence does not justify changing the MI350X M2
lower gate.

WS2/WS4 remain scaling diagnostics. Their non-monotonic frontiers reflect
different collective scaling and the same M42-to-M43 Iris/RCCL switch.

## Contract and completion

```text
predeclared hardware: MI350X gfx950
actual hardware:      8x MI355X gfx950
dtype / hidden:       bf16 / 6144
world sizes:          2, 4, 8
primary / diagnostic: 156 / 1 sequential sites
warmups / replays:    50 / 1000
statistic:             max rank per iteration, then p50/p95/p99
process isolation:    one fresh process per arm and shape
```

The retained matrix contains all 315 results and no incomplete arm triple.
Collection used 318 process attempts: 315 completed and three failed attempts
were retained before successful identity-validated retries. Two resumptions
were required. Authoritative collection, diagnosis, cooldowns, and resumes took
7,179.68 seconds (1:59:39.68).

Frozen device sets:

```text
WS2: HIP 1,2
WS4: HIP 1,2,4,5
WS8: HIP 0,1,2,3,4,5,6,7
```

All selected pairs are one-hop coherent bidirectional XGMI in the retained
topology audit. Shared-host preflight and nonintrusive KFD snapshots found no
foreign GPU process.

## Primary frontiers

Values below are the exact measured profitable M sets for the 156-site graph.
“Adjusted” uses each arm's serving-faithful estimate where available; here it
subtracts the changing-input reset copy from upstream-unfused while Triton is
unchanged. Raw replay remains the predeclared headline.

| WS | Triton vs unfused, raw | Triton vs unfused, adjusted | Triton vs Iris |
|---:|---|---|---|
| 2 | every sampled M, 1-256 | every sampled M at or above 4 | 1-128; loss at 256 |
| 4 | 2,4,8,16,24,32,36,40,43,44,48 | 4,8,16,24,43,44,48 | every measured M |
| 8 | 1-42 | 1-40 | every measured M |

At WS4, raw M41/M42 are small losses (+1.21%/+1.32%), M43-M48 become wins
after upstream switches to slow WS4 RCCL, and M64 is the first sustained loss.
The exact non-contiguous lists are preserved in `summary.json`.

## Model-faithful WS8 rows

P50 values are microseconds per site. Negative percentages favor Triton.

| M | Unfused | Iris | Triton | vs unfused | adjusted | vs Iris |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 21.140 | 21.824 | 18.457 | -12.69% | -5.21% | -15.43% |
| 2 | 22.175 | 22.138 | 17.819 | -19.64% | -13.09% | -19.51% |
| 40 | 38.546 | 40.915 | 35.486 | -7.94% | -2.97% | -13.27% |
| 41 | 39.343 | 41.917 | 37.422 | -4.88% | +0.17% | -10.72% |
| 42 | 39.722 | 42.352 | 38.580 | -2.87% | +2.19% | -8.90% |
| 43 | 28.305 | 44.201 | 37.814 | +33.59% | +43.66% | -14.45% |
| 64 | 27.555 | 58.946 | 53.426 | +93.89% | +109.68% | -9.36% |
| 256 | 46.646 | 262.247 | 240.419 | +415.42% | +450.19% | -8.32% |

The raw 156-site forward delta is -0.418 ms at M1, -0.680 ms at M2,
-0.477 ms at M40, -0.178 ms at M42, and +1.483 ms at M43.

## Transport and repeatability

M42 is 516,096 bytes and uses
`ordinary_iris_all_reduce+triton_residual_rmsnorm`. M43 is 528,384 bytes and
uses `rccl_all_reduce+triton_residual_rmsnorm`. All 105 candidate artifacts
resolved `oneshot_wholerow_padded`; all 105 fused-control artifacts resolved
Iris. Upstream produced 63 ordinary-Iris and 42 RCCL artifacts.

Maximum order-opposed p50 spread by WS:

| WS | Arm / M | Spread |
|---:|---|---:|
| 2 | Triton / 44 | 4.66% |
| 4 | Iris / 1 | 3.21% |
| 8 | Triton / 40 | 7.49% |

WS8 Triton M43 also spread 6.45%. All other confirmed rows were below 5%.

## Retained incidents and fixes

1. The first 20-process root revealed that the refactored WS2 path ignored the
   forced padded variant and executed blocked one-shot. That root is retained
   but excluded. Explicit WS2 variant dispatch was fixed and a fresh raw root
   was started.
2. WS2 upstream-unfused M43 and M44 timed out during teardown. The measured
   work had completed, but captured RCCL graphs remained alive while their
   communicator was destroyed. Releasing graphs before process-group teardown
   made bounded one-site and 156-site reproducers pass; both campaign rows then
   passed on resume.
3. The first WS8 upstream-unfused M40 attempt failed eager validation on rank 7
   for 350/245,760 values (0.1%; maximum relative error 0.02265 against 0.02).
   The order-opposed pass and isolated retry passed. The failure remains part of
   the evidence.

The source identity is commit `6e788124` plus the retained local architecture,
WS2 dispatch, and RCCL teardown fixes. Full source hashes, image/runtime,
model, topology, GPU serials, counts, and incident records are in
`summary.json`. After collection, resume validation was hardened to reject a
forced arm whose backend matches but whose kernel path does not.

## Reproduction

The qualified MI355X image/container and local model path were:

```text
container: jeremwan-ar-rmsnorm-profiler-mi355x-6e788124
image ID: sha256:554132d539e81c68245026124b99c296806213926a543b0816c8dcc38cef9114
model: /data/models/glm-5.2-fp8
raw root: benchmark/results/ar_rmsnorm/raw/current/glm-5.2-fp8/
          mi355x/2026-08-05/operator/definitive-graph-v1-attempt2
```

From the `tokenspeed-kernel` root inside that container:

```bash
source benchmark/profiles/ar_rmsnorm/glm_5_2_fp8_mi350x.env
export MODEL_PATH=/data/models/glm-5.2-fp8

PYTHONPATH=. python3 benchmark/run_ar_rmsnorm_graph_sweep.py \
  --spec benchmark/results/ar_rmsnorm/studies/mi350x/\
2026-08-glm-5.2-fp8-definitive-sweep/campaign.json \
  --devices 2=1,2 \
  --devices 4=1,2,4,5 \
  --devices 8=0,1,2,3,4,5,6,7 \
  --output-root benchmark/results/ar_rmsnorm/raw/current/glm-5.2-fp8/\
mi355x/2026-08-05/operator/definitive-graph-v1-attempt2
```

The immutable campaign remains `status=planned` because the resumable runner
requires that specification state; measured status belongs to this page and
`summary.json`.
