# AR+RMSNorm model profiles

Profiles make model identity, topology, serving arguments, allocation policy,
and validated dispatch boundaries explicit. A known-profile mismatch declines
`triton_shmem` before state creation and uses the complete unfused path.

## Available profiles

### GPT-OSS-120B MI350X

`gpt_oss_120b_mi350x.env` defines the TP=4/N=2880 core-v3 profile on HIP
`1,2,5,6`. It uses 72 graph-stable input/output sites, padded whole-row decode
through M64, blocked one-shot through M384, and ordinary fallback for larger
captured calls. Its gfx950 grid policy is profile-owned.

The profile is safety-qualified, but default-compatible performance did not
promote. See the
[GPT-OSS status](../../results/ar_rmsnorm/docs/gpt-oss-120b-status.md).

The completed current-machine definitive campaign used
`gpt_oss_120b_mi350x_definitive_ws{2,4,8}.env`. These wrappers require an
explicit preflight-qualified device set and bind profile identity to world size
and an optional actual-M gate (`0`, `64`, `91`, or `384`). They keep the 2048
workspace cap and 72-site ownership contract. The matrix selected gate `0` for
all world sizes; WS2 was inconclusive, WS4 passed only the five-pair screen, and
WS8 lost. These campaign wrappers are not deployment-qualified profiles. See
the [definitive study](../../results/ar_rmsnorm/studies/mi350x/2026-08-gpt-oss-120b-definitive-sweep/README.md).

### GLM-5.2-FP8 MI350X

`glm_5_2_fp8_mi350x.env` defines the diagnostic TP=8/N=6144 profile v2. It uses
156 graph-stable input/output sites and a four-warp padded whole-row kernel for
M=2-42. M=1 and M>=43 take the ordinary path.

Captured operator graphs and transitions are qualified; captured production
serving is not. See the
[GLM status](../../results/ar_rmsnorm/docs/glm-5.2-fp8-status.md) and the
[definitive sweep contract](../../results/ar_rmsnorm/studies/mi350x/2026-08-glm-5.2-fp8-definitive-sweep/README.md).

## Usage

```bash
source benchmark/profiles/ar_rmsnorm/<profile>.env
ENABLE_ALLREDUCE_FUSION=1 \
  bash benchmark/e2e_arnorm_serve.sh \
    "$AR_NORM_WORLD_SIZE" "$AR_NORM_DEVICES" \
    "$COMM_FUSION_MAX_NUM_TOKENS" triton_shmem
```

Profiles retain normal serving defaults unless the model status explicitly
documents otherwise. Benchmark-only controls and forced diagnostic boundaries
belong in campaign specifications, not deployment profiles.

For another model:

1. copy `model_template.env`;
2. set model, topology, width, cap, and site identity;
3. run correctness, graph, and shared-state transition gates;
4. run bounded model serving and marker-aligned traces;
5. run matched restart-randomized end-to-end comparisons;
6. promote only settings supported by the complete evidence ladder.

No profile is a universal kernel recommendation.

