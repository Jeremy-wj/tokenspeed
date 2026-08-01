# AR+RMSNorm model profiles

Profiles make model paths, hidden sizes, serving arguments, and validated
performance policies explicit.

The GPT-OSS profile was requalified after the `3f88dcc2` rebase. The model
template remains unqualified until a model completes the evidence ladder.

- `gpt_oss_120b_mi350x.env` is the post-rebase GPT-OSS-120B MI350X profile.
- `model_template.env` is a conservative starting point for another model.

The profile emits
`AR_NORM_PROFILE_ID=gpt-oss-120b-mi350x-triton-core-v3`,
`TS_TRITON_SHMEM_OUTPUT_RING=72`,
`TS_TRITON_SHMEM_INPUT_SITE_RING=72`, and
`TS_TRITON_SHMEM_BORROW_TWOSHOT_OUTPUT=1`. It selects padded whole-row decode
for M<=64 with four warps, blocked one-shot through M384, and eager two-shot
above that. Captured calls above M384 use the complete ordinary fallback. The
gfx950/TP=4 grid cap is profile-owned (`128` from M256), not a generic backend
default.

The profile does not override TokenSpeed's normal graph, memory, overlap, or
health behavior. Prefill graphs, the default decode capture ladder, 0.95 HBM
utilization, overlap scheduling, and standard generated health probes remain
enabled. Campaign runs validate those resolved server arguments as well as
architecture, topology, rank set, hidden size, dtype, token cap, rings, and
kernel policy. A mismatch declines triton-shmem before state creation and uses
the complete unfused path.

Core-v3's original restricted promotion is historical. Default-compatible
requalification did not promote, so explicit unfused is deployment default,
control, and fallback. Base overlap remains supported; disablement is only the
stable performance-measurement policy. The
[live status](../../results/ar_rmsnorm/docs/gpt-oss-120b-status.md) owns metrics
and deployment policy.

Usage:

```bash
source benchmark/profiles/ar_rmsnorm/gpt_oss_120b_mi350x.env
ENABLE_ALLREDUCE_FUSION=1 \
  bash benchmark/e2e_arnorm_serve.sh 4 1,2,5,6 2048 triton_shmem
# Add --disable-overlap-schedule only for matched performance reproduction.
```

For a new hidden size:

1. copy `model_template.env`;
2. set model identity and hidden size;
3. run the serving-faithful operator and graph probes;
4. run shared-state multigraph transitions;
5. run matched fused/unfused e2e;
6. promote grid/threshold overrides only after all gates pass.

Historical N=2880 data remains a gpt-oss regression profile, not a universal
kernel recommendation.

