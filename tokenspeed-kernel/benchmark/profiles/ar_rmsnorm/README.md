# AR+RMSNorm model profiles

Profiles make model paths, hidden sizes, serving arguments, and validated
performance policies explicit.

The GPT-OSS profile was requalified after the `3f88dcc2` rebase. The model
template remains unqualified until a model completes the evidence ladder.

- `gpt_oss_120b_mi350x.env` is the post-rebase GPT-OSS-120B MI350X profile.
- `model_template.env` is a conservative starting point for another model.

The profile emits
`AR_NORM_PROFILE_ID=gpt-oss-120b-mi350x-post-rebase-v1` and
`TS_TRITON_SHMEM_OUTPUT_RING=72`. Campaign runs also validate the resolved
server arguments; the profile ID alone does not hide command-line overrides.
The canonical deployment arm explicitly disables fusion; both Iris and
`triton_shmem` failed post-rebase performance promotion.

Usage:

```bash
source benchmark/profiles/ar_rmsnorm/gpt_oss_120b_mi350x.env
ENABLE_ALLREDUCE_FUSION=0 \
  bash benchmark/e2e_arnorm_serve.sh 4 1,2,3,5 2048 auto
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

