# AR+RMSNorm model profiles

Profiles make model paths, hidden sizes, serving arguments, and validated
performance policies explicit.

- `gpt_oss_120b_mi350x.env` is the qualified gpt-oss-120B MI350X profile.
- `model_template.env` is a conservative starting point for another model.

The qualified manual-serving profile emits
`AR_NORM_PROFILE_ID=gpt-oss-120b-mi350x-qualified-v4` and
`TS_TRITON_SHMEM_OUTPUT_RING=72`. Campaign runs also validate the resolved
server arguments; the profile ID alone does not hide command-line overrides.
Its graph-lifetime safety gates passed, but TP4 fusion remains opt-in because
the complete campaign rejected it on decode performance.

Usage:

```bash
source benchmark/profiles/ar_rmsnorm/gpt_oss_120b_mi350x.env
bash benchmark/e2e_arnorm_serve.sh 4 1,2,3,5 2048 triton_shmem
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

