# AR+RMSNorm model profiles

Profiles make model paths, hidden sizes, serving arguments, and validated
performance policies explicit.

- `gpt_oss_120b_mi350x.env` is the qualified gpt-oss-120B MI350X profile.
- `model_template.env` is a conservative starting point for another model.

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

