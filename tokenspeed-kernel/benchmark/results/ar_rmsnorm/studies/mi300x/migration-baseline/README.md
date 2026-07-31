# MI300X migration baseline

Legacy migration-era operator evidence across world sizes, model widths, and
noise-controlled repeats. It predates the MI350X serving work and upstream
`3f88dcc2`; do not transfer its numeric conclusions to current deployment.

- `triton_shmem_bench.csv` compares RCCL-unfused, triton-shmem, and native
  symmetric-memory paths across broad shapes.
- `ar_rmsnorm_extended_range.csv` extends model-targeted WS=8 shapes.
- `ar_rmsnorm_model_targeted/` and `ar_rmsnorm_noise_controlled/` retain paired
  repeat sweeps for WS=2/4/8.

The value of this study is historical algorithm coverage, especially the
power-of-two widths that later explained why N=2880 needed a padded whole-row
specialization.
