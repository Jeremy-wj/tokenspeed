# AR+RMSNorm consolidation summary

Date: 2026-07-24

## Counts

- Pre-migration artifacts inventoried: 504
- Verified empty/duplicate artifacts deleted: 84
- Raw artifacts retained: 293
  - current: 18
  - history: 219
  - quarantined for review: 56
- Study files regrouped: 127
- Post-migration raw + study artifacts: 420
- Local raw artifact size: approximately 411 MiB

## Safe deletions

- 17 zero-byte failed outputs
- 51 empty 42-byte Chrome-trace stubs
- 6 exact duplicate Hatchet copies
- 10 exact duplicates of tracked study files
- Empty output directories left by failed profiler attempts

Every deleted source, checksum, and reason remains in
`artifacts-pre-migration.csv` and `migration-map.csv`.

## Current evidence

```text
raw/current/gpt-oss-120b/mi350x/2026-07-24/
  e2e/
    tp4-fused-generic-block512/
    tp4-unfused/
  profiling/
    tp4-fused-generic-block512/
    tp4-unfused/
  traces/torch/
    tp4-fused-generic-block512/
    tp4-unfused/
```

Verification:

- fused traces: 1,080 `fused_ar_rmsnorm_oneshot_blocked_kernel` events per rank;
- unfused traces: 1,095 `amd_all_reduce_kernel` events per rank;
- fused server: `enable_allreduce_fusion=True`;
- unfused server: `enable_allreduce_fusion=False`.

## Review queue

Non-obvious artifacts were not deleted. They are under
`raw/review/delete_candidates/` and indexed in `deletion-review.md`.

## Removed legacy roots

The now-empty external roots were removed after checksum verification:

```text
/home/jeremwan/ar_rmsnorm_profiles
/home/jeremwan/ar_rmsnorm_e2e
```

