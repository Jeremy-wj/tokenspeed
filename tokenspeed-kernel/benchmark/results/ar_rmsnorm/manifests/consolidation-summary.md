# AR+RMSNorm consolidation summary

Date: 2026-07-24

Historical snapshot: this file records the 2026-07-24 migration only. It is
not the current campaign index. See `../README.md`, the gpt-oss status page,
and `../studies/mi350x/2026-07-repeatability/` for later evidence.
Every count and inventory statement below is as of that migration date, not a
description of the current tree. CSV checksums likewise describe the migration
snapshot and are not current-file integrity values for later-edited studies.

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

## Evidence current at migration time

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

At migration time, 56 non-obvious artifacts were quarantined rather than
deleted. The original set remains recorded in `artifacts-pre-migration.csv` and
`migration-map.csv`. For the current on-disk queue, use
`deletion-review.md`; do not derive current inventory from this snapshot.

## Removed legacy roots

The now-empty external roots were removed after checksum verification:

```text
/home/jeremwan/ar_rmsnorm_profiles
/home/jeremwan/ar_rmsnorm_e2e
```

