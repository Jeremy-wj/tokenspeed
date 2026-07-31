# Upstream-main rebase and AR+RMSNorm baseline reset

Updated: 2026-07-31

This is the durable record of the 2026-07-29 rebase boundary. Current deployment
policy belongs in [GPT-OSS-120B status](gpt-oss-120b-status.md).

## Rebase boundary

The six-commit AR+RMSNorm series moved from upstream base `f35ea4ef` to upstream
`main` at `3f88dcc2`.

```text
old local head: a031a98b1bb1a6fecf3a5ac5a6059d2e5b53d77b
rebased head:   7751b072460193d78f638026a113ca1fdca7d84e
backup ref:     refs/backup/jeremwan-triton-shmem-experiments-pre-rebase-20260729
```

Commit mapping:

```text
cd109677 -> b4a8e58d  First working version, very slow performance
1a263041 -> 5c1458ab  Perf tuning: allocation mode and barriers
42290463 -> b185e674  Barrier fixes and GPT-OSS-120B results
12c35bf0 -> 91427b46  Profiling and trace refactor
a1b46833 -> 5cfc190b  Project reorganization
a031a98b -> 7751b072  Serving/integration fixes and documentation
```

The external bundle and raw-evidence backup was written to
`/home/jeremwan/tokenspeed-pre-rebase-backups/20260729T2047Z/`. All 5,473
ignored evidence files passed checksum verification after promotion.

## Why the baseline reset was mandatory

Upstream changed both sides of the comparison:

- AMD fused AR+residual+RMSNorm selects Iris by default;
- ordinary small AMD all-reduce also selects Iris;
- AMD gained `all_reduce_two`;
- graph, profiling, serving, and communication runtime code moved materially.

Every pre-rebase latency, throughput, kernel, graph, and operator result is
therefore **legacy performance evidence**. The captured-pointer, padding,
synchronization, and complete-fallback findings remain valid safety evidence.

The post-rebase dispatch policy became:

```text
TS_ARNORM_BACKEND unset/auto   -> Iris -> native symm_mem -> caller fallback
TS_ARNORM_BACKEND=iris         -> Iris -> caller fallback
TS_ARNORM_BACKEND=symm_mem     -> native symm_mem -> caller fallback
TS_ARNORM_BACKEND=triton_shmem -> local backend -> caller fallback
```

`triton_shmem` remained explicit so local experiments could not silently replace
upstream behavior.

## Conflict-resolution decisions

1. Kept upstream Iris `all_reduce` and `all_reduce_two`; removed the local
   native symmetric-buffer replacement and its controls.
2. Kept Iris-first fused routing under `auto`; retained local
   `triton_shmem` only as an explicit selector.
3. Preserved the complete unfused fallback when any fused backend declines.
4. Kept upstream VizTracer-to-Proton flow events, metric filtering, finalize
   recovery, AMD visibility checks, and TP finalize barrier.
5. Retained local pre-graph profiling startup, graph scopes, import-time session
   shutdown, and forward markers.
6. Kept upstream graph valid-row replay and stale-tail clearing while preserving
   the local universal reserved sink and persistent per-site fused outputs;
   these solve distinct captured-address lifetime defects.
7. Kept upstream EPD/SIGTERM lifecycle and communication topology behavior,
   preserving only orthogonal local serving controls and safety fixes.

## Baseline-reset outcome

The 2026-07-30 GPT-OSS-120B TP=4 reset established upstream-unfused with
explicit fusion disablement as the control. Omitting the enable flag is not a
valid control because upstream may auto-enable fusion.

Both fused candidates completed 15/15 safe pairs but were slower:

- Iris fused: +2.55% median TPOT and -2.29% output throughput;
- explicit `triton_shmem`: +10.47% median TPOT and -10.58% throughput.

See the
[post-rebase baseline study](../studies/mi350x/2026-07-post-rebase-baseline/README.md)
for complete intervals, identity, operator, graph, transition, profile, and
campaign evidence.

Later studies deliberately build on this baseline:

- [realignment](../studies/mi350x/2026-07-triton-shmem-realignment/README.md)
  fixed copy-out and one-shot input lifetime but still failed promotion;
- [decomposition](../studies/mi350x/2026-07-triton-shmem-decomposition/README.md)
  isolated the blocked decode core;
- [core-v3 tuning](../studies/mi350x/2026-07-triton-shmem-core-tuning/README.md)
  replaced that core and cleared the capacity gate on qualified HIP `1,2,5,6`.

Those follow-ups supersede the reset's deployment conclusion, not its baseline
measurements or rebase record.

## Preserved implementation consequences

- A valid fused run must prove the resolved backend and observed per-rank kernel
  signatures; `enable_allreduce_fusion=True` alone is insufficient.
- `TS_TRITON_SHMEM_FUSION_MAX_M` affects only explicit local-backend runs.
- The ordinary unfused path may select Iris or RCCL by shape; analyses must
  identify the actual transport.
- Every fused decline must execute the missing all-reduce before ordinary
  residual-add + RMSNorm.
- `all_reduce_two` is a separate paired-reduction primitive and must not be
  folded into AR+RMSNorm evidence.
- Safety findings about captured pointers, sink padding, and output lifetime
  apply across backend revisions.

## Verification at the rebase

- `git range-diff` preserved all six logical commits.
- The promoted branch merge base was `3f88dcc2`.
- Python compilation passed for changed runtime, profiling, graph, and
  communication modules.
- Kernel profiling tests: 20 passed.
- Graph-analysis and repeatability-runner tests: 24 passed.
- Runtime collection was blocked by the then-stale `tokenspeed_scheduler`
  binary, which lacked `PagedCacheTransferPolicy`.
- Raw-evidence checksums passed for all 5,473 ignored files.
