# GPT-OSS-120B TP=4 serving root cause on MI350X (gfx950)

Investigation: 2026-07-28 through 2026-07-29

This is a durable pre-rebase incident record. Its padding, captured-address,
and output-lifetime findings remain applicable safety evidence, but its
performance observations are legacy after upstream `3f88dcc2`. See
[upstream-main rebase impact](upstream-main-rebase-impact-2026-07.md).

## Outcome

Two application-level graph-lifetime defects caused the recurrent GPU memory
faults:

1. ordinary CUDA/HIP graph padding aliased live request-pool slot 0;
2. the fused backend returned transient capture-time outputs to custom kernels
   instead of binding each site to persistent output storage.

GPT-OSS-120B decode captures only C32. When the real batch was smaller, the
wrapper padded request indices with zero. Slot 0 is scheduler-owned and mutable;
dummy attention and sampling rows therefore consumed a live or stale request's
page table and runtime state. The resulting negative/stale KV page indices
accessed memory immediately before 2 MiB-aligned KV allocations and raised GPU
memory aperture faults.

The fix maps every padded graph row to `max_req_pool_size`, the reserved sink
row already present in page-table, runtime-state, and sampling pools.

After that base fix, the fused-only fault was removed by assigning all 72
GPT-OSS-120B fused sites persistent output pairs allocated before graph capture.

## Regression

The safe sink-row behavior existed previously. Commit
`7777f490d32cbd3b854c3d3f1a51ed441bc2563e` restricted it to DFLASH while
addressing a Qwen3.5 TP8 performance regression. Ordinary decode graphs reverted
to slot-0 padding.

That assumption was unsafe for GPT-OSS-120B:

- startup warmup and later requests reuse request-pool slots;
- sliding/full attention page tables can contain stale or hole entries;
- long decode changes request and page ownership while graph padding persists;
- padded rows still execute captured model kernels even though their outputs
  are discarded.

## Address-level proof

GPT-OSS-120B TP=4 uses 8 KiB KV pages per layer and rank:

```text
64 tokens/page * 2 TP-sharded KV heads * 64 dimensions * 1 fp8 byte
= 8192 bytes
```

All 21 current-campaign fault addresses are exact 8 KiB multiples below the
next 2 MiB boundary. Observed underflows were 1, 2, 4, 5, 8, 12, 13, 15, 29,
83, 91, 94, and 205 KV pages.

Allocator snapshots and pointer logs independently show that each large K/V
tensor starts on a 2 MiB-aligned segment. Representative historical fault
addresses include:

```text
...b9fe000 = aligned KV base - 1 page
...139fc000 = aligned KV base - 2 pages
...ab9f8000 = aligned KV base - 4 pages
...e79f0000 = aligned KV base - 8 pages
...8b9e2000 = aligned KV base - 15 pages
```

This geometry is incompatible with generic HBM pressure or a random defective
board. It is the signature of a negative KV-page index.

## Positive control

The old behavior remains available only as a diagnostic override. With C32,
actual M=1, and metadata validation enabled, every rank failed closed before
graph replay:

```text
graph padding aliased a live request-pool row
expected=257 actual=[0, 0, ...]
```

Source:
`../studies/mi350x/2026-07-repeatability/graph-padding-sentinel-root-cause.json`.

## Why earlier conclusions were incomplete

Several controls changed fault incidence without changing the bad metadata:

- lowering KV capacity from 0.95 to 0.90 moved allocation boundaries and added
  margin;
- fully eager execution removed graph padding entirely;
- rotated rank sets changed allocator addresses and request timing;
- serialization and profiling changed synchronization/timing;
- short screens often ended before a stale slot/page combination appeared.

KV over-allocation was therefore a real amplifier, not the root cause. The
intermediate “HIP graph runtime” diagnosis correctly identified graph capture
as necessary but stopped one layer too early: application padding metadata,
not the HIP graph primitive, was corrupt.

## Falsified alternatives

- AR+RMSNorm fusion: faults reproduced unfused.
- Folded copy-in: faults reproduced with explicit copy-in and unfused.
- Prefill graphs: disabled in failing controls.
- Overlap scheduling and synthetic health traffic: disabled/passive.
- Standalone Triton all-reduce: RCCL-only controls faulted.
- Negative flat-KV helper: GPT-OSS-120B used the radix path in these runs.
- One physical GPU: faults appeared on nodes 3, 4, and 5; rotated sets passed.
- Hardware ECC/RAS/PCIe errors: selected-board counters were clean.
- Foreign contention: continuous KFD guards found no selected-GPU process.
- HIP graph or `torch.topk` alone: smaller raw graph probes passed.

## Fix and validation

Code fix:
`python/tokenspeed/runtime/execution/cuda_graph_wrapper.py`
`_pad_graph_req_pool_indices()`.

Regression test:
`test/runtime/test_dp_sampling_routing_metadata.py`
`test_cuda_graph_req_pool_padding_uses_reserved_sink_row`.

Post-fix evidence:

- 10/10 fresh M=1 C32 servers passed;
- 5/5 fresh long C32 servers passed
  (128 input, 512 output, 128 prompts, concurrency 32);
- the formal three-seed unfused fresh-container gate passed 3/3;
- no fault address or timeout occurred;
- profile v4 keeps 0.90/C32, eager prefill, passive health, disabled overlap,
  explicit copy-in, the original exit barrier, and graph-stable fused outputs.

## Fusion-specific captured-output lifetime

After the padding fix, unfused qualification passed but fusion still faulted
intermittently. The failure followed the captured fused operation across:

- triton-shmem and native symmetric-memory backends;
- coarse and fine-grained buffers;
- one-shot and forced two-shot paths;
- in-kernel and separate barriers;
- serialized launches.

It disappeared when the fused backend declined during capture (15/15), when
fusion ran fully eager (10/10), and when captured fused outputs used persistent
per-site storage (10/10 plus three fresh full-suite servers).

The fused entry had allocated `norm_out` and `residual_out` inside graph capture.
Those custom-kernel intermediates were owned/reused by the HIP graph private
pool across 72 sites. Profile v4 assigns one persistent output slot per
GPT-OSS-120B fused call site, outside graph-pool lifetime. The 72-site forward
returns to the same slot epoch only after prior consumers have completed.

Source:
`../studies/mi350x/2026-07-repeatability/fused-output-lifetime-root-cause.json`.

## Disposition

Profile `gpt-oss-120b-mi350x-qualified-v4` completed three restart blocks and
fifteen pairs without a safety failure. The underlying serving faults are
closed; its performance result is legacy after the upstream rebase.

Profile v2 and core-v3 preserve the reserved sink and 72 persistent captured
output sites. Their 72-site input ring and eager two-shot output ping-pong add
separate lifetime contracts without reopening either incident. See the
[realignment](../studies/mi350x/2026-07-triton-shmem-realignment/README.md) and
[core tuning](../studies/mi350x/2026-07-triton-shmem-core-tuning/README.md)
studies. Current deployment policy belongs in
[GPT-OSS-120B status](gpt-oss-120b-status.md).

## Harness findings

The investigation also exposed two independent startup-port races and one local
orchestration risk:

- Prometheus port allocation occurred too early;
- internal distributed ports used probe-then-release ephemeral clusters;
- concurrent local diagnostics could restart the shared dedicated container.

The launcher now defers the metrics-port choice, internal ports use a locked
rotating non-ephemeral cluster, health waits fail on first fatal log signature,
and the benchmark/reproducer share an exclusive GPU-campaign lock.
