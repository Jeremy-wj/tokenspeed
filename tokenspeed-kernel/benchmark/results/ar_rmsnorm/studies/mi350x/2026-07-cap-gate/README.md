# MI350X TP=4 fusion performance-gate screen

Date: 2026-07-26

**Superseded decision (2026-07-27):** the repeatability campaign rejected this
gate implementation after it timed out under the stable passive-health,
single-slot, no-overlap control. The dispatch trace and focused correctness
evidence below remain valid; deployment candidacy does not.

Purpose: test a performance eligibility gate at M=256 without reducing the
2048-row symmetric workspace/compatibility ceiling.

## Environment

- model: `/data/models/openai/gpt-oss-120b`
- hardware: MI350X (`gfx950`), TP=4
- devices: `HIP_VISIBLE_DEVICES=1,2,3,5`
- container: `jeremwan-tokenspeed-profiler`
- backend: explicit `TS_ARNORM_BACKEND=triton_shmem`
- candidate: `TS_TRITON_SHMEM_FUSION_MAX_M=256`
- workspace cap: `--comm-fusion-max-num-tokens 2048`

No TP=8 work was run. Physical GPU 3 remained excluded.

## What changed

`TS_TRITON_SHMEM_FUSION_MAX_M` is now an optional backend performance gate. A
zero value preserves prior behavior. With a value of 256, the backend fuses
M<=256 and declines larger valid shapes to the complete production unfused
fallback while retaining a 2048-row state allocation.

The generic launcher records the gate in `RUN_ENV`, passes it into the
container, defaults torch profiling to no-stack, creates host-owned output
directories, and correctly reports the background serve PID.

## Validation

A no-stack torch trace used 32 prompts, input 16, output 8, concurrency 32, and
eight profile steps. Every rank showed:

- 576 fused one-shot calls, exactly 8 decode forwards x 72 sites;
- no fused two-shot calls;
- 81 separate RMSNorm calls, covering the unfused prefill plus the remaining
  unfused final pair;
- a `triton_shmem` state with `max_tokens=2048`.

This proves that the independent gate preserves decode fusion and sends the
M=512 prefill region to the unfused path without shrinking the workspace.

Focused TP=4 validation passed:

```text
test_triton_shmem_arrms_separate_performance_gate_world4
test_triton_shmem_arrms_world4
test_triton_shmem_arrms_gridcap_graph_world4
3 passed
```

## End-to-end screen

Three fresh server starts screened cap=2048, cap=256, and cap=2048 plus the
independent M=256 gate. Prefill buckets used 128 prompts, concurrency 32,
output 8, and per-request input lengths chosen to target aggregate first-batch
M={128,256,512,1024,2048,4096}. Decode used the canonical 128/512 workload with
seeds 0 and 1.

The short prefill measurements did not resolve the expected operator-level
saving. For cap=2048+gate=256 versus cap=2048, median TTFT changes were:

- M=512: -0.9%;
- M=1024: +0.1%;
- M=2048: -1.1%.

M=4096 was unfused in both arms, but the first baseline run contained a large
tail outlier; it is excluded from the gate decision.

Decode is dispatch-identical across the three arms, yet two-seed means moved in
opposite directions:

- cap=256 versus cap=2048: median TPOT -1.5%, throughput +1.7%;
- cap=2048+gate=256 versus cap=2048: median TPOT +2.0%, throughput -2.4%.

That contradiction is direct evidence that this small, sequential screen is
below the run-to-run variance floor. It does not support promotion or rejection
on end-to-end latency.

## Historical decision

The 2026-07-26 screen was inconclusive and left deployment unchanged. Its
diagnostic-candidate disposition is superseded: the current gate implementation
is rejected by the repeatability study. Operator evidence above M=256 may inform
a future transition-safe fallback design, but rerunning the same policy is not
next work.

Machine-readable summary: `cap-gate-summary.json`.

Raw logs and traces:

```text
raw/current/gpt-oss-120b/mi350x/2026-07-26/cap-gate-screen/
```
