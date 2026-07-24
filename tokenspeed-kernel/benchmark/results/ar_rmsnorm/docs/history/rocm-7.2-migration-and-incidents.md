# ROCm 7.2 migration and incident history

Historical record. Current policy and commands:
[project index](../../README.md) and
[profiling workflow](../profiling-workflow.md).

## Runtime artifact

The validated serving image was created from the known-working TokenSpeed
environment and upgraded to released ROCm 7.2.4 userspace:

```text
serving image:   jeremwan/tokenspeed:rocm7.2.4-torch2.11
image ID:        sha256:96f8b38d54c7da59f7888def76be81e99bf7512117bb2769609fadc7f19d230f
profiling image: jeremwan/tokenspeed:rocm7.2.4-torch2.11-profiler
profiler ID:     sha256:ad3ea3f8cae8ca38cf12824b15c606d0630118c6e04b4087e191b04619a6c135
```

Torch reports a ROCm 7.2 compile label, while `libtorch_hip.so` resolves HIP,
HSA, ROCTX, RCCL, and roctracer from `/opt/rocm/lib` at runtime.

## HIP graph-capture failure

The old image used HIP runtime 7.2.26015. During graph capture, a watchdog-like
thread calling `hipEventQuery` could receive
`hipErrorStreamCaptureUnsupported`, invalidating another thread's graph.

`benchmark/probe_hip_event_query_capture.py` isolated this behavior without
TokenSpeed, Triton, symmetric memory, or RCCL. It failed on runtime 70226015 and
passed on HIP 7.2.53211 from ROCm 7.2.4.

`TORCH_NCCL_BLOCKING_WAIT=1` avoided the watchdog thread and was an old
compatibility workaround, not the final fix. Current qualification passes with
blocking wait disabled.

## Refreshed serving faults

The July refresh found failures initially attributed to RCCL. Single-variable
controls separated two kernel synchronization defects:

### Unfused Triton AR

The small-message AMD all-reduce used multiple wavefronts around a scalar
system-scope signal barrier. The scalar release/acquire did not represent sibling
wavefront peer loads or persistent-buffer reuse.

Fix: bracket the scalar barrier with workgroup barriers. Disabling the fix
reproduced ws=8 memory faults; enabling it completed the qualification matrix.

### Folded fused copy-in

Folded copy-in writes the coarse symmetric input, signals peers, pulls peer data,
and signals buffer reuse in one program. A scalar wavefront's system fence cannot
publish sibling-wave stores.

Fix: folded copy-in runs with one wavefront. A four-wave folded specialization
reproduced post-readiness faults; the single-wave version completed ws=4/8
serving.

## RCCL evidence boundary

The corrected RCCL probe captures `dist.all_reduce` itself. Eager all-reduce,
captured replay, and compute graph replay pass at ws=2/4/8 with blocking wait
disabled.

This proves:

- the refreshed ws=4/8 faults were not a size-dependent RCCL defect;
- the HIP event-query issue was real and is fixed by released ROCm userspace;
- not every historical report labeled “RCCL hang” can be attributed from the
  retained evidence.

If a similar stall recurs, treat it as a new issue and isolate eager execution,
overlap scheduling, RCCL-only execution, and mixed HIP-IPC/RCCL ordering.

## Profiler crash and fix

The original profiling crash mixed torch's bundled ROCm 7.2.0
`libroctracer64.so` with system ROCm 7.2.4 HIP/HSA/ROCTX. The crash occurred in
roctracer activity-buffer handling.

`benchmark/fix_torch_hip_bundling.sh` now relocates bundled roctracer along with
the other runtime libraries. The coherent system 7.2.4 runtime/tracing family
passes torch/Kineto graph traces and eager Proton traces.

An all-bundled ROCm 7.2.0 alternative was rejected because it reproduced the
HIP event-query capture defect.

## Proton graph qualification

- Chrome/trace output is suitable for eager execution.
- The installed AMD graph implementation uses tree/Hatchet output.
- Rocprofiler requires early initialization and a session active through graph
  replay.
- Legacy roctracer graph mode is less reliable for multi-rank Hatchet output.
- Empty and duplicate bring-up artifacts were removed during the project
  consolidation; retained environment evidence is under
  `../../raw/history/environment-qualification/`.

## Historical campaign evidence

- Qualification logs:
  `../../raw/history/qualification/2026-07-22-23/`
- RCCL diagnostic:
  `../../raw/history/diagnostics/rccl/`
- Environment qualification:
  `../../raw/history/environment-qualification/2026-07-23/`
- Historical torch profiler comparison:
  `../../raw/history/mi350x/2026-07-23/torch-profiler/tp4-auto-policy-era/`

These paths are local and Git-ignored. Checksums and source paths are recorded
in `../../manifests/`.

