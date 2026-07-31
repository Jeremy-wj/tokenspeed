# ROCm 7.2 migration and incident history

Historical record. Current policy and commands:
[project index](../../README.md) and
[profiling workflow](../profiling-workflow.md).
Artifact paths in this record are relative to `ar_rmsnorm/`.

## Runtime artifacts

The validated serving image was created from the known-working TokenSpeed
environment and upgraded to released ROCm 7.2.4 userspace:

```text
current serving/profiling image:
  jeremwan/tokenspeed:rocm7.2.4-torch2.11-profiler
  sha256:ad3ea3f8cae8ca38cf12824b15c606d0630118c6e04b4087e191b04619a6c135

retired historical tag:
  jeremwan/tokenspeed:rocm7.2.4-torch2.11
  sha256:96f8b38d54c7da59f7888def76be81e99bf7512117bb2769609fadc7f19d230f
```

Torch reports a ROCm 7.2 compile label, while `libtorch_hip.so` resolves HIP,
HSA, ROCTX, RCCL, and roctracer from `/opt/rocm/lib` at runtime.

## Image supersession audit and retirement

The two images were audited on 2026-07-24. Their package inventories and parent
layers matched; the functional difference was runtime library resolution. The
retired image loaded torch's bundled ROCm 7.2.0 roctracer and crashed in
activity-buffer handling. The profiler image loads the coherent system ROCm
7.2.4 stack and passed both torch/Kineto graph traces and eager Proton traces.

The retired tag and obsolete containers were removed. Its layers remain as the
profiler image's Docker parent, so tag removal did not reclaim shared image
bytes. A future reproducible rebuild should apply
`benchmark/fix_torch_hip_bundling.sh` in a clean layer and exclude generated
compiler/profiler caches.

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

Initial fix: folded copy-in ran with one wavefront. A four-wave specialization
reproduced post-readiness faults, while narrower single-wave screens passed.
The later full GPT-OSS promotion campaign nevertheless faulted on its first
separate fused M=4 prefill server after five decode seeds. Folded copy-in is
therefore disabled pending a producer-to-system-release publication proof.

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
  `raw/history/environment-qualification/`.

## 2026-07-27 TP=4 serving stability incident

Long fused serving showed illegal instructions, invalid ISA errors, memory
faults, and silent timeouts. Three controls isolated two causes:

1. The SMG custom deep-health RPC enqueued a one-token generation on every
   probe. It could report success after unrelated scheduler output, then cancel
   and abort the synthetic request while it was already entering execution.
   This injected asynchronous M=1 graph transitions. The project engine wrapper
   now makes custom deep health fully passive; standard gRPC liveness and the
   startup generation warmup remain enabled.
2. A two-slot prototype attempted to remove the original input-reuse exit
   barrier. It saved 5.37 us/site and passed several long runs, but later faulted
   because mutable host capture phase was not graph-stable slot identity. The
   stable one-slot path retains the exit barrier and is not the identified reuse
   bug. Separately, request-wave M changes required disabling overlap scheduling;
   fixed grid 32 stabilized decode but deadlocked prefill. Noncanonical rank
   sets are excluded because unfused runs also timed out.

The MI350X fabric itself is not asymmetric: every peer pair is a coherent,
bidirectional, one-hop XGMI link with the same topology weight. Rank-set
differences do not establish link locality or a one-slot reuse race; later
unfused timeouts showed that the one-slot path was not the common cause. The
subsequent root-cause closure is recorded below.

Curated evidence:
`studies/mi350x/2026-07-repeatability/serving-stability-summary.json`.

### 2026-07-28 graph-memory screens and supersession

Residual fused and unfused faults persisted after health, overlap, ring, and
standalone-all-reduce controls. They required CUDA-graph capture: three
fresh-container fully eager seeds passed. Decode capture restricted to C32
still faulted at `gpu_memory_utilization=0.95`, including under serialized
kernel/copy execution. The causal follow-up showed that pre-capture KV sizing
amplified the fault: 0.94/0.945 passed, and 0.95 passed when only KV capacity was
capped to the passing 0.94 size. Reserving more HBM headroom (`0.90`), capturing
only C32, and keeping prefill eager then passed three fresh canonical unfused
seeds, a fused screen, and marker trace.

Later formal 0.90/C32 campaigns faulted both fused and unfused. Fully eager
canonical controls passed, rotated C32 device sets passed, and hardware
counters were clean. Subsequent fault-address geometry and a positive control
identified the base cause: ordinary graph-padding rows aliased mutable request
slot 0 and produced exact negative 8-KiB KV-page underflows. Universal
reserved-sink padding closed that mechanism.

One fused-only fault remained after the base fix. It crossed collective,
barrier, and memory-substrate variants but disappeared without captured fusion
or with persistent per-site returned outputs. Profile v4 preallocates 72 output
slots outside graph-private-pool lifetime. Three fresh full transition servers
and the complete three-block/fifteen-pair campaign passed without a safety
failure; promotion was rejected only because decode performance regressed.
Sources:
`studies/mi350x/2026-07-repeatability/kv-cache-headroom-root-cause.json`,
`studies/mi350x/2026-07-repeatability/graph-padding-sentinel-root-cause.json`,
`studies/mi350x/2026-07-repeatability/fused-output-lifetime-root-cause.json`,
and
`studies/mi350x/2026-07-repeatability/e2e-stability-resolution-summary.json`.

## Historical campaign evidence

- Qualification logs:
  `raw/history/qualification/2026-07-22-23/`
- RCCL diagnostic:
  `raw/history/diagnostics/rccl/`
- Environment qualification:
  `raw/history/environment-qualification/2026-07-23/`
- Historical torch profiler comparison:
  `raw/history/mi350x/2026-07-23/torch-profiler/tp4-auto-policy-era/`

These paths are local and Git-ignored. Checksums and source paths are recorded
in `manifests/`.

