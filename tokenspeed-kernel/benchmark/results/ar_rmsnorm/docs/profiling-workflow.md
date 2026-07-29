# AR+RMSNorm profiling workflow

## Qualified environment

```text
image:     jeremwan/tokenspeed:rocm7.2.4-torch2.11-profiler
image ID:  sha256:ad3ea3f8cae8ca38cf12824b15c606d0630118c6e04b4087e191b04619a6c135
container: jeremwan-tokenspeed-profiler
torch:     2.11.0+rocm7.2
runtime:   system HIP 7.2.53211 / ROCm 7.2.4
tracing:   system roctracer 4.1.70204 / rocprofiler-sdk 1.1.0
```

This is the only supported image for both serving and profiling. The historical
`jeremwan/tokenspeed:rocm7.2.4-torch2.11` tag was retired on 2026-07-24. It had
the same Docker configuration and package sets, but `libtorch_hip.so` still
resolved torch's bundled ROCm 7.2.0 `libroctracer64.so`; that mixed tracing stack
is known to crash. Do not recreate or use the old tag as a supposedly leaner
serving image.

Torch's bundled HIP/HSA/ROCTX/RCCL/roctracer libraries must not mix with the
system 7.2.4 runtime family. Re-run:

```bash
cd tokenspeed-kernel  # from the TokenSpeed repository root
bash benchmark/fix_torch_hip_bundling.sh
```

after any torch reinstall.

Qualification after an image change or torch reinstall must confirm:

```bash
python3 -c 'import torch; print(torch.__version__, torch.version.hip)'
ldd /opt/venv/lib/python3.10/site-packages/torch/lib/libtorch_hip.so \
  | egrep 'amdhip|hsa|rccl|roctx|roctracer'
/opt/rocm/bin/rocprofv3 --version
```

HIP, HSA, RCCL, ROCTX, and roctracer must resolve from `/opt/rocm`; bundled
compute libraries such as rocBLAS may remain under `torch/lib`. Then run the HIP
event-query capture probe, a bounded torch/Kineto graph trace, and an eager
Proton trace. A successful serving run alone does not qualify the tracing stack.

The profiler image is a direct child of the retired image and therefore retains
its parent layers. Untagging the old image does not free the 77.2 GB shared
image data; disk reclamation comes from removing obsolete containers and their
writable caches. The detailed comparison and retirement record is in
[ROCm 7.2 migration and incidents](history/rocm-7.2-migration-and-incidents.md).

## Shared-host safety

Before each GPU run:

1. inspect `amd-smi` process and utilization output;
2. do not terminate other users' processes;
3. for TP<8, exclude physical GPU 3 / HIP index 0;
4. launch TP=8 only from a verified idle snapshot.

Canonical GPT-OSS MI350X TP=4 campaign rank set:

```text
HIP_VISIBLE_DEVICES=1,2,3,5 -> physical AMD-SMI GPUs 0,2,1,4
```

Other rank sets in dated studies are exploratory or incident controls and must
not be described as canonical.

## A/B proof requirements

A label is not evidence. For every arm, preserve:

- server `RUN_ENV`;
- resolved `enable_allreduce_fusion` server argument;
- per-rank trace files;
- expected kernel signatures.

TP=4 fused:

```text
TS_ARNORM_BACKEND=triton_shmem
--enable-allreduce-fusion
--comm-fusion-max-num-tokens 2048
```

Rejected historical performance gate, independent of the workspace cap:

```text
TS_TRITON_SHMEM_FUSION_MAX_M=256
```

Zero disables the additional gate. The value 256 proved dispatch mechanics but
timed out under the stable repeatability control; do not rerun it without a
transition-safe fallback redesign.

TP=4 matched unfused:

```text
ENABLE_ALLREDUCE_FUSION=0
--comm-fusion-max-num-tokens 2048
```

Expected fused signatures:

```text
fused_ar_rmsnorm_oneshot_wholerow_kernel
fused_ar_rmsnorm_oneshot_blocked_kernel
fused_ar_rmsnorm_twoshot_blocked_kernel
```

Expected unfused signatures:

```text
amd_all_reduce_kernel or RCCL kernels
_rmsnorm_kernel
```

Restart between variants so graph capture and state caches are clean. Keep
model, hardware, prompts, seed, world size, warmup, concurrency, and profile
window identical.

## Launchers

- Generic serving: `benchmark/e2e_arnorm_serve.sh`
- Generic benchmark: `benchmark/e2e_arnorm_bench.sh`
- Generic profiling wrapper: `benchmark/e2e_arnorm_profile_serve.sh`
- gpt-oss compatibility wrappers: `benchmark/e2e_gptoss_*.sh`
- Model profiles: `benchmark/profiles/ar_rmsnorm/`

The generic serve, benchmark, profile, and teardown paths all default to the
canonical `jeremwan-tokenspeed-profiler` container. Set `CONTAINER` only to
target an intentionally equivalent replacement that has passed the
qualification checks above.

New output roots default under:

```text
benchmark/results/ar_rmsnorm/raw/
  current/<model>/<hardware>/<date>/
  history/
  review/delete_candidates/
```

Raw artifacts are Git-ignored. The 2026-07-24 consolidation is recorded in
`manifests/`; later campaigns keep generated manifests and checksums inside
their campaign roots. Paths in this document are relative to `ar_rmsnorm/`
unless they are executable repository paths.

## Torch/Kineto production traces

Torch CPU+GPU traces are the primary graph-serving timeline. Use the control
sidecar at `http://127.0.0.1:8101`, bounded profile steps, no stack capture, and
shape recording only when needed.

The generic serving launcher defaults `TOKENSPEED_PROFILE_WITH_STACK=0`; enable
stack capture only for a specific attribution question.

Example benchmark arguments:

```text
--profile
--profile-num-steps 16
--profile-base-url http://127.0.0.1:8101
--no-profile-with-stack
--profile-record-shapes
--profile-activities CPU GPU
```

Every expected rank file must exist and be nonempty.

Analysis:

```bash
python3 benchmark/analyze_ar_rmsnorm_torch.py <traces...>
python3 benchmark/analyze_graph_replay.py <matched-arm-traces...> \
  --decode-m 32 --prefill-m <scheduled-or-padded-M>
```

`analyze_graph_replay.py` is a legacy heuristic. It merges whole-model and split
correlations by expected site count, but old traces have no authoritative
forward ID/mode/M marker. Its JSON is labeled `heuristic_legacy` and must not be
presented as exact. New torch profile servers enable
`tokenspeed.model_forward.v1` CPU markers carrying mode, actual/executed M, batch
sizes, and execution path. Analyze them with:

```bash
python3 benchmark/analyze_ar_rmsnorm_forwards.py <matched-rank-traces...> \
  --mode decode --expected-world-size 4 --output <summary.json>
```

The marker analyzer rejects legacy marker-free traces instead of silently
falling back to heuristic grouping.

## Proton

- Eager Perfetto-ready timeline: `data=trace`,
  `output_format=chrome_trace`, system roctracer.
- Graph aggregation: `data=tree`, `output_format=hatchet`.
- Rocprofiler must configure before HIP/HSA registration.
- Roctracer must start after runtime/model initialization but before graph
  capture.
- Keep graph sessions active through capture and replay.
- Use explicit replay scopes.

Relevant lifecycle environment:

```text
TOKENSPEED_KERNEL_PROFILE=1
TOKENSPEED_KERNEL_PROFILE_EARLY=1
TOKENSPEED_KERNEL_PROFILE_EARLY_KEEP_ACTIVE=1
TOKENSPEED_KERNEL_PROFILE_BEFORE_GRAPHS=1
TOKENSPEED_KERNEL_PROFILE_GRAPH_SCOPES=1
```

AMD Proton does not reliably map `HIP_VISIBLE_DEVICES`; set
`ROCR_VISIBLE_DEVICES` before profiler import.

## Focused tuning loop

1. Serving-faithful eager decomposition:

```bash
BENCH_WS=4 BENCH_N=<hidden> BENCH_M_VALUES=<tokens> \
  python3 -m benchmark.probe_ar_rmsnorm_decomp
```

2. Isolated graph replay:

```bash
BENCH_WS=4 BENCH_N=<hidden> BENCH_M=<tokens> \
  python3 -m benchmark.probe_ar_rmsnorm_graph_perf
```

3. Shared-state graph transition gate:

```bash
BENCH_WS=4 BENCH_N=<hidden> PROBE_MODE=multigraph \
  PROBE_RNG_SHARED=1 python3 -m benchmark.probe_inkernel_barrier_graph
```

4. Communication correctness:

```bash
pytest -q test/ops/test_triton_shmem_communication.py
```

5. Full matched serving A/B and production trace.

Fixed-shape replay is not a sufficient promotion gate. A candidate must survive
interleaved shared-state graph transitions and a complete multi-arm serve.

## Repeatability campaign

Fast stability diagnosis should precede a full campaign:

```bash
python3 -m benchmark.repro_ar_rmsnorm_serving \
  --label <case> --devices <HIP-indices> \
  --deep-health-mode passive --output-len 128 --timeout 90
```

The reproducer runs one bounded server/configuration, applies the same GPU PID
guard, saves detailed results, and tears down on the first fault or timeout.
Use 128 output tokens for screening, then 512 and multiple repeats only after a
case passes. Its repeats share one server and therefore do not satisfy a
fresh-container-per-seed stability gate.

Before paired qualification, require three fresh canonical unfused seeds:

```bash
python3 benchmark/run_ar_rmsnorm_repeatability.py \
  --stability-only --comparison unfused --decode-only \
  --blocks 1 --seeds 0,1,2 --devices 1,2,3,5 \
  --deep-health-mode passive --disable-overlap-schedule \
  --double-buffer-input 0 --barrier-grid 0 --skip-profiles
```

The runner stops on the first failure and writes `stability-summary.json` only
after all expected seeds complete.

The required serving harness is:

```bash
python3 benchmark/run_ar_rmsnorm_repeatability.py \
  --blocks 3 --seeds 0,1,2,3,4 \
  --devices <qualified-HIP-indices>
```

The runner records HIP-to-physical mapping, restarts the dedicated container
before every server, rejects foreign PIDs before measurement, and samples GPU
PIDs every two seconds from KFD sysfs during timed benchmarks. Do not poll
`amd-smi process` in the hot loop: AMD-SMI 26.2 creates a short-lived GPU
context on physical GPU 1 and can perturb the workload it is meant to observe.
The runner separates prefill from decode, saves detailed timelines and traces,
validates serve arguments/signatures, and computes paired hierarchical
bootstrap intervals. It cannot report promotion eligibility below three
complete blocks and fifteen paired observations.

GPT-OSS MI350X profile v4 uses passive health, disabled overlap, the original
one-slot exit barrier, 0.90 GPU-memory utilization, eager prefill, C32 decode
capture, explicit copy-in, and `TS_TRITON_SHMEM_OUTPUT_RING=72`. Profile ID
`gpt-oss-120b-mi350x-qualified-v4` and the output-ring value are mandatory proof
fields.

Universal reserved-sink graph padding and persistent per-site fused outputs are
required safety invariants. Rerun a full campaign only for a changed
implementation; do not weaken output ownership or the stability criteria. The
standard gRPC health service and startup generation warmup remain active; only
periodic synthetic generation probes are removed. The current promotion outcome
belongs in [GPT-OSS-120B status](gpt-oss-120b-status.md).

Use `--resume` only with the same campaign root. Completed decode seeds are
reused only after their result and serve log revalidate; every resume records a
new code-identity snapshot. Timeouts, GPU isolation failures, and incomplete
arms are evidence and must remain in the artifact root.

## Artifact naming

Use:

```text
<date>-<hardware>-<model>-tp<ws>-<arm>-<phase>-rank<rank>.<extension>
```

Required metadata:

- model path/name and hidden size;
- hardware/architecture and visible devices;
- world size;
- fusion integration flag, backend, and cap;
- kernel policy overrides;
- workload and seed;
- runtime/image identifiers.

Do not use `final`, `latest`, `ON`, or `OFF` in new artifact names.

## Reference performance traces

```text
raw/current/gpt-oss-120b/mi350x/2026-07-24/traces/torch/
  tp4-fused-generic-block512/
  tp4-unfused/
```

These are the matched 2026-07-24 performance traces, not the current stability
profile. Current stability and root-cause artifacts are indexed by the status
page and `studies/mi350x/2026-07-repeatability/`.

The historical environment and incident record is kept in
[ROCm 7.2 migration and incidents](history/rocm-7.2-migration-and-incidents.md).

