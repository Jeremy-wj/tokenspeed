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

Torch's bundled HIP/HSA/ROCTX/RCCL/roctracer libraries must not mix with the
system 7.2.4 runtime family. Re-run:

```bash
cd /home/jeremwan/tokenspeed/tokenspeed-kernel
bash benchmark/fix_torch_hip_bundling.sh
```

after any torch reinstall.

## Shared-host safety

Before each GPU run:

1. inspect `amd-smi` process and utilization output;
2. do not terminate other users' processes;
3. for TP<8, exclude physical GPU 3 / HIP index 0;
4. launch TP=8 only from a verified idle snapshot.

Validated TP=4 device list:

```text
HIP_VISIBLE_DEVICES=1,2,3,5
```

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

New output roots default under:

```text
benchmark/results/ar_rmsnorm/raw/
  current/<model>/<hardware>/<date>/
  history/
  review/delete_candidates/
```

Raw artifacts are Git-ignored. Their checksums and provenance belong in
`../manifests/`.

## Torch/Kineto production traces

Torch CPU+GPU traces are the primary graph-serving timeline. Use the control
sidecar at `http://127.0.0.1:8101`, bounded profile steps, no stack capture, and
shape recording only when needed.

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
python3 benchmark/analyze_graph_replay.py <traces...>
```

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

## Current traces

```text
../raw/current/gpt-oss-120b/mi350x/2026-07-24/traces/torch/
  tp4-fused-generic-block512/
  tp4-unfused/
```

The historical environment and incident record is kept in
[ROCm 7.2 migration and incidents](history/rocm-7.2-migration-and-incidents.md).

