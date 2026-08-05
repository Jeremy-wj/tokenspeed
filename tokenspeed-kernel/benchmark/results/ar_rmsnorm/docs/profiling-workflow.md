# AR+RMSNorm profiling workflow

Updated: 2026-08-03

Pre-rebase traces remain legacy. New runs must record the resolved ordinary AR
backend, fused backend, rank-side selected-backend log, kernel signatures, and
scope schema.

## Qualified environment

```text
image:     jeremwan/tokenspeed:rocm7.2.4-torch2.11-profiler
image ID:  sha256:ad3ea3f8cae8ca38cf12824b15c606d0630118c6e04b4087e191b04619a6c135
container: jeremwan-tokenspeed-profiler
torch:     2.11.0+rocm7.2
runtime:   system HIP 7.2.53211 / ROCm 7.2.4
tracing:   system roctracer 4.1.70204 / rocprofiler-sdk 1.1.0
```

This is the GPT-OSS qualified image identity. Every campaign must record its
actual image ID and package layer; tag equality is not qualification. The GLM
baseline used a different local image ID and therefore carries its own runtime
audit. The historical `jeremwan/tokenspeed:rocm7.2.4-torch2.11` tag was retired
on 2026-07-24 because `libtorch_hip.so` resolved torch's bundled ROCm 7.2.0
`libroctracer64.so`; do not recreate that mixed tracing stack.

The upstream rebase outgrew the image's original Python packages. The qualified
2026-07-30 writable layer uses source-tree `PYTHONPATH`, Transformers 5.12,
SMG 1.8.0.post20260728, gRPC proto 0.4.14.post20260728, gRPC servicer
0.7.0.post20260728, XGrammar 0.2.2, and a `tokenspeed-scheduler` 0.1.3 wheel
built from current source. Incompatible optional torchaudio is absent. Record
these package versions in addition to the immutable base image ID.

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

Post-rebase baseline rank set:

```text
HIP_VISIBLE_DEVICES=1,2,3,5 -> physical AMD-SMI GPUs 0,2,1,4
```

Core-v3 is qualified only on HIP `1,2,5,6` (physical `0,2,4,6`). The earlier
`1,2,3,5` reset remains valid baseline evidence but is not the promoted
deployment topology. Never merge percentages across rank sets; every topology
needs its own matched control and qualification.

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

The planned GPT definitive campaign does not reuse M256. It keeps the 2048
workspace identity and permits only predeclared actual-M gates at 64, 91, or
384 after two graph passes, eager evidence, and serving-M markers agree. The
selected gate is encoded in a fail-closed profile ID and requalified before any
end-to-end comparison.

`TS_TRITON_SHMEM_FUSION_MIN_M` is a separate lower performance gate. GLM
profile v2 sets it to 2 so M1 captures complete ordinary fallback; its
shared-state transition probe covers both sides of that boundary.

TP=4 matched unfused:

```text
TS_ARNORM_BACKEND=auto
ENABLE_ALLREDUCE_FUSION=0
--disable-allreduce-fusion
--comm-fusion-max-num-tokens 2048
```

The explicit disable flag is mandatory. Upstream auto-enables fusion when the
flag is merely absent.

Expected fused signatures:

```text
fused_ar_rmsnorm_oneshot_wholerow_padded_kernel
fused_ar_rmsnorm_oneshot_wholerow_kernel
fused_ar_rmsnorm_oneshot_blocked_kernel
fused_ar_rmsnorm_twoshot_blocked_kernel
```

Expected unfused signatures:

```text
iris_stage_one_shot_allreduce_kernel for eligible decode shapes
RCCL kernels for larger fallback shapes
_rmsnorm_kernel
```

Expected Iris fused signature:

```text
iris_allreduce_residual_rmsnorm_kernel
AR+RMSNorm backend resolved: requested=auto selected=iris
```

Restart between variants so graph capture and state caches are clean. Keep
model, hardware, prompts, seed, world size, warmup, concurrency, and profile
window identical.

## Launchers

- Generic serving: `benchmark/e2e_arnorm_serve.sh`
- Generic benchmark: `benchmark/e2e_arnorm_bench.sh`
- Generic profiling wrapper: `benchmark/e2e_arnorm_profile_serve.sh`
- Graph campaign runner: `benchmark/run_ar_rmsnorm_graph_sweep.py`
- Eager campaign runner: `benchmark/run_ar_rmsnorm_eager_sweep.py`
- GPT three-WS serving orchestrator: `benchmark/run_gpt_oss_definitive_e2e.py`
- gpt-oss compatibility wrappers: `benchmark/e2e_gptoss_*.sh`
- Model profiles: `benchmark/profiles/ar_rmsnorm/`

The GLM-5.2-FP8 profile is
`benchmark/profiles/ar_rmsnorm/glm_5_2_fp8_mi350x.env`. It retains ideal graph
serving defaults; eager/health/KVStore controls from the historical bring-up
are not encoded. See the [GLM status](glm-5.2-fp8-status.md). The generic
repeatability and reproducer paths derive model, artifact root, world size,
device set, and fusion cap from the sourced profile.

GPT definitive wrappers are
`benchmark/profiles/ar_rmsnorm/gpt_oss_120b_mi350x_definitive_ws{2,4,8}.env`.
They use `/data/models/openai-gpt-oss-120b` and require explicit current-machine
device sets. The old core-v3 image, path, and HIP mapping remain historical
identity and must not be copied without a topology audit.

The generic serve, benchmark, profile, and teardown paths accept `CONTAINER` or
`TOKENSPEED_CONTAINER`; the qualified lab value is
`jeremwan-tokenspeed-profiler`. `CONTAINER_REPO_ROOT` overrides the repository
mount inside the container. Host output roots are derived from the launcher
location unless `AR_RMSNORM_RESULT_ROOT` is set, so workflows do not require a
specific username or checkout path.

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

Final matched 32-step Perfetto-compatible traces:

```text
raw/current/gpt-oss-120b/mi350x/2026-08-01/serving-repros/
  final-perfetto-fused/traces/repro-repeat0/*-TP{0,1,2,3}.trace.json.gz
  final-perfetto-unfused/traces/repro-repeat0/*-TP{0,1,2,3}.trace.json.gz
```

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

1. Serving-faithful three-backend eager decomposition:

```bash
BENCH_WS=4 BENCH_N=<hidden> BENCH_M_VALUES=<tokens> \
  python3 -m benchmark.probe_ar_rmsnorm_backend_decomp
```

2. Captured full-site stage decomposition for GPT-OSS decode:

```bash
BENCH_WS=4 BENCH_N=2880 BENCH_M=32 BENCH_CALLS_PER_GRAPH=72 \
  BENCH_GRAPH_N_REPEAT=1000 \
  python3 -m benchmark.probe_ar_rmsnorm_backend_graph_decomp
```

Use cumulative-prefix marginal columns for additive accounting. The older
`probe_ar_rmsnorm_decomp` remains useful for triton-specific
separate/in-kernel/folded diagnostics.

3. Isolated graph replay:

```bash
BENCH_WS=4 BENCH_N=<hidden> BENCH_M=<tokens> \
  python3 -m benchmark.probe_ar_rmsnorm_graph_perf
```

Set `BENCH_MAX_TOKEN_NUM` when the actual M is smaller than the model profile's
allocation cap. Consolidate repeated per-site/full-site sweeps with
`benchmark/analyze_ar_rmsnorm_graph_sweep.py`; it preserves pass-level p50
values and emits compact JSON/CSV comparisons keyed by world size, hidden size,
site count, and M.

The predeclared GLM WS=2/4/8 campaign is in the
[definitive sweep study](../studies/mi350x/2026-08-glm-5.2-fp8-definitive-sweep/README.md).
Inspect its complete schedule without launching a benchmark:

```bash
source benchmark/profiles/ar_rmsnorm/glm_5_2_fp8_mi350x.env
PYTHONPATH=. python3 benchmark/run_ar_rmsnorm_graph_sweep.py \
  --spec benchmark/results/ar_rmsnorm/studies/mi350x/\
2026-08-glm-5.2-fp8-definitive-sweep/campaign.json \
  --devices 2=<two-idle-devices> \
  --devices 4=<four-idle-devices> \
  --devices 8=0,1,2,3,4,5,6,7 \
  --dry-run
```

Device sets are launch-time identity, not defaults in the campaign
specification. Freeze them only after the shared-host and topology preflight.

The [GPT definitive study](../studies/mi350x/2026-08-gpt-oss-120b-definitive-sweep/README.md)
uses 72-site graph sweeps plus a separate fresh-process eager sweep. Its graph
summary preserves both raw replay and reset-copy-adjusted unfused values; never
mix those columns or merge eager and graph rows.

4. Shared-state graph transition gate:

```bash
BENCH_IMPL=<production_unfused|auto|triton_shmem> \
  BENCH_WS=4 BENCH_N=<hidden> PROBE_REPLAYS=70 \
  python3 -m benchmark.probe_ar_rmsnorm_transitions
```

The legacy `probe_inkernel_barrier_graph` remains a `triton_shmem`-specific
barrier diagnostic. The post-rebase transition probe compares production
dispatch and retains failures across backend/path changes.

Before destroying ProcessGroupNCCL, release and garbage-collect every captured
collective graph, synchronize, and barrier. Keeping graph-owned collective
events alive through communicator teardown can hang an otherwise successful
probe.

5. Communication correctness:

```bash
pytest -q test/ops/test_triton_shmem_communication.py
```

Evaluate the separate paired-reduction primitive with
`python3 -m benchmark.bench_all_reduce_two`; do not merge its result into the
AR+RMSNorm arms.

6. Full matched serving A/B and production trace.

Fixed-shape replay is not a sufficient promotion gate. A candidate must survive
interleaved shared-state graph transitions and a complete multi-arm serve.

## Repeatability campaign

Fast stability diagnosis should precede a full campaign:

```bash
source tokenspeed-kernel/benchmark/profiles/ar_rmsnorm/gpt_oss_120b_mi350x.env
PYTHONPATH=tokenspeed-kernel \
  python3 -m benchmark.repro_ar_rmsnorm_serving \
  --label <case> --devices 1,2,5,6 --output-len 128 --timeout 90
```

The reproducer runs one bounded server/configuration, applies the same GPU PID
guard, saves detailed results, and tears down on the first fault or timeout.
Use 128 output tokens for screening, then 512 and multiple repeats only after a
case passes. Its repeats share one server and therefore do not satisfy a
fresh-container-per-seed stability gate.

Before paired qualification, require three fresh unfused seeds on the target
rank set:

```bash
PYTHONPATH=tokenspeed-kernel \
  python3 -m benchmark.run_ar_rmsnorm_repeatability \
  --stability-only --comparison iris --decode-only \
  --blocks 1 --seeds 0,1,2 --devices 1,2,5,6 \
  --double-buffer-input 0 --barrier-grid 0 --skip-profiles
```

The runner stops on the first failure and writes `stability-summary.json` only
after all expected seeds complete.

The required serving harness is:

```bash
PYTHONPATH=tokenspeed-kernel \
  python3 -m benchmark.run_ar_rmsnorm_repeatability \
  --comparison iris --blocks 3 --seeds 0,1,2,3,4 \
  --devices 1,2,5,6
```

Run explicit `triton_shmem` separately with
`--comparison triton_shmem`. Resume reuses a decode result only when result,
serve proof, and the complete phase GPU-guard history all revalidate.

Base-default safety campaigns leave overlap enabled. Final performance
qualification uses a matched, reversible no-overlap policy:

```bash
PYTHONPATH=tokenspeed-kernel \
  python3 -m benchmark.run_ar_rmsnorm_repeatability \
  --comparison triton_shmem --blocks 3 --seeds 0,1,2,3,4 \
  --decode-only --skip-profiles --disable-overlap-schedule \
  --devices 1,2,5,6
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

Qualified core-v3 proof requires:

```text
AR_NORM_PROFILE_ID=gpt-oss-120b-mi350x-triton-core-v3
TS_TRITON_SHMEM_PROFILE_PURE_TP=1
TS_TRITON_SHMEM_COARSE=1
TS_TRITON_SHMEM_INKERNEL_BARRIER=1
TS_TRITON_SHMEM_ONESHOT_VARIANT=padded
TS_TRITON_SHMEM_PADDED_MAX_M=64
TS_TRITON_SHMEM_ONESHOT_NUM_WARPS=4
TS_TRITON_SHMEM_INPUT_SITE_RING=72
TS_TRITON_SHMEM_OUTPUT_RING=72
TS_TRITON_SHMEM_BORROW_TWOSHOT_OUTPUT=1
TS_TRITON_SHMEM_GRID_CAP=128
TS_TRITON_SHMEM_GRID_CAP_MIN_M=256
```

Resolved server proof must retain base defaults:

```text
gpu_memory_utilization=0.95
disable_prefill_graph=False
cudagraph_capture_sizes=None
disable_overlap_schedule=False
deep health mode=generate
```

For the final performance-only policy,
`disable_overlap_schedule=True` replaces only that one line and must match in
both arms.

Decode proof must contain
`fused_ar_rmsnorm_oneshot_wholerow_padded_kernel`. Captured direct-M512 proof
must contain ordinary all-reduce plus RMSNorm and no triton-shmem two-shot;
the eager/standalone transition probe must still prove the two-shot kernel.
The qualified campaign rank set is HIP `1,2,5,6`.

Universal reserved-sink graph padding and persistent per-site fused outputs are
required safety invariants. Rerun a full campaign only for a changed
implementation; do not weaken output ownership or the stability criteria. The
standard gRPC health service and startup generation warmup remain active;
periodic generated health probes also remain active. The current promotion
outcome belongs in [GPT-OSS-120B status](gpt-oss-120b-status.md).

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

The historical environment and incident record is kept in
[ROCm 7.2 migration and incidents](history/rocm-7.2-migration-and-incidents.md).

