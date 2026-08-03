# GLM-5.2-FP8 definitive AR+RMSNorm sweep

Status: **planned; not run**

This study predeclares the final captured-operator comparison for the current
GLM optimization effort. It does not contain results or change the
[live deployment decision](../../../docs/glm-5.2-fp8-status.md).

The immutable matrix is in [campaign.json](campaign.json).

## Question

At GLM's N=6144 width, how do explicit upstream-unfused, default Iris fused,
and the optimized four-warp padded Triton kernel compare across WS=2/4/8, and
where does fusion stop helping?

WS=8 is model-faithful TP evidence. WS=2/4 measure scaling of the same operator
shape and are not GLM deployment qualification.

## Contract

```text
hardware / dtype: MI350X gfx950 / bf16
world sizes: 2, 4, 8
hidden size: 6144
primary graph: 156 sequential sites
diagnostic graph: 1 site
warmups / measured replays: 50 / 1000
statistic: max rank per iteration, then p50/p95/p99
process isolation: one fresh process per arm and shape
```

Arms:

1. `upstream_unfused` — `BENCH_IMPL=production_unfused`; ordinary Iris through
   512 KiB, then RCCL, followed by residual RMSNorm.
2. `iris_fused` — `BENCH_IMPL=auto`; require the default fused Iris backend on
   every rank.
3. `triton_forced` — unqualified four-warp padded whole-row Triton with 156
   input/output sites. It is forced at M1 and above M42 to measure both loss
   regions; profile v2 itself falls back there.

## Sweep space

The primary 156-site pass covers:

```text
M=1,2,4,8,16,24,32,36,40,41,42,43,44,48,64,96,128,256
```

The order-opposed confirmation pass covers:

```text
M=1,2,16,32,40,41,42,43,44,48,64,128
```

The one-site diagnostic covers:

```text
M=1,2,42,43,64
```

This is 315 fresh graph processes:

```text
primary:      18 M x 3 WS x 3 arms = 162
confirmation: 12 M x 3 WS x 3 arms = 108
diagnostic:    5 M x 3 WS x 3 arms =  45
```

M1 resolves the lower loss border. M40-44 distinguishes a gradual candidate
crossover from the discrete M42-to-M43 unfused Iris-to-RCCL switch. M48 through
M256 provide multiple points after fusion has ceased helping.

## Runtime estimate

Eight clean WS=8/156-site confirmation processes averaged 46.88 seconds.
Applying that conservative rate to all 315 processes gives 4.10 hours of graph
work. New WS compilation, orchestration, two block cooldowns, the separate
transition proof, and bounded reruns bring the planning estimate to about
5.7 hours. Reserve **6-8 hours**; do not treat that as a completion guarantee.

## Preflight and dry run

Run inside the profiler container from the `tokenspeed-kernel` root. Select
WS2/WS4 device sets only after verifying topology and an idle shared-host
snapshot; the campaign deliberately does not encode availability-dependent
device IDs.

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

The dry run must report 315 processes and performs no writes or GPU work.

## Collection command

After freezing device sets and an immutable raw root, remove `--dry-run` and
provide `--output-root`:

```bash
PYTHONPATH=. python3 benchmark/run_ar_rmsnorm_graph_sweep.py \
  --spec benchmark/results/ar_rmsnorm/studies/mi350x/\
2026-08-glm-5.2-fp8-definitive-sweep/campaign.json \
  --devices 2=<frozen-two-device-set> \
  --devices 4=<frozen-four-device-set> \
  --devices 8=0,1,2,3,4,5,6,7 \
  --output-root benchmark/results/ar_rmsnorm/raw/current/\
glm-5.2-fp8/mi350x/<date>/operator/definitive-graph-v1
```

The runner is resumable after validating result identity. It retains invalid
existing results, process logs, timeouts, failures, the resolved device map,
the specification hash, and elapsed time.

## Result integration

Only summarize a complete retained three-arm matrix. Generate the tracked
artifacts with:

```bash
python3 benchmark/analyze_ar_rmsnorm_graph_sweep.py <raw-root> \
  --max-m 256 \
  --output-json benchmark/results/ar_rmsnorm/studies/mi350x/\
2026-08-glm-5.2-fp8-definitive-sweep/graph-sweep-summary.json \
  --output-csv benchmark/results/ar_rmsnorm/studies/mi350x/\
2026-08-glm-5.2-fp8-definitive-sweep/graph-sweep.csv
```

Then add `summary.json` with:

- code, image, runtime, profile-spec hash, topology, and device identity;
- attempted/completed/failed process counts and elapsed time;
- per-WS profitable measured M values against both controls;
- the first measured loss after profit and the M42/M43 transport paths;
- pass spreads and every retained incident;
- a recommendation limited to operator evidence.

Replace this page's planned status with the measured decision and compact
tables, update the study index, and update the live GLM status last. Do not turn
WS2/WS4 scaling or forced post-cap rows into deployment claims.
