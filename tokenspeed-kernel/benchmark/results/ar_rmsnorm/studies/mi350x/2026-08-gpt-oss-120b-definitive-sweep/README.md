# GPT-OSS-120B definitive AR+RMSNorm campaign

Status: **planned; not run**

This study predeclares the current-machine WS=2/4/8 eager, captured-graph, and
end-to-end comparison. It does not change the
[live deployment decision](../../../docs/gpt-oss-120b-status.md).

The immutable matrix and stopping rules are in [campaign.json](campaign.json).

## Purpose

Prior GPT-OSS performance data spans multiple implementations, rank sets, and
serving policies. This campaign replaces those environment-tied percentages
with one current-code comparison of:

1. explicit production unfused;
2. default Iris fused;
3. the WS-specific optimized Triton profile.

Reserved-sink graph padding, persistent outputs, the 72-site lifetime contract,
complete fallback, and closed tuning conclusions remain durable engineering
evidence.

## Current-machine identity

The model path is `/data/models/openai-gpt-oss-120b`. Before collection, record:

- kernel and parent TokenSpeed commits, dirty state, and campaign-file hashes;
- container tag, immutable image ID, mounts, imports, and package versions;
- model realpath plus config, tokenizer, index, and shard manifest hashes;
- all GPU identities, PCI/KFD/NUMA mapping, XGMI and peer topology, clocks,
  RAS/ECC state, and HIP-to-physical mapping;
- verified-idle, nested, topology-balanced WS2/4/8 device sets.

The historical image ID and old model path are evidence identities, not
qualification for this machine.

## Microbenchmark contract

All arms use N=2880 bf16 and max-rank-per-iteration statistics. Eager and graph
results are reported separately.

Captured 72-site primary pass:

```text
M=1,32,64,65,91,92,128,256,384,385,512
50 warmups, 1000 measured replays
```

Order-opposed graph confirmation:

```text
M=32,64,65,91,92,384,385
```

Two order-opposed eager passes:

```text
M=1,32,64,65,91,92,128,256,384,385,512,1024,2048
30 warmups, 150 measured iterations
```

This is 162 graph processes, 18 eager processes, and three WS-specific
transition probes. The boundaries resolve:

- M64-to-M65 padded versus blocked one-shot;
- M91-to-M92 ordinary Iris versus RCCL in the unfused control;
- M384-to-M385 Triton one-shot versus captured ordinary fallback.

The graph report preserves raw replay time and separately reports the unfused
benchmark-reset copy. Reset-adjusted values are serving-shaped diagnostics, not
directly measured kernels.

Dry-run both micro schedules from the `tokenspeed-kernel` root:

```bash
SPEC=benchmark/results/ar_rmsnorm/studies/mi350x/\
2026-08-gpt-oss-120b-definitive-sweep/campaign.json

PYTHONPATH=. python3 benchmark/run_ar_rmsnorm_graph_sweep.py \
  --spec "$SPEC" \
  --devices 2=<two-devices> --devices 4=<four-devices> \
  --devices 8=<eight-devices> --dry-run

PYTHONPATH=. python3 benchmark/run_ar_rmsnorm_eager_sweep.py \
  --spec "$SPEC" \
  --devices 2=<two-devices> --devices 4=<four-devices> \
  --devices 8=<eight-devices> --dry-run
```

The graph dry run must report 162 processes; eager must report 18. Actual
collection additionally requires an immutable `--output-root`. Run the
transition probe once per WS after freezing `selected-policy.json`.

For each frozen WS profile:

```bash
export AR_NORM_DEVICES=<frozen-devices>
export GPT_OSS_DEFINITIVE_FUSION_MAX_M=<selected-cap>
source benchmark/profiles/ar_rmsnorm/\
gpt_oss_120b_mi350x_definitive_ws<WS>.env

HIP_VISIBLE_DEVICES="$AR_NORM_DEVICES" \
TS_TRITON_SHMEM_VISIBLE_DEVICES="$AR_NORM_DEVICES" \
BENCH_IMPL=triton_shmem_profile BENCH_WS=<WS> BENCH_N=2880 \
PROBE_MS=1,32,64,65,91,92,384,385,512,2048 \
PROBE_REPLAYS=1000 PROBE_JSON=<raw-root>/transitions-ws<WS>.json \
PYTHONPATH=python:. python3 -m benchmark.probe_ar_rmsnorm_transitions
```

## Profile and cap decision

Keep `COMM_FUSION_MAX_NUM_TOKENS=2048` as workspace and capture identity.
Lowering it is not a free launcher optimization: it changes allocation sizes,
profile identity, graph topology, and fallback capture.

After microbenchmarks and before serving, freeze one actual-M eligibility
policy per WS:

```text
disable fusion, cap M64, cap M91, cap M384, or retain no additional gate
```

A cap requires consistent graph passes, relevant eager direction, margin beyond
pass spread, and the observed serving M distribution. Multiple caps must never
be tried against end-to-end results. Every selected profile reruns correctness
and transitions.

## End-to-end contract

The core campaign uses three matched three-arm triplets per WS: 27 fresh server
lifecycles total. Every observation uses:

```text
input/output: 128 / 512
prompts/concurrency: 128 / 32
request rate: infinite
temperature: 0
ignore EOS: true
overlap scheduling: disabled symmetrically
```

Output throughput is primary; median TPOT is the guardrail. Arm order is
balanced, one unfused observation is shared by both contrasts, and every
observation gets a fresh server because server reuse previously reproduced
scheduler/GPU stalls.

The first triplet at each WS also collects untimed concurrency-64 and direct
sequential-M512 marker traces. `tokenspeed.model_forward.v1` must prove
actual/executed M, graph/eager mode, and one rank-consistent backend path.

Extend from three to five triplets per WS only when signs disagree, a retained
failure is ambiguous, or the result lands within the existing promotion
thresholds. A deployment promotion still requires 15 pairs; stopping earlier
produces a definitive current-machine screen, not promotion evidence.

Inspect the three-WS serving schedule without starting a server:

```bash
PYTHONPATH=. python3 benchmark/run_gpt_oss_definitive_e2e.py \
  --spec "$SPEC" \
  --selected-policy benchmark/results/ar_rmsnorm/studies/mi350x/\
2026-08-gpt-oss-120b-definitive-sweep/selected-policy.json \
  --devices 2=<two-devices> --devices 4=<four-devices> \
  --devices 8=<eight-devices> --stage core --dry-run
```

Dry-run accepts the pending policy as ungated for schedule validation. Real
serving refuses to start until that file is `frozen` with one allowed cap per
WS. Run `--stage extension` into the same output root to add blocks 3-4, and
`--stage promotion` to add blocks 5-14. Stage directories remain separate; the
orchestrator rebuilds the combined per-WS paired analysis after each stage.

## Runtime

Historical process rates imply 2-3 hours for microbenchmarks and transitions,
plus 2.25-3.25 hours for 27 server lifecycles. Reserve **4-6 hours** for the core
campaign. The five-triplet extension adds about 1.5-2 hours. A full 15-triplet
promotion campaign is a separate 16-22-hour commitment.

## Result integration

After collection, track:

- [reporting-schema.json](reporting-schema.json);
- [selected-policy.json](selected-policy.json), changed from pending to frozen;
- `summary.json`;
- `graph-sweep.csv` and `graph-sweep-summary.json`;
- `eager-sweep.csv` and `eager-sweep-summary.json`;
- `forward-cohorts.json`;
- `end-to-end-summary.json`.

The final report must join executed-M forward cohorts to matching micro rows,
then keep operator delta, target-stage delta, full max-rank forward period,
TPOT, and throughput as separate outcomes. Update the live status only after
all artifacts for the selected stopping stage are complete.

Generate compact micro summaries with:

```bash
python3 benchmark/analyze_ar_rmsnorm_graph_sweep.py <graph-raw-root> \
  --max-m 512 --output-json graph-sweep-summary.json \
  --output-csv graph-sweep.csv

python3 benchmark/analyze_ar_rmsnorm_eager_sweep.py <eager-raw-root> \
  --output-json eager-sweep-summary.json --output-csv eager-sweep.csv
```

For each arm/WS trace set, run `analyze_ar_rmsnorm_forwards.py` with
`--graph-summary`, `--eager-summary`, `--arm`, and `--expected-world-size`.
Reject missing cohorts or cross-rank path disagreement before interpreting
end-to-end metrics.

## Time-saving exclusions

Do not repeat overlap root-cause work, M256 gating, blocked-core/grid/block
searches, folded copy-in, mutable two-slot rings, old hardware-fault theories,
marker-free grouping, or `all_reduce_two`. Do not flush L2 for this warm,
repeated-serving target, and do not use profiler request latency as a
performance metric.
