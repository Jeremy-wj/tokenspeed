# GLM-5.2-FP8 baseline and AR+RMSNorm screening on MI350X

Date: 2026-08-01

## Decision

Keep explicit upstream-unfused as the default for GLM-5.2-FP8.

The new `glm-5.2-fp8-mi350x-triton-v1` candidate is promising at the
AR+RMSNorm operator level but is not deployment- or performance-qualified:

- a captured 156-call M32 graph improved from 35.38 to 31.98 us/site
  versus upstream-unfused, a 9.61% operator reduction;
- the 1000-replay shared-state transition probe passed all eight ranks with
  1,690 checked operations/rank and no failed step;
- a bounded fused server resolved the intended profile on every rank and
  completed five canary requests;
- the matched five-pair, one-restart canary screen measured a **+0.59% mean
  TPOT regression**; this is below the 1% noise boundary and has the wrong
  direction;
- no marker-aligned full-model graph trace or three-block/fifteen-pair campaign
  is available, so the evidence gate is not met.

More importantly, the present GLM FP8 model stack is not a useful production
baseline on this image. The only bounded serving configuration used eager
execution and measured 84.55 s median TPOT at concurrency 16. AR+RMSNorm saves
roughly 0.53 ms per 156-site synthetic forward, which cannot materially move a
model step dominated by the current FP8 GEMM/MoE path.

Machine-readable results are in [summary.json](summary.json). Raw logs, traces,
and full probe samples are ignored under `raw/current/glm-5.2-fp8/`.

## Scope and identity

Model:

- checkpoint: `zai-org/GLM-5.2-FP8`;
- local snapshot:
  `/data/models/hf/hub/models--zai-org--GLM-5.2-FP8/snapshots/70311cfa0158cce7dd2cf5d2e04f68e3fdc3efc1`;
- indexed checkpoint size: 755,617,140,416 bytes;
- architecture: `GlmMoeDsaForCausalLM`;
- hidden size: 6144;
- main decoder layers: 78;
- quantization: block FP8, 128x128 scales.

The expected full-forward fused-site count is 156: 78 post-attention reductions,
77 next-layer input reductions, and one final reduction/norm. This count is
covered by the synthetic full-site graph but has not yet been proven by a
marker-aligned production graph trace.

Hardware and runtime:

- 8x MI350X, gfx950, all HIP devices `0,1,2,3,4,5,6,7`;
- container `jeremwan-tokenspeed-profiler`;
- image `jeremwan/tokenspeed:rocm7.2.4-torch2.11-profiler`;
- local image ID
  `sha256:04df0a0098e677846ecdd83981124c3e671cc5ea0361076ae88827ffe2bd2555`;
- torch 2.11.0+rocm7.2;
- system HIP runtime 7.2.53211 and rocprofiler-sdk 1.1.0.

The image ID differs from the historical GPT-OSS machine, so this study does
not inherit that machine's environment qualification by identity. The runtime
library audit showed HIP, HSA, RCCL, ROCTX, and roctracer resolving from
`/opt/rocm`, and the cross-thread HIP event-query capture probe passed.

## Method

The evidence ladder followed
[the project methodology](../../../docs/benchmark-methodology-recommendations-2026-07.md):

1. inspect model structure, dispatch, ownership, and fallback;
2. rerun the complete triton-shmem correctness suite on all eight devices;
3. collect order-opposed eager decomposition at N=6144;
4. screen blocked versus padded one-shot kernels and warp counts;
5. compare 156-call M32 graph replay with 1,000 measured replays;
6. run shared-state eager/graph transitions across M `1,8,16,31,32`;
7. bring up the real model with explicit fusion-off;
8. run a bounded fused server and a matched five-seed canary screen.

Every operator iteration reduces to max rank before the median. Cold startup,
JIT compilation, gateway registration, and failed configurations are retained
as incidents rather than folded into serving latency.

## Environment and bring-up incidents

### Unsupported documented MoE backend

The repository GLM recipe requested `--moe-backend flashinfer_trtllm`. The
first startup failed before weight loading:

```text
No kernel found for moe.apply ... solution 'flashinfer_trtllm'
on AMD Radeon Graphics
```

The AMD block-FP8 registry provides the Triton precomputed-routing kernels, so
the working launch uses `--moe-backend triton`.

### Flat scheduler incompatibility

The image carried `tokenspeed-scheduler` 0.1.3 built with
`TOKENSPEED_FLAT_KVCACHE=ON`. GLM's `DSATokenToKVPool` publishes no flat
paged-cache groups, and startup failed after loading the full checkpoint.

The same source was rebuilt in the container with the default radix path:

```bash
python3 -m pip install --no-cache-dir --force-reinstall --no-deps \
  ./tokenspeed-scheduler
```

The rebuilt wheel reported `tokenspeed_scheduler.FLAT_KVCACHE=False`.

### Unbounded default graph startup

Both attention-TP8/MoE-EP8 and attention-TP8/MoE-TP8 launches with normal graph
defaults remained at 100% GPU utilization without new progress logs for 40 and
18 minutes respectively. They were stopped as bounded-startup failures. Eager
execution reached readiness and is the only serving mode characterized here.

This means the GLM profile does not inherit GPT-OSS graph qualification. The
156-slot output and input rings are proven synthetically, not in captured model
serving.

### Health and gateway timing

The original generated deep health probe could not register the very slow GLM
worker. A single generated warmup token initially took one to two minutes, while
the gateway's worker-health timeout was much shorter.

The bounded screen therefore uses:

```text
TS_SERVE_ENGINE_MODULE=tokenspeed.runtime.entrypoints.safe_smg_server
TOKENSPEED_DEEP_HEALTH_MODE=passive
TOKENSPEED_PASSIVE_HEALTH_STUCK_SEC=3600
--gateway-startup-timeout 300
```

The extended passive threshold is required because a real request can remain
inside one model forward for much longer than the historical 30-second stuck
threshold.

### KVStore allocation

Even with `--disable-kvstore`, the radix runtime reserves a host tier for
retraction. The default ratio attempted 226 GB per rank and added about five
minutes to startup. `--kvstore-ratio 0` removed that allocation for the bounded
screen. This is a serving-policy change, not an AR+RMSNorm optimization.

### Cold JIT and steady state

The first successful 8-input/2-output canary took 762 seconds. Repeating after
the Triton caches were populated reduced the same workload to about 18 seconds.
Cold compilation is therefore excluded from TPOT, but retained as startup
evidence.

### Harness generalization faults

The new model exposed three GPT-OSS assumptions in the benchmark tools:

- graph replay looked up the old four-field triton-shmem cache key after state
  identity had gained profile/policy fields;
- graph replay removed `TS_TRITON_SHMEM_ONESHOT_BLOCK_N=0`, causing exact known
  profiles to decline;
- backend decomposition labeled every one-shot call as a 72-site, no-exit
  profile even when no site ring was configured.

The probes now use the production state-cache key, preserve sourced profile
policy, and derive ring/output/barrier labels from the resolved state.

A persistent interactive shell also preserved old diagnostic environment
values across `source`. The known profile correctly declined that launch. The
valid fused server was relaunched from a clean profile environment and proved
all resolved values on every rank.

## Operator characterization

The initial generic `triton_shmem` policy is not suitable for N=6144:

- M16: 86.8 us public total versus 46.5 us upstream-unfused;
- M32: 87.2 us versus 46.4 us.

The useful changed mechanism was the GLM-width equivalent of GPT-OSS core-v3:

- scratch-free padded 8192-lane whole-row core;
- four warps;
- in-kernel barriers;
- one persistent input and output pair per 156 model sites;
- one-shot limited to M<=32;
- fusion eligibility limited to M<=32 at the serving layer.

Corrected eager public totals for the site-156 candidate were:

- M1: 34.0 us versus 49.7 us upstream-unfused;
- M16: 33.6 us versus 46.3 us;
- M32: 37.4 us versus 46.1 us.

These eager rows are screening evidence. The authoritative fixed-shape result
is the full-site graph:

- upstream-unfused: 5.519 ms/graph, 35.38 us/site;
- Iris fused: 5.849 ms/graph, 37.49 us/site;
- triton profile v1: 4.988 ms/graph, 31.98 us/site.

Profile v1 reduces the synthetic max-rank median by 9.61% versus unfused and
14.72% versus Iris. Its p95 was 32.19 us/site.

## Correctness and transitions

The pre-change communication suite passed:

```text
23 passed in 271.88 s
```

It included ws=8 eager and graph coverage. The exact GLM profile then passed:

- 1,000 full 156-call M32 graph replays with changing inputs;
- persistent output and input-site rings;
- max-rank median 31.98 us/site;
- 1,000 interleaved transition replays;
- M `1,8,16,31,32`;
- odd/even calls per graph;
- eager-to-graph and graph-to-eager transitions;
- five captured graph variants;
- 1,690 checked operations/rank;
- zero failed steps and complete rank-path agreement.

The transition probe cannot safely inspect the private signal pad and records
`not_exposed_by_safe_public_api`, consistent with the existing methodology.

## Serving characterization

The bounded baseline is intentionally narrow:

```text
attention TP=8
MoE EP=8
max_model_len=80000
max_num_seqs=16
eager execution
passive deep health
KVStore normal use disabled, host ratio 0
comm fusion cap=32
```

Explicit upstream-unfused at concurrency 16, 16 requests, input 128 and output
8 measured:

- duration: 605.05 s;
- output throughput: 0.212 tokens/s;
- median TTFT: 13.188 s;
- median TPOT: 84.551 s.

This is a valid measurement of the current stack, but not a satisfactory
production baseline. The block-FP8 GEMM path repeatedly reported missing tuned
configs, and the Triton FP8 MoE path dominates the step.

The matched canary used five seeds, one server per arm, input 8, output 2, and
concurrency 1. Paired fused TPOT changes were:

```text
+1.206%, +1.632%, -1.938%, +0.750%, +1.294%
```

The paired mean is +0.59% and median +1.21%; positive is slower. One restart
block and five very short requests are diagnostic only. The result does not
meet the three-block/fifteen-pair evidence gate and does not support promotion.

## Extensibility changes

This study added:

- a GLM-5.2-FP8 MI350X model profile;
- exact profile validation for gfx950, WS=8, N=6144, cap32, and all eight
  visible devices;
- model-derived default world size, device set, cap, artifact root, and profile
  identity in the repeatability runner;
- generic serving/benchmark/teardown use instead of GPT-OSS wrappers;
- generic model-profile proof in serving campaigns;
- generic model support in the bounded reproducer;
- passive-health stuck-threshold propagation in the serving launcher;
- corrected graph-state lookup and decomposition metadata.

The GPT-OSS profile remains unchanged except for exporting its profile path,
world size, device set, and ignored physical GPU so the generic runner preserves
its prior defaults.

## Reproduction

From the repository root in a fresh shell:

```bash
source tokenspeed-kernel/benchmark/profiles/ar_rmsnorm/glm_5_2_fp8_mi350x.env

docker exec jeremwan-tokenspeed-profiler bash -lc '
  cd /home/jeremwan/tokenspeed/tokenspeed-kernel
  source benchmark/profiles/ar_rmsnorm/glm_5_2_fp8_mi350x.env
  HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  TS_TRITON_SHMEM_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  BENCH_WS=8 BENCH_N=6144 BENCH_M=32 \
  BENCH_CALLS_PER_GRAPH=156 BENCH_N_REPEAT=1000 \
  BENCH_IMPL=triton_shmem \
  PYTHONPATH=$PWD/python:$PWD \
  python3 -m benchmark.probe_ar_rmsnorm_graph_perf
'

PYTHONPATH=tokenspeed-kernel \
  python3 -m benchmark.repro_ar_rmsnorm_serving \
  --label glm52-v1 --backend triton_shmem --fusion 1 \
  --input-len 8 --output-len 2 --prompts 1 --concurrency 1 \
  --deep-health-mode passive --timeout 900
```

The reproducer is a safety screen, not promotion evidence. A future promotion
attempt must first establish a usable GLM FP8 model baseline, then add a
marker-aligned production graph trace and the full restart-randomized campaign.

## Next steps

1. Keep explicit `--disable-allreduce-fusion` as deployment/control default.
2. Fix or replace the current AMD block-FP8 GEMM and Triton FP8 MoE path before
   spending more time on end-to-end AR+RMSNorm optimization.
3. Add tuned N=6144 block-FP8 GEMM configurations and remeasure warmed model
   forwards.
4. Make default graph startup bounded for GLM and capture an authoritative
   156-site model trace.
5. Revalidate generated health and normal KVStore only after model steps are
   fast enough for their existing timeouts.
6. Run three restart blocks and fifteen matched pairs only after the unfused
   control reaches representative throughput.
