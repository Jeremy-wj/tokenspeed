# GLM-5.2-FP8 representative AR+RMSNorm characterization

Date: 2026-08-01 through 2026-08-03

## Decision

Captured WS=8 operator graphs establish a diagnostic profile-v2 opportunity,
not a serving result. At N=6144 with 156 AR+RMSNorm sites:

- four-warp padded Triton is 3.4%-20.9% faster than upstream-unfused for
  M=2-42;
- M=1 loses and must fall back;
- extending the padded kernel removes the old M33 profile-policy cliff;
- M=43 switches unfused from ordinary Iris to RCCL and makes padded Triton
  25.6% slower.

Profile v2 therefore implements:

```text
M=1:     ordinary fallback
M=2-42:  padded whole-row Triton, four warps
M>=43:   ordinary fallback
```

The profile passes captured graph and shared-state transition checks. Explicit
upstream-unfused remains the deployment default because production graph
serving and end-to-end performance are not qualified.

## Artifacts

- [summary.json](summary.json) — decision, environment, and validation counts;
- [padded-extension-summary.json](padded-extension-summary.json) —
  authoritative padded-range, M1, and profile-v2 results;
- [graph-sweep.csv](graph-sweep.csv) — initial three-arm sweep rows;
- [graph-sweep-summary.json](graph-sweep-summary.json) — initial pass-level
  statistics and comparisons.

The initial graph-sweep Triton rows above M32 use the superseded blocked
diagnostic path. Use `padded-extension-summary.json` for profile-v2 conclusions
at M=33-43. Raw logs and 1,000-sample rank arrays are ignored under
`raw/current/glm-5.2-fp8/`.

## Benchmark contract

The representative comparison used:

```text
hardware: 8x MI350X (gfx950)
world size / hidden size / dtype: 8 / 6144 / bf16
graph sizes: 1 site and 156 sites
warmup / measured replays: 50 / 1000
aggregation: max rank per iteration, then percentiles
arms: explicit upstream-unfused, default Iris fused, explicit Triton
```

The 156-site graph represents the expected 78-layer GLM reduction/norm count
and amortizes graph replay overhead. It is a synthetic forward proxy; the site
count has not yet been confirmed by marker-aligned production traces.

Representative 156-site max-rank p50 values in microseconds/site:

```text
M      unfused   Iris fused   padded Triton   Triton vs unfused
1       21.116       22.856           27.735              +31.4%
2       23.143       23.197           18.303              -20.9%
16      25.792       28.688           21.765              -15.6%
32      35.517       38.294           30.938              -12.9%
33      35.815       39.511           32.401               -9.5%
42      40.724       43.985           39.341               -3.4%
43      30.239       45.160           37.964              +25.6%
48      29.642       48.514           41.959              +41.6%
64      29.591       59.953           57.076              +92.9%
```

The M42 value is the exact-profile clean retry. Two forced padded runs measured
38.942 and 37.312 us/site; one retained exact-profile run entered a 90.887
us/site high-tail mode. M42 is therefore a smaller, less repeatable win than
M=2-33.

## Why the borders occur

### M=1

The original scalar-M specialization expanded the persistent row loop into a
209-KB branch-heavy program. Marking M non-specialized restores the compact M2
code shape. On a bounded WS=4, 156-site check, Triton improved from 15.639 to
12.225 us/site but remained 3.9% behind the 11.764 us/site unfused control.
Profile v2 retains ordinary M1 fallback.

### M=33

Profile v1 switched from scratch-free padded to scratch-backed blocked
one-shot above M32. Forcing padded through M33 changed the 156-site result from
71.609 to 32.401 us/site. The old cliff was a policy boundary, not a padded
kernel limit.

### M=43

A bf16 N=6144 row is 12,288 bytes. The 512-KiB ordinary-Iris limit ends at
M=42; M=43 selects RCCL in the unfused control. The resulting discontinuity
drops unfused from 40.724 to 30.239 us/site, while padded Triton continues to
scale upward. Larger-M measurements confirm that fusion has stopped helping.

## Safety and validation

Profile v2 sets the M=2 lower gate, M=42 cap, four warps, and 156 input/output
sites. Its exact-profile validation completed:

- 1,000 full-site M42 graph replays with changing inputs;
- 1,000 interleaved transition replays across M
  `1,2,16,32,33,40,42`;
- ordinary fallback at M1 and padded Triton at M=2-42;
- odd/even graph calls and eager/graph transitions;
- 1,649 checked operations/rank with zero failed steps.

The communication suite passed 25 tests. Two long exploratory campaign
sequences ended in transient ordinary Iris/RCCL lifecycle failures and passed
when isolated; an M192 eight-warp diagnostic stalled, and an M1024 unfused
single-site process reached the NCCL watchdog. These incidents are retained in
raw evidence and prevent treating the sweep as a serving safety gate.

## Serving pivot

Full-model bring-up established that the current GLM FP8 serving stack is not a
useful AR+RMSNorm performance baseline:

- the documented MoE backend was unsupported on AMD and required Triton;
- the flat KV-cache scheduler was incompatible with GLM DSA;
- default graph startup did not reach bounded readiness;
- eager serving was dominated by immature block-FP8 GEMM/MoE paths.

A narrow unfused screen measured 0.212 output tokens/s and 84.551 s median
TPOT. A five-pair canary measured fused mean TPOT 0.59% slower. These values
explain the pivot to operator evidence; they do not validate or refute the
captured operator result and are not deployment estimates.

## Identity

```text
model: zai-org/GLM-5.2-FP8
snapshot: 70311cfa0158cce7dd2cf5d2e04f68e3fdc3efc1
architecture: GlmMoeDsaForCausalLM
hidden size / layers: 6144 / 78
container: jeremwan-tokenspeed-profiler
image: jeremwan/tokenspeed:rocm7.2.4-torch2.11-profiler
image ID: sha256:04df0a0098e677846ecdd83981124c3e671cc5ea0361076ae88827ffe2bd2555
torch / HIP: 2.11.0+rocm7.2 / 7.2.53211
```

This image ID differs from the GPT-OSS qualified machine. The study records its
own runtime-library audit and does not inherit qualification by tag.

## Reproduction

From the TokenSpeed repository root, in a fresh shell:

```bash
source tokenspeed-kernel/benchmark/profiles/ar_rmsnorm/glm_5_2_fp8_mi350x.env

docker exec jeremwan-tokenspeed-profiler bash -lc '
  cd /home/jeremwan/tokenspeed/tokenspeed-kernel
  source benchmark/profiles/ar_rmsnorm/glm_5_2_fp8_mi350x.env
  HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  TS_TRITON_SHMEM_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  BENCH_WS=8 BENCH_N=6144 BENCH_M=42 BENCH_MAX_TOKEN_NUM=42 \
  BENCH_CALLS_PER_GRAPH=156 BENCH_N_WARMUP=50 BENCH_N_REPEAT=1000 \
  BENCH_IMPL=triton_shmem \
  PYTHONPATH=$PWD/python:$PWD \
  python3 -m benchmark.probe_ar_rmsnorm_graph_perf
'
```

The planned successor is the
[definitive WS=2/4/8 sweep](../2026-08-glm-5.2-fp8-definitive-sweep/README.md).
Do not combine its future results with this baseline until its complete
three-arm matrix and retained failures are available.
