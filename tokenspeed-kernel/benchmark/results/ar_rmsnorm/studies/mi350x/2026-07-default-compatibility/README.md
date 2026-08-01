# GPT-OSS-120B default compatibility

Date: 2026-07-31 through 2026-08-01

## Decision

Core-v3 is safe with TokenSpeed's base serving defaults on GPT-OSS-120B TP=4,
MI350X, HIP `1,2,5,6`:

- prefill graphs enabled with the normal 40-bucket ladder through M2048;
- automatic decode capture sizes;
- 0.95 GPU memory utilization;
- overlap scheduling enabled;
- standard generated health probes.

The earlier eager-prefill, C32-only, 0.90-HBM, passive-health, and
disabled-overlap settings were incident controls, not required fixes. The
reserved graph-padding sink row, 72 persistent output sites, 72 input sites,
explicit copy-in, and complete unfused fallback remain required.

The default-compatible profile is **not performance-promoted**. Base overlap
scheduling is correct, but fresh servers repeatedly occupied distinct ~29 s and
~37 s performance modes in both arms. A matched no-overlap policy removed that
variance and produced the final clean 15-pair result:

- output throughput: **+0.45%** (95% CI -0.05% to +0.92%);
- median TPOT: **-0.51%** (95% CI -0.90% to -0.06%);
- mean TPOT: **-0.48%** (95% CI -0.93% to +0.01%).

Median TPOT improves with its interval excluding zero, but throughput does not
clear the +1% capacity threshold and latency does not reach the -1.5%
objective. `--disable-overlap-schedule` is therefore a reversible benchmark
performance policy, not a correctness requirement or a production default.
Explicit upstream-unfused remains the deployment default and fallback. The
earlier +1.29% capacity promotion remains evidence for the restricted
historical configuration, not current policy.

Machine-readable results are in [summary.json](summary.json). Raw logs and
campaign trees are ignored under `raw/`.

## Campaign lineage

Newest canonical evidence first:

1. `2026-08-01-stable-no-overlap-qualification` — final clean 15-pair
   performance decision; not promoted.
2. `2026-08-01-stable-default-qualification` — overlap-on diagnostic; safe but
   fresh-server performance modes prevent useful promotion inference.
3. `2026-07-31-compat-captured-fallback-qualification` — superseded noisy
   evidence retained only for incident history.
4. `2026-07-triton-shmem-core-tuning` — historical +1.29% restricted-control
   capacity promotion, no longer deployment policy.
5. `final-perfetto-{fused,unfused}` — canonical matched no-overlap end-to-end
   traces.

## Integration changes

The compatibility pass:

1. removed `--disable-prefill-graph`, `--cudagraph-capture-sizes 32`,
   `--gpu-memory-utilization 0.90`, and `--disable-overlap-schedule` from the
   GPT-OSS profile;
2. restored the standard SMG engine and generated health mode;
3. kept `TS_ARNORM_BACKEND=auto` Iris-first and triton-shmem explicit-only;
4. moved the gfx950/TP=4 grid cap and in-kernel barrier enablement into the
   qualified profile instead of generic defaults;
5. validates known profile architecture, rank set, topology, hidden size,
   dtype, token cap, and complete kernel/lifetime policy before allocation;
6. keys communication state by profile and every allocation/synchronization
   policy;
7. synchronizes coarse HIP-IPC failures across ranks and falls back to
   fine-grained symmetric data buffers;
8. declines captured calls above the persistent M384 output-ring cap and
   captures the complete ordinary fallback instead of transient custom-kernel
   outputs;
9. parameterizes repository/container paths and removes launcher variables
   with no runtime consumer.

Unknown or mismatched profiles decline before triton-shmem state construction.
The layernorm caller then performs the missing all-reduce and ordinary
residual-add + RMSNorm.

## Most important optimizations

1. **Scratch-free padded decode core.** Non-power-of-two hidden 2880 uses one
   masked 4096-lane program per row at M<=64. This removed the fp32 scratch
   write/reload and improved the 72-call captured graph by 35.2%.
2. **Graph-stable 72-site input ownership.** One symmetric input view per fused
   site delays reuse for a complete GPT-OSS forward, allowing the qualified
   one-shot path to remove its exit barrier without mutable host phase.
3. **Persistent captured outputs.** Seventy-two preallocated output pairs keep
   decode custom-kernel pointers stable. Captured calls above M384 decline to
   ordinary AR+RMSNorm rather than retaining transient two-shot outputs.
4. **Coarse data, fine signals.** Bulk buffers use cached HBM exported through
   HIP IPC while the small signal pad remains fine-grained for system-scope
   atomics. Collective failure agreement safely falls back to fine memory.
5. **Eager two-shot output borrowing.** Two symmetric output pairs ping-pong
   above M384, removing eager copy-out while preventing residual aliasing.
6. **Profile-owned launch policy.** Four-warps padded decode and the gfx950/TP=4
   grid cap are explicit profile choices; generic triton-shmem defaults remain
   conservative and divergence-safe.

## Critical bugs closed

- Graph padding formerly aliased live request slot 0, producing negative
  8-KiB KV-page underflows. Every padded row now uses the reserved sink.
- Capture-time `torch.empty_like` outputs were transient graph-pool storage.
  Persistent per-site outputs and captured-oversize decline close that lifetime
  class.
- Folded copy-in lacked a serving-qualified producer publication contract and
  remains disabled.
- Scalar system-scope barriers could run before sibling wavefront memory
  operations. Workgroup synchronization now brackets every multi-wave barrier.
- A fused-backend decline previously normalized rank-local partials. The caller
  now restores the missing all-reduce before ordinary RMSNorm.
- State caches formerly ignored environment-owned lifetime policy. Profile,
  allocation, synchronization, device, and shape now participate in identity.
- Launcher port races, hard-coded checkout paths, and no-op environment
  controls were removed or made explicit.

## Validation

Static and unit coverage:

- launcher and locked-port tests: 51 passed;
- runtime argument/profile tests: 18 passed;
- repeatability and graph-analysis tests: 27 passed;
- prefill-graph and sink-padding tests: 18 passed;
- triton-shmem communication suite: 20 passed, 3 WS=8 tests deselected;
- RMSNorm/Gemma fused-decline fallback: 1 two-rank test passed;
- shell syntax, Python compilation, diff whitespace, and IDE diagnostics: pass.

The 1000-replay transition probe covered M
`1,32,64,256,384,385,512,1024,2048`, nine captured graphs, odd/even call
counts, and 1,652 checked operations per rank with zero failed steps. It proved:

```text
M=1,32,64: padded whole-row one-shot
M=256,384: blocked one-shot
M=385,512,1024,2048: two-shot
```

Bounded serving:

- a narrow prefill-graph smoke captured `[128,256,2048]` and completed 8/8;
- triton-selected default full ladder completed 128/128 at every M128-M4096
  workload and long decode; captured calls above M384 proved ordinary fallback;
- explicit upstream-unfused completed the same full ladder without failures;
- profile resolution, state policy, graph engagement, and backend selection
  were proven on all four ranks.

The final performance run used three restart blocks, five paired seeds per
block, a fresh server for every seed, explicit fusion-off controls, and the
paired hierarchical bootstrap. All 15 measured pairs were clean; one
pre-measurement startup port collision was rerun through the normal resume path.

## Perfetto traces

Final 32-step end-to-end Kineto/Perfetto traces were captured for all four ranks
under the matched no-overlap policy:

```text
fused:   raw/.../2026-08-01/serving-repros/final-perfetto-fused/
         traces/repro-repeat0/*-TP{0,1,2,3}.trace.json.gz
unfused: raw/.../2026-08-01/serving-repros/final-perfetto-unfused/
         traces/repro-repeat0/*-TP{0,1,2,3}.trace.json.gz
```

Each fused rank contains 2,232 padded one-shot signatures and no Iris fused
signature. Each unfused rank contains 2,190 ordinary Iris all-reduce, 4,704
RCCL, and 2,336 RMSNorm signatures.

## Reproduction

From the repository root:

```bash
source tokenspeed-kernel/benchmark/profiles/ar_rmsnorm/gpt_oss_120b_mi350x.env

PYTHONPATH=tokenspeed-kernel \
  python3 -m benchmark.repro_ar_rmsnorm_serving \
  --label compat-defaults-full --devices 1,2,5,6 \
  --backend triton_shmem --fusion 1 --suite full --timeout 600

PYTHONPATH=tokenspeed-kernel \
  python3 -m benchmark.run_ar_rmsnorm_repeatability \
  --comparison triton_shmem --blocks 3 --seeds 0,1,2,3,4 \
  --decode-only --skip-profiles --disable-overlap-schedule \
  --devices 1,2,5,6
```

Physical GPU 3 remained occupied, so WS=8 was not run and remains unqualified.
Earlier noisy and foreign-process-contaminated roots remain ignored historical
evidence; they do not contribute to the final summary.
