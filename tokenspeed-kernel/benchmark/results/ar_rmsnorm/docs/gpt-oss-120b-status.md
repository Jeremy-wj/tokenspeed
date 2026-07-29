# GPT-OSS-120B fused AR+RMSNorm status

Updated: 2026-07-29

This is the sole live deployment decision and priority page for GPT-OSS-120B,
TP=4, on MI350X (gfx950).

## Deployment decision

- There is no post-rebase project performance baseline or promotion decision.
- Upstream `main` at `3f88dcc2` now makes AMD `auto` use Iris for fused
  AR+RMSNorm and ordinary small all-reduce. Upstream may auto-enable fusion for
  supported single-node AMD TP mappings.
- `TS_ARNORM_BACKEND=triton_shmem` is an explicit experimental candidate; it
  no longer replaces upstream `auto`.
- Profile `gpt-oss-120b-mi350x-qualified-v4` is legacy performance evidence.
  It qualified the old `triton_shmem` integration, not Iris or the rebased
  runtime.
- The graph-padding and captured-output lifetime faults remain closed
  historical incidents. Their invariants still apply to captured buffers.

The old final campaign completed three independent restart blocks and fifteen
fused/unfused pairs without a safety failure, but is not a post-rebase
baseline. Raw campaign:
`../raw/current/gpt-oss-120b/mi350x/2026-07-29/2026-07-29-output-ring-v4-fused-vs-unfused/`.

See [upstream-main rebase impact](upstream-main-rebase-impact-2026-07.md) for
the backend analysis and baseline reset.

## Legacy qualified TP=4 profile

Scope:

- model: `/data/models/openai/gpt-oss-120b`, hidden size 2880 bf16 elements;
- hardware: AMD Instinct MI350X (gfx950), canonical HIP set `1,2,3,5`;
- runtime: torch 2.11 with released ROCm 7.2.4 userspace;
- shared-host rule: exclude physical GPU 3 / HIP index 0 for TP<8.

The pre-rebase profile-v4 reproduction requires:

```text
AR_NORM_PROFILE_ID=gpt-oss-120b-mi350x-qualified-v4
TS_TRITON_SHMEM_FOLD_COPYIN=0
TS_TRITON_SHMEM_DOUBLE_BUFFER_INPUT=0
TS_TRITON_SHMEM_OUTPUT_RING=72
TS_TRITON_SHMEM_BARRIER_GRID=0
TS_SERVE_ENGINE_MODULE=tokenspeed.runtime.entrypoints.safe_smg_server
TOKENSPEED_DEEP_HEALTH_MODE=passive
--gpu-memory-utilization 0.90
--disable-overlap-schedule
--disable-prefill-graph
--cudagraph-capture-sizes 32
```

TP=4 opt-in additionally requires:

```text
TS_ARNORM_BACKEND=triton_shmem
--enable-allreduce-fusion
--comm-fusion-max-num-tokens 2048
```

A valid fused run must show `enable_allreduce_fusion=True` in server arguments
and fused kernel signatures in every rank trace.

## Historical 2026-07-24 evidence

The favorable 2026-07-24 matched pair used an older serving profile. It is
historical mechanism evidence and was superseded first by profile v4 and then
by the upstream baseline reset; it must not be described as current.

- generic fused decode median: 34.1–36.5 µs across ranks;
- maximum same-rank unfused AR + RMSNorm median sum: 36.919 µs;
- GPU kernels per rank: 29,967 fused versus 31,333 unfused;
- profiled GPU-window span: about 313.8 ms fused versus 340.5 ms unfused
  (-7.8%);
- mean median TPOT: 12.560 ms fused versus 12.755 ms unfused (-1.5%).

Sources:
`../studies/mi350x/2026-07-profile-guided-followup/corrected_profile_comparison.json`,
`../studies/mi350x/2026-07-profile-guided-followup/e2e_summary.json`, and
`../raw/current/gpt-oss-120b/mi350x/2026-07-24/`.

## Legacy dispositions

- Profile-v4's `triton_shmem` performance rejection remains the final decision
  for the pre-rebase implementation only.
- Keep the M=256 performance gate rejected.
- Keep the host-alternated two-slot/no-exit input ring rejected until slot
  identity is graph-stable across captured variants and request waves.
- Preserve reserved-sink graph padding, persistent per-site output storage,
  explicit copy-in, the original one-shot exit barrier, passive health, eager
  prefill, and disabled overlap.
- Treat legacy trace grouping as heuristic; future traces must use
  `tokenspeed.model_forward.v1` markers.

## Priorities

1. Establish an upstream-unfused control on the rebased code and record the
   ordinary AR backend and kernel signatures.
2. Establish the upstream Iris-first fused baseline with correctness, decline,
   graph, and transition evidence.
3. Compare explicit `triton_shmem` against those controls on the identical
   code, image, topology, and workload. Do not transfer old percentages.
4. Evaluate `all_reduce_two` and the NVIDIA lane/latent-norm APIs as separate
   primitives; they are not GPT-OSS AR+RMSNorm results.
5. Only after operator, graph, transition, and marker-aligned profiling gates
   pass, run the repeatability runner's three-block/fifteen-pair campaign with
   fresh servers, GPU isolation, signature proof, checksums, and failure gates.
6. Preserve graph-stable output lifetime, universal sink padding, complete
   fallback, and explicit completion/barrier contracts in every candidate.

Technical ownership:
[serving root cause](gpt-oss-120b-serving-root-cause.md),
[backend design](backend-design-and-safety.md),
[producer lifetime contract](producer-lifetime-contract.md), and
[profiling workflow](profiling-workflow.md).

