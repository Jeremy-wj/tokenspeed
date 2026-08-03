# AR+RMSNorm integration roadmap

Updated: 2026-08-03

This document records remaining engineering directions and their dependencies.
Deployment policy remains in the model status pages:
[GPT-OSS-120B](gpt-oss-120b-status.md) and
[GLM-5.2-FP8](glm-5.2-fp8-status.md).

## GPT-OSS-120B baseline

Core-v3 is the qualified triton baseline for GPT-OSS-120B TP=4 on HIP
`1,2,5,6`. It combines:

- a padded scratch-free whole-row decode kernel for M<=64;
- blocked one-shot through M384, eager two-shot above M384, and captured
  oversize decline to ordinary fallback;
- a 72-site graph-stable input ring with no one-shot exit barrier;
- 72 persistent captured output sites;
- eager two-shot borrowed-output ping-pong.

Its restricted historical campaign cleared the capacity gate at +1.29% output
throughput and -0.80% median TPOT. The
[core-tuning study](../studies/mi350x/2026-07-triton-shmem-core-tuning/README.md)
owns those measurements.

The compatibility pass removed historical serving controls that were not root
causes. The profile now retains TokenSpeed defaults for prefill/decode graphs,
memory utilization, overlap scheduling, and health probes. Profile validation,
collective coarse-IPC fallback, policy-aware state keys, and complete unfused
decline make those defaults fail closed rather than relying on launcher
restrictions. The final clean matched no-overlap campaign measured +0.45%
throughput and -0.51% median TPOT and did not promote. Base overlap remains
correct but produced fresh-server performance mode variance; overlap
disablement is a reversible measurement policy. Explicit unfused remains the
deployment default. See the dated default-compatibility study.

## GLM-5.2-FP8 baseline

Profile v2 is a diagnostic TP=8/N=6144 operator candidate:

- four-warp padded whole-row Triton for M=2-42;
- ordinary fallback at M=1 and M>=43;
- 156 graph-stable input and output sites;
- 1,000-replay M-boundary transition coverage on all eight ranks.

The M43 boundary is structural for the current comparison: unfused switches
from ordinary Iris to RCCL above 512 KiB. Captured operator graphs are
qualified; model graph serving and end-to-end performance are not. The
[baseline study](../studies/mi350x/2026-08-glm-5.2-fp8-baseline/README.md)
owns the current evidence, and the
[definitive sweep](../studies/mi350x/2026-08-glm-5.2-fp8-definitive-sweep/README.md)
owns the pending WS=2/4/8 campaign contract.

## GPT-OSS-120B closed work

Do not repeat these local searches without a changed mechanism or precondition:

- blocked-core block width, grid cap, XCD mapping, or fast-path sweeps;
- one-shot caps above M384;
- fixed barrier grids;
- folded copy-in;
- mutable two-slot input reuse;
- eager two-shot copy-out removal;
- one-shot exit-barrier removal for the qualified 72-site profile;
- M=256 deployment gating;
- M2048 tuning without a different communication algorithm.
- eager prefill, C32-only decode capture, 0.90 HBM utilization, passive health,
  and globally disabled overlap as incident-control requirements.

The [realignment](../studies/mi350x/2026-07-triton-shmem-realignment/README.md)
and [decomposition](../studies/mi350x/2026-07-triton-shmem-decomposition/README.md)
studies contain the closure evidence.

## GPT-OSS-120B remaining work

### 1. Rank-set and world-size qualification

Requalify each topology independently. A profile must record logical rank, HIP
index, physical GPU, NUMA placement, and peer-access matrix. Decline the fused
path unless every rank resolves the same supported profile. WS=8 remains
deferred.

Completion requires the full safety ladder, marker-aligned trace, and 15-pair
campaign on the target rank set.

### 2. Mixed transport transitions

The synthetic sequence that mixes ordinary Iris and RCCL fallback under graph
capture timed out, although bounded serving passed. Isolate graph-pool sharing,
communicator epochs, and capture order while preserving:

- complete unfused fallback;
- separate state identities for incompatible graph/stream roles;
- zeroed signal slots after every successful transition;
- no weakening of production graph isolation.

### 3. Overlap performance mode stability

Base overlap scheduling passes the full safety ladder, but repeated fresh
servers occupy distinct ~29 s and ~37 s decode modes in both fused and unfused
arms. Isolate scheduler state, host readiness, and launch cadence before using
overlap for promotion evidence. Do not recast `--disable-overlap-schedule` as a
kernel correctness requirement.

### 4. Small decode code generation

The padded core remains about 1.13 us/site behind Iris and the complete fused
operation about 1.43 us/site behind serving-faithful upstream-unfused. A further
kernel investigation is justified only if it changes generated code or register
behavior for the padded whole-row specialization. It must be ranked under the
72-call captured graph, not eager timing.

### 5. Producer-direct input and progress publication

Treat this as one cross-layer systems project, not independent copy or barrier
experiments. Required pieces:

1. dense row-parallel GEMM accepts an exact caller-owned output view;
2. active MXFP4 MoE finalize forms its exact reserved `[M,N]` output;
3. the compiler/runtime reserves graph-stable symmetric storage before launch;
4. producer completion is published with ROCm-supported system-scope ordering;
5. fused decline retains a valid rank-local partial for complete fallback;
6. input, residual, and norm outputs have explicit graph epochs and ownership.

The target regime is M512-M1024, where the two-shot core is competitive but
copy and required synchronization cost roughly 25-30 us. Copy removal alone is
insufficient. See the
[producer lifetime contract](producer-lifetime-contract.md).

Stop if the design replaces one staging copy with another, cannot support both
dense and active MoE paths, or requires weaker fallback/lifetime semantics.

## GLM-5.2-FP8 remaining work

1. Run the definitive operator campaign without changing its predeclared
   matrix; retain failures and report WS=2/4 as scaling evidence only.
2. Resolve the AMD block-FP8 GEMM/MoE serving baseline and bounded graph
   startup before another AR+RMSNorm end-to-end campaign.
3. Capture `tokenspeed.model_forward.v1` markers and compare executed decode M
   with the M=2-42 operator window.
4. Re-run the shared-state transition gate after any profile, topology, cap, or
   site-count change.
5. Require a restart-randomized serving campaign before changing the explicit
   unfused deployment default.

## State and dispatch requirements

Every communication state needs a stable identity containing:

```text
group generation, model profile, world size, hidden size, dtype,
workspace cap, allocation generation, graph owner, signal range
```

Construct and rendezvous all planned states before capture. Warm every selected
kernel, run a checked dry call, and verify signal slots return to zero. Reject
state destruction or recreation while a captured graph references its
addresses.

Known model profiles must resolve identically on every rank before state
construction. Configuration changes that affect allocation, synchronization,
or graph lifetime must create a different cache key. Generic triton-shmem runs
default to separate barriers and no architecture-specific grid cap; qualified
profiles opt into faster policies explicitly.

Resolve one inspectable profile from:

```text
(architecture, topology, rank set, world size, hidden, dtype,
 actual/executed M, graph mode, divergence capability)
  -> (fuse/decline, backend, kernel family, grid, synchronization,
      workspace, output and lifetime policy)
```

Environment variables may remain diagnostic overrides; production behavior
must log the resolved profile and per-rank path.

## Design invariants

- Preserve complete unfused semantics on every decline or failure.
- Keep coarse HIP-IPC data separate from fine-grained signal memory.
- Require graph-stable ownership for addresses, epochs, sites, and channels.
- Retain generic barriers on every path without a complete delayed-reuse proof.
- Keep captured outputs in persistent caller-owned or profile-owned storage.
- Treat graph/operator savings as screening evidence, not TPOT forecasts.
- Qualify every model, topology, world size, and crossover independently.
- Require integration complexity to purchase a measurable serving effect.
