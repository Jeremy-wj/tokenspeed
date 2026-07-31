# AR+RMSNorm integration roadmap

Updated: 2026-07-31

This document records remaining engineering directions and their dependencies.
Deployment policy and execution priority belong only in
[GPT-OSS-120B status](gpt-oss-120b-status.md).

## Current baseline

Core-v3 is the qualified triton baseline for GPT-OSS-120B TP=4 on HIP
`1,2,5,6`. It combines:

- a padded scratch-free whole-row decode kernel for M<=64;
- blocked one-shot through M384 and two-shot above M384;
- a 72-site graph-stable input ring with no one-shot exit barrier;
- 72 persistent captured output sites;
- eager two-shot borrowed-output ping-pong.

It clears the capacity gate at +1.29% output throughput and -0.80% median TPOT.
The [core-tuning study](../studies/mi350x/2026-07-triton-shmem-core-tuning/README.md)
owns the measurements.

## Closed work

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

The [realignment](../studies/mi350x/2026-07-triton-shmem-realignment/README.md)
and [decomposition](../studies/mi350x/2026-07-triton-shmem-decomposition/README.md)
studies contain the closure evidence.

## Remaining work

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

### 3. Small decode code generation

The padded core remains about 1.13 us/site behind Iris and the complete fused
operation about 1.43 us/site behind serving-faithful upstream-unfused. A further
kernel investigation is justified only if it changes generated code or register
behavior for the padded whole-row specialization. It must be ranked under the
72-call captured graph, not eager timing.

### 4. Producer-direct input and progress publication

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
