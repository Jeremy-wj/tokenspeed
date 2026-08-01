# AR+RMSNorm benchmarking and promotion methodology

Updated: 2026-08-01

## Purpose

This document defines the evidence needed to promote an AR+RMSNorm integration
change. Qualified environments, commands, implemented harness behavior, and
artifact layout belong in the [profiling workflow](profiling-workflow.md).

Upstream auto-enables fusion on supported AMD TP mappings. Therefore an
unfused arm is valid only when it records `--disable-allreduce-fusion` and
resolved `enable_allreduce_fusion=False`; omission of the enable flag is not a
control.

> The candidate is the complete serving state machine, not only the fused
> kernel.

Allocation, rendezvous, graph capture, health transitions, prefill/decode,
buffer lifetime, fallback, topology, scheduling, and restart behavior are part
of the measured object. Fixed-shape timing is screening evidence only.

The qualification history demonstrates why safety and performance are separate:
profile v4 and the post-rebase fused baselines passed safety but failed
performance, while core-v3 later passed both the safety/evidence gates and the
capacity objective under restricted serving controls. Default-compatible
requalification remained safe but did not promote. Dated results belong in the
[study index](../studies/README.md); the live decision belongs in
[GPT-OSS-120B status](gpt-oss-120b-status.md).

The methodology must answer, independently:

1. Is the candidate numerically and synchronously correct?
2. Does it survive actual serving transitions?
3. Does it reduce the max-rank model critical path?
4. Does it improve latency or capacity across independent restarts?

## Identity and phase rules

A pair is valid only when code/runtime, model, rank topology, graph/scheduler
policy, communication state, workload, and health sequence match except for the
declared candidate. Preserve resolved state and per-rank signatures; labels such
as `fused`, `gate`, or `ring` are insufficient.

Record initialization, symmetric allocation/rendezvous, compile warmup, graph
capture, passive readiness, functional canary, workload warmup, prefill, steady
decode, and teardown as separate phases. An M=1 generation canary must be
explicit and timestamped, not an implicit health probe inside measurement.
Cold-start, transition, and teardown failures must not be folded into TPOT.

## Required evidence ladder

Higher levels do not excuse missing lower-level proof. Retain every failure as
a result.

### Level 0: static dispatch and lifetime proof

Document:

- integration call path, eligibility, and complete fallback;
- owner and lifetime of every input, output, scratch, and signal allocation;
- participant/channel set and graph capture/replay assumptions;
- expected one-shot, two-shot, fused, and unfused signatures;
- odd/even site-count validity.

The normative contracts are
[backend design and safety](backend-design-and-safety.md) and the
[producer/buffer lifetime contract](producer-lifetime-contract.md).

### Level 1: eager correctness and operator screening

Cover changing random inputs, every M dispatch boundary, arbitrary and
power-of-two model widths, qualified world sizes, one-/two-shot paths, and
complete fallback. Compare both outputs with fp32-aware tolerances and verify
the assigned signal range returns to zero.

Keep setup outside timing. Use randomized or order-opposed passes with enough
warmup and repeats to expose tails. For each iteration, take the maximum rank
first and then compute percentiles; max-of-rank-medians can hide the collective
straggler. Retain raw samples and exact path, grid, clocks, and code identity.
Subtraction-derived phase costs are diagnostic because phases overlap.

### Level 2: fixed-shape graph replay

Preallocate and rendezvous before capture; replay changing source inputs and
validate outputs afterward. Require at least 1,000 screening replays, even and
odd calls per graph, a rank-skew correctness probe, signal-zero checks, and
agreement between graph and eager direction.

This level can expose a graph-only opportunity, but it does not exercise graph
variants or the serving scheduler.

### Level 3: shared-state transitions

Exercise both directions across:

- M=1, steady decode M, and each dispatch boundary;
- one-shot and two-shot;
- graph and eager;
- repeated and interleaved graph variants;
- shared and intentionally separate state identities;
- both ring epochs and odd/even call chains;
- functional canary and rank-skew controls.

Compare graph ID, actual/executed M, path, grid, epoch, channel, and slot across
ranks. Any disagreement is a failure even if execution completes.

### Level 4: bounded full-server reproduction

Reproduce startup, health, prefill/decode, and teardown on the real
model/scheduler path before expensive benchmarking. Preserve first-failure logs
and signal state and use fused-off, communication-only, or barrier controls when
needed to isolate a transition.

Success requires expected output counts and per-rank path signatures. A timeout
is a safety failure, not an outlier.

### Level 5: marker-aligned production trace

New traces require the versioned `tokenspeed.model_forward.v1` marker with
forward ID, mode, actual/executed M, batch/padded size, and execution path.
Analyze each rank independently, align by forward ID and mode, and reject
missing, duplicated, reordered, or cross-rank-inconsistent forwards.

For every aligned forward, compute target-stage time per rank, select that
forward's max rank, and report distribution, start-to-start period, and target
share separately for decode, prefill, mixed, eager, and graph paths. Retain
per-forward rows.

Pooled kernel medians are not a model-step budget. Legacy marker-free grouping
is `heuristic_legacy`, not promotion evidence. Kineto establishes signatures,
counts, order, and within-trace periods; profiler-perturbed request latency is
not the serving effect.

### Level 6: restart-randomized end-to-end campaign

Require:

- at least three independent restart blocks;
- five paired seeds per block and a fresh decode server per seed;
- separate prefill servers and randomized arm order within each block;
- identical workload, warmup, and health policy within each pair;
- GPU-process isolation, hard timeouts, request timelines, serve proof, and
  per-rank signatures;
- resume only after artifact revalidation;
- retained failures and partial blocks.

Fifteen successful pairs are the minimum performance sample. A safety failure
can reject a candidate earlier.

## Objective and workload separation

Declare one primary objective before running:

- latency: paired median TPOT;
- capacity: output tokens/s;
- prefill: TTFT at declared token buckets;
- startup: time to qualified readiness.

Treat other metrics as guardrails. Do not infer capacity from TPOT, mix
saturated-burst and finite-arrival workloads, or mix startup with steady-state
latency.

The canonical steady-decode workload is useful for M=32, but it must be
complemented when the mechanism touches other regimes:

- concurrency bands for graph and scheduling transitions;
- fixed prefill/TTFT token buckets around dispatch boundaries;
- explicit health and prefill-to-decode transitions;
- representative finite-arrival production mixes after safety qualification.

Report correctness, start/transition success, failure phase, TPOT/ITL/TTFT,
output throughput, aligned max-rank forward period and target-stage time, and
offered versus achieved load as separate outcomes.

## Statistical design

### Pairing and hierarchy

The e2e unit is a candidate/control pair sharing seed, prompts, qualified rank
set, restart block, workload, and health sequence, with randomized order.
Report every paired percentage change; do not average arm medians first and form
an unpaired ratio.

Variance has at least two levels:

1. server restart block;
2. seed/workload pair within block.

Use a paired hierarchical bootstrap that resamples blocks and then pairs within
blocks. Report the point estimate, 95% interval, block/pair counts, and all pair
values. Fewer than three complete blocks may produce diagnostics but cannot
produce promotion eligibility.

### Failure-aware reporting

Performance and reliability are joint outcomes:

- report attempted and completed pairs;
- never impute a timeout or drop its paired control;
- preserve partial blocks;
- stop for repeated safety failures rather than accumulate only passing runs.

If a candidate fails safety, failed qualification is the primary result even if
its completed requests are faster.

## Decomposition-to-serving transfer rules

Use cumulative stage prefixes for additive critical-path decomposition. Sums of
independently timed copies, barriers, and kernels can over-count launch overlap.

An operator trend can explain serving only when all of these match:

- executed M and one-/two-shot dispatch;
- eager versus captured execution;
- rank set and max-rank aggregation;
- output ownership and copy-out policy;
- graph site count and barrier/lifetime contract;
- ordinary Iris versus RCCL transport selection.

In particular:

- eager M32 is not decode evidence; capture amortizes Iris launch/barrier costs
  enough to reverse the backend ordering;
- aggregate prefill bucket names are not executed M proof;
- prefill TTFT improvements do not satisfy decode TPOT guardrails;
- probe reset copies for in-place unfused transport must be labeled and removed
  from serving-stage accounting;
- graph and marker target sums explain direction, while only paired campaign
  TPOT/throughput can promote deployment.

The reference implementation and current measurements are in the
[three-backend decomposition study](../studies/mi350x/2026-07-triton-shmem-decomposition/README.md).

## Promotion gates

### Universal safety gate

Require eager correctness, 1,000+ fixed-graph replays, the relevant transition
matrix, interleaved shared-state graph coverage, explicit M=1 transition, a
bounded full serve, complete fallback, and no signal/channel/epoch disagreement.
Model-profile site rings additionally require a full-site graph (72 calls for
GPT-OSS), wraparound, interleaved graph variants, and proof that every unknown
site-count path retains its exit barrier.

### Evidence gate

Require immutable environment/code identity, resolved state and per-rank
signatures, authoritative forward markers, marker-aligned max-rank analysis,
three complete restart blocks, fifteen paired observations, and no unaccounted
GPU process.

### Performance gate

Current campaign thresholds are:

- latency: at least **1.5%** paired TPOT improvement, 95% interval excluding
  zero, and no material throughput regression;
- capacity: at least **1%** output-throughput improvement, interval excluding
  zero, and no material TPOT regression;
- below **1% TPOT** or **0.5% throughput**: noise/inconclusive.

These are campaign decisions, not universal constants. Greater integration
complexity should require a larger effect.

Candidate-specific gates extend, never replace, the universal gates. Examples
include graph-stable ring epochs, complete large-M fallback, disjoint channels
for concurrent graph/stream roles, and overwrite-safe caller-owned producer
output. Mechanism details belong in the
[integration roadmap](integration-optimization-roadmap-2026-07.md).

Prefill signature workloads must prove executed dispatch, not an aggregate
token label. With the post-rebase scheduler, aggregate “M512” can execute near
M144. The realigned profile therefore uses sequential direct 512-token requests
to require two-shot.

## Minimal decision record

Every completed campaign must record:

1. exact mechanism and expected lifecycle or critical-path effect;
2. code, image, model, topology, profile, and executed dispatch path;
3. transition, fallback, and safety outcomes;
4. marker-aligned max-rank forward effect;
5. every paired TPOT, throughput, and applicable TTFT result;
6. restart-level estimate and confidence interval;
7. attempted/completed starts, transitions, blocks, and pairs;
8. predeclared threshold and whether it was cleared;
9. recommendation: diagnostic-only, opt-in, or default.

If identity, safety, aligned-forward effect, or paired e2e results are missing,
the campaign is not promotion evidence.
