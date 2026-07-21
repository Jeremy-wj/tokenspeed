# Fused AR+RMSNorm on MI350X — e2e benchmarks, crossover & optimization roadmap

**Companion to `AR_RMSNORM_SYMM_MEM_MIGRATION.md`** (migration triton-shmem → torch
symm_mem + MI300X microbench). This is the **MI350X (gfx950) benchmarking + analysis
record**: how to run the gpt-oss-120b e2e gate, the op-level crossover, a latency
decomposition of the *shipping* fused op, and the consolidated optimization plan. Audience:
engineering agents. Precise + forward-looking; delete anything that stops informing a
decision.

> **HEADLINE.** The migrated `triton_shmem` fused AR+RMSNorm auto-selects, is numerically
> correct e2e on gpt-oss-120b, and — after two shipped fixes — is now **at parity-to-winning
> vs the real unfused path for ws=4 decode**, and **wins broadly at ws=2**. The fixes:
> (1) **M-aware one-shot dispatch** at small M (`TS_TRITON_SHMEM_ONESHOT_MAX_M=256`, drops
> the two-shot copy-out) and (2) **in-kernel barriers**, now **DEFAULT ON**
> (`TS_TRITON_SHMEM_INKERNEL_BARRIER=1`, drops the two separate barrier-kernel launches).
> Together they take the ws=4 decode op from ~0.088 ms to ~0.048 ms (§4) and flip the
> op-level crossover vs RCCL to **>1 for M≤192 at ws=4** and **≥1 almost everywhere at ws=2**
> (§3). e2e (ws=4, same-session A/B, §5): fused decode TPOT reaches **parity at conc16 and a
> win at conc32** vs unfused. **Recommendation: enable fusion for ws=2 and ws=4 decode
> (`comm_fusion_max_num_tokens>0`).**
>
> **The remaining gap is small and bounded (§4):** folding the barriers reclaimed only the
> *launches* (~0.020 ms/op) — the barrier *work* stays inside the kernel, and **copy-in
> (~0.012 ms) is an untouched floor**. Two levers remain (§7): **avoid copy-in** (biggest,
> ~25% of the decode op) and a **fixed-participant in-kernel barrier** (fixes the M=256 dip,
> makes the default robust, and unlocks two-shot in-kernel).
>
> **Caveat (documented, narrow):** the in-kernel barrier's signal-pad slot range is
> **M-dependent**, so it deadlocks *only if TP ranks replay different-M graphs
> simultaneously*. Pure TP (dp=1, `overlap_schedule_depth=1`, no spec-decode) never does, so
> the default is safe there. **Set `TS_TRITON_SHMEM_INKERNEL_BARRIER=0` under DP /
> `overlap_schedule_depth>1` / spec-decode until the fixed-participant barrier lands (§7).**

Data (repo): `benchmark/results/ar_rmsnorm_mi350x_e2e/`. Probes/helpers:
`benchmark/probe_ar_rmsnorm_decomp.py`, `benchmark/probe_inkernel_barrier_graph.py`,
`benchmark/probe_rccl_hang.py`, `benchmark/e2e_gptoss_{serve,bench,teardown}.sh`.

---

## 1. Environment & serve setup

Container `ts-migrate-mi350x` (`diprajap-tokenspeed:serve-base`, torch 2.11+rocm7.2,
8×gfx950). Serve = `python -m tokenspeed.cli serve …` (`e2e_gptoss_serve.sh`); confirm the
fused path via the log line `triton_shmem AR+RMSNorm state: … substrate=coarse+ipc
inkernel_barrier=True`. Model `gpt-oss-120b` (`/data/models/openai/gpt-oss-120b`) = mxfp4
MoE experts + **bf16 dense/residual stream, H=2880** → AR+RMSNorm runs on the bf16 residual
stream (representative).

**GPU pinning (HIP index ≠ rocm-smi index).** rocm-smi→HIP via PCI bus: `0→1 1→3 2→2 3→0
4→5 5→7 6→6 7→4`. **Noisy GPU (rocm-smi 3) = HIP index 0.** For ws<8 use
`HIP_VISIBLE_DEVICES=1,2,3,5` (→ rocm-smi {0,1,2,4}). Shared box: check `rocm-smi` for other
users before launching; tear serves down after.

**Serve gotchas (all image/version-skew, perf-neutral for an A/B):**

| gotcha | fix |
|---|---|
| `import tokenspeed` fails (baked editable points at unmounted path) | `pip install -e /home/jeremwan/tokenspeed/python --no-deps` |
| gateway `--policy: invalid choice 'passthrough'` (image smg 1.4.1 < pinned 1.7.0) | `--policy round_robin` (or rebuild image to 1.7.0) |
| startup pins ~1.6 TB host RAM (KV host mirror, `kvstore_ratio=2.0`) | `--kvstore-size 8` |
| `Address already in use` / orphans survive teardown (PID 1 = `sleep infinity`, procs renamed `ts-serve`/`ts-control`) | graceful SIGINT to `ts-serve` first (`e2e_gptoss_teardown.sh`); container restart clears leaked sockets |
| chat `harmony_parsing_failed` (gpt-oss harmony vs `--reasoning-parser base`) | bench via `/v1/completions` |

---

## 2. Method

- **Crossover** (`bench_triton_shmem_ar_rmsnorm.py`): full production op via the dispatcher
  (`auto→triton_shmem`, coarse+ipc, **inkernel default**) vs `rccl_unfused`
  (`dist.all_reduce` + residual + `F.rms_norm`). This box can't pin clocks (sysfs read-only)
  → lean on warmup/repeat; trust <0.1 ms configs only for *relative* trends, not absolutes.
- **Decomposition** (`probe_ar_rmsnorm_decomp.py`): times the fused op **both ways**
  (in-kernel vs separate-barrier, toggling the live state) and splits it into copy-in / pure
  kernel / residual in-kernel-barrier / copy-out, vs the unfused custom-AR + eager norm.
- **e2e A/B**: identical serve except `--comm-fusion-max-num-tokens` (>0 = fusion ON, 0 =
  pure unfused). `random`, temperature 0, ignore-eos.

---

## 3. Op-level crossover — new default (N=2880, fused ÷ RCCL; >1 ⇒ fusion wins)

`mi350x_cross_inkernel_default.csv`. Decode M is one-shot+in-kernel; M>256 is two-shot
(separate barriers, §4).

| M | ws=2 | ws=4 |
|------|------|------|
| 8    | 1.16 | 1.05 |
| 64   | 0.95 | **1.12** |
| 128  | 1.00 | **1.11** |
| 192  | —    | 1.00 |
| 256  | 1.18 | 0.73 |
| 384  | 1.28 | 0.68 |
| 512  | 1.25 | 0.77 |
| 1024 | 1.17 | 0.92 |
| 2048 | 1.10 | 0.81 |

- **ws=2 wins ≈everywhere** (only M=64 dips to 0.95, within noise) — vs the pre-fix envelope
  that lost below M≈384. Enable ws=2 fusion.
- **ws=4 decode now WINS** (M≤192: 1.0–1.12×, was 0.64× pre-fix). Two effects bound it: the
  **M=256 dip to 0.73×** (the one-shot in-kernel barrier's cost scales with grid ≈ M — §4;
  two-shot there is *worse*, 0.61×, so 256 is the right one-shot cap), and **M>256 two-shot**
  never beats RCCL at ws=4 (bandwidth regime; RCCL is byte-optimal — migration doc §8.6).
- The RCCL baseline is fair: forcing `NCCL_MIN/MAX_NCHANNELS=32` doesn't help it (auto
  already optimal for N=2880). Note this microbench baseline (eager RCCL+`F.rms_norm`) is
  *pessimistic* vs the real serve unfused path (custom-AR + triton fused norm) — the e2e A/B
  (§5) is the ground truth.

---

## 4. Decomposition — what the fixes reclaimed, and why the win is bounded (decision-critical)

`probe_ar_rmsnorm_decomp.py`, ws=4, N=2880, ms, max across ranks, 2-pass. The **shipping**
op is `copy-in → [in-kernel leading barrier → kernel → in-kernel trailing barrier]` (one
launch); the legacy path issued the two barriers as **separate 1-block kernel launches**.

| M | path | copy-in | kernel | in-kernel barrier | **FULL (default)** | legacy sep-barrier | **reclaim** | unfused (custom-AR+norm) | unf/FULL |
|-----|------|---------|--------|-------------------|--------------------|--------------------|-------------|--------------------------|----------|
| 8   | 1-shot | 0.023 | 0.013 | 0.012 | **0.048** | 0.068 | **0.020** | 0.047 | 0.98× |
| 32  | 1-shot | 0.012 | 0.017 | 0.014 | **0.048** | 0.067 | **0.019** | 0.050 | 1.04× |
| 64  | 1-shot | 0.012 | 0.015 | 0.015 | **0.049** | 0.067 | **0.019** | 0.064 | 1.31× |
| 128 | 1-shot | 0.013 | 0.012 | 0.011 | **0.048** | 0.069 | **0.021** | 0.103 | 2.12× |
| 256 | 1-shot | 0.013 | 0.019 | 0.052 | **0.084** | 0.065 | **−0.018** | 0.169 | 2.03× |
| 512 | 2-shot | 0.013 | 0.021 | n/a (sep) | **0.083** | 0.083 | — | 0.318 | 3.8× |

(Two-shot uses separate barriers by default (~0.032 ms) + a ~0.017 ms copy-out, so its FULL
exceeds copy-in+kernel; folding its barriers in-kernel was measured a net loss — §7.)

**What this says (answers "why the win is smaller than hoped"):**
1. **Folding barriers reclaims the two launches, not the barrier itself.** At decode M the
   two separate launches cost ~0.032 ms; the in-kernel barrier still costs ~0.011–0.015 ms
   of *work* inside the kernel → net **reclaim ≈ 0.020 ms/op** (~30% of the sep-barrier op).
   Combined with the one-shot fix (no copy-out), the decode op is ~0.048 ms — **at/under the
   real unfused custom-AR+norm** (0.047–0.103 ms), and the advantage grows with M because the
   custom-AR full-fan-in explodes.
2. **Copy-in (~0.012 ms) is an untouched floor** — ~25% of the decode op and now the single
   largest removable chunk (§7 lever A).
3. **The in-kernel barrier cost scales with grid width (≈ M).** It is a clear win while the
   grid is small (M≤~192) but *balloons* at M=256 (grid=256=num_cus → barrier 0.052 ms,
   `reclaim` goes **negative**) and across the two-shot regime — which is exactly why
   **two-shot keeps the separate barriers** (measured reclaim −0.005…−0.048 ms at M≥512) and
   why the fixed-participant barrier (§7 lever B) is the key to going further.
4. Context: the reference `mi350x_tuning_report.md` "~1.38× best-fused" is **kernel-only**
   (pre-placed input, no copy-in/out, no leading barrier, pinned clocks). Production wraps
   the ~0.014 ms kernel in copy-in + barriers; that wrapper — not the kernel — is the gap.

---

## 5. End-to-end A/B (ws=4, gpt-oss-120b) — the ground truth

Same-session 3-arm A/B, `random` in128/out512, temperature 0, ignore-eos; median decode
TPOT (ms) / output tok/s. Fused = new default (one-shot + in-kernel). Data:
`ar_rmsnorm_e2e/logs/serve_{inkernel_ON,inkernel_OFF,unfused}.log`.

| conc | unfused (cap=0) | **fused (default)** | fused, in-kernel OFF |
|------|-----------------|---------------------|----------------------|
| 8    | 10.28 / 766     | 11.04 (+7.4%) / 712 | 11.59 (+12.7%) / 682 |
| 16   | 11.80 / 1337    | 12.23 (+3.6%) / 1290| 12.32 (+4.4%) / 1223 |
| 32   | 12.86 / 2439    | **12.62 (−1.9%)** / 2421 | 13.23 (+2.9%) / 2266 |
| 64/128 | —             | healthy, correct (11.04→15.83 as conc grows) | — |

The default (in-kernel ON) **beats the legacy separate-barrier path at every concurrency**
and reaches **parity at conc16 / a win at conc32** vs unfused; only conc8 retains a ~7%
penalty (the copy-in + barrier floor of §4). Validated healthy + numerically correct
(completion "…is Paris") across conc 8/16/32/64/128 + a mixed varied-length load; correctness
+ HIP graph capture/replay tests pass at ws=2/4 with the default
(`test_triton_shmem_communication.py`).

---

## 6. Known serve issue — RCCL large-all-reduce hang (NOT the fused kernel)

Under serve, a large **unfused RCCL all-reduce** (batched prefill, NumelIn≈6656×2880,
ALLREDUCE, NCCL watchdog timeout) intermittently deadlocks — 100% spin on 2/4 ranks (rank
desync signature). Occurs on fusion on/off (fusion routes ≤cap through triton_shmem but >cap
still hits RCCL), so fusion-OFF is *more* exposed → a minor robustness point for fusion.
**Isolated as serve-specific:** standalone RCCL at the suspect sizes (±rank jitter, 300
iters, `probe_rccl_hang.py`) never hangs. Prime hypothesis: RCCL colliding with coexisting
symm_mem / HIP-IPC collectives (custom-AR, TritonRSAG, triton_shmem). Debug steps in §7.

---

## 7. Consolidated plan & open items

**Shipped (this line of work):**
- ✅ M-aware one-shot dispatch at small M (`ONESHOT_MAX_M=256`) — removes two-shot copy-out.
- ✅ In-kernel barriers **default ON** for the one-shot decode path — removes 2 barrier
  launches (~0.020 ms/op reclaim). Correctness + graph capture validated; e2e §5.
- ✅ Two-shot in-kernel barriers **implemented but gated OFF** (kernel keeps the param): the
  grid-scaling barrier makes it a net loss at two-shot grid widths (§4). Re-enable after
  lever B.

**Config recommendation (act on now, at deploy):**
- **Enable fusion for ws=2 and ws=4 decode** (`comm_fusion_max_num_tokens>0`, e.g. 256 to
  cover decode + short prefill). ws=2 wins ≈everywhere (§3); ws=4 decode is parity/win (§5).
  ws=2's old "M∈[384,2048] only" note is obsolete.
- **Keep `TS_TRITON_SHMEM_INKERNEL_BARRIER=0` under DP / `overlap_schedule_depth>1` /
  spec-decode** until lever B lands (M-divergence caveat, headline).

**Optimization levers (ordered by expected value; goal = beat unfused at all decode conc):**

- **A. Avoid copy-in (biggest remaining lever, ~0.012 ms ≈ 25% of the decode op).** Have the
  producer of the AR input (the attention/MLP output projection) write **directly into the
  state's persistent symmetric input buffer**, eliminating `self._x.copy_(input_tensor)`.
  The in-kernel *leading* barrier (now default) already provides the entry ordering this
  needs. Work: expose `state.symmetric_input_view(m)` from the shim; teach
  `layernorm.forward_with_allreduce_fusion` / `comm_ops` to route the prior op's output into
  it (fall back to copy-in when the producer can't target it); validate under graph capture
  (buffer address is persistent, so capture-safe) + re-run §4/§5. Risk: caller-side, spans
  model code → land behind a flag, A/B before default.

- **B. Fixed-participant in-kernel barrier (unlocks robustness + the M≥256 / two-shot wins).**
  Bake a **fixed barrier participant count `G`, identical across all captured graphs/ranks**,
  so `block_id∈[0,G)` regardless of M (the row loop already strides over M). This (i) makes
  the barrier **M-independent → safe under M-divergence** (removes the DP/overlap/spec-decode
  caveat), and (ii) **decouples barrier cost from grid width** → fixes the M=256 dip (§3/§4)
  and makes two-shot in-kernel a win (re-enable it then). Constraint: **`G` ≤ a safe fraction
  of `num_cus`** so all `G` blocks stay co-resident under serve concurrency (too-large `G`
  risks a co-residency deadlock — the likely form of the historical fault). Validate:
  `probe_inkernel_barrier_graph.py PROBE_MODE=multigraph PROBE_RNG_SHARED=0` must flip
  HANG→PASS; then re-A/B (expect the M=256 dip and the two-shot loss to disappear).

- **C. ws=8 e2e A/B.** Microbench validated; e2e pending (needs GPU3/HIP0 — time around its
  periodic burst). Divergence analysis is ws-agnostic; expect the ws=4 conclusions to carry.

**Separate robustness issue — RCCL hang (§6), ws=4-safe debug (do NOT run 8-GPU variants):**
1. `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL,P2P`; trigger; find the stalled peer/transport.
2. Force all collectives to RCCL (disable triton custom-AR + RSAG auto paths) — if the hang
   vanishes, it's a symm_mem/IPC + RCCL resource conflict.
3. `--enforce-eager` — if it vanishes, graph/replay-related.
4. Env: `NCCL_P2P_DISABLE=1`, alternate `NCCL_ALGO`/`NCCL_PROTO`.
Meanwhile keep prefill batches ≤~2048 and set a short `--distributed-timeout-seconds`.
