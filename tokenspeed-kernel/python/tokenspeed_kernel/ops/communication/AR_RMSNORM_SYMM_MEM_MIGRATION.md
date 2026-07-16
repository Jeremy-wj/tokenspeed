# Fused AllReduce + Residual + RMSNorm: triton-shmem → PyTorch Symmetric Memory

**Status:** Phases 0–4 complete through microbenchmarking; end-to-end server run
**held** (no local model fits the ~57 GB free disk; gpt-oss-120B needs ~240 GB —
awaiting instruction). The migrated backend is named **`triton_shmem`** (after its
source repo; the old rocSHMEM port that previously held that name is **deleted** —
see §7 Phase 4). It is wired into the `TS_ARNORM_BACKEND` dispatch and **passes on
8× MI300X**: packaged comm test (ws=4, two-shot), dedicated sweep ws=1/2/4/8 +
power-of-two whole-row (all three kernel variants), and HIP graph capture+replay at
ws=2 (one-shot) and ws=8 (two-shot). Code: `ops/communication/triton_shmem.py` +
`ops/communication/_triton_shmem_kernels.py`; test
`test/ops/test_triton_shmem_communication.py`; microbench
`benchmark/bench_triton_shmem_ar_rmsnorm.py`. No new runtime deps; no
`rocshmem4py`/upstream-`triton_shmem` imports.

> **HEADLINE FINDING (§8):** correctness and graph-capture safety are solid, but
> torch symm_mem on ROCm hands back **fine-grained** memory (~105 GB/s bulk local
> vs. ~3200 GB/s coarse-grained), so this backend is **2–25× slower than RCCL**
> and slower than the rocSHMEM reference for large tensors. It is faster than the
> already-shipping native `symm_mem` kernel (2–7×). The regression is a substrate
> property, not a port bug. Read §8 before shipping.
**Branch:** `jeremwan/triton-shmem-experiments` (tokenspeed repo)
**Hardware:** dev box is 8× AMD Instinct MI300X (gfx942); work runs in a ROCm 7.2 +
torch 2.11 container (§5.1). MI300X is sufficient — the MI350X(gfx950)-only attention
kernels are dispatch-gated and untouched by this migration.
**Audience:** engineering agents continuing this migration. Read top-to-bottom before
touching code. Environment setup is done and documented in §5 — just follow it.

---

## 1. Goal & motivation

Bring the fused **all-reduce + residual-add + RMSNorm** Triton kernels developed in
the external `triton-shmem` repo (`/home/jeremwan/triton-shmem`) into TokenSpeed as a
production AMD path, but **backed by PyTorch symmetric memory
(`torch.distributed._symmetric_memory`) instead of rocSHMEM (`rocshmem4py`)**.

Why migrate the backend (four independent reasons, strongest last):

1. **Dependency avoidance.** `rocshmem4py` is *not* pip-installable. It requires apt
   deps (openmpi, cmake, ninja), a from-source build of the rocSHMEM C library with
   specific GPU targets, and `ROCSHMEM_HOME`-pointed editable install (see
   `/home/jeremwan/triton-shmem/README.md`). TokenSpeed has **no container/env setup
   yet**; adding rocSHMEM to it is a large, undesirable burden.
2. **`torch` symmetric memory is already used pervasively in TokenSpeed** for AMD —
   see `ops/communication/triton.py` (all-reduce, RS/AG, the *native* fused
   AR+RMSNorm kernel `amd_allreduce_residual_rmsnorm_kernel`, and DP-sampling). It
   ships inside `torch`, so it adds **zero** new runtime deps.
3. **The pointer-translation paradigm is nearly identical** across the two backends
   (proven below), so the kernels port with minimal change.
4. **Graph-capture safety (decisive).** The fused op runs inside captured CUDA/HIP
   decode graphs (`runtime/execution/cuda_graph_wrapper.py` captures forward;
   `runtime/layers/layernorm.py:184` calls the op). symm_mem's in-kernel signal-pad
   barrier is pure device code and is **verified graph-capture-safe on this box**
   (Phase 0 probe #2). rocSHMEM's host `barrier_all_on_stream` under HIP graph capture
   is unproven and a real risk.

**Scope guardrails:**
- This change is **TokenSpeed-only**. Do **not** modify the `triton-shmem` repo.
- Production will **not** ship `triton-shmem` publicly → **vendor (copy-paste) the
  kernels** into TokenSpeed rather than importing the library. This also removes the
  `rocshmem4py` import entirely.
- Target contract is unchanged: AMD ROCm, bf16, 2-D `(num_tokens, hidden)` input,
  `weight` of shape `(hidden,)`, whole-world TP (`tp_size == world_size`).

---

## 2. Current integration state (what exists today)

All on branch `jeremwan/triton-shmem-experiments`, **uncommitted**. The migration is
complete; the backend is named `triton_shmem` (the historical rocSHMEM port that
originally carried this name has been deleted — see §7 Phase 4).

- **`ops/communication/triton_shmem.py`** (untracked): the symm_mem-backed shim
  (`TritonShmemAllReduceResidualRMSNorm`, `create_triton_shmem_ar_rmsnorm_state`,
  `triton_shmem_allreduce_residual_rmsnorm`, `TRITON_SHMEM_AR_RMSNORM_STATES`). It
  drives the vendored kernels over a `torch.distributed._symmetric_memory` heap (no
  rocSHMEM, no library import).
- **`ops/communication/_triton_shmem_kernels.py`** (untracked): vendored `@triton.jit`
  kernels + host tuning helpers + the signal-pad barrier kernel.
- **`ops/communication/triton.py`**: adds a `TS_ARNORM_BACKEND` env switch
  (`auto`|`iris`|`triton_shmem`|`symm_mem`) inside `allreduce_residual_rmsnorm` that
  routes to the migrated backend.

Dispatch chain:
```
layernorm.forward_with_allreduce_fusion()   runtime/layers/layernorm.py:184
  -> triton.allreduce_residual_rmsnorm()     ops/communication/triton.py
      -> [TS_ARNORM_BACKEND=triton_shmem] triton_shmem.py shim
          -> _triton_shmem_kernels.py fused kernels
              -> inlined symmetric_ptr + symm_mem buffer_ptrs_dev tables
```

The upstream kernels (`/home/jeremwan/triton-shmem/triton_shmem/ccl/fused_ar_rmsnorm.py`,
419 lines) are **pure `@triton.jit`** (no host launcher). Three variants:
- `fused_ar_rmsnorm_twoshot_blocked_kernel` — two-shot push, N-blocked, arbitrary N.
  Needs **3 symmetric buffers** (input, output, residual_out) + a trailing barrier.
- `fused_ar_rmsnorm_oneshot_blocked_kernel` — one-shot pull, N-blocked, arbitrary N.
  Needs **1 symmetric buffer** (input only); output/residual_out are local; no
  trailing barrier.
- `fused_ar_rmsnorm_oneshot_wholerow_kernel` — one-shot pull, whole-row, power-of-two N.
  1 symmetric buffer; no trailing barrier.

Plus pure-Python host tuning helpers in the same file: `ArchProfile`, `_MI300X`,
`_MI350X`, `recommended_kernel(ws, N)`, `recommended_grid(kernel, ws, work_rows,
num_cus)`, `recommended_num_warps(kernel)`, `recommended_block_n(dtype, N)`.

**Key dispatch fact:** on MI300X, `_MI300X.oneshot_max_ws = 2`, so
`recommended_kernel(ws=8, ...)` returns **`twoshot_blocked`**. The production 8-GPU
path is therefore the two-shot kernel → the migration MUST handle the 3-symmetric-buffer
case, not just the trivial one-shot case.

---

## 3. Validation of the existing port (Task-1 findings)

The port is **structurally in sync** with the current upstream kernels: all three
kernel launch sites and the three helper calls match the current signatures
argument-for-argument (verified). The kernel-name refactor and tuning framework
already landed upstream (`git log` in triton-shmem: `a589e0c`, `8739e40`, `881dbba`),
and the port was written against the post-refactor names. Real issues found:

1. **Reimplements `recommended_block_n`.** `triton_shmem.py::_blocked_block_n`
   hardcodes the MI300X value (`min(n, max(128, 1024//itemsize))`) instead of importing
   upstream `recommended_block_n`. Correct on gfx942, wrong on other archs. When
   vendoring, use the vendored `recommended_block_n`.
2. **Not validated under CUDA-graph capture** (see §1.4). The rocSHMEM `barrier_all()`
   calls inside `fused()` are the risk. The migration removes this risk (Phase 0 proved
   the symm_mem replacement is capture-safe).
3. **rocSHMEM heap [2 GiB, 4 GiB) hang band.** `triton-shmem`'s own tests warn rocSHMEM
   hangs for per-PE heaps in `[2 GiB, 4 GiB)`. `_ensure_rocshmem_initialized` sizes
   `max(512 MiB, 8*need)`, which can land in the band for large `max_token_num`. Moot
   after migration (symm_mem has no such band).
4. **Local CU count** (`torch.cuda.get_device_properties(...).multi_processor_count`)
   vs upstream's rank-min `all_reduce(MIN)`. Safe on a homogeneous node; note it.

Conclusion: the rocSHMEM port only needs cosmetic fixes to be "current," but is a
dead-end for production because of deps + graph capture. Migrate rather than polish.

---

## 4. The core technical insight: symmetric_ptr ≡ buffer_ptrs_dev

rocSHMEM device translation (from `rocshmem4py/interop/torch.py::get_heap_bases`
docstring) is **exactly**:
```
peer_ptr = heap_bases[peer] + (local_ptr - heap_bases[my_pe])
```
which is exactly what `triton_shmem/utils/symmetric.py::symmetric_ptr` computes.

PyTorch symm_mem exposes `handle.buffer_ptrs_dev`: a device tensor of **per-peer
pointers to a specific buffer**. TokenSpeed's existing AMD kernels already use it as
`buffer_ptrs[peer] + offset` (e.g. `amd_all_reduce_kernel`,
`amd_allreduce_residual_rmsnorm_kernel` in `triton.py`).

**Equivalence:** if you pass a buffer's `buffer_ptrs_dev` as the `heap_bases` argument
to `symmetric_ptr`, the math is unchanged:
```
buffer_ptrs[peer] + (local_ptr - buffer_ptrs[my_pe]) = buffer_ptrs[peer] + offset_in_buffer
```
So **`symmetric_ptr` works verbatim on symm_mem**, provided each symmetric tensor is
translated with **its own** pointer table.

The one real difference:
- rocSHMEM shares **one** `heap_bases` across *all* symmetric allocations (same per-PE
  heap offset invariant).
- symm_mem gives a **separate** `buffer_ptrs_dev` per allocation. Do **not** assume a
  single table works across allocations — use per-tensor tables.

Consequences per kernel:
- **one-shot kernels:** only `input` is symmetric → **one** table, `symmetric_ptr`
  unchanged, no trailing barrier. Trivial.
- **two-shot kernel:** input + output + residual_out symmetric → **three** tables →
  edit the (vendored) kernel signature to take three base arrays and translate each
  tensor with its own. Contained edit since we own the copy.

Reusable helpers already in `triton.py`: `_alloc_symm(shape, dtype, device, group)`
and `_peer_ptrs_dev(handle, shape, dtype, world_size, device)` (builds a
`buffer_ptrs_dev`-style uint64 table via `handle.get_buffer(peer,...)`).

Barriers:
- rocSHMEM leading `barrier_all()` (all inputs visible before pull) → in-kernel entry
  signal-pad barrier OR a tiny dedicated barrier kernel between the input `copy_` and
  the fused kernel. Needed for one-shot and two-shot.
- rocSHMEM trailing `barrier_all()` (two-shot peer writes visible) → in-kernel exit
  barrier / barrier kernel after the fused kernel.
- Reuse `symm_mem_barrier` / `blockwise_barrier` from `triton.py`. Reserve the signal
  pad at state creation exactly like `nvidia_create_rsag_state` does:
  `symm_mem.set_signal_pad_size(max(cur, max_blocks * ws * 4))`.

---

## 5. Environment setup (validated end-to-end)

Everything runs in **one ROCm 7.2 container**. The dev box is 8× MI300X (gfx942) on a
ROCm 7.1 host driver; the container ships its own ROCm 7.2 userspace, so the host
version is irrelevant. MI300X is fully sufficient for this migration — the only
MI350X(gfx950)-specific code is the attention kernels, which are dispatch-gated by arch
and never execute on this path (they register at import, then get filtered out on
gfx942).

### 5.1 Container
No single `rocm/pytorch` tag ships both ROCm 7.2 **and** torch 2.11 (the torch-2.11
images are a separate, newer ROCm line). You don't need one: start from any ROCm 7.2
base and let the install step pull `torch==2.11.0+rocm7.2`. Validated base image:
`rocm/pytorch:rocm7.2.2_ubuntu22.04_py3.10_pytorch_release_2.7.1`.

```
docker run -d --name ts-migrate \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  --ipc=host --shm-size=16g --network=host \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /home/jeremwan:/home/jeremwan -w /home/jeremwan/tokenspeed \
  rocm/pytorch:rocm7.2.2_ubuntu22.04_py3.10_pytorch_release_2.7.1 sleep infinity
```

Mounting `/home/jeremwan` keeps paths identical to the host (the probes and
`triton-shmem` source resolve unchanged). Run commands with `docker exec ts-migrate …`.

### 5.2 Install (inside the container)
Authoritative recipe: `test/ci_system/install_deps_rocm.sh`. The comm-op subset below
is all this migration needs (it skips the serving stack — scheduler + `./python`):

```
apt-get update && apt-get install -y openmpi-bin libopenmpi-dev libssl-dev pkg-config
python3 -m pip install --upgrade pip "setuptools<82" wheel
pip install --force-reinstall --no-deps ./tokenspeed-kernel-amd --no-build-isolation
TOKENSPEED_KERNEL_BACKEND=rocm PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/rocm7.2 \
  pip install tokenspeed-kernel/python/ --no-build-isolation
```

- **Install order matters.** Install the in-tree **`tokenspeed-kernel-amd`** (v0.1.1)
  *first*. The package `__init__` imports `mha_decode_gfx950` / `mha_extend_gfx950`,
  which the published PyPI wheel does not ship but the in-tree source does. Its version
  matches the `rocm.txt` pin, so the next command treats the pin as satisfied and keeps
  the source build.
- The second command installs `torch==2.11.0+rocm7.2` and
  `tokenspeed-triton/-proton/-iris/-triton-kernels`. On ROCm the native CUDA build is
  skipped (pure-Python; no nvcc/CUDA toolkit needed).
- Vendored kernels must import `tl`/`triton` from `tokenspeed_kernel._triton` (the
  `tokenspeed_triton` distribution), not stock `triton` — see Phase 1.

### 5.3 Validated working
This exact container + install was run and verified: **torch 2.11.0+rocm7.2 (HIP 7.2),
8 GPUs visible**, and all of the following pass —

- `import tokenspeed_kernel.ops.communication.triton` and the attention-gluon `__init__`
  chain import cleanly on gfx942.
- Substrate probes (`/home/jeremwan/symm_probe.py`, `/home/jeremwan/symm_graph_probe.py`,
  run with `python3 <probe>.py`): peer all-reduce **PASSED**, and HIP-graph
  capture+replay **PASSED**.
- Packaged correctness test **passes**:
  `pytest tokenspeed-kernel/test/ops/test_communcation.py` (world=4, through the
  `allreduce_residual_rmsnorm` dispatcher).

This de-risks the whole Phase-2 substrate — symm_mem allocation, per-peer pointer
translation in Triton, the in-kernel signal-pad barrier, multi-process rccl spawn, and
graph capture/replay. Remaining work is integration, not feasibility. `symm_mem` needs
no group opt-in here (plain `empty`+`rendezvous` works).

---

## 6. Testing & benchmarking plan (Task-2 findings)

### 6.1 Correctness — reuse existing infra
- **`test/ops/test_communcation.py::check_allreduce_residual_rmsnorm`** already tests
  this exact op via `mp.spawn` (nccl), world=4, hidden=2880, fp32 reference, through
  the `allreduce_residual_rmsnorm` dispatcher. Because the dispatcher reads
  `TS_ARNORM_BACKEND`, running it with `TS_ARNORM_BACKEND=triton_shmem` (the migrated
  backend) exercises the migrated path with **zero new test code**. Fastest lever.
- **`test/ops/test_iris_communication.py` Suite 3** (lines ~285–421) is a near-1:1
  template for the dedicated `test_triton_shmem_communication.py`: `mp.spawn`, world
  1/2/4/8, token cases `[1, 64, 256, 1024, 8192]`, non-identity linspace weight, fp32
  reference, `atol/rtol=2e-2`, persistent/non-persistent parametrization. Swap
  `create_iris_ar_rmsnorm_state` → the new `create_*_state`.
- Upstream `triton-shmem/tests/test_fused_ar_rmsnorm.py` is the numerics oracle for the
  kernels themselves (all 3 variants, sweeps M/N/dtype/gamma/fusion) — runs in the
  triton-shmem venv with rocSHMEM. Use it to confirm kernel logic before/after the
  symmetric_ptr edits, but it is torchrun+rocSHMEM-bound.

### 6.2 Microbenchmark — does not exist; build a small one
- The `tokenspeed_kernel.benchmark` framework (`BenchmarkRunner`, `KernelRegistry`,
  `benchmark_op`) is **single-process / single-GPU / registry-based** (GEMM/attention).
  It cannot benchmark a distributed collective without major surgery — **do not** force
  the op into it.
- **Recommendation:** one self-contained multi-process microbench (~200–300 lines,
  one file), modeled on `test_iris_communication.py` Suite 3 but timing with CUDA
  events, comparing backends via `TS_ARNORM_BACKEND` (iris / symm_mem native /
  triton_shmem migrated) plus an RCCL `dist.all_reduce` + `F.rms_norm` baseline. Report
  p50 latency + skew across ranks. Environment: torch.distributed + 8 GPUs; **no new
  deps** post-migration. **Built and run — see `bench_triton_shmem_ar_rmsnorm.py` and
  §8.**
- Upstream `triton-shmem/benchmark/bench_fused_ar_rmsnorm.py` + `benchmark/bench.py`
  is a reference for axis sweeps and the RCCL baseline design (native-dtype comm for a
  fair comparison), but it is rocSHMEM/torchrun-bound; port ideas, not the harness.

### 6.3 End-to-end (final gate)
Run the model server with `TS_ARNORM_BACKEND=triton_shmem`, compare tokens/s and output
correctness against `iris` and the native `symm_mem` kernel. This is validation, not
part of the core migration. **HELD** — see §7 Phase 4 (no model fits disk).

---

## 7. Multi-phase execution plan

### Phase 0 — environment + substrate (DONE ✅)
See §5. Container + install fully set up; substrate probes and the packaged comm test
pass on hardware. Nothing left to discover here — follow §5 to reproduce the env.

### Phase 1 — vendor the kernels + tuning (DONE ✅)
- Kernels + tuning live in `ops/communication/_triton_shmem_kernels.py`; the
  shim/state lives in `ops/communication/triton_shmem.py`. `tl`/`triton` import from
  `tokenspeed_kernel._triton`. `symmetric_ptr` is inlined (6 lines). No `rocshmem4py`
  / upstream-`triton_shmem` library imports. Actual footprint ≈ 430 (kernels) + 400
  (shim) lines.
- **Decision — one shared `symmetric_ptr`, per-tensor tables at the call site.**
  `symmetric_ptr` keeps its `(local_ptr, my_pe, peer, bases)` signature; the one-shot
  kernels pass the single input table as `heap_bases` and are otherwise byte-identical
  to upstream. The two-shot kernel is the **only** device-code edit: its `heap_bases`
  param is split into `input_bases`/`output_bases`/`residual_out_bases` and each
  symmetric tensor is translated with its own (per §4). `recommended_block_n` is the
  vendored (arch-correct) one — resolves §3 finding #1.

### Phase 2 — swap the symmetric-memory substrate (DONE ✅)
- Allocation via `_alloc_symm` (`symm_mem.empty` under `inference_mode(False)` +
  `rendezvous`); per-tensor `buffer_ptrs_dev` via `_peer_ptrs_dev` (both reused from
  `triton.py`). one-shot builds 1 table (input); two-shot builds 3.
- **Decision — barriers are a dedicated signal-pad kernel, leading + trailing on
  EVERY variant.** `symm_grid_barrier_kernel` launches a **single block per rank**,
  doing `symm_mem_barrier(sig, block_id=0, rank, ws)`; each rank's block signals every
  peer and waits for every peer (all-to-all), and the kernel-launch boundary is the
  global barrier. (Initially this launched `grid_sms` blocks matching the fused grid;
  that was reduced to 1 block during Phase-4 microbench — a global barrier needs only
  one representative per rank, and the multi-block version dominated the small-M decode
  latency. Correctness + graph capture re-verified after the change.)
  Leading barrier = all peers' inputs visible before any pull. **Trailing barrier is
  issued for one-shot too** (not just two-shot): the persistent symmetric `input`
  buffer is reused across calls (repeated decode graphs), so peers must finish reading
  it before the next call's `copy_` overwrites it. This matches the native kernel
  (entry+exit) and Iris (`device_barrier` before+after); upstream's "one-shot needs no
  trailing barrier" only holds without cross-call buffer reuse. Signal pad reserved
  before the first `empty`: `set_signal_pad_size(max(cur, num_cus * ws * 4))`.

### Phase 3 — wire the dispatch (DONE ✅)
- `triton_shmem` backend value in the `TS_ARNORM_BACKEND` switch in
  `allreduce_residual_rmsnorm` (`triton.py`): `(id(group), max_token_num, hidden_dim,
  dtype)` cache key, `create_*` returns `None` to fall back when ineligible. (During
  Phase 4 this backend was briefly named `symm_shmem` alongside the old rocSHMEM
  `triton_shmem` branch for A/B; the old branch + module are now deleted and the
  migrated backend took the `triton_shmem` name — see Phase 4.)

### Phase 4 — validate & benchmark (DONE ✅ except e2e, which is HELD)
- **DONE — Correctness:** `test_communcation.py` with `TS_ARNORM_BACKEND=triton_shmem`
  passes (ws=4, two-shot). `test/ops/test_triton_shmem_communication.py` passes at
  ws=1/2/4/8 (hidden=2880 → one-shot for ws≤2, two-shot for ws≥4) plus ws=2 hidden=4096
  (whole-row) — all three kernel variants covered, fp32 reference, 2e-2 tol.
- **DONE — Graph capture:** capture+replay cases at ws=2 (one-shot) and ws=8
  (two-shot) pass 3 replays with changing input. Capture-safety confirmed on the
  production two-shot path.
- **DONE — Microbench (§6.2, §8):** `benchmark/bench_triton_shmem_ar_rmsnorm.py`
  (self-contained, multi-process, CUDA-event timed, correctness-gated per config).
  Swept ws=2/4/8 × M∈{1024,4096,16384} × N∈{1024,2880,4096,16384} vs. an RCCL
  unfused baseline and the native `symm_mem` kernel. CSV: `results/triton_shmem_bench.csv`.
  Key result in §8. iris left out of the sweep (its singleton heap can't grow across
  the multi-config bench — an iris-shim limitation, not this backend's).
- **DONE — Refactor/rename:** the migrated backend was renamed `symm_shmem` →
  `triton_shmem` across the whole repo (module, kernels, test, bench, dispatch branch,
  env value, class/factory/state-cache symbols) and the old rocSHMEM `triton_shmem.py`
  port + its dispatch branch were **deleted**. Rationale: (a) the two are
  correctness-identical (literally the same kernels, only the substrate differs), (b)
  the old port is non-runnable here (rocSHMEM is not installed — the whole point of the
  migration), and (c) all future triton-shmem ports undergo the same rocSHMEM→symm_mem
  migration, so one unified name suffices. Tests re-run green after the rename.
- **HELD — End-to-end server run.** No local model is available and gpt-oss-120B
  (~240 GB) does not fit the ~57 GB free disk (host 94% full). Per instruction, held
  pending guidance rather than downloading. NOTE: given §8, an e2e run today would show
  the fused `triton_shmem` path **losing** to the default — worth weighing before
  investing disk/time.

---

## 8. Benchmark findings (Phase 4) — READ BEFORE SHIPPING

Full-op p50 latency (copy-in → leading barrier → fused kernel → trailing barrier →
copy-out, at the dispatcher level), 8× MI300X, bf16, fusion=residual. `results/
triton_shmem_bench.csv` has the full grid; representative `triton_shmem` vs. RCCL:

| ws | M×N | RCCL (ms) | triton_shmem (ms) | vs RCCL | native symm_mem (ms) |
|----|-----|-----------|-------------------|---------|----------------------|
| 2 | 16384×2880 | 2.20 | 5.30 | 0.42× | 21.3 |
| 4 | 16384×2880 | 1.21 | 12.8 | 0.09× | 43.1 |
| 8 | 16384×2880 | 0.73 | 14.8 | 0.05× | 99.3 |
| 8 | 1024×2880  | 0.11 | 1.07 | 0.11× | 5.82 |
| 8 | 16384×16384| 4.00 | 99.0 | 0.04× | 165.9 |

Three conclusions, all internally consistent:

1. **Methodology is sound.** Our RCCL baseline matches the external `triton-shmem`
   reference (`reverified_baseline.csv`, `dist_unfused_ar_rmsnorm … residual`) within
   ~2% (e.g. ws=2 16384²: 12.30 vs 12.35 ms; ws=8 16384²: 3.995 vs 4.14 ms). So the
   numbers are directly comparable to the reference.
2. **The port is correct and well-built.** `triton_shmem` beats the already-shipping
   native `symm_mem` kernel by **2–7×** (same fine-grained substrate; our grid-strided
   kernel + minimal 2-barrier design vs. the native per-row kernel that launches
   `token_num` blocks each doing a per-row barrier).
3. **But it loses badly to RCCL (0.04–0.76×)** and to the rocSHMEM reference (which is
   ~0.88–1.65× RCCL, kernel-only). **Root cause = memory coherence grain**, measured
   directly:

   | buffer | bulk local read/write |
   |---|---|
   | plain `torch.empty` (coarse-grained) | ~4200 GB/s |
   | `symm_mem.empty`, CUDA/HIP backend (fine-grained) | **~105 GB/s** |
   | `symm_mem.empty`, NCCL backend (coarse-grained) | ~3200 GB/s |

   torch symm_mem's default CUDA/HIP backend allocates **fine-grained** memory (needed
   for the signal-pad atomics; it bypasses L2 → ~40× slower bulk access). The upstream
   rocSHMEM heap is coarse-grained, which is why the reference is fast. Every symm_mem
   access pays this: copy-in, copy-out, and the kernel's own reads/writes. It hits the
   two-shot (peer-push + copy-out) hardest, but forcing one-shot is *worse* (its higher
   read fan-in also hits fine-grained memory) — so the rocSHMEM dispatch (two-shot at
   ws≥4) remains the right choice.

**Things tried that do NOT fix it:**
- `symm_mem.set_backend("NCCL")` gives coarse-grained fast memory (table above) **but
  faults** (`hipErrorIllegalAddress`) on the direct `buffer_ptrs_dev` peer load/store
  the vendored kernels require — the NCCL backend is for NCCL window collectives, not
  raw peer pointers. Not compatible with this kernel design.
- Forcing one-shot everywhere (measured): slower, not faster (see above).

**Paths forward (not attempted — would need a decision / more scope):**
- Coarse-grained peer buffers for the *data* (input/output/residual_out) with a
  fine-grained signal pad only — exactly what rocSHMEM does. torch's CUDA/HIP symm_mem
  allocator is hardcoded fine-grained; getting this means either patching torch's
  allocator or hand-rolling coarse-grained HIP allocations + `hipIpc*` peer-handle
  exchange (≈ reimplementing a slice of rocSHMEM). Biggest potential win.
- Accept the regression only where it doesn't matter: at tiny decode batch (M≈1–64)
  the absolute latency is small; profile the real serving decode shape before judging.
- Revisit on a newer torch/ROCm where symm_mem may expose a coherence knob.

**Bottom line:** the migration meets its correctness / dependency-avoidance /
graph-capture goals, but at a real large-tensor perf cost rooted in the symm_mem
substrate. Shipping it as the default is **not** advisable until the coarse-grained
data-buffer path (or an equivalent) lands.

## 9. Risks & open items
- **[RESOLVED] symm_mem cross-allocation offset is NOT assumed** — per-tensor
  `buffer_ptrs_dev` (3 tables for two-shot) implemented and confirmed by the two-shot
  correctness tests (ws=4/8 pass).
- **[RESOLVED] Grid-level barrier semantics for persistent grids** — chose the
  dedicated signal-pad barrier kernel (`symm_grid_barrier_kernel`, a **single block
  per rank** doing one `symm_mem_barrier`) rather than in-kernel entry/exit. Confirmed
  by correctness + graph capture at ws=2/4/8. Ordering for the two-shot push is
  protected by the trailing barrier before copy-out. (Reduced from `grid_sms` blocks to
  1 during Phase-4 microbench — see §7 Phase 2.)
- **[KEY FINDING — see §8] Fine-grained symm_mem memory is the dominant perf cost.**
  torch symm_mem on ROCm (CUDA/HIP backend) is fine-grained (~105 GB/s vs ~3200 GB/s
  coarse-grained), making this backend 0.04–0.76× RCCL for large tensors. Substrate
  property, not a port bug. A coarse-grained data-buffer path is the main open
  optimization; do not ship as default until addressed.
- **[RESOLVED] two-shot correctness** — output and residual_out each translated with
  their own table + trailing barrier before copy-out; ws=4/8 correctness + ws=8 graph
  capture pass.
- **[OPEN] Buffer-reuse race for one-shot** — mitigated by issuing the trailing
  barrier for one-shot too (see Phase 2 decision). If a future perf pass wants to drop
  it, first prove no peer can still be reading the symmetric `input` when the next
  call's `copy_` runs.
- **[OPEN] torch/rocm version** — validated env is torch 2.11+rocm7.2 in the §5
  container. Re-check symm_mem behavior (`enable_symm_mem_for_group`, signal-pad
  sizing) if moving to another version.
- **[OPEN] Tuning portability** — `ArchProfile` numbers are MI300X/MI350X-specific.
  `detect_arch()` auto-selection is preserved; re-tune for other archs.
- **[NOTE] Sub-group generality** — unlike the rocSHMEM shim (which required
  `group == world`), `symm_mem` rendezvous accepts any process group, so `triton_shmem`
  is not restricted to whole-world TP. Only whole-world TP is exercised today; validate
  a proper sub-group before relying on it.

## 10. Key file reference
- **Canonical AMD env setup (authoritative):** `test/ci_system/install_deps_rocm.sh`
- AMD kernels source to build (§5.2): `tokenspeed-kernel-amd/` (in-tree, v0.1.1)
- ROCm requirements pins: `tokenspeed-kernel/python/requirements/{rocm,rocm-thirdparty,common}.txt`
- Upstream kernels: `/home/jeremwan/triton-shmem/triton_shmem/ccl/fused_ar_rmsnorm.py`
- Upstream ptr translate: `/home/jeremwan/triton-shmem/triton_shmem/utils/symmetric.py`
- Upstream test/bench: `triton-shmem/tests/test_fused_ar_rmsnorm.py`,
  `triton-shmem/benchmark/bench_fused_ar_rmsnorm.py`
- rocSHMEM interop (reference semantics):
  `/home/jeremwan/rocm-systems/projects/rocshmem/python/rocshmem4py/interop/torch.py`
- **Migrated backend (this project): `ops/communication/triton_shmem.py` (shim/state) +
  `ops/communication/_triton_shmem_kernels.py` (vendored kernels + tuning + barrier).**
  (The old rocSHMEM port that previously held the `triton_shmem.py` name is deleted.)
- Dispatch + native symm_mem kernel + reusable helpers (`_alloc_symm`,
  `_peer_ptrs_dev`, `symm_mem_barrier`, `amd_allreduce_residual_rmsnorm_kernel`):
  `ops/communication/triton.py`
- Sibling backend template (Iris): `ops/communication/iris.py`
- Callers: `runtime/layers/layernorm.py`, `runtime/distributed/comm_ops.py`
- Existing tests: `test/ops/test_communcation.py`, `test/ops/test_iris_communication.py`
- **Migrated-backend test: `test/ops/test_triton_shmem_communication.py`**
- **Microbench: `benchmark/bench_triton_shmem_ar_rmsnorm.py`; results:
  `results/triton_shmem_bench.csv`. Reference: `triton-shmem/benchmark/results/
  ar_rmsnorm_opt_sweep/reverified_baseline.csv`.**
- Phase 0 probes: `/home/jeremwan/symm_probe.py`, `/home/jeremwan/symm_graph_probe.py`
