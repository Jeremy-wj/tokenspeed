# Fused AllReduce + Residual + RMSNorm: triton-shmem → PyTorch Symmetric Memory

**Status:** Migration complete + microbench-validated (MI300X §8) **and e2e-validated on
8× MI350X gpt-oss-120b** (§0.3 + companion doc `AR_RMSNORM_MI350X_E2E_BENCHMARKS.md`).
The e2e gate is DONE: `triton_shmem` auto-selects and is numerically correct end-to-end.
On MI350X fusion originally lost ~15–20% decode TPOT at ws=4 (staging overhead, not
compute). Two shipped fixes close it: **M-aware one-shot dispatch**
(`TS_TRITON_SHMEM_ONESHOT_MAX_M=256`, drops the two-shot copy-out) + **in-kernel barriers,
now DEFAULT ON** (`TS_TRITON_SHMEM_INKERNEL_BARRIER=1`, drops the two barrier-kernel
launches). Result: the ws=4 decode op drops ~0.088→~0.048 ms; fusion is **parity-to-winning
vs the real unfused path for ws=4 decode** (e2e: parity at conc16, win at conc32) and **wins
broadly at ws=2**. **Recommendation: enable fusion for ws=2 and ws=4 decode
(`comm_fusion_max_num_tokens>0`).** The remaining gap is small and bounded — folding barriers
reclaimed only the *launches*; the barrier *work* and **copy-in (~0.012 ms)** remain — so the
next levers are **avoid copy-in** and a **fixed-participant barrier** (companion §7).
**Caveat:** the in-kernel barrier's slot range is M-dependent, so it deadlocks only if TP
ranks replay **different-M graphs simultaneously**, which pure TP (dp=1,
`overlap_schedule_depth=1`) never does; set `INKERNEL_BARRIER=0` under DP / overlap>1 /
spec-decode until the fixed-participant barrier lands. Full record + plan: companion doc.
The migrated backend is named **`triton_shmem`** (after its source repo; the old rocSHMEM
port that previously held that name is **deleted**). It is the AMD fused default in the
`TS_ARNORM_BACKEND` dispatch and **passes on 8× MI300X**: packaged comm test, dedicated
sweep ws=1/2/4/8 (all three kernel variants), and HIP graph capture+replay (ws=2 one-shot,
ws=8 two-shot). Code: `ops/communication/triton_shmem.py` +
`ops/communication/_triton_shmem_kernels.py` + `ops/communication/_coarse_shmem.py`;
test `test/ops/test_triton_shmem_communication.py`; microbench
`benchmark/bench_triton_shmem_ar_rmsnorm.py` + driver
`benchmark/run_ar_rmsnorm_noise_controlled.sh`. No new runtime deps.

> **HEADLINE (§8).** torch symm_mem on ROCm hands back **fine-grained** memory
> (~107 GB/s bulk local vs ~3255 GB/s coarse, 30×), which crippled bulk transfer.
> **Fixed** by backing the *data* buffers with **coarse-grained HBM shared over HIP
> IPC** (rocSHMEM's own model), signal pad only left fine-grained
> (`TS_TRITON_SHMEM_COARSE=1`, default). Noise-controlled result (8× MI300X, ≥0.5 ms
> configs trustworthy): `triton_shmem` is **parity-to-winning vs RCCL at ws=2**,
> **0.6–0.7× RCCL at ws=4/8** (the rocSHMEM envelope), and **1.7–5.4× faster than iris
> at ws≥4**. Dispatch now routes `auto → triton_shmem` (iris demoted to explicit-only);
> a **model-targeted sweep** (DeepSeek/Kimi/GLM/gpt-oss hidden + MLA-compressed widths)
> shows fusion is profitable vs RCCL only at ws=2 (M≳256), peaking at moderate M and
> eroding with M/ws at ws=4/8 (§8.6). Correctness + graph capture pass. See §8.
**Branch:** `jeremwan/triton-shmem-experiments` (tokenspeed repo)
**Hardware (MI300X, prior):** 8× AMD Instinct MI300X (gfx942); ROCm 7.2 + torch 2.11
container (§5.1). MI300X microbench + correctness complete (§8).
**Hardware (MI350X, active):** 8× AMD Instinct MI350X (gfx950); host ROCm **7.1.1**
driver. gfx950 attention/MoE kernels are present on this branch — **gpt-oss-120b serves
end-to-end** (§0.3). AR+RMSNorm migration code is arch-agnostic (`ArchProfile`
auto-selects MI350X tuning; verified in the serve log).
**Audience:** engineering agents continuing this migration. Read top-to-bottom before
touching code. MI300X env: §5. MI350X env: §0.

---

## 0. MI350X environment (Jul 2026 — active box)

Repo copied file-for-file from the MI300X dev box. Relative in-repo paths unchanged;
host paths outside the repo differ.

### 0.1 Docker

| Image | torch | Notes |
|-------|-------|-------|
| **`diprajap-tokenspeed:serve-base`** | **2.11.0+rocm7.2** | **Use this.** Pre-built ROCm 7.2 userspace + torch 2.11; prior `diprajap-tokenspeed-serve` container used it with `/data` bind-mount. |
| `rocm/pytorch:rocm7.2_ubuntu22.04_py3.10_pytorch_release_2.9.1` | 2.9.1+rocm7.2 | Fallback base; needs torch 2.11 upgrade via §5.2 install. |
| `rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0` | 2.10.0+rocm7.2.4 | Available; not validated for this project. |

The MI300X-validated base (`rocm/pytorch:rocm7.2.2_ubuntu22.04_py3.10_pytorch_release_2.7.1`)
is **not** present locally. The §5.2 install recipe upgrades any ROCm 7.2 base to torch 2.11.

**Active container:** `ts-migrate-mi350x` (`diprajap-tokenspeed:serve-base`, 8 GPUs visible,
mounts `/home/jeremwan` + `/data`, `HSA_ENABLE_IPC_MODE_LEGACY=1`). Re-create:

```
docker run -d --name ts-migrate-mi350x \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  --ipc=host --shm-size=16g --network=host \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /home/jeremwan:/home/jeremwan -v /data:/data \
  -w /home/jeremwan/tokenspeed -e HSA_ENABLE_IPC_MODE_LEGACY=1 \
  diprajap-tokenspeed:serve-base sleep infinity
```

Re-install in-tree packages before e2e (`GFX_ARCH=gfx950 bash test/ci_system/install_deps_rocm.sh`;
subset in §5.2 suffices for comm-op work). **Install notes (Jul 2026):** run `apt-get update`
first (stale apt cache 404s); Step 5 (`pip install -e ./python`) fails on private dep
`tokenspeed-mooncake` — the image's preinstalled `tokenspeed==0.1.0` is sufficient for serve;
use **`pip install -e tokenspeed-kernel/python/`** (editable) so uncommitted migration changes
(`triton.py` dispatch, `_coarse_shmem.py`) are live. Phase-0 probes (`symm_probe.py`, etc.)
were **not** copied to this box — re-run comm tests (`test_triton_shmem_communication.py`) instead.

### 0.2 Models & disk

**Storage layout on this box:** `/data/dev/<user>/` is per-user (requires admin to create
a new dir — no `jeremwan` entry). **`/data/models/`** is the team model store; bringup
scripts (`fw-bringup/scripts/mi450-gptoss-setup.sh`, `perf-tracking/scripts/serve.sh`) use
paths under it. Personal downloads go in `$HOME` (~502 GB free on `/`).

**E2e checkpoint (use this):** **`/data/models/openai/gpt-oss-120b`** — admin symlink to
the on-disk weights at `/data/dev/morhuang/models/gpt-oss-120b` (183 GB, complete
`openai/gpt-oss-120b` BF16 safetensors, H=2880, matches §8.6). Read-only for all users;
TokenSpeed with a local `--model` path does not write into the weight tree. This is the
canonical shared path — not a private dev-only location.

```
GPT_OSS_MODEL=/data/models/openai/gpt-oss-120b
GPT_OSS_WORLD_SIZE=8
```

See `test/runtime/models/test_gpt_oss.py`. Coordinate GPU use on the shared node before
an 8-GPU serve; disk I/O is read-only.

**Do not use for this e2e:** `amd/gpt-oss-120b-w-mxfp4-a-fp8` (CI perf target, ~65 GB on
HF, not gated) — quantized weight/MoE path skews end-to-end timing and is not representative
of the full-precision serve this gate targets. Microbench already covers the comm op.

**Fallbacks:** download `openai/gpt-oss-120b` to `$HOME/models/` if the shared symlink
breaks (no HF credentials required — public model). Smoke-test only: `openai/gpt-oss-20b`
(~41 GB) in `$HOME/models/` — different scale, not a substitute for this gate.

### 0.3 E2e gate — DONE (see `AR_RMSNORM_MI350X_E2E_BENCHMARKS.md`)

Served gpt-oss-120b on MI350X; `triton_shmem` auto-selected + correct e2e. Key results,
methodology, serve gotchas, and the fusion-threshold decision live in the companion doc.
Two setup facts needed beyond §5.2: (1) the serve package also needs editing in —
`pip install -e /home/jeremwan/tokenspeed/python --no-deps --no-build-isolation` (the
baked editable `tokenspeed` points at an unmounted path → `import tokenspeed` fails);
(2) serve needs `--policy round_robin --kvstore-size 8` (see companion §1 gotchas).
**Bottom line:** fusion is not a served win on gfx950 (loses ws=4/8, helps ws=2 M≳384);
**set `comm_fusion_max_num_tokens=0` at ws≥4.**

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
- **`ops/communication/_coarse_shmem.py`** (untracked): coarse-grained + HIP-IPC
  peer-buffer allocator (`alloc_coarse_symm`, `CoarseSymmBuffer`) — the §8.1 perf fix.
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

## 5. Environment setup (validated end-to-end on MI300X)

Everything runs in **one ROCm 7.2 container**. The MI300X dev box is 8× gfx942 on a
ROCm 7.1 host driver; the container ships its own ROCm 7.2 userspace, so the host
version is irrelevant. For **MI350X**, use §0.1 container recipe instead of §5.1 base
image (or reuse `diprajap-tokenspeed:serve-base` which already ships torch 2.11).

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

### 6.2 Microbenchmark — built (`bench_triton_shmem_ar_rmsnorm.py`)
Self-contained multi-process bench: CUDA-event p50 + cross-rank skew, per-config
correctness-gated, backends via `TS_ARNORM_BACKEND` (triton_shmem / iris / native
symm_mem) plus an RCCL-unfused baseline. Env-overridable axes; run under noise control
with `benchmark/run_ar_rmsnorm_noise_controlled.sh`. All backends (incl. iris — its
singleton heap is pre-sized for the sweep) share one process per config so cross-backend
noise cancels. Results + interpretation: §8.

### 6.3 End-to-end (final gate) — DONE on MI350X
Completed on 8× MI350X gpt-oss-120b; full record in
`AR_RMSNORM_MI350X_E2E_BENCHMARKS.md`. `triton_shmem` auto-selects and is correct e2e.
The honest unfused baseline on AMD is the auto all-reduce (**triton custom-AR for ≤512 KiB
≈ ≤91 tokens at H=2880, else RCCL**) + triton fused add-RMSNorm — `custom_all_reduce` and
`trtllm_allreduce` are NVIDIA-only and disabled. Fusion is served only for
`num_tokens ≤ comm_fusion_max_num_tokens`. Verdict: fusion is **not** a served throughput
win on gfx950 (ws=4 ~20% decode regression when on); disable at ws≥4. (MI300X e2e never
run — no disk.)

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
- **Decision — leading + trailing signal-pad barriers on EVERY variant** (originally all
  via a dedicated `symm_grid_barrier_kernel`, a **single block per rank** doing
  `symm_mem_barrier(sig, block_id=0, …)` all-to-all, the kernel-launch boundary being the
  global barrier; reduced from `grid_sms` blocks to 1 during Phase-4 as it dominated small-M
  latency). **UPDATE (default now): the one-shot decode path folds both barriers IN-KERNEL**
  (`INKERNEL_BARRIER=1`, per-block at `block_id=pid`), removing the two launches; the
  separate 1-block kernel remains for the two-shot path and the `INKERNEL_BARRIER=0`
  fallback. See companion doc §4/§7 (reclaim, M-dependence caveat, fixed-participant plan).
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

### Phase 4 — validate & benchmark (DONE ✅, incl. MI350X e2e)
- **Correctness:** `test_triton_shmem_communication.py` passes ws=1/2/4/8 (all three
  kernel variants) + `test_communcation.py` via the dispatcher; fp32 ref, 2e-2 tol.
- **Graph capture:** capture+replay pass at ws=2 (one-shot) and ws=8 (two-shot).
- **Microbench + noise control:** §8. iris is included (heap pre-sized).
- **e2e:** DONE on MI350X (gpt-oss-120b) — see `AR_RMSNORM_MI350X_E2E_BENCHMARKS.md`.

### Phase 5 — coarse-grained substrate + hardened benchmark (DONE ✅)
- Coarse-grained HIP-IPC data buffers (`_coarse_shmem.py`, `TS_TRITON_SHMEM_COARSE`,
  default 1) — the perf fix (§8.1). Noise-controlled multi-backend sweep incl. iris,
  input-range validation, and the RCCL-unfused/e2e analysis (§8.2–§8.5).

---

## 8. Benchmark findings

Harness: `benchmark/bench_triton_shmem_ar_rmsnorm.py` (multi-process, CUDA-event p50,
per-config correctness-gated, fp32 ref, 2e-2 tol). Full op = copy-in → leading barrier
→ fused kernel → trailing barrier → copy-out, at the dispatcher level. Axes are
env-overridable (`BENCH_{WORLD_SIZES,BACKENDS,M_VALUES,N_VALUES,N_WARMUP,N_REPEAT}`).

### 8.1 Substrate fix — coarse-grained data buffers via HIP IPC (default on)

torch's ROCm symm_mem allocator (`CUDASymmetricMemory.cu`, `USE_ROCM`) allocates via
**HIP VMM** (`hipMemCreate` + `hipMemAllocationTypePinned`) with **no coherence knob**
(confirmed in shipped headers — only `TORCH_SYMMMEM_NBLOCKS` exists), so the mapping is
**fine-grained**: ~107 GB/s bulk local vs ~3255 GB/s coarse-grained (measured, 30×).
Every access paid it (copy-in/out + the kernel's local reads/writes), making the legacy
backend 0.04–0.11× RCCL at ws=8.

Fix: allocate the data buffers (`input`/`output`/`residual_out`) as ordinary
**coarse-grained `torch.empty`** tensors and expose them peer-to-peer via **HIP IPC**
(`hipIpcGetMemHandle`/`OpenMemHandle`, offsets via `hipMemGetAddressRange`), building the
same `buffer_ptrs_dev` uint64 peer table the kernels already consume. Only the signal pad
stays fine-grained symm_mem (barrier atomics need it). This is the rocSHMEM/MSCCL++/vLLM
P2P pattern. Code: `ops/communication/_coarse_shmem.py`; wired into `triton_shmem.py`
behind `TS_TRITON_SHMEM_COARSE` (default `1`, `0` = legacy). Kernels, barrier, dispatch,
and graph-capture path unchanged. Coherence validated (`/home/jeremwan/coarse_probe.py`):
in-kernel peer **read and write** are coherent across the signal-pad barrier
(`sem=release/acquire scope=sys`); remote xGMI is fabric-bound (~0.9× either grain), so
the win is on local access — exactly why two-shot (remote push once, rest local) is the
right dispatch at ws≥4.

Constraint: HIP IPC needs the torch caching allocator **not** in expandable-segments
(VMM) mode. If IPC export fails, `create_*` declines and the dispatcher falls back to the
fine-grained path rather than crashing.

### 8.2 Noise control

This box **cannot pin GPU clocks** — sysfs perf-control is read-only in the unprivileged
container, so `rocm-smi --setperfdeterminism` is a no-op. Substitutes (driver:
`benchmark/run_ar_rmsnorm_noise_controlled.sh`): world-size isolation (each ws in its own
process + 20 s cooldown, no heat-soak carry-over), high warmup/repeat (30/150), and a
**2-pass variance floor**. Measured floor `|pass1−pass2|/min`: median 0.1–0.8%; **every
config ≥0.5 ms is stable to <5%**; all larger deltas are sub-0.5 ms latency-bound configs
(consistent with the reference's finding). **Trust only ≥0.5 ms configs**; small-size
margins are within noise.

### 8.3 Results (noise-controlled, ≥0.5 ms configs, vs RCCL-unfused baseline)

`results/ar_rmsnorm_noise_controlled/` (per-ws, 2 passes). Representative large-tensor
p50 (ms) and speedup vs RCCL (>1 = fused wins):

| ws | M×N | RCCL | triton_shmem | iris | triton_shmem / iris |
|----|-----|------|--------------|------|---------------------|
| 2 | 16384×4096  | 3.11 | 2.95 (1.05×) | 2.94 (1.06×) | ~1.0× |
| 2 | 16384×16384 | 12.30| 12.00 (1.03×)| 12.30 (1.00×)| ~1.0× |
| 4 | 16384×2880  | 1.22 | 1.89 (0.64×) | 3.14 (0.39×) | **1.66×** |
| 4 | 16384×16384 | 6.75 | 10.4 (0.65×) | 18.2 (0.37×) | **1.75×** |
| 8 | 16384×2880  | 0.73 | 1.12 (0.65×) | 5.17 (0.14×) | **4.6×** |
| 8 | 16384×16384 | 3.99 | 6.56 (0.61×) | 35.3 (0.11×) | **5.4×** |

- **triton_shmem wins/parity at ws=2** (1.0–1.15×), settles at **0.6–0.7× RCCL at
  ws=4/8** — the rocSHMEM reference envelope (ws=2 win, ws=8 RCCL wins on large tensors as
  it keeps gaining xGMI bandwidth with PE count).
- **triton_shmem beats iris everywhere at ws≥4 (1.7× → 5.4×)** and matches it at ws=2.
  iris uses a one-shot full-fan-in kernel (each row reads all peers), so its cost scales
  with ws; triton_shmem's two-shot at ws≥4 is bandwidth-optimal. triton_shmem is the AMD
  fused default.

### 8.4 Input range — no restriction for triton_shmem

The triton_shmem kernels take **arbitrary M and N**: the blocked variants mask N with a
power-of-two `BLOCK_N`; two-shot `cdiv`-shards M (guards non-divisible M); `oneshot_wholerow`
(the only pow2-N kernel) is auto-selected **only** at ws≤2 for pow2 N, else the dispatch
picks a blocked kernel. Verified beyond the reference grid at ws=8 (non-pow2 N ∈ {3584,
5120}, non-divisible M ∈ {2000, 32768}): all correct, same 0.61–0.69× envelope
(`results/ar_rmsnorm_extended_range.csv`). The prior "iris only" restriction on the sweep
was **not** a kernel/input limit: it was the iris shim's process-global singleton heap
(sized at first use, never grows). The bench now pre-sizes it for the whole sweep
(`_presize_iris_heap`), so iris runs the full grid too (it also handles arbitrary M/N, just
uncompetitively at ws≥4).

### 8.5 "RCCL-unfused" baseline vs. the real TokenSpeed unfused path (e2e caveats)

The bench baseline is `dist.all_reduce(bf16)` + residual-add + eager `F.rms_norm`. The
**real** TokenSpeed unfused path (`models/base/comm_ops.py::AllReduceNormOp`, else-branch)
is `all_reduce(x, group)` + `norm_module(x, residual)` where (a) `all_reduce` goes through
`get_global_backend()` (auto → **custom IPC all-reduce** if registered, else triton, else
RCCL) and (b) the norm is a **Triton fused add-RMSNorm**, not eager. Two consequences:
- The bench baseline is a **pessimistic** proxy: real unfused AR often uses the custom IPC
  path (faster than plain RCCL at small sizes), so fused's true e2e advantage is **smaller**
  than the bench's RCCL ratios suggest.
- Fusion is only taken for `num_tokens ≤ comm_fusion_max_num_tokens` (default **2048**) and
  only when `enable_allreduce_fusion` (auto-on for single-node AMD TP). **Large prefill
  (>2048 tokens) always runs unfused** — so the bench's large-M rows (4096/16384) map to no
  served fused decision; they are robustness/scaling data. The e2e-relevant fused regime is
  **M ≤ 2048** (decode + small prefill), where the bench shows fused competitive.

**Microbench → e2e extrapolation is bounded, not direct:** it is a faithful *kernel-level*
latency comparison, but (i) only M≤2048 is a served fused shape, (ii) the honest e2e
baseline is the auto/custom AR, not the RCCL proxy, and (iii) AR+RMSNorm is a small fraction
of per-layer time (attention + MLP GEMMs dominate), so op-level ratios do not linearly map to
tokens/s. The e2e server run (§6.3) — **now DONE** (companion doc §5) — is the real gate.

**Bottom line:** correctness, dependency-avoidance, graph-capture, and performance goals are
met. `triton_shmem` (coarse default) is the AMD fused backend — competitive with RCCL at
ws=2, 0.6–0.7× at ws=4/8 (rocSHMEM envelope), and 1.7–5.4× faster than iris at ws≥4.

### 8.6 Model-targeted sweep — where fusion is profitable (ws=4/8 crossover)

The dispatch now routes `auto → triton_shmem` (native symm_mem fallback); iris is
demoted to explicit-only (`TS_ARNORM_BACKEND=iris`). The migrated backend carries **no
input-size/dtype gate beyond the shared eligibility** (bf16 contract + the caller's
`comm_fusion_max_num_tokens=2048` cap); the old iris-era routing was the only real
activation gate and is gone. The sweep was retargeted to the **row widths actually
encountered** by the fused AR+RMSNorm / comm+norm ops in the production models rather than
a synthetic pow2 grid (`results/ar_rmsnorm_model_targeted/`, 2-pass, iris excluded):

- **N = 512** — DeepSeek-V3/V4 & Kimi-K2 `kv_lora_rank` (compressed-KV latent norm).
- **N = 1536** — DeepSeek/Kimi `q_lora_rank` (compressed-Q latent norm) = GLM `moe_intermediate_size`.
- **N = 2880 / 5120 / 7168** — hidden sizes of gpt-oss-120B / GLM-4.6 / DeepSeek-V3-V4·Kimi-K2 (the residual stream).
- **M ∈ {1…4096}** — decode/low-concurrency → chunked-prefill cap (2048) → one point past.

**Fusion profitability vs the (pessimistic) RCCL baseline — speedup = RCCL ÷ triton_shmem:**

| ws | profitable regime (>1.0×) | peak vs RCCL | large-M (≥0.5 ms, trustworthy) |
|----|---------------------------|--------------|--------------------------------|
| 2  | M ≳ 256 (N≥5120), ≳512 (2880), ≳1024 (1536); N=512 ~never | 1.05–1.14× @ M 256–1024 | ~0.97–1.01× (parity) |
| 4  | **none** — never reaches parity | ~0.83× @ M≈512, N=5120 | 0.66–0.68× @ M=4096 |
| 8  | **none** — never reaches parity | ~0.78× @ M≈1024, N=2880 | 0.62–0.66× @ M=4096 |

**Where fusion stops being profitable at ws=4/8:** against the RCCL microbench baseline it
is *not* profitable anywhere in the served range — the ratio **peaks at moderate M
(≈512–1024) and then erodes monotonically** toward the large-M floor (~0.62–0.66×). So the
marginal case for fusion is strongest around M≈512–1024 and weakens past ~1024 tokens; the
small compressed-latent widths (N=512, 1536) are the least fusion-favorable at every ws
(latency-bound, too few bytes to amortize the barrier/copy overhead), while the wins — where
they exist (ws=2) — concentrate on the large residual streams (2880–7168).

**Hypothesis (the fused kernel is already highly tuned, so this is algorithmic, not a tuning
gap).** Fusion's benefit is a **fixed-overhead saving**: one kernel launch instead of three
(AR + add + norm), no intermediate materialization, one pass over the row. That saving is a
near-constant; it dominates only while the op is latency-bound (small/moderate M), which is
why parity/wins appear at ws=2 and mid-M. As M and ws grow the op becomes **bandwidth-bound**,
and there RCCL's ring/tree all-reduce is asymptotically byte-optimal and gains effective xGMI
bandwidth as link/PE count rises with ws, whereas the one-shot/two-shot fused pattern moves
more fabric bytes per rank. A constant overhead saving cannot outrun a growing
byte-movement gap — hence profitability humps at moderate M and falls off with M/ws even
for a perfectly-tuned kernel. **Caveat (see §8.5):** the RCCL baseline is pessimistic (real
e2e unfused uses the custom IPC AR), so these are conservative crossovers; the served fused
regime is M≤2048 and the e2e gate (§6.3) is **DONE** (companion doc §5: fusion parity/win at
ws=4 decode with the shipped fixes).

## 9. Design invariants & open items

Invariants (hold today; preserve them):
- **Per-tensor peer tables.** Offsets are not shared across allocations — two-shot uses
  3 tables (input/output/residual_out); one-shot uses 1.
- **Barrier.** Leading + trailing signal-pad barriers on every variant (trailing guards
  cross-call reuse of the persistent `input`; two-shot push ordering). **Default: the
  one-shot decode path folds them IN-KERNEL** (`INKERNEL_BARRIER=1`, per-block at
  `block_id=pid`); the two-shot path and the legacy fallback use the separate 1-block
  `symm_grid_barrier_kernel`. The in-kernel form is M-dependent (see open items).
- **Substrate.** Coarse-grained `torch.empty` data buffers + HIP-IPC peer table; only
  the signal pad is fine-grained symm_mem. Requires the caching allocator **not** in
  expandable-segments mode; on IPC-export failure the backend declines → dispatcher
  falls back to fine-grained (correct, slow).

Open items (full plan + data: **companion doc §7**):
- **Fused small-M loss — mostly removed; two levers remain.** SHIPPED: one-shot dispatch
  (`ONESHOT_MAX_M=256`, no copy-out) + in-kernel barriers (default ON, no barrier launches) →
  ws=4 decode op ~0.088→~0.048 ms, parity/win vs unfused. Remaining bounded gap: the barrier
  *work* stays in-kernel and **copy-in (~0.012 ms) is an untouched floor**. Next levers
  (companion §7): **(A) avoid copy-in** (producer writes into the symmetric input buffer;
  biggest lever, ~25% of the decode op) and **(B) fixed-participant in-kernel barrier**
  (fixes the in-kernel barrier's grid-scaling cost, the M=256 op-level dip, makes it robust
  to M-divergence, and unlocks two-shot in-kernel — currently gated OFF as a measured loss).
- **In-kernel barrier is M-divergence-fragile** (`block_id∈[0,grid_sms)`, M-dependent slot
  range): deadlocks only if TP ranks replay different-M graphs at once — never in dp=1 pure
  TP (the separate 1-block barrier is M-independent, hence robust). Set `INKERNEL_BARRIER=0`
  under DP / overlap>1 / spec-decode until lever B lands. Repro:
  `benchmark/probe_inkernel_barrier_graph.py PROBE_MODE=multigraph`.
- **Fusion gate:** enable `comm_fusion_max_num_tokens>0` for ws=2 (wins broadly) and ws=4
  decode (parity/win). Companion §3/§5/§7.
- **Large unfused RCCL all-reduce hang is serve-specific, NOT RCCL and NOT the fused
  kernel.** Standalone ws=4 RCCL stress (incl. rank jitter) passes; under serve 2/4 ranks
  spin. Prime hypothesis: RCCL colliding with coexisting symm_mem/HIP-IPC collectives.
  ws=4-safe debug plan in companion §6/§7 (do NOT run 8-GPU variants — GPU 3 contention).
- **Serve setup skew:** image ships smg 1.4.1 but the checkout pins smg 1.7.0 → rebuild the
  image to drop the `--policy round_robin`/socket-leak workarounds (all perf-neutral).
  Companion §1.
- **Decouple from caching-allocator mode** — a dedicated
  `hipExtMallocWithFlags(hipDeviceMallocDefault)` allocator would remove the
  expandable-segments constraint above.
- **Kernel tuning** — `ArchProfile` is MI300X/MI350X-specific (`detect_arch()`
  auto-selects); the remaining gap to RCCL at ws≥4 is kernel-level (grid/block, xGMI push
  scheduling), not substrate. Re-tune for other archs.
- **torch/rocm version** — validated on torch 2.11+rocm7.2; re-check symm_mem signal-pad
  behavior on other versions.
- **Sub-group TP** — symm_mem rendezvous + IPC accept any process group, but only
  whole-world TP is exercised; validate a sub-group before relying on it.

## 10. Key file reference
- **MI350X e2e + fusion-threshold record (companion):**
  `ops/communication/AR_RMSNORM_MI350X_E2E_BENCHMARKS.md`. Data:
  `benchmark/results/ar_rmsnorm_mi350x_e2e/` (crossover CSVs + e2e A/B). Helpers:
  `benchmark/e2e_gptoss_{serve,bench,teardown}.sh`.
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
  `ops/communication/_triton_shmem_kernels.py` (vendored kernels + tuning + barrier) +
  `ops/communication/_coarse_shmem.py` (coarse-grained HIP-IPC data buffers, §8.1).**
  (The old rocSHMEM port that previously held the `triton_shmem.py` name is deleted.)
- Dispatch + native symm_mem kernel + reusable helpers (`_alloc_symm`,
  `_peer_ptrs_dev`, `symm_mem_barrier`, `amd_allreduce_residual_rmsnorm_kernel`):
  `ops/communication/triton.py`
- Sibling backend template (Iris): `ops/communication/iris.py`
- Callers: `runtime/layers/layernorm.py`, `runtime/distributed/comm_ops.py`
- Existing tests: `test/ops/test_communcation.py`, `test/ops/test_iris_communication.py`
- **Migrated-backend test: `test/ops/test_triton_shmem_communication.py`**
- **Microbench: `benchmark/bench_triton_shmem_ar_rmsnorm.py`; noise-controlled driver:
  `benchmark/run_ar_rmsnorm_noise_controlled.sh`. Results:
  `results/ar_rmsnorm_noise_controlled/pass{1,2}_ws{2,4,8}.csv` (2-pass variance floor),
  `results/ar_rmsnorm_extended_range.csv` (non-pow2 N / non-divisible M). Reference:
  `triton-shmem/benchmark/results/ar_rmsnorm_opt_sweep/reverified_baseline.csv`.**
- Phase 0 probes: `/home/jeremwan/symm_probe.py`, `/home/jeremwan/symm_graph_probe.py`
- Coarse-grained fix probe (§8.1: local/remote BW + peer read/write coherence):
  `/home/jeremwan/coarse_probe.py`
