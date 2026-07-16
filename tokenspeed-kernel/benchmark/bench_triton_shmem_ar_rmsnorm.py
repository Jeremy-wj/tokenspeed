# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Multi-process microbenchmark for the fused AR + residual + RMSNorm backends.

Self-contained (``torch.distributed`` + ``mp.spawn``, no new deps) latency
microbench for the migrated ``triton_shmem`` backend. Times, per ``(world_size,
M, N)``, the full production op (input copy-in -> barrier -> fused kernel ->
barrier -> copy-out) for each backend selectable through the
``TS_ARNORM_BACKEND`` dispatch, plus an RCCL ``all_reduce`` + residual +
``F.rms_norm`` unfused baseline:

* ``rccl_unfused`` -- ``dist.all_reduce`` (native bf16 transport) + residual add
  + eager ``F.rms_norm``. The baseline (the reference bench's
  ``dist_unfused_ar_rmsnorm``).
* ``triton_shmem``   -- the migrated PyTorch symmetric-memory backend (this project).
* ``symm_mem``     -- the native TokenSpeed symm_mem fused kernel.
* ``iris``         -- the Iris backend.

For each config it CUDA-event-times every rank, reports the collective latency
as the **max p50 across ranks** (a collective is bounded by its slowest rank)
plus cross-rank skew, and prints the fused-vs-RCCL speedup. Axes mirror the
external ``triton-shmem`` reference bench
(``benchmark/results/ar_rmsnorm_opt_sweep/reverified_baseline.csv``) so the
migrated numbers can be compared against the rocSHMEM port directly:
``num_ranks in {2,4,8}``, ``M,N`` power-of-two, ``fusion=residual``, bf16.

Correctness is asserted (fp32 reference, 2e-2 tol) on the first iteration of
every config for every backend before timing, so a numerically-broken backend
fails loudly rather than reporting a fast-but-wrong latency.

Run (inside the ROCm container, from the tokenspeed repo root)::

    python -m benchmark.bench_triton_shmem_ar_rmsnorm            # default sweep
    BENCH_WORLD_SIZES=8 python -m benchmark.bench_triton_shmem_ar_rmsnorm
    BENCH_BACKENDS=triton_shmem,rccl_unfused python -m benchmark.bench_triton_shmem_ar_rmsnorm
    BENCH_CSV=results/triton_shmem_bench.csv python -m benchmark.bench_triton_shmem_ar_rmsnorm
"""
from __future__ import annotations

import os
import socket
import statistics
import traceback
from typing import List, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

_EPS = 1e-6

# Axes (mirror the triton-shmem reverified_baseline.csv grid). ``2880`` is the
# production gpt-oss hidden size (non-pow2 -> exercises the blocked kernels);
# the power-of-two Ns additionally reach oneshot_wholerow at ws<=2.
_DEFAULT_WORLD_SIZES: List[int] = [2, 4, 8]
_M_VALUES: List[int] = [1024, 4096, 16384]
_N_VALUES: List[int] = [1024, 2880, 4096, 16384]
_DEFAULT_BACKENDS: List[str] = ["rccl_unfused", "triton_shmem", "symm_mem", "iris"]

_N_WARMUP = 25
_N_REPEAT = 100


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _env_list(name: str, default: List) -> List:
    raw = os.environ.get(name)
    if not raw:
        return default
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if default and isinstance(default[0], int):
        return [int(p) for p in parts]
    return parts


def _reference(x, residual, weight, world_size, hidden, device):
    reduced = torch.full(
        (x.shape[0], hidden),
        world_size * (world_size + 1) // 2,
        dtype=torch.float32,
        device=device,
    )
    ref_residual = reduced + residual.float()
    ref_norm = ref_residual * torch.rsqrt(
        ref_residual.pow(2).mean(dim=-1, keepdim=True) + _EPS
    )
    ref_norm = ref_norm * weight.float()
    return ref_residual, ref_norm


def _make_inputs(tokens, hidden, rank, device):
    # rank contributes rank+1 (sum = ws*(ws+1)/2); residual deterministic and
    # replicated across ranks (as under TP) so a bug can't be masked.
    x = torch.full((tokens, hidden), rank + 1, dtype=torch.bfloat16, device=device)
    residual = (
        torch.arange(tokens * hidden, dtype=torch.float32, device=device)
        .reshape(tokens, hidden)
        .mul_(0.001)
        .to(torch.bfloat16)
    )
    return x, residual


def _run_backend(backend, x, residual, weight, rank, group, max_token_num):
    """Run one fused op for ``backend``; return (norm_out, residual_out)."""
    hidden = x.shape[1]
    if backend == "rccl_unfused":
        acc = x.detach().clone()
        dist.all_reduce(acc, group=group)
        residual_out = acc + residual
        norm_out = F.rms_norm(residual_out, [hidden], weight, _EPS)
        return norm_out, residual_out

    # All fused backends share the production dispatcher; TS_ARNORM_BACKEND
    # (set by the caller) selects which one runs.
    from tokenspeed_kernel.ops.communication.triton import (
        allreduce_residual_rmsnorm,
    )

    norm_out, residual_out, _, _ = allreduce_residual_rmsnorm(
        input_tensor=x,
        residual=residual,
        weight=weight,
        rank=rank,
        group=group,
        eps=_EPS,
        max_token_num=max_token_num,
    )
    return norm_out, residual_out


def _time_backend(backend, x, residual, weight, rank, group, max_token_num) -> float:
    """Return this rank's p50 latency (ms) for ``backend`` on the given inputs."""
    for _ in range(_N_WARMUP):
        _run_backend(backend, x, residual, weight, rank, group, max_token_num)
    torch.cuda.synchronize()
    dist.barrier(group=group)

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(_N_REPEAT)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(_N_REPEAT)]
    for i in range(_N_REPEAT):
        starts[i].record()
        _run_backend(backend, x, residual, weight, rank, group, max_token_num)
        ends[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return statistics.median(times)


def _worker_fn(rank, world_size, port, backends, result_dict, error_dict):
    try:
        _worker_main(rank, world_size, port, backends, result_dict)
    except Exception:
        error_dict[rank] = traceback.format_exc()


def _worker_main(rank, world_size, port, backends, result_dict):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        group = dist.group.WORLD
        max_token_num = max(_M_VALUES)
        for n in _N_VALUES:
            weight = torch.linspace(0.5, 1.5, n, dtype=torch.bfloat16, device=device)
            for m in _M_VALUES:
                x, residual = _make_inputs(m, n, rank, device)
                ref_residual, ref_norm = _reference(
                    x, residual, weight, world_size, n, device
                )
                for backend in backends:
                    if backend != "rccl_unfused":
                        os.environ["TS_ARNORM_BACKEND"] = backend
                    # Correctness gate before timing.
                    norm_out, residual_out = _run_backend(
                        backend, x, residual, weight, rank, group, max_token_num
                    )
                    if norm_out is None:
                        result_dict[(rank, backend, m, n)] = float("nan")
                        continue
                    torch.testing.assert_close(
                        residual_out.float(), ref_residual, atol=2e-2, rtol=2e-2
                    )
                    torch.testing.assert_close(
                        norm_out.float(), ref_norm, atol=2e-2, rtol=2e-2
                    )
                    p50 = _time_backend(
                        backend, x, residual, weight, rank, group, max_token_num
                    )
                    result_dict[(rank, backend, m, n)] = p50
    finally:
        os.environ.pop("TS_ARNORM_BACKEND", None)
        dist.destroy_process_group()


def _aggregate(result_dict, world_size, backends) -> List[dict]:
    """Collapse per-rank p50s into per-config collective latency + skew."""
    rows = []
    for n in _N_VALUES:
        for m in _M_VALUES:
            for backend in backends:
                vals = [
                    result_dict.get((r, backend, m, n)) for r in range(world_size)
                ]
                vals = [v for v in vals if v is not None]
                if not vals or any(v != v for v in vals):  # NaN -> unsupported
                    rows.append(
                        {"backend": backend, "M": m, "N": n, "lat_ms": float("nan"),
                         "skew_pct": float("nan")}
                    )
                    continue
                lat = max(vals)  # collective bounded by slowest rank
                lo = min(vals)
                skew = (lat - lo) / lat * 100.0 if lat > 0 else 0.0
                rows.append(
                    {"backend": backend, "M": m, "N": n, "lat_ms": lat,
                     "skew_pct": skew}
                )
    return rows


def _run_one_world(world_size: int, backends: List[str]) -> List[dict]:
    manager = mp.Manager()
    result_dict = manager.dict()
    error_dict = manager.dict()
    port = _get_open_port()
    mp.spawn(
        _worker_fn,
        args=(world_size, port, backends, result_dict, error_dict),
        nprocs=world_size,
        join=True,
    )
    if error_dict:
        msg = "\n".join(f"Rank {r}: {e}" for r, e in error_dict.items())
        raise RuntimeError(f"worker failure at ws={world_size}:\n{msg}")
    return _aggregate(dict(result_dict), world_size, backends)


def _print_world(world_size: int, rows: List[dict], backends: List[str]) -> None:
    baseline = "rccl_unfused"
    by_cfg = {}
    for r in rows:
        by_cfg.setdefault((r["M"], r["N"]), {})[r["backend"]] = r
    print(f"\n===== world_size = {world_size} =====")
    hdr = f"{'M':>6} {'N':>6} | " + " | ".join(
        f"{b:>16}" for b in backends
    ) + " | " + " ".join(f"{b.split('_')[0]:>7}x" for b in backends if b != baseline)
    print(hdr)
    print("-" * len(hdr))
    for n in _N_VALUES:
        for m in _M_VALUES:
            cfg = by_cfg.get((m, n), {})
            base = cfg.get(baseline, {}).get("lat_ms", float("nan"))
            lat_cells = []
            spd_cells = []
            for b in backends:
                lat = cfg.get(b, {}).get("lat_ms", float("nan"))
                lat_cells.append(f"{lat:16.4f}" if lat == lat else f"{'n/a':>16}")
                if b != baseline:
                    spd = base / lat if (lat == lat and lat > 0) else float("nan")
                    spd_cells.append(f"{spd:7.2f}x" if spd == spd else f"{'n/a':>8}")
            print(f"{m:>6} {n:>6} | " + " | ".join(lat_cells) + " | " + " ".join(spd_cells))


def _write_csv(path: str, all_rows: List[Tuple[int, dict]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("world_size,backend,M,N,lat_ms,skew_pct\n")
        for ws, r in all_rows:
            f.write(
                f"{ws},{r['backend']},{r['M']},{r['N']},"
                f"{r['lat_ms']:.6f},{r['skew_pct']:.3f}\n"
            )


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA/ROCm required")
    world_sizes = _env_list("BENCH_WORLD_SIZES", _DEFAULT_WORLD_SIZES)
    backends = _env_list("BENCH_BACKENDS", _DEFAULT_BACKENDS)
    ndev = torch.cuda.device_count()
    csv_path = os.environ.get("BENCH_CSV")

    all_rows: List[Tuple[int, dict]] = []
    for ws in world_sizes:
        if ws > ndev:
            print(f"skip ws={ws}: only {ndev} GPUs")
            continue
        rows = _run_one_world(ws, backends)
        _print_world(ws, rows, backends)
        all_rows.extend((ws, r) for r in rows)

    if csv_path:
        _write_csv(csv_path, all_rows)
        print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
