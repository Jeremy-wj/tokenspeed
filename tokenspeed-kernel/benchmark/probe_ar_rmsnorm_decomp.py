"""Decompose fused AR+RMSNorm and compare with the serving unfused stack.

The serving baseline follows TokenSpeed's production transport gate:

* bf16 payload <= 512 KiB: Triton shared-memory all-reduce
* larger payload: RCCL
* both paths then run TokenSpeed's Triton residual-add RMSNorm

Input resets needed by in-place transports happen before timing events. Directly
timed and subtraction-derived columns are labeled separately.

Examples:
    HIP_VISIBLE_DEVICES=1,2 BENCH_WS=2 \
      BENCH_CSV=results/decomp_ws2.csv \
      python -m benchmark.probe_ar_rmsnorm_decomp
"""
from __future__ import annotations

import csv
import os
import socket
import statistics
from typing import Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from benchmark.shape_axes import default_hidden_size

_EPS = 1e-6
_TRITON_AR_MAX_BYTES = 512 * 1024


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def _env_ints(name: str, default: list[int]) -> list[int]:
    raw = os.environ.get(name)
    return [int(v.strip()) for v in raw.split(",") if v.strip()] if raw else default


_N = default_hidden_size()
_MS = _env_ints("BENCH_M_VALUES", [8, 32, 64, 128, 256, 512, 1024])
_WARM = _env_int("BENCH_N_WARMUP", 30)
_REP = _env_int("BENCH_N_REPEAT", 100)


def _port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _time(
    fn: Callable[[], object],
    group: dist.ProcessGroup,
    setup: Callable[[], object] | None = None,
) -> float:
    """Return p50 after taking the maximum rank for every iteration."""
    for _ in range(_WARM):
        if setup is not None:
            setup()
        fn()
    torch.cuda.synchronize()
    dist.barrier(group=group)

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(_REP)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(_REP)]
    for i in range(_REP):
        if setup is not None:
            setup()
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()

    samples = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(starts, ends)],
        dtype=torch.float64,
        device="cuda",
    )
    dist.all_reduce(samples, op=dist.ReduceOp.MAX, group=group)
    return statistics.median(samples.cpu().tolist())


def _worker(rank: int, ws: int, port: int, out) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=ws,
    )
    group = dist.group.WORLD
    device = torch.device(f"cuda:{rank}")

    from tokenspeed_kernel.ops.communication import triton as tri
    from tokenspeed_kernel.ops.communication import triton_shmem as ts
    from tokenspeed_kernel.ops.layernorm.triton import rmsnorm as triton_rmsnorm

    max_m = max(_MS)
    fused_state = ts.create_triton_shmem_ar_rmsnorm_state(
        group=group,
        rank_in_group=rank,
        max_token_num=max_m,
        hidden_dim=_N,
        dtype=torch.bfloat16,
    )
    if fused_state is None:
        raise RuntimeError("triton_shmem state creation failed")
    triton_ar_state = tri.create_state(
        group=group,
        rank_in_group=rank,
        device=device,
        max_numel=_TRITON_AR_MAX_BYTES
        // torch.empty((), dtype=torch.bfloat16).element_size(),
    )
    weight = torch.linspace(0.5, 1.5, _N, dtype=torch.bfloat16, device=device)

    rows = []
    for m in _MS:
        x = torch.full((m, _N), rank + 1, dtype=torch.bfloat16, device=device)
        residual = (
            torch.arange(m * _N, dtype=torch.float32, device=device)
            .reshape(m, _N)
            .mul_(0.001)
            .to(torch.bfloat16)
        )
        norm_out = torch.empty_like(x)
        residual_out = torch.empty_like(x)
        scratch = torch.empty_like(x)

        uses_twoshot = fused_state._is_twoshot and m > fused_state._oneshot_max_m
        path = (
            "twoshot"
            if uses_twoshot
            else f"oneshot:{fused_state._oneshot_kernel.split('_')[-1]}"
        )

        # Directly timed fused phases and complete variants.
        t_copyin = _time(lambda: fused_state._x[:m].copy_(x), group)
        t_barrier = _time(lambda: fused_state._barrier(), group)
        if uses_twoshot:
            t_copyout = _time(
                lambda: (
                    norm_out.copy_(fused_state._y[:m]),
                    residual_out.copy_(fused_state._residual_out[:m]),
                ),
                group,
            )
        else:
            t_copyout = 0.0

        default_fold_copyin = fused_state._fold_copyin
        fused_state._inkernel = False
        fused_state._fold_copyin = False
        t_full_sep = _time(
            lambda: fused_state.fused(x, residual, weight, _EPS), group
        )
        fused_state._inkernel = True
        fused_state._fold_copyin = False
        t_full_ink = _time(
            lambda: fused_state.fused(x, residual, weight, _EPS), group
        )
        fused_state._fold_copyin = True
        t_full_fold = _time(
            lambda: fused_state.fused(x, residual, weight, _EPS), group
        )
        t_full_default = t_full_fold if default_fold_copyin else t_full_ink
        fused_state._fold_copyin = default_fold_copyin

        # Subtraction-derived estimates. They can be noisy at sub-0.1 ms.
        t_kernel_derived = (
            t_full_sep - t_copyin - 2 * t_barrier - t_copyout
        )
        t_inkbarrier_derived = (
            t_full_ink - t_copyin - t_kernel_derived - t_copyout
        )

        use_triton_ar = x.numel() * x.element_size() <= _TRITON_AR_MAX_BYTES
        transport = "triton_ar" if use_triton_ar else "rccl"

        def reset_scratch():
            scratch.copy_(x)

        def run_transport():
            if use_triton_ar:
                return tri.all_reduce(triton_ar_state, scratch)
            return dist.all_reduce(scratch, group=group)

        def run_unfused():
            run_transport()
            return triton_rmsnorm(
                scratch,
                weight,
                _EPS,
                residual=residual,
            )

        # Numerical gate against the same transport and norm outside timing.
        reset_scratch()
        baseline_norm, baseline_residual = run_unfused()
        reference = x.clone()
        dist.all_reduce(reference, group=group)
        reference_norm, reference_residual = triton_rmsnorm(
            reference,
            weight,
            _EPS,
            residual=residual,
        )
        torch.testing.assert_close(
            baseline_residual.float(),
            reference_residual.float(),
            atol=2e-2,
            rtol=2e-2,
        )
        torch.testing.assert_close(
            baseline_norm.float(),
            reference_norm.float(),
            atol=2e-2,
            rtol=2e-2,
        )

        t_transport = _time(run_transport, group, setup=reset_scratch)
        # Norm phase on the already reduced reference; no transport in this timer.
        t_norm = _time(
            lambda: triton_rmsnorm(
                reference,
                weight,
                _EPS,
                residual=residual,
            ),
            group,
        )
        t_unfused = _time(run_unfused, group, setup=reset_scratch)

        rows.append(
            {
                "world_size": ws,
                "M": m,
                "N": _N,
                "path": path,
                "baseline_transport": transport,
                "copyin_ms": t_copyin,
                "barrier_ms": t_barrier,
                "kernel_derived_ms": t_kernel_derived,
                "inkbarrier_derived_ms": t_inkbarrier_derived,
                "copyout_ms": t_copyout,
                "full_sep_ms": t_full_sep,
                "full_ink_ms": t_full_ink,
                "full_fold_ms": t_full_fold,
                "full_default_ms": t_full_default,
                "serve_transport_ms": t_transport,
                "serve_norm_ms": t_norm,
                "serve_unfused_ms": t_unfused,
                "serve_speedup": t_unfused / t_full_default,
            }
        )

    if rank == 0:
        out.extend(rows)
    dist.destroy_process_group()


def _write_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ws = _env_int("BENCH_WS", 4)
    manager = mp.Manager()
    out = manager.list()
    mp.spawn(_worker, args=(ws, _port(), out), nprocs=ws, join=True)
    rows = list(out)

    print(f"\n===== ws={ws} N={_N} (ms, max-rank p50) =====")
    print(
        f"{'M':>5} {'path':>18} {'base':>10} {'copy':>8} {'kern*':>8} "
        f"{'inkbar*':>8} {'full':>8} {'unfused':>8} {'speedup':>8}"
    )
    for row in rows:
        print(
            f"{row['M']:5d} {row['path']:>18} "
            f"{row['baseline_transport']:>10} {row['copyin_ms']:8.4f} "
            f"{row['kernel_derived_ms']:8.4f} "
            f"{row['inkbarrier_derived_ms']:8.4f} "
            f"{row['full_default_ms']:8.4f} "
            f"{row['serve_unfused_ms']:8.4f} "
            f"{row['serve_speedup']:8.2f}"
        )
    print("* subtraction-derived estimate")

    csv_path = os.environ.get("BENCH_CSV")
    if csv_path:
        _write_csv(csv_path, rows)
        print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
