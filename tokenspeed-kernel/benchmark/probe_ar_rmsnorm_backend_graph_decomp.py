"""Captured 72-call stage decomposition for GPT-OSS decode M=32.

Each reported value is max-rank p50 microseconds per call. Cumulative captured
graphs provide additive marginal critical-path stages, avoiding sums of
independently timed kernels that can over-count launch overlap.
"""

from __future__ import annotations

import csv
import json
import os
import socket
import statistics
from typing import Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from benchmark import probe_ar_rmsnorm_backend_decomp as eager


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


_M = _env_int("BENCH_M", 32)
_N = _env_int("BENCH_N", 2880)
_WS = _env_int("BENCH_WS", 4)
_CALLS = _env_int("BENCH_CALLS_PER_GRAPH", 72)
_WARM = _env_int("BENCH_GRAPH_N_WARMUP", 30)
_REP = _env_int("BENCH_GRAPH_N_REPEAT", 1000)


def _port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _capture(fn: Callable[[], object], group: dist.ProcessGroup) -> torch.cuda.CUDAGraph:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    dist.barrier(group=group)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        fn()
    torch.cuda.synchronize()
    dist.barrier(group=group)
    return graph


def _time_graph_us(
    graph: torch.cuda.CUDAGraph,
    group: dist.ProcessGroup,
) -> float:
    for _ in range(_WARM):
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier(group=group)

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(_REP)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(_REP)]
    for index in range(_REP):
        starts[index].record()
        graph.replay()
        ends[index].record()
    torch.cuda.synchronize()
    samples = torch.tensor(
        [
            start.elapsed_time(end) * 1000.0 / _CALLS
            for start, end in zip(starts, ends)
        ],
        dtype=torch.float64,
        device="cuda",
    )
    dist.all_reduce(samples, op=dist.ReduceOp.MAX, group=group)
    return statistics.median(samples.cpu().tolist())


def _make_row(backend: str, path: str, **values) -> dict:
    row = {
        "world_size": _WS,
        "M": _M,
        "N": _N,
        "calls_per_graph": _CALLS,
        "backend": backend,
        "path": path,
        "copy_prefix_us_per_call": 0.0,
        "entry_prefix_us_per_call": 0.0,
        "kernel_prefix_us_per_call": 0.0,
        "full_us_per_call": 0.0,
        "marginal_copy_us_per_call": 0.0,
        "marginal_entry_us_per_call": 0.0,
        "marginal_transport_us_per_call": 0.0,
        "marginal_comm_norm_us_per_call": 0.0,
        "marginal_norm_us_per_call": 0.0,
        "marginal_exit_us_per_call": 0.0,
        "marginal_lifetime_us_per_call": 0.0,
        "marginal_sum_us_per_call": 0.0,
        "notes": "",
    }
    row.update(values)
    return row


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

    from tokenspeed_kernel.ops.communication import _triton_shmem_kernels as tk
    from tokenspeed_kernel.ops.communication import iris as iris_mod
    from tokenspeed_kernel.ops.communication import triton as tri
    from tokenspeed_kernel.ops.communication import triton_shmem as ts
    from tokenspeed_kernel.ops.layernorm import triton as layernorm_mod

    iris_mod._get_or_create_iris_context(1 << 29)
    ordinary_state = tri.create_state(
        group=group,
        rank_in_group=rank,
        device=device,
        max_numel=512 * 1024 // torch.bfloat16.itemsize,
    )
    iris_state = iris_mod.create_iris_ar_rmsnorm_state(
        group=group,
        rank_in_group=rank,
        max_token_num=_M,
        hidden_dim=_N,
        dtype=torch.bfloat16,
        heap_size=1 << 29,
        device=device,
        persistent=False,
    )
    triton_state = ts.create_triton_shmem_ar_rmsnorm_state(
        group=group,
        rank_in_group=rank,
        max_token_num=_M,
        hidden_dim=_N,
        dtype=torch.bfloat16,
        device=device,
    )
    if triton_state is None:
        raise RuntimeError("triton_shmem state creation failed")
    if triton_state._input_site_ring_size < _CALLS:
        raise RuntimeError(
            "input site ring must cover BENCH_CALLS_PER_GRAPH: "
            f"{triton_state._input_site_ring_size} < {_CALLS}"
        )

    weight = torch.linspace(0.5, 1.5, _N, dtype=torch.bfloat16, device=device)
    residual = (
        torch.arange(_M * _N, dtype=torch.float32, device=device)
        .reshape(_M, _N)
        .mul_(0.001)
        .to(torch.bfloat16)
    )
    sources = [
        torch.full(
            (_M, _N),
            rank + 1 + (index % 3) * 0.125,
            dtype=torch.bfloat16,
            device=device,
        )
        for index in range(_CALLS)
    ]
    rows = []

    # Production-unfused graph. The scratch copy is a benchmark reset, not a
    # serving AR stage, but is isolated so it can be removed from interpretation.
    unfused_scratch = [torch.empty_like(sources[0]) for _ in range(_CALLS)]
    unfused_norm = [torch.empty_like(sources[0]) for _ in range(_CALLS)]
    unfused_residual = [torch.empty_like(sources[0]) for _ in range(_CALLS)]

    def unfused_copy():
        for index in range(_CALLS):
            unfused_scratch[index].copy_(sources[index])

    def unfused_copy_transport():
        for index in range(_CALLS):
            unfused_scratch[index].copy_(sources[index])
            tri.all_reduce(ordinary_state, unfused_scratch[index])

    def unfused_full():
        for index in range(_CALLS):
            unfused_scratch[index].copy_(sources[index])
            tri.all_reduce(ordinary_state, unfused_scratch[index])
            eager._launch_rmsnorm_prealloc(
                layernorm_mod,
                unfused_scratch[index],
                residual,
                weight,
                unfused_norm[index],
                unfused_residual[index],
            )

    unfused_copy_us = _time_graph_us(_capture(unfused_copy, group), group)
    unfused_transport_us = _time_graph_us(
        _capture(unfused_copy_transport, group),
        group,
    )
    unfused_full_us = _time_graph_us(_capture(unfused_full, group), group)
    rows.append(
        _make_row(
            "production_unfused",
            "ordinary_iris+triton_rmsnorm",
            copy_prefix_us_per_call=unfused_copy_us,
            entry_prefix_us_per_call=unfused_transport_us,
            full_us_per_call=unfused_full_us,
            marginal_copy_us_per_call=unfused_copy_us,
            marginal_transport_us_per_call=(
                unfused_transport_us - unfused_copy_us
            ),
            marginal_norm_us_per_call=(
                unfused_full_us - unfused_transport_us
            ),
            marginal_sum_us_per_call=unfused_full_us,
            notes=(
                "copy is harness reset; production producer writes transport "
                "input directly; Iris transport internals are opaque"
            ),
        )
    )

    # Iris fused cumulative stage graphs.
    iris_norm = [torch.empty_like(sources[0]) for _ in range(_CALLS)]
    iris_residual = [torch.empty_like(sources[0]) for _ in range(_CALLS)]
    iris_input = iris_state._input_buf[:_M]

    def iris_copy():
        for index in range(_CALLS):
            iris_input.copy_(sources[index])

    def iris_entry():
        for index in range(_CALLS):
            iris_input.copy_(sources[index])
            iris_state._ctx.device_barrier()

    def iris_kernel():
        for index in range(_CALLS):
            iris_input.copy_(sources[index])
            iris_state._ctx.device_barrier()
            eager._launch_iris_fused_kernel(
                iris_mod,
                iris_state,
                iris_input,
                residual,
                weight,
                iris_norm[index],
                iris_residual[index],
            )

    def iris_full():
        for index in range(_CALLS):
            iris_state.fused(
                sources[index],
                residual,
                weight,
                eager._EPS,
                norm_out=iris_norm[index],
                residual_out=iris_residual[index],
            )

    iris_copy_us = _time_graph_us(_capture(iris_copy, group), group)
    iris_entry_us = _time_graph_us(_capture(iris_entry, group), group)
    iris_kernel_us = _time_graph_us(_capture(iris_kernel, group), group)
    iris_full_us = _time_graph_us(_capture(iris_full, group), group)
    rows.append(
        _make_row(
            "iris_fused",
            "iris_fused_nonpersistent",
            copy_prefix_us_per_call=iris_copy_us,
            entry_prefix_us_per_call=iris_entry_us,
            kernel_prefix_us_per_call=iris_kernel_us,
            full_us_per_call=iris_full_us,
            marginal_copy_us_per_call=iris_copy_us,
            marginal_entry_us_per_call=iris_entry_us - iris_copy_us,
            marginal_comm_norm_us_per_call=(
                iris_kernel_us - iris_entry_us
            ),
            marginal_exit_us_per_call=iris_full_us - iris_kernel_us,
            marginal_sum_us_per_call=iris_full_us,
            notes=(
                "copy, device barriers, and fused peer-pull+residual+RMSNorm "
                "are cumulative captured stages"
            ),
        )
    )

    # Realigned triton_shmem cumulative site-ring graphs.
    triton_norm = [torch.empty_like(sources[0]) for _ in range(_CALLS)]
    triton_residual = [torch.empty_like(sources[0]) for _ in range(_CALLS)]
    site_views = [
        triton_state._input_site_tensor[
            index
            * triton_state._input_site_max_m : index
            * triton_state._input_site_max_m
            + _M
        ]
        for index in range(_CALLS)
    ]

    def triton_copy():
        for index in range(_CALLS):
            site_views[index].copy_(sources[index])

    def triton_core():
        for index in range(_CALLS):
            site_views[index].copy_(sources[index])
            eager._launch_triton_oneshot_core(
                tk,
                triton_state,
                site_views[index],
                triton_state._input_site_bases,
                residual,
                weight,
                triton_norm[index],
                triton_residual[index],
                _M,
                _N,
            )

    def triton_entry_kernel():
        for index in range(_CALLS):
            site_views[index].copy_(sources[index])
            triton_state._run_oneshot(
                site_views[index],
                triton_state._input_site_bases,
                sources[index],
                residual,
                weight,
                eager._EPS,
                _M,
                _N,
                ws,
                triton_norm[index],
                triton_residual[index],
                False,
                False,
            )

    def triton_full():
        for index in range(_CALLS):
            triton_state.fused(
                sources[index],
                residual,
                weight,
                eager._EPS,
                norm_out=triton_norm[index],
                residual_out=triton_residual[index],
            )

    triton_copy_us = _time_graph_us(_capture(triton_copy, group), group)
    triton_core_us = _time_graph_us(_capture(triton_core, group), group)
    triton_entry_us = _time_graph_us(
        _capture(triton_entry_kernel, group),
        group,
    )
    triton_full_us = _time_graph_us(_capture(triton_full, group), group)
    rows.append(
        _make_row(
            "triton_shmem_realigned",
            f"{triton_state._oneshot_kernel_for_m(_M)}_site_ring",
            copy_prefix_us_per_call=triton_copy_us,
            entry_prefix_us_per_call=triton_core_us,
            kernel_prefix_us_per_call=triton_entry_us,
            full_us_per_call=triton_full_us,
            marginal_copy_us_per_call=triton_copy_us,
            marginal_comm_norm_us_per_call=(
                triton_core_us - triton_copy_us
            ),
            marginal_entry_us_per_call=(
                triton_entry_us - triton_core_us
            ),
            marginal_lifetime_us_per_call=(
                triton_full_us - triton_entry_us
            ),
            marginal_sum_us_per_call=triton_full_us,
            notes=(
                "entry sync is in-kernel; exit sync omitted by site ring; "
                "lifetime remainder is public-wrapper vs explicit site calls"
            ),
        )
    )

    # Correctness after one full replay for representative final outputs.
    _capture(unfused_full, group).replay()
    _capture(iris_full, group).replay()
    _capture(triton_full, group).replay()
    torch.cuda.synchronize()
    ref = sources[-1].float()
    dist.all_reduce(ref, group=group)
    ref_residual = ref + residual.float()
    ref_norm = ref_residual * torch.rsqrt(
        ref_residual.square().mean(-1, keepdim=True) + eager._EPS
    )
    ref_norm *= weight.float()
    eager._assert_outputs(
        unfused_norm[-1],
        unfused_residual[-1],
        ref_norm,
        ref_residual,
    )
    eager._assert_outputs(
        iris_norm[-1],
        iris_residual[-1],
        ref_norm,
        ref_residual,
    )
    eager._assert_outputs(
        triton_norm[-1],
        triton_residual[-1],
        ref_norm,
        ref_residual,
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
    ws = _WS
    manager = mp.Manager()
    out = manager.list()
    mp.spawn(_worker, args=(ws, _port(), out), nprocs=ws, join=True)
    rows = list(out)
    print(json.dumps(rows, indent=2, sort_keys=True))

    csv_path = os.environ.get("BENCH_CSV")
    if csv_path:
        _write_csv(csv_path, rows)
        print(f"wrote {csv_path}")
    json_path = os.environ.get("BENCH_JSON")
    if json_path:
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
