"""Measure the profiled AR+RMSNorm shape under HIP graph replay.

Examples:
    HIP_VISIBLE_DEVICES=1,2,3,5 BENCH_IMPL=generic \
      python3 -m benchmark.probe_ar_rmsnorm_graph_perf
    HIP_VISIBLE_DEVICES=1,2,3,5 BENCH_IMPL=block2048 \
      python3 -m benchmark.probe_ar_rmsnorm_graph_perf
    HIP_VISIBLE_DEVICES=1,2,3,5 BENCH_IMPL=unfused \
      python3 -m benchmark.probe_ar_rmsnorm_graph_perf
"""
from __future__ import annotations

import os
import socket
import statistics

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from benchmark.shape_axes import default_hidden_size


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def _port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _worker(rank: int, ws: int, port: int, out) -> None:
    impl = os.environ.get("BENCH_IMPL", "auto")
    if impl.startswith("block") and impl[5:].isdigit():
        os.environ["TS_TRITON_SHMEM_ONESHOT_BLOCK_N"] = impl[5:]
    elif impl == "generic":
        os.environ["TS_TRITON_SHMEM_ONESHOT_BLOCK_N"] = "512"
    else:
        os.environ.pop("TS_TRITON_SHMEM_ONESHOT_BLOCK_N", None)

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=ws,
    )
    group = dist.group.WORLD
    device = torch.device(f"cuda:{rank}")
    m = _env_int("BENCH_M", 32)
    n = default_hidden_size()
    warmup = _env_int("BENCH_N_WARMUP", 50)
    repeat = _env_int("BENCH_N_REPEAT", 300)
    eps = 1e-6

    from tokenspeed_kernel.ops.communication import triton as tri
    from tokenspeed_kernel.ops.communication import triton_shmem as ts
    from tokenspeed_kernel.ops.layernorm.triton import rmsnorm as triton_rmsnorm

    x = torch.full((m, n), rank + 1, dtype=torch.bfloat16, device=device)
    residual = (
        torch.arange(m * n, dtype=torch.float32, device=device)
        .reshape(m, n)
        .mul_(0.001)
        .to(torch.bfloat16)
    )
    weight = torch.linspace(0.5, 1.5, n, dtype=torch.bfloat16, device=device)

    if impl == "unfused":
        state = tri.create_state(
            group=group,
            rank_in_group=rank,
            device=device,
            max_numel=512 * 1024 // x.element_size(),
        )

        def launch():
            tri.all_reduce(state, x)
            return triton_rmsnorm(x, weight, eps, residual=residual)

    else:
        state = ts.create_triton_shmem_ar_rmsnorm_state(
            group=group,
            rank_in_group=rank,
            max_token_num=m,
            hidden_dim=n,
            dtype=torch.bfloat16,
        )
        if state is None:
            raise RuntimeError("triton_shmem state creation failed")
        norm_out = torch.empty_like(x)
        residual_out = torch.empty_like(x)

        def launch():
            return ts.triton_shmem_allreduce_residual_rmsnorm(
                state,
                input_tensor=x,
                residual=residual,
                weight=weight,
                eps=eps,
                norm_out=norm_out,
                residual_out=residual_out,
            )

    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for _ in range(warmup):
            launch()
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()
    dist.barrier(group=group)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        launch()
    dist.barrier(group=group)

    if impl != "unfused":
        x.fill_(rank + 1)
        graph.replay()
        torch.cuda.synchronize()
        reference_residual = (
            torch.full_like(residual, ws * (ws + 1) // 2, dtype=torch.float32)
            + residual.float()
        )
        reference_norm = reference_residual * torch.rsqrt(
            reference_residual.pow(2).mean(dim=-1, keepdim=True) + eps
        )
        reference_norm *= weight.float()
        torch.testing.assert_close(
            residual_out.float(), reference_residual, atol=2e-2, rtol=2e-2
        )
        torch.testing.assert_close(
            norm_out.float(), reference_norm, atol=2e-2, rtol=2e-2
        )
        dist.barrier(group=group)

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    for idx in range(repeat):
        starts[idx].record()
        graph.replay()
        ends[idx].record()
    torch.cuda.synchronize()
    rank_median = statistics.median(
        start.elapsed_time(end) for start, end in zip(starts, ends)
    )
    medians = [None] * ws
    dist.all_gather_object(medians, rank_median, group=group)
    if rank == 0:
        out.append(
            {
                "impl": impl,
                "world_size": ws,
                "M": m,
                "N": n,
                "max_rank_median_us": max(medians) * 1000,
                "rank_medians_us": [value * 1000 for value in medians],
            }
        )
    dist.destroy_process_group()


def main() -> None:
    ws = _env_int("BENCH_WS", 4)
    manager = mp.Manager()
    out = manager.list()
    mp.spawn(_worker, args=(ws, _port(), out), nprocs=ws, join=True)
    print(dict(out[0]))


if __name__ == "__main__":
    main()
