"""Measure post-rebase AR+RMSNorm implementations under HIP graph replay.

``BENCH_IMPL`` accepts ``production_unfused``, ``auto``, ``iris``,
``symm_mem``, and ``triton_shmem``. Legacy ``unfused`` is an alias for
``production_unfused``; ``generic`` and ``block<N>`` select ``triton_shmem``
with the historical block override.

Examples:
    HIP_VISIBLE_DEVICES=1,2,3,5 BENCH_IMPL=auto \
      python3 -m benchmark.probe_ar_rmsnorm_graph_perf
    HIP_VISIBLE_DEVICES=1,2,3,5 BENCH_IMPL=block2048 \
      python3 -m benchmark.probe_ar_rmsnorm_graph_perf
    HIP_VISIBLE_DEVICES=1,2,3,5 BENCH_IMPL=production_unfused \
      python3 -m benchmark.probe_ar_rmsnorm_graph_perf
"""
from __future__ import annotations

import json
import os
import socket
import statistics
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from benchmark.shape_axes import default_hidden_size


_ORDINARY_AR_MAX_BYTES = 512 * 1024
_FUSED_IMPLS = {"auto", "iris", "symm_mem", "triton_shmem"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "min_us": min(values),
        "p50_us": statistics.median(values),
        "p95_us": _percentile(values, 0.95),
        "p99_us": _percentile(values, 0.99),
        "max_us": max(values),
        "mean_us": statistics.fmean(values),
    }


def _port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _resolve_impl(requested: str) -> tuple[str, int | None]:
    requested = requested.strip().lower()
    if requested == "unfused":
        return "production_unfused", None
    if requested == "generic":
        return "triton_shmem", 512
    if requested.startswith("block") and requested[5:].isdigit():
        block_n = int(requested[5:])
        if block_n <= 0:
            raise ValueError("block override must be positive")
        return "triton_shmem", block_n
    if requested == "production_unfused" or requested in _FUSED_IMPLS:
        return requested, None
    supported = sorted(_FUSED_IMPLS | {"production_unfused"})
    raise ValueError(
        f"unsupported BENCH_IMPL={requested!r}; expected one of {supported} "
        "or legacy unfused/generic/block<N>"
    )


def _set_inputs(
    xs: list[torch.Tensor],
    rank: int,
    epoch: int,
    epoch_stride: int,
) -> None:
    for call, x in enumerate(xs):
        x.fill_(rank + call + 1 + epoch * epoch_stride)


def _check_outputs(
    outputs: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    expected_calls: int,
    residual: torch.Tensor,
    weight: torch.Tensor,
    world_size: int,
    epoch: int,
    epoch_stride: int,
    eps: float,
) -> None:
    if len(outputs) != expected_calls:
        raise AssertionError(
            f"output count {len(outputs)} != calls per graph {expected_calls}"
        )
    for call, (norm_out, residual_out) in enumerate(outputs):
        rank_sum = (
            world_size * (world_size + 1) // 2
            + call * world_size
            + epoch * epoch_stride * world_size
        )
        reference_residual = (
            torch.full_like(residual, rank_sum, dtype=torch.float32)
            + residual.float()
        )
        reference_norm = reference_residual * torch.rsqrt(
            reference_residual.pow(2).mean(dim=-1, keepdim=True) + eps
        )
        reference_norm *= weight.float()
        torch.testing.assert_close(
            residual_out.float(),
            reference_residual,
            atol=2e-2,
            rtol=2e-2,
        )
        torch.testing.assert_close(
            norm_out.float(),
            reference_norm,
            atol=2e-2,
            rtol=2e-2,
        )


def _worker(rank: int, ws: int, port: int, out) -> None:
    requested_impl = os.environ.get("BENCH_IMPL", "auto")
    impl, block_n_override = _resolve_impl(requested_impl)
    if block_n_override is not None:
        os.environ["TS_TRITON_SHMEM_ONESHOT_BLOCK_N"] = str(block_n_override)
    if impl in _FUSED_IMPLS:
        os.environ["TS_ARNORM_BACKEND"] = impl
    else:
        os.environ.pop("TS_ARNORM_BACKEND", None)

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
    calls_per_graph = _env_int("BENCH_CALLS_PER_GRAPH", 1)
    if calls_per_graph < 1:
        raise ValueError("BENCH_CALLS_PER_GRAPH must be positive")
    warmup = _env_int("BENCH_N_WARMUP", 50)
    repeat = _env_int("BENCH_N_REPEAT", 1000)
    if repeat < 1000:
        raise ValueError("BENCH_N_REPEAT must be at least 1000 for graph screening")
    double_buffer_input = (
        os.environ.get("TS_TRITON_SHMEM_DOUBLE_BUFFER_INPUT", "0")
        not in ("0", "false", "False")
    )
    eps = 1e-6
    epoch_stride = ws + calls_per_graph + 1

    from tokenspeed_kernel.ops.communication import triton as tri
    from tokenspeed_kernel.ops.layernorm.triton import rmsnorm as triton_rmsnorm

    xs = [
        torch.empty((m, n), dtype=torch.bfloat16, device=device)
        for _ in range(calls_per_graph)
    ]
    _set_inputs(xs, rank, epoch=0, epoch_stride=epoch_stride)
    residual = (
        torch.arange(m * n, dtype=torch.float32, device=device)
        .reshape(m, n)
        .mul_(0.001)
        .to(torch.bfloat16)
    )
    weight = torch.linspace(0.5, 1.5, n, dtype=torch.bfloat16, device=device)

    ordinary_state = None
    ordinary_uses_iris = False
    scratches: list[torch.Tensor] = []
    if impl == "production_unfused":
        if xs[0].numel() * xs[0].element_size() <= _ORDINARY_AR_MAX_BYTES:
            ordinary_state = tri.create_state(
                group=group,
                rank_in_group=rank,
                device=device,
                max_numel=_ORDINARY_AR_MAX_BYTES // xs[0].element_size(),
            )
            ordinary_uses_iris = tri.all_reduce_can_run(ordinary_state, xs[0])
        scratches = [torch.empty_like(x) for x in xs]

        def launch():
            outputs = []
            for x, scratch in zip(xs, scratches):
                scratch.copy_(x)
                if ordinary_uses_iris:
                    tri.all_reduce(ordinary_state, scratch)
                else:
                    dist.all_reduce(scratch, group=group)
                result = triton_rmsnorm(
                    scratch,
                    weight,
                    eps,
                    residual=residual,
                )
                if not isinstance(result, tuple):
                    raise RuntimeError("residual RMSNorm did not return both outputs")
                outputs.append(result)
            return outputs

    else:
        def launch():
            outputs = []
            for x in xs:
                norm_out, residual_out, _, _ = tri.allreduce_residual_rmsnorm(
                    input_tensor=x,
                    residual=residual,
                    weight=weight,
                    rank=rank,
                    group=group,
                    eps=eps,
                    max_token_num=m,
                )
                if norm_out is None or residual_out is None:
                    raise RuntimeError(
                        f"production fused dispatcher declined BENCH_IMPL={impl}"
                    )
                outputs.append((norm_out, residual_out))
            return outputs

    # Force every lazy allocation, symmetric-memory rendezvous, Iris context,
    # or RCCL communicator setup to happen before stream warmup and capture.
    eager_outputs = launch()
    torch.cuda.synchronize()
    dist.barrier(group=group)
    _check_outputs(
        eager_outputs,
        expected_calls=calls_per_graph,
        residual=residual,
        weight=weight,
        world_size=ws,
        epoch=0,
        epoch_stride=epoch_stride,
        eps=eps,
    )
    dist.barrier(group=group)

    if impl == "production_unfused":
        expected_backend = "iris" if ordinary_uses_iris else "rccl"
        expected_path = (
            "ordinary_iris_all_reduce+triton_residual_rmsnorm"
            if ordinary_uses_iris
            else "rccl_all_reduce+triton_residual_rmsnorm"
        )
        path_details = {}
    elif impl in ("auto", "iris"):
        expected_backend = "iris"
        expected_path = "fused_iris_allreduce_residual_rmsnorm"
        path_details = {}
    elif impl == "symm_mem":
        expected_backend = "symm_mem"
        expected_path = "fused_native_symm_mem"
        path_details = {}
    else:
        # Keep all triton_shmem-specific state inspection inside its explicit
        # arm; Iris and native production paths must not depend on local state.
        from tokenspeed_kernel.ops.communication import triton_shmem as ts

        state_key = ts.triton_shmem_state_cache_key(
            group,
            m,
            n,
            torch.bfloat16,
        )
        triton_shmem_state = ts.TRITON_SHMEM_AR_RMSNORM_STATES.get(state_key)
        if triton_shmem_state is None:
            raise RuntimeError("triton_shmem dispatcher state was not precreated")
        uses_oneshot = (not triton_shmem_state._is_twoshot) or (
            triton_shmem_state._oneshot_max_m > 0
            and m <= triton_shmem_state._oneshot_max_m
        )
        expected_backend = "triton_shmem"
        expected_path = (
            triton_shmem_state._oneshot_kernel_for_m(m)
            if uses_oneshot
            else "twoshot_blocked"
        )
        path_details = {
            "coarse": bool(triton_shmem_state._coarse),
            "double_buffer_input": bool(
                triton_shmem_state._double_buffer_input
            ),
            "input_ring_size": len(triton_shmem_state._input_ring),
            "oneshot_block_n": triton_shmem_state._oneshot_block_n,
        }

    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for _ in range(warmup):
            warmup_outputs = launch()
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()
    dist.barrier(group=group)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        captured_outputs = launch()
    dist.barrier(group=group)

    # A value set only after capture proves replay consumes changing source
    # tensors rather than values accidentally frozen during capture.
    _set_inputs(xs, rank, epoch=1, epoch_stride=epoch_stride)
    graph.replay()
    torch.cuda.synchronize()
    _check_outputs(
        captured_outputs,
        expected_calls=calls_per_graph,
        residual=residual,
        weight=weight,
        world_size=ws,
        epoch=1,
        epoch_stride=epoch_stride,
        eps=eps,
    )
    dist.barrier(group=group)

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    for idx in range(repeat):
        epoch = 2 + idx % 2
        _set_inputs(xs, rank, epoch=epoch, epoch_stride=epoch_stride)
        starts[idx].record()
        graph.replay()
        ends[idx].record()
    torch.cuda.synchronize()
    final_epoch = 2 + (repeat - 1) % 2
    _check_outputs(
        captured_outputs,
        expected_calls=calls_per_graph,
        residual=residual,
        weight=weight,
        world_size=ws,
        epoch=final_epoch,
        epoch_stride=epoch_stride,
        eps=eps,
    )
    dist.barrier(group=group)

    rank_samples_us = [
        start.elapsed_time(end) * 1000
        for start, end in zip(starts, ends)
    ]
    all_rank_samples_us = [None] * ws
    dist.all_gather_object(all_rank_samples_us, rank_samples_us, group=group)
    rank_path_identity = {
        "rank": rank,
        "expected_backend": expected_backend,
        "expected_path": expected_path,
        "path_details": path_details,
    }
    rank_path_identities = [None] * ws
    dist.all_gather_object(
        rank_path_identities,
        rank_path_identity,
        group=group,
    )
    if rank == 0:
        if any(len(samples) != repeat for samples in all_rank_samples_us):
            raise RuntimeError("rank timing sample counts do not match")
        path_signatures = {
            json.dumps(
                {
                    "expected_backend": identity["expected_backend"],
                    "expected_path": identity["expected_path"],
                    "path_details": identity["path_details"],
                },
                sort_keys=True,
            )
            for identity in rank_path_identities
        }
        if len(path_signatures) != 1:
            raise RuntimeError(
                f"backend/path identity differs across ranks: "
                f"{rank_path_identities}"
            )
        max_rank_samples_us = [
            max(samples[idx] for samples in all_rank_samples_us)
            for idx in range(repeat)
        ]
        rank_medians_us = [
            statistics.median(samples) for samples in all_rank_samples_us
        ]
        max_rank_stats = _stats(max_rank_samples_us)
        result = {
            "impl": requested_impl,
            "resolved_impl": impl,
            "expected_backend": expected_backend,
            "expected_path": expected_path,
            "path_details": path_details,
            "world_size": ws,
            "M": m,
            "N": n,
            "payload_bytes": xs[0].numel() * xs[0].element_size(),
            "ordinary_all_reduce_max_bytes": _ORDINARY_AR_MAX_BYTES,
            "calls_per_graph": calls_per_graph,
            "calls_per_graph_parity": (
                "even" if calls_per_graph % 2 == 0 else "odd"
            ),
            "double_buffer_input": double_buffer_input,
            "warmup": warmup,
            "repeat": repeat,
            "changing_inputs": True,
            "block_n_override": block_n_override,
            "max_rank_samples_stats_us": max_rank_stats,
            "max_rank_samples_per_call_stats_us": {
                key: value / calls_per_graph
                for key, value in max_rank_stats.items()
            },
            "max_rank_samples_us": max_rank_samples_us,
            "rank_samples_us": all_rank_samples_us,
            "rank_path_identities": rank_path_identities,
            # Compatibility fields retained for existing JSON consumers. New
            # comparisons should use max_rank_samples_stats_us, which takes the
            # max rank per iteration before computing percentiles.
            "max_rank_median_us": max(rank_medians_us),
            "max_rank_median_per_call_us": (
                max(rank_medians_us) / calls_per_graph
            ),
            "rank_medians_us": rank_medians_us,
        }
        if impl == "triton_shmem":
            result["signal_zero_status"] = "not_exposed_by_safe_public_api"
        out.append(result)
    dist.destroy_process_group()


def main() -> None:
    ws = _env_int("BENCH_WS", 4)
    manager = mp.Manager()
    out = manager.list()
    mp.spawn(_worker, args=(ws, _port(), out), nprocs=ws, join=True)
    result = dict(out[0])
    output = os.environ.get("BENCH_JSON")
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
