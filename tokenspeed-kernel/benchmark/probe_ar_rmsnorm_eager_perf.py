"""Measure one AR+RMSNorm arm across an eager M sweep."""

from __future__ import annotations

import json
import os
import socket
import statistics
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from benchmark.probe_ar_rmsnorm_graph_perf import (
    _ORDINARY_AR_MAX_BYTES,
    _check_outputs,
    _resolve_impl,
    _set_inputs,
    _stats,
)
from benchmark.shape_axes import default_hidden_size


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def _env_ints(name: str, default: list[int]) -> list[int]:
    raw = os.environ.get(name)
    return [int(value) for value in raw.split(",")] if raw else default


def _port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _worker(rank: int, ws: int, port: int, out) -> None:
    requested_impl = os.environ.get("BENCH_IMPL", "auto")
    impl, block_n_override = _resolve_impl(requested_impl)
    if block_n_override is not None:
        os.environ["TS_TRITON_SHMEM_ONESHOT_BLOCK_N"] = str(block_n_override)
    if impl in {"auto", "iris", "symm_mem", "triton_shmem"}:
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
    n = default_hidden_size()
    m_values = _env_ints("BENCH_M_VALUES", [1, 32, 64, 65, 91, 92])
    max_token_num = _env_int("BENCH_MAX_TOKEN_NUM", max(m_values))
    warmup = _env_int("BENCH_N_WARMUP", 30)
    repeat = _env_int("BENCH_N_REPEAT", 150)
    eps = 1e-6

    from tokenspeed_kernel.ops.communication import triton as tri
    from tokenspeed_kernel.ops.layernorm.triton import rmsnorm as triton_rmsnorm

    rows = []
    for case_index, m in enumerate(m_values):
        x = torch.empty((m, n), dtype=torch.bfloat16, device=device)
        xs = [x]
        residual = (
            torch.arange(m * n, dtype=torch.float32, device=device)
            .reshape(m, n)
            .mul_(0.001)
            .to(torch.bfloat16)
        )
        weight = torch.linspace(0.5, 1.5, n, dtype=torch.bfloat16, device=device)
        epoch_stride = ws + 2

        ordinary_state = None
        ordinary_uses_iris = False
        scratch = torch.empty_like(x)
        if (
            impl in ("production_unfused", "triton_shmem")
            and x.numel() * x.element_size() <= _ORDINARY_AR_MAX_BYTES
        ):
            ordinary_state = tri.create_state(
                group=group,
                rank_in_group=rank,
                device=device,
                max_numel=_ORDINARY_AR_MAX_BYTES // x.element_size(),
            )
            ordinary_uses_iris = tri.all_reduce_can_run(ordinary_state, x)

        def reset_ordinary(scratch_tensor=scratch, source=x) -> None:
            scratch_tensor.copy_(source)

        def launch_ordinary(
            use_iris=ordinary_uses_iris,
            state=ordinary_state,
            scratch_tensor=scratch,
            norm_weight=weight,
            norm_residual=residual,
        ):
            if use_iris:
                tri.all_reduce(state, scratch_tensor)
            else:
                dist.all_reduce(scratch_tensor, group=group)
            result = triton_rmsnorm(
                scratch_tensor,
                norm_weight,
                eps,
                residual=norm_residual,
            )
            if not isinstance(result, tuple):
                raise TypeError("residual RMSNorm did not return both outputs")
            return [result]

        fallback_used = False

        def launch_fused(
            input_tensor=x,
            norm_residual=residual,
            norm_weight=weight,
            ordinary_launch=launch_ordinary,
            ordinary_reset=reset_ordinary,
        ):
            nonlocal fallback_used
            norm_out, residual_out, _, _ = tri.allreduce_residual_rmsnorm(
                input_tensor=input_tensor,
                residual=norm_residual,
                weight=norm_weight,
                rank=rank,
                group=group,
                eps=eps,
                max_token_num=max_token_num,
            )
            if norm_out is None or residual_out is None:
                if impl != "triton_shmem":
                    raise RuntimeError(
                        f"production fused dispatcher declined BENCH_IMPL={impl}"
                    )
                fallback_used = True
                ordinary_reset()
                return ordinary_launch()
            return [(norm_out, residual_out)]

        launch = launch_ordinary if impl == "production_unfused" else launch_fused
        _set_inputs(xs, rank, epoch=0, epoch_stride=epoch_stride)
        if impl == "production_unfused":
            reset_ordinary()
        outputs = launch()
        if fallback_used:
            launch = launch_ordinary
        torch.cuda.synchronize()
        _check_outputs(
            outputs,
            expected_calls=1,
            residual=residual,
            weight=weight,
            world_size=ws,
            epoch=0,
            epoch_stride=epoch_stride,
            eps=eps,
        )

        for warm_index in range(warmup):
            _set_inputs(
                xs,
                rank,
                epoch=1 + warm_index % 2,
                epoch_stride=epoch_stride,
            )
            if impl == "production_unfused" or fallback_used:
                reset_ordinary()
            outputs = launch()
        torch.cuda.synchronize()
        dist.barrier(group=group)

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
        for index, (start, end) in enumerate(zip(starts, ends)):
            epoch = 3 + index % 2
            _set_inputs(xs, rank, epoch=epoch, epoch_stride=epoch_stride)
            if impl == "production_unfused" or fallback_used:
                reset_ordinary()
                torch.cuda.synchronize()
            start.record()
            outputs = launch()
            end.record()
        torch.cuda.synchronize()
        final_epoch = 3 + (repeat - 1) % 2
        _check_outputs(
            outputs,
            expected_calls=1,
            residual=residual,
            weight=weight,
            world_size=ws,
            epoch=final_epoch,
            epoch_stride=epoch_stride,
            eps=eps,
        )

        rank_samples_us = [
            start.elapsed_time(end) * 1000 for start, end in zip(starts, ends)
        ]
        all_rank_samples_us = [None] * ws
        dist.all_gather_object(all_rank_samples_us, rank_samples_us, group=group)

        if impl == "production_unfused" or fallback_used:
            expected_backend = "iris" if ordinary_uses_iris else "rccl"
            expected_path = (
                "ordinary_iris_all_reduce+triton_residual_rmsnorm"
                if ordinary_uses_iris
                else "rccl_all_reduce+triton_residual_rmsnorm"
            )
        elif impl in ("auto", "iris"):
            expected_backend = "iris"
            expected_path = "fused_iris_allreduce_residual_rmsnorm"
        elif impl == "symm_mem":
            expected_backend = "symm_mem"
            expected_path = "fused_native_symm_mem"
        else:
            from tokenspeed_kernel.ops.communication import triton_shmem as ts

            state_key = ts.triton_shmem_state_cache_key(
                group,
                max_token_num,
                n,
                torch.bfloat16,
            )
            state = ts.TRITON_SHMEM_AR_RMSNORM_STATES.get(state_key)
            if state is None:
                raise RuntimeError("triton_shmem dispatcher state was not precreated")
            uses_oneshot = (not state._is_twoshot) or (
                state._oneshot_max_m > 0 and m <= state._oneshot_max_m
            )
            expected_backend = "triton_shmem"
            expected_path = (
                state._oneshot_kernel_for_m(m) if uses_oneshot else "twoshot_blocked"
            )

        identities = [None] * ws
        dist.all_gather_object(
            identities,
            {
                "rank": rank,
                "expected_backend": expected_backend,
                "expected_path": expected_path,
            },
            group=group,
        )
        if rank == 0:
            signatures = {
                (identity["expected_backend"], identity["expected_path"])
                for identity in identities
            }
            if len(signatures) != 1:
                raise RuntimeError(f"path identity differs across ranks: {identities}")
            max_rank_samples_us = [
                max(samples[index] for samples in all_rank_samples_us)
                for index in range(repeat)
            ]
            rows.append(
                {
                    "M": m,
                    "expected_backend": expected_backend,
                    "expected_path": expected_path,
                    "dispatcher_declined": fallback_used,
                    "max_rank_samples_stats_us": _stats(max_rank_samples_us),
                    "max_rank_samples_us": max_rank_samples_us,
                    "rank_medians_us": [
                        statistics.median(samples) for samples in all_rank_samples_us
                    ],
                    "rank_path_identities": identities,
                }
            )
        dist.barrier(group=group)

    if rank == 0:
        out.append(
            {
                "schema_version": 1,
                "mode": "eager",
                "impl": requested_impl,
                "resolved_impl": impl,
                "world_size": ws,
                "N": n,
                "max_token_num": max_token_num,
                "warmup": warmup,
                "repeat": repeat,
                "M_values": m_values,
                "rows": rows,
            }
        )
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
