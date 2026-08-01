"""Granular eager decomposition for the three GPT-OSS AR+RMSNorm stacks.

The probe compares production-unfused, fused Iris, and the active realigned
triton_shmem profile. GPU stages use events and report per-iteration max-rank
p50 latency. Host output-allocation time is reported separately and is not
additive with GPU-event totals.

Examples:
    HIP_VISIBLE_DEVICES=1,2,5,6 BENCH_WS=4 BENCH_N=2880 \
      BENCH_M_VALUES=32,91,92,256,384,512,1024,2048 \
      BENCH_CSV=/tmp/backend-decomp.csv \
      python -m benchmark.probe_ar_rmsnorm_backend_decomp
"""

from __future__ import annotations

import csv
import os
import socket
import statistics
import time
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


_N = _env_int("BENCH_N", default_hidden_size())
_MS = _env_ints(
    "BENCH_M_VALUES",
    [32, 91, 92, 256, 384, 512, 1024, 2048],
)
_WARM = _env_int("BENCH_N_WARMUP", 30)
_REP = _env_int("BENCH_N_REPEAT", 150)
_HOST_REP = _env_int("BENCH_HOST_REPEAT", 1000)


def _port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _time_gpu(
    fn: Callable[[], object],
    group: dist.ProcessGroup,
    setup: Callable[[], object] | None = None,
) -> float:
    """Return milliseconds after per-iteration max-rank and then p50."""
    for _ in range(_WARM):
        if setup is not None:
            setup()
        fn()
    torch.cuda.synchronize()
    dist.barrier(group=group)

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(_REP)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(_REP)]
    for index in range(_REP):
        if setup is not None:
            setup()
        starts[index].record()
        fn()
        ends[index].record()
    torch.cuda.synchronize()

    samples = torch.tensor(
        [start.elapsed_time(end) for start, end in zip(starts, ends)],
        dtype=torch.float64,
        device="cuda",
    )
    dist.all_reduce(samples, op=dist.ReduceOp.MAX, group=group)
    return statistics.median(samples.cpu().tolist())


def _time_host_us(fn: Callable[[], object], group: dist.ProcessGroup) -> float:
    """Return max-rank median host dispatch/allocation time in microseconds."""
    for _ in range(50):
        fn()
    torch.cuda.synchronize()
    dist.barrier(group=group)
    samples = []
    for _ in range(_HOST_REP):
        begin = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - begin) / 1000.0)
    local = torch.tensor(
        [statistics.median(samples)],
        dtype=torch.float64,
        device="cuda",
    )
    dist.all_reduce(local, op=dist.ReduceOp.MAX, group=group)
    return float(local.item())


def _launch_rmsnorm_prealloc(
    layernorm_mod,
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    norm_out: torch.Tensor,
    residual_out: torch.Tensor,
) -> None:
    hidden = x.shape[-1]
    x_2d = x.view(-1, hidden)
    layernorm_mod._rmsnorm_kernel[(x_2d.shape[0],)](
        x_2d,
        residual,
        weight,
        norm_out.view(-1, hidden),
        residual_out,
        hidden,
        _EPS,
        BLOCK=layernorm_mod.triton.next_power_of_2(hidden),
        HAS_RESIDUAL=True,
    )


def _launch_iris_fused_kernel(
    iris_mod,
    state,
    in_view: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    norm_out: torch.Tensor,
    residual_out: torch.Tensor,
) -> None:
    m = in_view.shape[0]
    block_size = iris_mod.triton.next_power_of_2(state.hidden_dim)
    if state.persistent:
        kernel = iris_mod.iris_allreduce_residual_rmsnorm_kernel_persistent
        grid = (min(m, state._num_programs),)
    else:
        kernel = iris_mod.iris_allreduce_residual_rmsnorm_kernel
        grid = (m,)
    kernel[grid](
        in_view,
        residual,
        weight,
        norm_out,
        residual_out,
        m,
        state._ctx.get_heap_bases(),
        iris_rank=state._iris_rank,
        world_size=state.world_size,
        rank_start=state._rank_start,
        rank_stride=state._rank_stride,
        HIDDEN_SIZE=state.hidden_dim,
        BLOCK_SIZE=block_size,
        EPS=_EPS,
        num_warps=8,
    )


def _launch_triton_oneshot_core(
    kernels,
    state,
    x_view: torch.Tensor,
    input_bases: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    norm_out: torch.Tensor,
    residual_out: torch.Tensor,
    m: int,
    n: int,
) -> None:
    kernel_name = state._oneshot_kernel_for_m(m)
    grid_sms = state._grid_width(
        kernel_name,
        state.world_size,
        m,
        True,
    )
    grid = (grid_sms,)
    num_warps = state._oneshot_num_warps
    if kernel_name == "oneshot_wholerow":
        kernels.fused_ar_rmsnorm_oneshot_wholerow_kernel[grid](
            x_view,
            norm_out,
            _EPS,
            weight,
            state.my_pe,
            input_bases,
            residual,
            residual_out,
            norm_out,
            m,
            state._signal_pad,
            x_view,
            N=n,
            ws=state.world_size,
            NUM_SMS=grid_sms,
            HAS_RESIDUAL=True,
            HAS_ADD=False,
            RANK=state.my_pe,
            INKERNEL_BARRIER=False,
            FOLD_COPYIN=False,
            EXIT_BARRIER=False,
            WORKGROUP_SYNC=state._workgroup_sync,
            num_warps=num_warps,
        )
    elif kernel_name == "oneshot_wholerow_padded":
        kernels.fused_ar_rmsnorm_oneshot_wholerow_padded_kernel[grid](
            x_view,
            norm_out,
            _EPS,
            weight,
            state.my_pe,
            input_bases,
            residual,
            residual_out,
            norm_out,
            m,
            n,
            state._signal_pad,
            x_view,
            BLOCK_N=state._oneshot_padded_block_n,
            ws=state.world_size,
            NUM_SMS=grid_sms,
            HAS_RESIDUAL=True,
            HAS_ADD=False,
            RANK=state.my_pe,
            INKERNEL_BARRIER=False,
            FOLD_COPYIN=False,
            EXIT_BARRIER=False,
            WORKGROUP_SYNC=state._workgroup_sync,
            num_warps=num_warps,
        )
    else:
        kernels.fused_ar_rmsnorm_oneshot_blocked_kernel[grid](
            x_view,
            norm_out,
            state._oneshot_scratch,
            _EPS,
            weight,
            state.my_pe,
            input_bases,
            residual,
            residual_out,
            norm_out,
            m,
            n,
            state._signal_pad,
            x_view,
            BLOCK_N=state._oneshot_block_n,
            ws=state.world_size,
            NUM_SMS=grid_sms,
            HAS_RESIDUAL=True,
            HAS_ADD=False,
            RANK=state.my_pe,
            INKERNEL_BARRIER=False,
            FOLD_COPYIN=False,
            EXIT_BARRIER=False,
            WORKGROUP_SYNC=state._workgroup_sync,
            num_warps=num_warps,
        )


def _launch_triton_twoshot_core(
    kernels,
    state,
    x_view: torch.Tensor,
    input_bases: torch.Tensor,
    y_view: torch.Tensor,
    output_bases: torch.Tensor,
    residual: torch.Tensor,
    residual_view: torch.Tensor,
    residual_bases: torch.Tensor,
    weight: torch.Tensor,
    m: int,
    n: int,
) -> None:
    work_rows = kernels.triton.cdiv(m, state.world_size)
    grid_sms = state._grid_width(
        "twoshot_blocked",
        state.world_size,
        work_rows,
        False,
    )
    kernels.fused_ar_rmsnorm_twoshot_blocked_kernel[(grid_sms,)](
        x_view,
        y_view,
        state._scratch,
        _EPS,
        weight,
        state.my_pe,
        input_bases,
        output_bases,
        residual_bases,
        residual,
        residual_view,
        y_view,
        m,
        n,
        state._signal_pad,
        BLOCK_N=state._twoshot_block_n,
        ws=state.world_size,
        NUM_SMS=grid_sms,
        HAS_RESIDUAL=True,
        HAS_ADD=False,
        RANK=state.my_pe,
        INKERNEL_BARRIER=False,
        WORKGROUP_SYNC=state._workgroup_sync,
        num_warps=kernels.recommended_num_warps("twoshot_blocked"),
    )


def _assert_outputs(
    norm_out: torch.Tensor,
    residual_out: torch.Tensor,
    ref_norm: torch.Tensor,
    ref_residual: torch.Tensor,
) -> None:
    torch.testing.assert_close(
        residual_out.float(),
        ref_residual,
        atol=2e-2,
        rtol=2e-2,
    )
    torch.testing.assert_close(
        norm_out.float(),
        ref_norm,
        atol=2e-2,
        rtol=2e-2,
    )


def _base_row(
    *,
    backend: str,
    path: str,
    transport: str,
    ws: int,
    m: int,
) -> dict:
    return {
        "world_size": ws,
        "M": m,
        "N": _N,
        "backend": backend,
        "path": path,
        "transport": transport,
        "profile_id": os.environ.get("AR_NORM_PROFILE_ID", ""),
        "oneshot_max_m": os.environ.get("TS_TRITON_SHMEM_ONESHOT_MAX_M", ""),
        "input_site_ring": os.environ.get("TS_TRITON_SHMEM_INPUT_SITE_RING", ""),
        "borrow_twoshot_output": os.environ.get(
            "TS_TRITON_SHMEM_BORROW_TWOSHOT_OUTPUT", ""
        ),
        "output_alloc_host_us": 0.0,
        "output_alloc_method": "preallocated_or_ring",
        "benchmark_reset_copy_ms": 0.0,
        "copy_in_ms": 0.0,
        "copy_in_method": "none",
        "entry_sync_ms": 0.0,
        "entry_sync_method": "none",
        "transport_ms": 0.0,
        "transport_method": "none",
        "core_kernel_ms": 0.0,
        "core_kernel_method": "none",
        "comm_norm_kernel_ms": 0.0,
        "comm_norm_kernel_method": "none",
        "rmsnorm_residual_ms": 0.0,
        "rmsnorm_method": "none",
        "exit_sync_ms": 0.0,
        "exit_sync_method": "none",
        "copy_out_ms": 0.0,
        "copy_out_method": "none",
        "stage_sum_ms": 0.0,
        "total_prealloc_ms": 0.0,
        "total_public_ms": 0.0,
        "unaccounted_prealloc_ms": 0.0,
        "sync_derived_ms": 0.0,
        "sync_derived_method": "none",
        "marginal_copy_in_ms": 0.0,
        "marginal_entry_sync_ms": 0.0,
        "marginal_transport_ms": 0.0,
        "marginal_comm_norm_ms": 0.0,
        "marginal_rmsnorm_ms": 0.0,
        "marginal_exit_sync_ms": 0.0,
        "marginal_copy_out_ms": 0.0,
        "marginal_sum_ms": 0.0,
        "marginal_method": "direct_cumulative_difference",
        "notes": "",
    }


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

    max_m = max(_MS)
    # Iris uses a process-global heap. Size it before either ordinary or fused
    # state is created so sweep order cannot change the available heap.
    iris_mod._get_or_create_iris_context(1 << 29)
    ordinary_state = tri.create_state(
        group=group,
        rank_in_group=rank,
        device=device,
        max_numel=_TRITON_AR_MAX_BYTES
        // torch.empty((), dtype=torch.bfloat16).element_size(),
    )
    iris_state = iris_mod.create_iris_ar_rmsnorm_state(
        group=group,
        rank_in_group=rank,
        max_token_num=max_m,
        hidden_dim=_N,
        dtype=torch.bfloat16,
        heap_size=1 << 29,
        device=device,
        persistent=False,
    )
    triton_state = ts.create_triton_shmem_ar_rmsnorm_state(
        group=group,
        rank_in_group=rank,
        max_token_num=max_m,
        hidden_dim=_N,
        dtype=torch.bfloat16,
        device=device,
    )
    if triton_state is None:
        raise RuntimeError("triton_shmem state creation failed")

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
        reference = x.float()
        dist.all_reduce(reference, group=group)
        ref_residual = reference + residual.float()
        ref_norm = ref_residual * torch.rsqrt(
            ref_residual.square().mean(-1, keepdim=True) + _EPS
        )
        ref_norm *= weight.float()

        # ------------------------------------------------------------------
        # Production unfused: producer tensor is reduced in place. The copy
        # below resets benchmark input and is deliberately outside total timing.
        # ------------------------------------------------------------------
        scratch = torch.empty_like(x)
        unfused_norm = torch.empty_like(x)
        unfused_residual = torch.empty_like(x)
        use_ordinary_iris = (
            x.numel() * x.element_size() <= _TRITON_AR_MAX_BYTES
        )
        transport = "ordinary_iris" if use_ordinary_iris else "rccl"

        def reset_scratch():
            scratch.copy_(x)

        def run_transport():
            if use_ordinary_iris:
                return tri.all_reduce(ordinary_state, scratch)
            return dist.all_reduce(scratch, group=group)

        def run_unfused_kernel():
            return _launch_rmsnorm_prealloc(
                layernorm_mod,
                reference,
                residual,
                weight,
                unfused_norm,
                unfused_residual,
            )

        def run_unfused_prealloc():
            run_transport()
            return _launch_rmsnorm_prealloc(
                layernorm_mod,
                scratch,
                residual,
                weight,
                unfused_norm,
                unfused_residual,
            )

        def run_unfused_public():
            run_transport()
            return layernorm_mod.rmsnorm(
                scratch,
                weight,
                _EPS,
                residual=residual,
            )

        t_reset = _time_gpu(reset_scratch, group)
        t_transport = _time_gpu(run_transport, group, setup=reset_scratch)
        t_norm = _time_gpu(run_unfused_kernel, group)
        t_unfused_prealloc = _time_gpu(
            run_unfused_prealloc,
            group,
            setup=reset_scratch,
        )
        t_unfused_public = _time_gpu(
            run_unfused_public,
            group,
            setup=reset_scratch,
        )
        reset_scratch()
        run_transport()
        _launch_rmsnorm_prealloc(
            layernorm_mod,
            scratch,
            residual,
            weight,
            unfused_norm,
            unfused_residual,
        )
        _assert_outputs(
            unfused_norm,
            unfused_residual,
            ref_norm,
            ref_residual,
        )
        alloc_host_us = _time_host_us(
            lambda: (torch.empty_like(x), torch.empty_like(x)),
            group,
        )
        row = _base_row(
            backend="production_unfused",
            path=f"{transport}+triton_rmsnorm",
            transport=transport,
            ws=ws,
            m=m,
        )
        row.update(
            {
                "output_alloc_host_us": alloc_host_us,
                "output_alloc_method": "host_direct_two_empty_like",
                "benchmark_reset_copy_ms": t_reset,
                "transport_ms": t_transport,
                "transport_method": (
                    "direct_opaque_iris_kernel"
                    if use_ordinary_iris
                    else "direct_opaque_rccl"
                ),
                "rmsnorm_residual_ms": t_norm,
                "rmsnorm_method": "direct_preallocated_kernel",
                "stage_sum_ms": t_transport + t_norm,
                "total_prealloc_ms": t_unfused_prealloc,
                "total_public_ms": t_unfused_public,
                "unaccounted_prealloc_ms": (
                    t_unfused_prealloc - t_transport - t_norm
                ),
                "marginal_transport_ms": t_transport,
                "marginal_rmsnorm_ms": (
                    t_unfused_prealloc - t_transport
                ),
                "marginal_sum_ms": t_unfused_prealloc,
                "notes": (
                    "benchmark reset copy excluded from op; transport internal "
                    "staging/sync is opaque"
                ),
            }
        )
        rows.append(row)

        # ------------------------------------------------------------------
        # Fused Iris: explicit input staging, entry barrier, fused comm+norm,
        # exit barrier. All four GPU stages are directly callable.
        # ------------------------------------------------------------------
        iris_in = iris_state._input_buf[:m]
        iris_norm = torch.empty_like(x)
        iris_residual = torch.empty_like(x)
        t_iris_copy = _time_gpu(lambda: iris_in.copy_(x), group)
        t_iris_barrier = _time_gpu(
            lambda: iris_state._ctx.device_barrier(),
            group,
        )
        iris_in.copy_(x)
        iris_state._ctx.device_barrier()
        t_iris_kernel = _time_gpu(
            lambda: _launch_iris_fused_kernel(
                iris_mod,
                iris_state,
                iris_in,
                residual,
                weight,
                iris_norm,
                iris_residual,
            ),
            group,
        )
        iris_state._ctx.device_barrier()

        def run_iris_copy_entry():
            iris_in.copy_(x)
            iris_state._ctx.device_barrier()

        def run_iris_copy_entry_kernel():
            run_iris_copy_entry()
            _launch_iris_fused_kernel(
                iris_mod,
                iris_state,
                iris_in,
                residual,
                weight,
                iris_norm,
                iris_residual,
            )

        t_iris_copy_entry = _time_gpu(run_iris_copy_entry, group)
        t_iris_copy_entry_kernel = _time_gpu(
            run_iris_copy_entry_kernel,
            group,
        )
        t_iris_prealloc = _time_gpu(
            lambda: iris_state.fused(
                x,
                residual,
                weight,
                _EPS,
                norm_out=iris_norm,
                residual_out=iris_residual,
            ),
            group,
        )
        t_iris_public = _time_gpu(
            lambda: iris_state.fused(x, residual, weight, _EPS),
            group,
        )
        iris_state.fused(
            x,
            residual,
            weight,
            _EPS,
            norm_out=iris_norm,
            residual_out=iris_residual,
        )
        _assert_outputs(
            iris_norm,
            iris_residual,
            ref_norm,
            ref_residual,
        )
        iris_stage_sum = t_iris_copy + 2 * t_iris_barrier + t_iris_kernel
        row = _base_row(
            backend="iris_fused",
            path="iris_fused_nonpersistent",
            transport="iris_peer_pull",
            ws=ws,
            m=m,
        )
        row.update(
            {
                "output_alloc_host_us": alloc_host_us,
                "output_alloc_method": "host_direct_two_empty_like",
                "copy_in_ms": t_iris_copy,
                "copy_in_method": "direct_symmetric_heap_copy",
                "entry_sync_ms": t_iris_barrier,
                "entry_sync_method": "direct_device_barrier",
                "comm_norm_kernel_ms": t_iris_kernel,
                "comm_norm_kernel_method": "direct_fused_kernel",
                "exit_sync_ms": t_iris_barrier,
                "exit_sync_method": "direct_device_barrier",
                "stage_sum_ms": iris_stage_sum,
                "total_prealloc_ms": t_iris_prealloc,
                "total_public_ms": t_iris_public,
                "unaccounted_prealloc_ms": t_iris_prealloc - iris_stage_sum,
                "marginal_copy_in_ms": t_iris_copy,
                "marginal_entry_sync_ms": (
                    t_iris_copy_entry - t_iris_copy
                ),
                "marginal_comm_norm_ms": (
                    t_iris_copy_entry_kernel - t_iris_copy_entry
                ),
                "marginal_exit_sync_ms": (
                    t_iris_prealloc - t_iris_copy_entry_kernel
                ),
                "marginal_sum_ms": t_iris_prealloc,
                "notes": (
                    "comm, residual add, and RMSNorm are inseparable inside "
                    "one Iris kernel; marginal columns are additive cumulative "
                    "critical-path increments"
                ),
            }
        )
        rows.append(row)

        # ------------------------------------------------------------------
        # Realigned triton_shmem.
        # ------------------------------------------------------------------
        uses_twoshot = (
            triton_state._is_twoshot and m > triton_state._oneshot_max_m
        )
        triton_norm = torch.empty_like(x)
        triton_residual = torch.empty_like(x)
        if uses_twoshot:
            triton_x = triton_state._x[:m]
            input_bases = triton_state._input_bases
        elif (
            triton_state._input_site_ring_size
            and m <= triton_state._input_site_max_m
        ):
            triton_x = triton_state._input_site_tensor[:m]
            input_bases = triton_state._input_site_bases
        else:
            triton_x = triton_state._x[:m]
            input_bases = triton_state._input_bases

        t_triton_copy = _time_gpu(lambda: triton_x.copy_(x), group)
        t_triton_barrier = _time_gpu(lambda: triton_state._barrier(), group)
        triton_x.copy_(x)
        triton_state._barrier()

        if uses_twoshot:
            y_view = triton_state._twoshot_y_ring[0][:m]
            res_view = triton_state._twoshot_residual_ring[0][:m]
            output_bases = triton_state._twoshot_output_bases_ring[0]
            residual_bases = triton_state._twoshot_residual_bases_ring[0]
            t_triton_core = _time_gpu(
                lambda: _launch_triton_twoshot_core(
                    tk,
                    triton_state,
                    triton_x,
                    input_bases,
                    y_view,
                    output_bases,
                    residual,
                    res_view,
                    residual_bases,
                    weight,
                    m,
                    _N,
                ),
                group,
            )
            triton_state._barrier()
            t_triton_copyout = _time_gpu(
                lambda: (
                    triton_norm.copy_(y_view),
                    triton_residual.copy_(res_view),
                ),
                group,
            )

            def run_twoshot_copy_entry():
                triton_x.copy_(x)
                triton_state._barrier()

            def run_twoshot_copy_entry_kernel():
                run_twoshot_copy_entry()
                _launch_triton_twoshot_core(
                    tk,
                    triton_state,
                    triton_x,
                    input_bases,
                    y_view,
                    output_bases,
                    residual,
                    res_view,
                    residual_bases,
                    weight,
                    m,
                    _N,
                )

            t_twoshot_copy_entry = _time_gpu(
                run_twoshot_copy_entry,
                group,
            )
            t_twoshot_copy_entry_kernel = _time_gpu(
                run_twoshot_copy_entry_kernel,
                group,
            )
            t_triton_prealloc = _time_gpu(
                lambda: triton_state.fused(
                    x,
                    residual,
                    weight,
                    _EPS,
                    norm_out=triton_norm,
                    residual_out=triton_residual,
                ),
                group,
            )
            t_triton_public = _time_gpu(
                lambda: triton_state.fused(x, residual, weight, _EPS),
                group,
            )
            public_norm, public_residual = triton_state.fused(
                x,
                residual,
                weight,
                _EPS,
            )
            _assert_outputs(
                public_norm,
                public_residual,
                ref_norm,
                ref_residual,
            )
            triton_stage_sum = (
                t_triton_copy + 2 * t_triton_barrier + t_triton_core
            )
            row = _base_row(
                backend="triton_shmem_realigned",
                path="twoshot_blocked_borrowed_output",
                transport="coarse_ipc_peer_pull_push",
                ws=ws,
                m=m,
            )
            row.update(
                {
                    "output_alloc_method": "borrowed_symmetric_pingpong",
                    "copy_in_ms": t_triton_copy,
                    "copy_in_method": "direct_coarse_ipc_copy",
                    "entry_sync_ms": t_triton_barrier,
                    "entry_sync_method": "direct_signal_barrier",
                    "comm_norm_kernel_ms": t_triton_core,
                    "comm_norm_kernel_method": "direct_fused_twoshot_kernel",
                    "exit_sync_ms": t_triton_barrier,
                    "exit_sync_method": "direct_signal_barrier",
                    "copy_out_ms": 0.0,
                    "copy_out_method": "omitted_borrowed_output",
                    "stage_sum_ms": triton_stage_sum,
                    "total_prealloc_ms": t_triton_prealloc,
                    "total_public_ms": t_triton_public,
                    "unaccounted_prealloc_ms": (
                        t_triton_public - triton_stage_sum
                    ),
                    "marginal_copy_in_ms": t_triton_copy,
                    "marginal_entry_sync_ms": (
                        t_twoshot_copy_entry - t_triton_copy
                    ),
                    "marginal_comm_norm_ms": (
                        t_twoshot_copy_entry_kernel
                        - t_twoshot_copy_entry
                    ),
                    "marginal_exit_sync_ms": (
                        t_triton_public
                        - t_twoshot_copy_entry_kernel
                    ),
                    "marginal_sum_ms": t_triton_public,
                    "notes": (
                        f"caller-output compatibility total={t_triton_prealloc:.6f} "
                        f"ms includes direct copyout={t_triton_copyout:.6f} ms; "
                        "production eager path borrows outputs"
                    ),
                }
            )
        else:
            use_input_site = (
                triton_state._input_site_ring_size > 0
                and m <= triton_state._input_site_max_m
            )
            needs_exit_barrier = not (
                triton_state._double_buffer_input or use_input_site
            )
            use_output_ring = (
                triton_state._output_ring_size > 0
                and m <= triton_state._output_ring_max_m
            )
            t_triton_core = _time_gpu(
                lambda: _launch_triton_oneshot_core(
                    tk,
                    triton_state,
                    triton_x,
                    input_bases,
                    residual,
                    weight,
                    triton_norm,
                    triton_residual,
                    m,
                    _N,
                ),
                group,
            )
            triton_state._barrier()
            t_triton_kernel_entry = _time_gpu(
                lambda: triton_state._run_oneshot(
                    triton_x,
                    input_bases,
                    x,
                    residual,
                    weight,
                    _EPS,
                    m,
                    _N,
                    ws,
                    triton_norm,
                    triton_residual,
                    False,
                    False,
                ),
                group,
            )
            t_triton_kernel_full = _time_gpu(
                lambda: triton_state._run_oneshot(
                    triton_x,
                    input_bases,
                    x,
                    residual,
                    weight,
                    _EPS,
                    m,
                    _N,
                    ws,
                    triton_norm,
                    triton_residual,
                    False,
                    needs_exit_barrier,
                ),
                group,
            )
            t_triton_prealloc = _time_gpu(
                lambda: triton_state.fused(
                    x,
                    residual,
                    weight,
                    _EPS,
                    norm_out=triton_norm,
                    residual_out=triton_residual,
                ),
                group,
            )
            t_triton_public = _time_gpu(
                lambda: triton_state.fused(x, residual, weight, _EPS),
                group,
            )
            triton_state.fused(
                x,
                residual,
                weight,
                _EPS,
                norm_out=triton_norm,
                residual_out=triton_residual,
            )
            _assert_outputs(
                triton_norm,
                triton_residual,
                ref_norm,
                ref_residual,
            )
            entry_derived = t_triton_kernel_entry - t_triton_core
            exit_derived = t_triton_kernel_full - t_triton_kernel_entry
            triton_stage_sum = (
                t_triton_copy + t_triton_core + entry_derived + exit_derived
            )
            row = _base_row(
                backend="triton_shmem_realigned",
                path=(
                    f"{triton_state._oneshot_kernel_for_m(m)}"
                    + ("_site_ring" if use_input_site else "_persistent_input")
                ),
                transport="coarse_ipc_peer_pull",
                ws=ws,
                m=m,
            )
            row.update(
                {
                    "output_alloc_method": (
                        "model_profile_output_ring"
                        if use_output_ring
                        else "per_call_empty_like"
                    ),
                    "copy_in_ms": t_triton_copy,
                    "copy_in_method": (
                        "direct_site_ring_copy"
                        if use_input_site
                        else "direct_persistent_input_copy"
                    ),
                    "entry_sync_ms": entry_derived,
                    "entry_sync_method": "full_entry_kernel_minus_no_sync_core",
                    "core_kernel_ms": t_triton_core,
                    "core_kernel_method": "direct_fused_kernel_no_sync_diagnostic",
                    "comm_norm_kernel_ms": t_triton_kernel_full,
                    "comm_norm_kernel_method": (
                        "direct_fused_kernel_including_required_sync"
                    ),
                    "exit_sync_ms": exit_derived,
                    "exit_sync_method": (
                        "omitted_by_site_lifetime"
                        if not needs_exit_barrier
                        else "full_kernel_minus_entry_only_kernel"
                    ),
                    "stage_sum_ms": triton_stage_sum,
                    "total_prealloc_ms": t_triton_prealloc,
                    "total_public_ms": t_triton_public,
                    "unaccounted_prealloc_ms": (
                        t_triton_prealloc - triton_stage_sum
                    ),
                    "sync_derived_ms": entry_derived + exit_derived,
                    "sync_derived_method": (
                        "entry_and_exit_cumulative_kernel_differences"
                    ),
                    "marginal_copy_in_ms": t_triton_copy,
                    "marginal_entry_sync_ms": entry_derived,
                    "marginal_comm_norm_ms": t_triton_core,
                    "marginal_exit_sync_ms": exit_derived,
                    "marginal_sum_ms": t_triton_prealloc,
                    "notes": (
                        "one-shot exit barrier "
                        + (
                            "omitted by graph-stable input-site lifetime"
                            if not needs_exit_barrier
                            else "retained for persistent-input reuse"
                        )
                    ),
                }
            )
        rows.append(row)

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
        f"{'M':>5} {'backend':>25} {'copy':>8} {'entry':>8} "
        f"{'transport':>10} {'core/fused':>10} {'norm':>8} {'exit':>8} "
        f"{'total':>8}"
    )
    for row in rows:
        kernel = row["comm_norm_kernel_ms"] or row["core_kernel_ms"]
        print(
            f"{row['M']:5d} {row['backend']:>25} "
            f"{row['copy_in_ms']:8.4f} {row['entry_sync_ms']:8.4f} "
            f"{row['transport_ms']:10.4f} {kernel:10.4f} "
            f"{row['rmsnorm_residual_ms']:8.4f} {row['exit_sync_ms']:8.4f} "
            f"{row['total_public_ms']:8.4f}"
        )

    csv_path = os.environ.get("BENCH_CSV")
    if csv_path:
        _write_csv(csv_path, rows)
        print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
