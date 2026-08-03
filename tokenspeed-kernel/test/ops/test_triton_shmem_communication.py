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

"""Correctness tests for the symm_mem fused AR+residual+RMSNorm backend.

A near-1:1 clone of ``test_iris_communication.py`` Suite 3, retargeted at the
``triton_shmem`` shim (``create_triton_shmem_ar_rmsnorm_state`` +
``triton_shmem_allreduce_residual_rmsnorm``). Same mp.spawn / fp32-reference /
non-identity linspace-weight design and 2e-2 tolerances.

World sizes and hidden dims are chosen to cover all three vendored kernel
variants via the ``recommended_kernel`` dispatch (``oneshot_max_ws=2`` on
MI300X/MI350X):

* ``hidden=2880`` (not a power of two): ws<=2 -> ``oneshot_blocked``;
  ws>=4 -> ``twoshot_blocked`` (the ws=8 production path, 3 pointer tables +
  trailing barrier -- the highest-risk kernel).
* ``hidden=4096`` (power of two) at ws=2 -> ``oneshot_wholerow``.

Also includes a HIP graph capture+replay case (the capability the rocSHMEM host
barrier could not prove; see the migration doc).
"""

import os
import socket
import traceback
from typing import List

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.platform import current_platform

_TOKEN_CASES: List[int] = [1, 64, 256, 1024, 8192]
_EPS = 1e-6


def test_padded_kernel_keeps_m_dynamic():
    from tokenspeed_kernel.ops.communication._triton_shmem_kernels import (
        fused_ar_rmsnorm_oneshot_wholerow_padded_kernel,
    )

    assert "M" in fused_ar_rmsnorm_oneshot_wholerow_padded_kernel.do_not_specialize


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _skip_if_unsupported(world_size: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm is required for triton_shmem tests")
    if world_size > torch.cuda.device_count():
        pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")
    if not current_platform().is_amd:
        pytest.skip("triton_shmem backend only targets AMD ROCm")


def _spawn_and_collect(worker_fn, args, world_size: int) -> None:
    error_dict = mp.Manager().dict()
    mp.spawn(worker_fn, args=args + (error_dict,), nprocs=world_size, join=True)
    if error_dict:
        raise RuntimeError("\n".join(f"Rank {r}: {e}" for r, e in error_dict.items()))


def _reference(x, residual, weight, world_size, hidden, eps, device):
    reduced = torch.full(
        (x.shape[0], hidden),
        world_size * (world_size + 1) // 2,
        dtype=torch.float32,
        device=device,
    )
    ref_residual = reduced + residual.float()
    ref_norm = ref_residual * torch.rsqrt(
        ref_residual.pow(2).mean(dim=-1, keepdim=True) + eps
    )
    ref_norm = ref_norm * weight.float()
    return ref_residual, ref_norm


def test_core_v3_profile_validation_is_explicit(monkeypatch):
    _skip_if_unsupported(1)
    from tokenspeed_kernel.ops.communication.triton_shmem import (
        _dynamic_grid_cap,
        _dynamic_grid_cap_min_m,
        _profile_validation_errors,
    )

    monkeypatch.delenv("TS_TRITON_SHMEM_GRID_CAP", raising=False)
    monkeypatch.delenv("TS_TRITON_SHMEM_GRID_CAP_MIN_M", raising=False)
    assert _dynamic_grid_cap() == 0
    assert _dynamic_grid_cap_min_m() == 0

    profile_env = {
        "TS_TRITON_SHMEM_COARSE": "1",
        "TS_TRITON_SHMEM_INKERNEL_BARRIER": "1",
        "TS_TRITON_SHMEM_FOLD_COPYIN": "0",
        "TS_TRITON_SHMEM_WORKGROUP_SYNC": "1",
        "TS_TRITON_SHMEM_OUTPUT_RING": "72",
        "TS_TRITON_SHMEM_INPUT_SITE_RING": "72",
        "TS_TRITON_SHMEM_BORROW_TWOSHOT_OUTPUT": "1",
        "TS_TRITON_SHMEM_DOUBLE_BUFFER_INPUT": "0",
        "TS_TRITON_SHMEM_ONESHOT_MAX_M": "384",
        "TS_TRITON_SHMEM_ONESHOT_BLOCK_N": "0",
        "TS_TRITON_SHMEM_ONESHOT_VARIANT": "padded",
        "TS_TRITON_SHMEM_PADDED_MAX_M": "64",
        "TS_TRITON_SHMEM_ONESHOT_NUM_WARPS": "4",
        "TS_TRITON_SHMEM_TWOSHOT_BLOCK_N": "0",
        "TS_TRITON_SHMEM_GRID_CAP": "128",
        "TS_TRITON_SHMEM_GRID_CAP_MIN_M": "256",
        "TS_TRITON_SHMEM_BARRIER_GRID": "0",
        "TS_TRITON_SHMEM_FUSION_MIN_M": "0",
        "TS_TRITON_SHMEM_FUSION_MAX_M": "0",
        "TS_TRITON_SHMEM_PROFILE_PURE_TP": "1",
    }
    for name, value in profile_env.items():
        monkeypatch.setenv(name, value)

    kwargs = {
        "profile_id": "gpt-oss-120b-mi350x-triton-core-v3",
        "arch": "gfx950",
        "world_size": 4,
        "max_token_num": 2048,
        "hidden_dim": 2880,
        "dtype": torch.bfloat16,
        "visible_devices": "1,2,5,6",
    }
    assert _profile_validation_errors(**kwargs) == []
    kwargs["world_size"] = 2
    assert "world_size=2 expected 4" in _profile_validation_errors(**kwargs)


def test_glm52_v2_profile_validation_is_explicit(monkeypatch):
    _skip_if_unsupported(1)
    from tokenspeed_kernel.ops.communication.triton_shmem import (
        _profile_validation_errors,
    )

    profile_env = {
        "TS_TRITON_SHMEM_COARSE": "1",
        "TS_TRITON_SHMEM_INKERNEL_BARRIER": "1",
        "TS_TRITON_SHMEM_FOLD_COPYIN": "0",
        "TS_TRITON_SHMEM_WORKGROUP_SYNC": "1",
        "TS_TRITON_SHMEM_OUTPUT_RING": "156",
        "TS_TRITON_SHMEM_INPUT_SITE_RING": "156",
        "TS_TRITON_SHMEM_BORROW_TWOSHOT_OUTPUT": "0",
        "TS_TRITON_SHMEM_DOUBLE_BUFFER_INPUT": "0",
        "TS_TRITON_SHMEM_ONESHOT_MAX_M": "42",
        "TS_TRITON_SHMEM_ONESHOT_BLOCK_N": "0",
        "TS_TRITON_SHMEM_ONESHOT_VARIANT": "padded",
        "TS_TRITON_SHMEM_PADDED_MAX_M": "42",
        "TS_TRITON_SHMEM_ONESHOT_NUM_WARPS": "4",
        "TS_TRITON_SHMEM_TWOSHOT_BLOCK_N": "0",
        "TS_TRITON_SHMEM_GRID_CAP": "0",
        "TS_TRITON_SHMEM_GRID_CAP_MIN_M": "0",
        "TS_TRITON_SHMEM_BARRIER_GRID": "0",
        "TS_TRITON_SHMEM_FUSION_MIN_M": "2",
        "TS_TRITON_SHMEM_FUSION_MAX_M": "0",
        "TS_TRITON_SHMEM_PROFILE_PURE_TP": "1",
    }
    for name, value in profile_env.items():
        monkeypatch.setenv(name, value)

    kwargs = {
        "profile_id": "glm-5.2-fp8-mi350x-triton-v2",
        "arch": "gfx950",
        "world_size": 8,
        "max_token_num": 42,
        "hidden_dim": 6144,
        "dtype": torch.bfloat16,
        "visible_devices": "0,1,2,3,4,5,6,7",
    }
    assert _profile_validation_errors(**kwargs) == []
    kwargs["max_token_num"] = 2048
    assert "max_token_num=2048 expected 42" in _profile_validation_errors(**kwargs)


def test_glm52_v1_profile_validation_remains_reproducible(monkeypatch):
    _skip_if_unsupported(1)
    from tokenspeed_kernel.ops.communication.triton_shmem import (
        _profile_validation_errors,
    )

    profile_env = {
        "TS_TRITON_SHMEM_COARSE": "1",
        "TS_TRITON_SHMEM_INKERNEL_BARRIER": "1",
        "TS_TRITON_SHMEM_FOLD_COPYIN": "0",
        "TS_TRITON_SHMEM_WORKGROUP_SYNC": "1",
        "TS_TRITON_SHMEM_OUTPUT_RING": "156",
        "TS_TRITON_SHMEM_INPUT_SITE_RING": "156",
        "TS_TRITON_SHMEM_BORROW_TWOSHOT_OUTPUT": "0",
        "TS_TRITON_SHMEM_DOUBLE_BUFFER_INPUT": "0",
        "TS_TRITON_SHMEM_ONESHOT_MAX_M": "32",
        "TS_TRITON_SHMEM_ONESHOT_BLOCK_N": "0",
        "TS_TRITON_SHMEM_ONESHOT_VARIANT": "padded",
        "TS_TRITON_SHMEM_PADDED_MAX_M": "32",
        "TS_TRITON_SHMEM_ONESHOT_NUM_WARPS": "4",
        "TS_TRITON_SHMEM_TWOSHOT_BLOCK_N": "0",
        "TS_TRITON_SHMEM_GRID_CAP": "0",
        "TS_TRITON_SHMEM_GRID_CAP_MIN_M": "0",
        "TS_TRITON_SHMEM_BARRIER_GRID": "0",
        "TS_TRITON_SHMEM_FUSION_MIN_M": "0",
        "TS_TRITON_SHMEM_FUSION_MAX_M": "0",
        "TS_TRITON_SHMEM_PROFILE_PURE_TP": "1",
    }
    for name, value in profile_env.items():
        monkeypatch.setenv(name, value)

    assert (
        _profile_validation_errors(
            profile_id="glm-5.2-fp8-mi350x-triton-v1",
            arch="gfx950",
            world_size=8,
            max_token_num=32,
            hidden_dim=6144,
            dtype=torch.bfloat16,
            visible_devices="0,1,2,3,4,5,6,7",
        )
        == []
    )


def _make_inputs(tokens, hidden, rank, device):
    # Each rank contributes rank+1 (sum across ranks = ws*(ws+1)/2); residual is
    # non-uniform (deterministic, identical across ranks -> replicated per TP)
    # so a weight/residual bug can't be masked.
    x = torch.full((tokens, hidden), rank + 1, dtype=torch.bfloat16, device=device)
    residual = (
        torch.arange(tokens * hidden, dtype=torch.float32, device=device)
        .reshape(tokens, hidden)
        .mul_(0.001)
        .to(torch.bfloat16)
    )
    return x, residual


# ---------------------------------------------------------------------------
# Suite 1: correctness sweep over token counts
# ---------------------------------------------------------------------------
def _corr_worker_fn(rank, world_size, port, hidden, error_dict):
    try:
        _corr_worker_main(rank, world_size, port, hidden)
    except Exception:
        error_dict[rank] = traceback.format_exc()


def _corr_worker_main(rank: int, world_size: int, port: int, hidden: int) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        from tokenspeed_kernel.ops.communication.triton_shmem import (
            create_triton_shmem_ar_rmsnorm_state,
            triton_shmem_allreduce_residual_rmsnorm,
        )

        max_token_num = max(_TOKEN_CASES)
        state = create_triton_shmem_ar_rmsnorm_state(
            group=dist.group.WORLD,
            rank_in_group=rank,
            max_token_num=max_token_num,
            hidden_dim=hidden,
            dtype=torch.bfloat16,
        )
        assert state is not None, "triton_shmem state creation returned None"

        weight = torch.linspace(0.5, 1.5, hidden, dtype=torch.bfloat16, device=device)

        for tokens in _TOKEN_CASES:
            x, residual = _make_inputs(tokens, hidden, rank, device)
            norm_out, residual_out = triton_shmem_allreduce_residual_rmsnorm(
                state,
                input_tensor=x,
                residual=residual,
                weight=weight,
                eps=_EPS,
            )
            ref_residual, ref_norm = _reference(
                x, residual, weight, world_size, hidden, _EPS, device
            )
            torch.testing.assert_close(
                residual_out.float(), ref_residual, atol=2e-2, rtol=2e-2
            )
            torch.testing.assert_close(norm_out.float(), ref_norm, atol=2e-2, rtol=2e-2)
    finally:
        dist.destroy_process_group()


def _run_corr(world_size: int, hidden: int) -> None:
    _skip_if_unsupported(world_size)
    port = _get_open_port()
    _spawn_and_collect(_corr_worker_fn, (world_size, port, hidden), world_size)


def test_triton_shmem_arrms_world1():
    # ws=1: oneshot_blocked (self-only reduce; exercises the single-rank barrier).
    _run_corr(world_size=1, hidden=2880)


def test_triton_shmem_arrms_world2():
    # ws=2: oneshot_blocked (arbitrary-N one-shot pull).
    _run_corr(world_size=2, hidden=2880)


def test_triton_shmem_arrms_world4():
    # ws=4: twoshot_blocked (3 pointer tables + trailing barrier).
    _run_corr(world_size=4, hidden=2880)


def _random_corr_worker(rank, world_size, port, hidden, error_dict):
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://localhost:{port}",
            rank=rank,
            world_size=world_size,
        )
        try:
            os.environ["TS_TRITON_SHMEM_FOLD_COPYIN"] = "0"
            from tokenspeed_kernel.ops.communication.triton_shmem import (
                create_triton_shmem_ar_rmsnorm_state,
                triton_shmem_allreduce_residual_rmsnorm,
            )

            device = torch.device(f"cuda:{rank}")
            state = create_triton_shmem_ar_rmsnorm_state(
                group=dist.group.WORLD,
                rank_in_group=rank,
                max_token_num=256,
                hidden_dim=hidden,
                dtype=torch.bfloat16,
            )
            assert state is not None
            weight = torch.linspace(
                0.5, 1.5, hidden, dtype=torch.bfloat16, device=device
            )
            iterations = 16  # 80 calls total, wrapping the profile's 72-site ring.
            for tokens in (1, 4, 32, 128, 256):
                for iteration in range(iterations):
                    generator = torch.Generator(device=device).manual_seed(
                        10_000 * iteration + rank
                    )
                    x = torch.randn(
                        (tokens, hidden),
                        dtype=torch.bfloat16,
                        device=device,
                        generator=generator,
                    )
                    residual_generator = torch.Generator(device=device).manual_seed(
                        20_000 + iteration
                    )
                    residual = torch.randn(
                        (tokens, hidden),
                        dtype=torch.bfloat16,
                        device=device,
                        generator=residual_generator,
                    )
                    norm_out, residual_out = triton_shmem_allreduce_residual_rmsnorm(
                        state,
                        input_tensor=x,
                        residual=residual,
                        weight=weight,
                        eps=_EPS,
                    )
                    reduced = x.float()
                    dist.all_reduce(reduced)
                    ref_residual = reduced + residual.float()
                    ref_norm = ref_residual * torch.rsqrt(
                        ref_residual.pow(2).mean(dim=-1, keepdim=True) + _EPS
                    )
                    ref_norm *= weight.float()
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
            if state._output_ring_size:
                assert (
                    state._output_ring_index
                    == (len((1, 4, 32, 128, 256)) * iterations)
                    % state._output_ring_size
                )
        finally:
            dist.destroy_process_group()
    except Exception:
        error_dict[rank] = traceback.format_exc()


def test_triton_shmem_arrms_random_correctness_world4():
    _skip_if_unsupported(4)
    _spawn_and_collect(
        _random_corr_worker,
        (4, _get_open_port(), 2880),
        4,
    )


def test_triton_shmem_arrms_output_ring_random_world4(monkeypatch):
    _skip_if_unsupported(4)
    monkeypatch.setenv("TS_TRITON_SHMEM_OUTPUT_RING", "72")
    _spawn_and_collect(
        _random_corr_worker,
        (4, _get_open_port(), 2880),
        4,
    )


def test_triton_shmem_arrms_input_site_ring_random_world4(monkeypatch):
    _skip_if_unsupported(4)
    monkeypatch.setenv("TS_TRITON_SHMEM_INKERNEL_BARRIER", "1")
    monkeypatch.setenv("TS_TRITON_SHMEM_INPUT_SITE_RING", "72")
    monkeypatch.setenv("TS_TRITON_SHMEM_DOUBLE_BUFFER_INPUT", "0")
    _spawn_and_collect(
        _random_corr_worker,
        (4, _get_open_port(), 2880),
        4,
    )


def test_triton_shmem_arrms_padded_wholerow_random_world4(monkeypatch):
    _skip_if_unsupported(4)
    monkeypatch.setenv("TS_TRITON_SHMEM_INKERNEL_BARRIER", "1")
    monkeypatch.setenv("TS_TRITON_SHMEM_INPUT_SITE_RING", "72")
    monkeypatch.setenv("TS_TRITON_SHMEM_DOUBLE_BUFFER_INPUT", "0")
    monkeypatch.setenv("TS_TRITON_SHMEM_ONESHOT_VARIANT", "padded")
    monkeypatch.setenv("TS_TRITON_SHMEM_ONESHOT_NUM_WARPS", "8")
    _spawn_and_collect(
        _random_corr_worker,
        (4, _get_open_port(), 2880),
        4,
    )


def _twoshot_borrow_worker(rank, world_size, port, hidden, error_dict):
    try:
        os.environ["TS_TRITON_SHMEM_BORROW_TWOSHOT_OUTPUT"] = "1"
        os.environ["TS_TRITON_SHMEM_ONESHOT_MAX_M"] = "256"
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://localhost:{port}",
            rank=rank,
            world_size=world_size,
        )
        try:
            from tokenspeed_kernel.ops.communication.triton_shmem import (
                create_triton_shmem_ar_rmsnorm_state,
                triton_shmem_allreduce_residual_rmsnorm,
            )

            device = torch.device(f"cuda:{rank}")
            state = create_triton_shmem_ar_rmsnorm_state(
                group=dist.group.WORLD,
                rank_in_group=rank,
                max_token_num=2048,
                hidden_dim=hidden,
                dtype=torch.bfloat16,
            )
            assert state is not None
            assert len(state._twoshot_y_ring) == 2
            assert len(state._twoshot_residual_ring) == 2
            weight = torch.linspace(
                0.5, 1.5, hidden, dtype=torch.bfloat16, device=device
            )

            expected_norm_ptrs = {tensor.data_ptr() for tensor in state._twoshot_y_ring}
            expected_residual_ptrs = {
                tensor.data_ptr() for tensor in state._twoshot_residual_ring
            }
            previous_norm_ptr = None
            previous_residual_ptr = None
            total_calls = 0
            for tokens, calls in ((257, 72), (512, 8), (1024, 8), (2048, 8)):
                residual = torch.linspace(
                    -0.5,
                    0.5,
                    tokens * hidden,
                    dtype=torch.bfloat16,
                    device=device,
                ).reshape(tokens, hidden)
                reference_residual_input = residual.float()
                for call in range(calls):
                    x = torch.full(
                        (tokens, hidden),
                        rank + 1 + (call % 3) * 0.125,
                        dtype=torch.bfloat16,
                        device=device,
                    )
                    norm_out, residual_out = triton_shmem_allreduce_residual_rmsnorm(
                        state,
                        input_tensor=x,
                        residual=residual,
                        weight=weight,
                        eps=_EPS,
                    )
                    reduced = x.float()
                    dist.all_reduce(reduced)
                    reference_residual = reduced + reference_residual_input
                    reference_norm = reference_residual * torch.rsqrt(
                        reference_residual.pow(2).mean(dim=-1, keepdim=True) + _EPS
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
                    assert norm_out.data_ptr() in expected_norm_ptrs
                    assert residual_out.data_ptr() in expected_residual_ptrs
                    if previous_norm_ptr is not None:
                        assert norm_out.data_ptr() != previous_norm_ptr
                        assert residual_out.data_ptr() != previous_residual_ptr
                    previous_norm_ptr = norm_out.data_ptr()
                    previous_residual_ptr = residual_out.data_ptr()
                    residual = residual_out
                    reference_residual_input = residual_out.float()
                    total_calls += 1

            assert state._twoshot_output_index == total_calls % 2

            # Explicit caller ownership retains the copied compatibility path
            # and does not consume a borrowed-output slot.
            x, residual = _make_inputs(512, hidden, rank, device)
            norm_out = torch.empty_like(x)
            residual_out = torch.empty_like(x)
            slot_before = state._twoshot_output_index
            returned_norm, returned_residual = triton_shmem_allreduce_residual_rmsnorm(
                state,
                input_tensor=x,
                residual=residual,
                weight=weight,
                eps=_EPS,
                norm_out=norm_out,
                residual_out=residual_out,
            )
            assert returned_norm.data_ptr() == norm_out.data_ptr()
            assert returned_residual.data_ptr() == residual_out.data_ptr()
            assert state._twoshot_output_index == slot_before
        finally:
            dist.destroy_process_group()
    except Exception:
        error_dict[rank] = traceback.format_exc()


def test_triton_shmem_arrms_twoshot_borrow_output_world4():
    _skip_if_unsupported(4)
    _spawn_and_collect(
        _twoshot_borrow_worker,
        (4, _get_open_port(), 2880),
        4,
    )


def _fusion_gate_worker(rank, world_size, port, error_dict):
    try:
        os.environ["TS_ARNORM_BACKEND"] = "triton_shmem"
        os.environ["TS_TRITON_SHMEM_FUSION_MIN_M"] = "2"
        os.environ["TS_TRITON_SHMEM_FUSION_MAX_M"] = "256"
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://localhost:{port}",
            rank=rank,
            world_size=world_size,
        )

        from tokenspeed_kernel.ops.communication import triton, triton_shmem

        hidden = 2880
        weight = torch.linspace(0.5, 1.5, hidden, dtype=torch.bfloat16, device=device)
        x, residual = _make_inputs(1, hidden, rank, device)
        declined = triton.allreduce_residual_rmsnorm(
            x,
            residual,
            weight,
            rank,
            dist.group.WORLD,
            max_token_num=2048,
        )
        assert declined[:2] == (None, None)

        x, residual = _make_inputs(256, hidden, rank, device)
        norm_out, residual_out, *_ = triton.allreduce_residual_rmsnorm(
            x,
            residual,
            weight,
            rank,
            dist.group.WORLD,
            max_token_num=2048,
        )
        assert norm_out is not None and residual_out is not None
        state_key = triton_shmem.triton_shmem_state_cache_key(
            dist.group.WORLD, 2048, hidden, torch.bfloat16
        )
        state = triton_shmem.TRITON_SHMEM_AR_RMSNORM_STATES[state_key]
        assert state.max_token_num == 2048

        x, residual = _make_inputs(512, hidden, rank, device)
        declined = triton.allreduce_residual_rmsnorm(
            x,
            residual,
            weight,
            rank,
            dist.group.WORLD,
            max_token_num=2048,
        )
        assert declined[:2] == (None, None)
    except Exception:
        error_dict[rank] = traceback.format_exc()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_triton_shmem_arrms_separate_performance_gate_world4():
    """M=1/M=512 decline while M=256 fuses in a 2048-row workspace."""
    _skip_if_unsupported(4)
    error_dict = mp.Manager().dict()
    mp.spawn(
        _fusion_gate_worker,
        args=(4, _get_open_port(), error_dict),
        nprocs=4,
        join=True,
    )
    if error_dict:
        raise RuntimeError("\n".join(f"Rank {r}: {e}" for r, e in error_dict.items()))


def test_triton_shmem_arrms_world8():
    # ws=8: twoshot_blocked -- the production path.
    _run_corr(world_size=8, hidden=2880)


def test_triton_shmem_arrms_world2_wholerow():
    # ws=2, power-of-two hidden -> oneshot_wholerow.
    _run_corr(world_size=2, hidden=4096)


# ---------------------------------------------------------------------------
# Suite 1b: subgroup TP pointer/rank semantics.
# ---------------------------------------------------------------------------
def _subgroup_worker(rank, world_size, port, interleaved, error_dict):
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://localhost:{port}",
            rank=rank,
            world_size=world_size,
        )
        rank_sets = [(0, 2), (1, 3)] if interleaved else [(0, 1), (2, 3)]
        groups = [dist.new_group(ranks) for ranks in rank_sets]
        group_idx = next(i for i, ranks in enumerate(rank_sets) if rank in ranks)
        ranks = rank_sets[group_idx]
        group = groups[group_idx]

        from tokenspeed_kernel.ops.communication.triton_shmem import (
            create_triton_shmem_ar_rmsnorm_state,
            triton_shmem_allreduce_residual_rmsnorm,
        )

        hidden = 2880
        device = torch.device(f"cuda:{rank}")
        state = create_triton_shmem_ar_rmsnorm_state(
            group=group,
            rank_in_group=ranks.index(rank),
            max_token_num=512,
            hidden_dim=hidden,
            dtype=torch.bfloat16,
        )
        assert state is not None
        weight = torch.linspace(0.5, 1.5, hidden, dtype=torch.bfloat16, device=device)
        expected_reduced = float(sum(r + 1 for r in ranks))
        for tokens in (64, 256, 512):
            x, residual = _make_inputs(tokens, hidden, rank, device)
            norm_out, residual_out = triton_shmem_allreduce_residual_rmsnorm(
                state, x, residual, weight, _EPS
            )
            ref_residual = (
                torch.full_like(x, expected_reduced, dtype=torch.float32)
                + residual.float()
            )
            ref_norm = ref_residual * torch.rsqrt(
                ref_residual.square().mean(-1, keepdim=True) + _EPS
            )
            ref_norm *= weight.float()
            torch.testing.assert_close(
                residual_out.float(), ref_residual, atol=2e-2, rtol=2e-2
            )
            torch.testing.assert_close(norm_out.float(), ref_norm, atol=2e-2, rtol=2e-2)
    except Exception:
        error_dict[rank] = traceback.format_exc()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_subgroup(interleaved: bool) -> None:
    _skip_if_unsupported(4)
    error_dict = mp.Manager().dict()
    mp.spawn(
        _subgroup_worker,
        args=(4, _get_open_port(), interleaved, error_dict),
        nprocs=4,
        join=True,
    )
    if error_dict:
        raise RuntimeError("\n".join(f"Rank {r}: {e}" for r, e in error_dict.items()))


def test_triton_shmem_arrms_disjoint_subgroups():
    _run_subgroup(interleaved=False)


def test_triton_shmem_arrms_interleaved_subgroups():
    _run_subgroup(interleaved=True)


# ---------------------------------------------------------------------------
# Suite 2: HIP graph capture + replay (capture-safety is the decisive reason to
# migrate off the rocSHMEM host barrier). Replays with changing input must
# recompute the reduction correctly.
# ---------------------------------------------------------------------------
def _graph_worker_fn(
    rank, world_size, port, hidden, tokens, fold_copyin, replays, error_dict
):
    try:
        _graph_worker_main(rank, world_size, port, hidden, tokens, fold_copyin, replays)
    except Exception:
        error_dict[rank] = traceback.format_exc()


def _graph_worker_main(
    rank: int,
    world_size: int,
    port: int,
    hidden: int,
    tokens: int,
    fold_copyin: bool,
    replays: int,
) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        os.environ["TS_TRITON_SHMEM_FOLD_COPYIN"] = "1" if fold_copyin else "0"
        os.environ["TS_TRITON_SHMEM_INKERNEL_BARRIER"] = "1" if fold_copyin else "0"
        from tokenspeed_kernel.ops.communication.triton_shmem import (
            create_triton_shmem_ar_rmsnorm_state,
            triton_shmem_allreduce_residual_rmsnorm,
        )

        state = create_triton_shmem_ar_rmsnorm_state(
            group=dist.group.WORLD,
            rank_in_group=rank,
            max_token_num=tokens,
            hidden_dim=hidden,
            dtype=torch.bfloat16,
        )
        assert state is not None

        weight = torch.linspace(0.5, 1.5, hidden, dtype=torch.bfloat16, device=device)
        x = torch.empty((tokens, hidden), dtype=torch.bfloat16, device=device)
        residual = (
            torch.arange(tokens * hidden, dtype=torch.float32, device=device)
            .reshape(tokens, hidden)
            .mul_(0.001)
            .to(torch.bfloat16)
        )
        norm_out = torch.empty_like(x)
        residual_out = torch.empty_like(x)

        def launch():
            triton_shmem_allreduce_residual_rmsnorm(
                state,
                input_tensor=x,
                residual=residual,
                weight=weight,
                eps=_EPS,
                norm_out=norm_out,
                residual_out=residual_out,
            )

        # Warmup on a side stream (required before capture).
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            x.fill_(rank + 1)
            launch()
        torch.cuda.current_stream().wait_stream(s)
        dist.barrier()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            launch()
        dist.barrier()

        # Replay with changing input; the captured graph must recompute the AR.
        for it in range(1, replays + 1):
            x.fill_((rank + 1) * it)
            g.replay()
            if it % 10 == 0 or it == replays:
                torch.cuda.synchronize()
                reduced = torch.full(
                    (tokens, hidden),
                    world_size * (world_size + 1) // 2 * it,
                    dtype=torch.float32,
                    device=device,
                )
                ref_residual = reduced + residual.float()
                ref_norm = ref_residual * torch.rsqrt(
                    ref_residual.pow(2).mean(dim=-1, keepdim=True) + _EPS
                )
                ref_norm = ref_norm * weight.float()
                torch.testing.assert_close(
                    residual_out.float(), ref_residual, atol=2e-2, rtol=2e-2
                )
                torch.testing.assert_close(
                    norm_out.float(), ref_norm, atol=2e-2, rtol=2e-2
                )
                dist.barrier()
    finally:
        dist.destroy_process_group()


def _run_graph(
    world_size: int,
    hidden: int,
    *,
    tokens: int = 256,
    fold_copyin: bool = False,
    replays: int = 3,
) -> None:
    _skip_if_unsupported(world_size)
    port = _get_open_port()
    _spawn_and_collect(
        _graph_worker_fn,
        (world_size, port, hidden, tokens, fold_copyin, replays),
        world_size,
    )


def test_triton_shmem_arrms_graph_capture_world2():
    # oneshot_blocked under graph capture/replay.
    _run_graph(world_size=2, hidden=2880)


def test_triton_shmem_arrms_graph_capture_world8():
    # M=256 uses the small-M one-shot overlay at ws=8.
    _run_graph(world_size=8, hidden=2880)


def test_triton_shmem_arrms_folded_copyin_graph_world4():
    # Serving regression: repeated captured one-shot blocked calls with folded
    # phase-0 copy-in and no host synchronization between every replay.
    _run_graph(
        world_size=4,
        hidden=2880,
        tokens=64,
        fold_copyin=True,
        replays=100,
    )


def test_triton_shmem_arrms_gridcap_graph_world4(monkeypatch):
    # Exercises the gfx950 ws4 M>=256 integration cap (256 -> 128 CTAs).
    monkeypatch.setenv("TS_TRITON_SHMEM_GRID_CAP", "128")
    monkeypatch.setenv("TS_TRITON_SHMEM_GRID_CAP_MIN_M", "256")
    _run_graph(
        world_size=4,
        hidden=2880,
        tokens=256,
        fold_copyin=True,
        replays=100,
    )


def _double_buffer_graph_worker(
    rank, world_size, port, hidden, tokens, replays, error_dict
):
    try:
        os.environ["TS_TRITON_SHMEM_DOUBLE_BUFFER_INPUT"] = "1"
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://localhost:{port}",
            rank=rank,
            world_size=world_size,
        )
        from tokenspeed_kernel.ops.communication.triton_shmem import (
            create_triton_shmem_ar_rmsnorm_state,
            triton_shmem_allreduce_residual_rmsnorm,
        )

        state = create_triton_shmem_ar_rmsnorm_state(
            group=dist.group.WORLD,
            rank_in_group=rank,
            max_token_num=tokens,
            hidden_dim=hidden,
            dtype=torch.bfloat16,
        )
        assert state is not None
        assert len(state._input_ring) == 2
        weight = torch.linspace(0.5, 1.5, hidden, dtype=torch.bfloat16, device=device)
        residual = (
            torch.arange(tokens * hidden, dtype=torch.float32, device=device)
            .reshape(tokens, hidden)
            .mul_(0.001)
            .to(torch.bfloat16)
        )
        xs = [
            torch.empty((tokens, hidden), dtype=torch.bfloat16, device=device)
            for _ in range(2)
        ]
        norm_outs = [torch.empty_like(xs[0]) for _ in range(2)]
        residual_outs = [torch.empty_like(xs[0]) for _ in range(2)]

        def launch_pair():
            for index in range(2):
                triton_shmem_allreduce_residual_rmsnorm(
                    state,
                    input_tensor=xs[index],
                    residual=residual,
                    weight=weight,
                    eps=_EPS,
                    norm_out=norm_outs[index],
                    residual_out=residual_outs[index],
                )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for index, x in enumerate(xs):
                x.fill_(rank + index + 1)
            launch_pair()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        dist.barrier()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            launch_pair()
        assert state._input_ring_index == 0
        dist.barrier()

        for iteration in range(1, replays + 1):
            for index, x in enumerate(xs):
                x.fill_((rank + index + 1) * iteration)
            graph.replay()
            if iteration % 10 == 0 or iteration == replays:
                torch.cuda.synchronize()
                for index in range(2):
                    rank_sum = world_size * (world_size + 1) // 2 + index * world_size
                    reduced = torch.full(
                        (tokens, hidden),
                        rank_sum * iteration,
                        dtype=torch.float32,
                        device=device,
                    )
                    ref_residual = reduced + residual.float()
                    ref_norm = ref_residual * torch.rsqrt(
                        ref_residual.square().mean(-1, keepdim=True) + _EPS
                    )
                    ref_norm *= weight.float()
                    torch.testing.assert_close(
                        residual_outs[index].float(),
                        ref_residual,
                        atol=2e-2,
                        rtol=2e-2,
                    )
                    torch.testing.assert_close(
                        norm_outs[index].float(),
                        ref_norm,
                        atol=2e-2,
                        rtol=2e-2,
                    )
                dist.barrier()
    except Exception:
        error_dict[rank] = traceback.format_exc()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_triton_shmem_double_buffer_even_graph_world4():
    """Two captured calls safely reuse two input slots without exit barriers."""
    _skip_if_unsupported(4)
    error_dict = mp.Manager().dict()
    mp.spawn(
        _double_buffer_graph_worker,
        args=(4, _get_open_port(), 2880, 32, 100, error_dict),
        nprocs=4,
        join=True,
    )
    if error_dict:
        raise RuntimeError("\n".join(f"Rank {r}: {e}" for r, e in error_dict.items()))


def _input_site_ring_graph_worker(
    rank, world_size, port, hidden, tokens, replays, error_dict
):
    try:
        os.environ["TS_TRITON_SHMEM_INKERNEL_BARRIER"] = "1"
        os.environ["TS_TRITON_SHMEM_INPUT_SITE_RING"] = "72"
        os.environ["TS_TRITON_SHMEM_DOUBLE_BUFFER_INPUT"] = "0"
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://localhost:{port}",
            rank=rank,
            world_size=world_size,
        )
        try:
            from tokenspeed_kernel.ops.communication.triton_shmem import (
                create_triton_shmem_ar_rmsnorm_state,
                triton_shmem_allreduce_residual_rmsnorm,
            )

            device = torch.device(f"cuda:{rank}")
            state = create_triton_shmem_ar_rmsnorm_state(
                group=dist.group.WORLD,
                rank_in_group=rank,
                max_token_num=tokens,
                hidden_dim=hidden,
                dtype=torch.bfloat16,
            )
            assert state is not None
            assert state._input_site_ring_size == 72
            weight = torch.linspace(
                0.5, 1.5, hidden, dtype=torch.bfloat16, device=device
            )
            residual = (
                torch.arange(tokens * hidden, dtype=torch.float32, device=device)
                .reshape(tokens, hidden)
                .mul_(0.001)
                .to(torch.bfloat16)
            )

            def make_graph_inputs():
                xs = [
                    torch.empty((tokens, hidden), dtype=torch.bfloat16, device=device)
                    for _ in range(72)
                ]
                norm_outs = [torch.empty_like(xs[0]) for _ in range(72)]
                residual_outs = [torch.empty_like(xs[0]) for _ in range(72)]
                return xs, norm_outs, residual_outs

            def launch_chain(xs, norm_outs, residual_outs):
                for index in range(72):
                    triton_shmem_allreduce_residual_rmsnorm(
                        state,
                        input_tensor=xs[index],
                        residual=residual,
                        weight=weight,
                        eps=_EPS,
                        norm_out=norm_outs[index],
                        residual_out=residual_outs[index],
                    )

            xs_a, norms_a, residuals_a = make_graph_inputs()
            xs_b, norms_b, residuals_b = make_graph_inputs()
            for graph_id, (xs, norms, residuals) in enumerate(
                ((xs_a, norms_a, residuals_a), (xs_b, norms_b, residuals_b))
            ):
                for index, x in enumerate(xs):
                    x.fill_(rank + 1 + graph_id + (index % 3))
                launch_chain(xs, norms, residuals)
            torch.cuda.synchronize()
            assert state._input_site_ring_index == 0
            dist.barrier()

            graph_a = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph_a):
                launch_chain(xs_a, norms_a, residuals_a)
            graph_b = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph_b):
                launch_chain(xs_b, norms_b, residuals_b)
            assert state._input_site_ring_index == 0
            dist.barrier()

            for iteration in range(1, replays + 1):
                for index, x in enumerate(xs_a):
                    x.fill_((rank + 1 + (index % 3)) * iteration)
                graph_a.replay()
                for index, x in enumerate(xs_b):
                    x.fill_((rank + 2 + (index % 3)) * iteration)
                graph_b.replay()
                if iteration % 10 == 0 or iteration == replays:
                    torch.cuda.synchronize()
                    for index in (0, 35, 71):
                        rank_sum = world_size * (world_size + 1) // 2 + world_size * (
                            1 + (index % 3)
                        )
                        reduced = torch.full(
                            (tokens, hidden),
                            rank_sum * iteration,
                            dtype=torch.float32,
                            device=device,
                        )
                        ref_residual = reduced + residual.float()
                        ref_norm = ref_residual * torch.rsqrt(
                            ref_residual.square().mean(-1, keepdim=True) + _EPS
                        )
                        ref_norm *= weight.float()
                        torch.testing.assert_close(
                            residuals_b[index].float(),
                            ref_residual,
                            atol=2e-2,
                            rtol=2e-2,
                        )
                        torch.testing.assert_close(
                            norms_b[index].float(),
                            ref_norm,
                            atol=2e-2,
                            rtol=2e-2,
                        )
                    dist.barrier()
        finally:
            dist.destroy_process_group()
    except Exception:
        error_dict[rank] = traceback.format_exc()


def test_triton_shmem_input_site_ring_graph_world4():
    _skip_if_unsupported(4)
    _spawn_and_collect(
        _input_site_ring_graph_worker,
        (4, _get_open_port(), 2880, 32, 100),
        4,
    )


def test_triton_shmem_padded_wholerow_graph_world4(monkeypatch):
    _skip_if_unsupported(4)
    monkeypatch.setenv("TS_TRITON_SHMEM_ONESHOT_VARIANT", "padded")
    monkeypatch.setenv("TS_TRITON_SHMEM_ONESHOT_NUM_WARPS", "8")
    _spawn_and_collect(
        _input_site_ring_graph_worker,
        (4, _get_open_port(), 2880, 32, 100),
        4,
    )


def test_triton_shmem_capture_requires_persistent_output(monkeypatch):
    _skip_if_unsupported(1)
    from tokenspeed_kernel.ops.communication.triton_shmem import (
        triton_shmem_can_run,
    )

    state = type(
        "State",
        (),
        {"_output_ring_size": 72, "_output_ring_max_m": 384},
    )()
    small = type("Input", (), {"shape": (384, 2880)})()
    large = type("Input", (), {"shape": (512, 2880)})()

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert triton_shmem_can_run(state, small)
    assert not triton_shmem_can_run(state, large)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert triton_shmem_can_run(state, large)


def test_triton_shmem_arrms_twoshot_graph_world8():
    _run_graph(world_size=8, hidden=2880, tokens=512)
