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

"""``triton_shmem`` fused all-reduce + residual + RMSNorm backend (symm_mem).

The fused kernels originate in the external ``triton-shmem`` repo (hence the
name); this module vendors them (see :mod:`._triton_shmem_kernels`) and drives
them over **PyTorch symmetric memory** (``torch.distributed._symmetric_memory``)
rather than the rocSHMEM heap they used upstream. That rocSHMEM->symm_mem
re-backing IS the migration: zero new runtime deps (symm_mem ships in ``torch``;
rocSHMEM is not pip-installable) and a graph-capture-safe in-kernel signal-pad
barrier (rocSHMEM's host ``barrier_all_on_stream`` under HIP graph capture was
unproven). See the migration doc for the full rationale.

It exposes the same shim contract as ``communication.iris``
(``create_*_state`` + ``*_allreduce_residual_rmsnorm`` + a ``*_STATES`` cache) so
it drops into the ``TS_ARNORM_BACKEND`` switch as the ``triton_shmem`` backend.

Substrate mapping (migration doc §4):

* Allocation: ``symm_mem.empty`` + ``rendezvous`` (via :func:`._alloc_symm`).
* Pointer translation: a **per-tensor** ``buffer_ptrs_dev`` table (via
  :func:`._peer_ptrs_dev`). The one-shot kernels translate only ``input`` (one
  table); the two-shot kernel pushes into peers' ``output`` and ``residual_out``
  too, so it takes three tables.
* Barriers: a single-block signal-pad barrier kernel
  (:func:`._triton_shmem_kernels.symm_grid_barrier_kernel`). We issue a
  **leading** barrier (all peers' inputs visible before any pull) and a
  **trailing** barrier (all peer pushes visible + all peer reads done) for
  **every** variant. The trailing barrier is required even for the one-shot
  kernels -- although their outputs are purely local, the persistent symmetric
  ``input`` buffer is reused across calls (repeated captured decode graphs), so
  peers must finish reading it before the next call overwrites it. This matches
  the native ``amd_allreduce_residual_rmsnorm_kernel`` (entry + exit barrier) and
  Iris (``device_barrier`` before + after).

PERFORMANCE NOTE (migration doc §8): torch symm_mem on ROCm allocates
**fine-grained** memory (needed for the signal-pad atomics) which bypasses L2
and delivers only ~105 GB/s for bulk local access vs. ~3200 GB/s coarse-grained.
The upstream rocSHMEM heap is coarse-grained, so this backend is materially
slower than the rocSHMEM numbers for large tensors. It is still correct and
graph-capture-safe; the regression is a substrate property, not a port bug.

Scope (matches the Iris/native contract): AMD ROCm, bf16, 2-D
``(num_tokens, hidden)`` input, ``weight`` of shape ``(hidden,)``. The reduction
spans ``group.size()`` ranks; symm_mem rendezvous accepts any process group, so
this is not restricted to the whole world, but the current dispatch only
exercises whole-world TP.
"""

import logging

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from tokenspeed_kernel._triton import triton
from tokenspeed_kernel.platform import current_platform

from . import _triton_shmem_kernels as _k
from .triton import _alloc_symm, _peer_ptrs_dev

logger = logging.getLogger(__file__)

_platform = current_platform()

__all__ = [
    "TritonShmemAllReduceResidualRMSNorm",
    "create_triton_shmem_ar_rmsnorm_state",
    "triton_shmem_allreduce_residual_rmsnorm",
    "is_available",
    "TRITON_SHMEM_AR_RMSNORM_STATES",
]


# State cache keyed identically to the Iris shim so the two are drop-in
# interchangeable: (id(group), max_token_num, hidden_dim, dtype).
TRITON_SHMEM_AR_RMSNORM_STATES: dict = {}


def is_available() -> bool:
    """Whether the triton_shmem (symm_mem) fused backend can run here."""
    return _platform.is_amd


def _num_cus(device: torch.device) -> int:
    # Single-node MI300X box is homogeneous, so the local CU count is already
    # rank-consistent (no cross-rank MIN needed).
    return torch.cuda.get_device_properties(device).multi_processor_count


class TritonShmemAllReduceResidualRMSNorm:
    """symm_mem-backed fused all-reduce + residual-add + RMSNorm.

    Holds persistent symmetric ``(max_token_num, hidden_dim)`` input (and, for
    the two-shot kernel, output + residual_out) buffers plus their per-tensor
    peer-pointer tables and a local fp32 scratch, and dispatches to the tuned
    kernel variant per call.
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        rank_in_group: int,
        max_token_num: int,
        hidden_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | None = None,
    ) -> None:
        assert _platform.is_amd, (
            "TritonShmemAllReduceResidualRMSNorm targets AMD ROCm; "
            f"got non-AMD platform: {_platform}"
        )
        assert dist.is_initialized(), (
            "torch.distributed must be initialized before constructing "
            "TritonShmemAllReduceResidualRMSNorm."
        )

        self.group = group
        self.rank_in_group = rank_in_group
        self.max_token_num = max_token_num
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        self.device = device or torch.device(f"cuda:{torch.cuda.current_device()}")
        self.world_size = group.size()

        # The (ws, hidden) pair is fixed for this state's lifetime, so the tuned
        # kernel variant is chosen once here.
        self.kernel = _k.recommended_kernel(self.world_size, hidden_dim)
        self._is_twoshot = self.kernel == "twoshot_blocked"
        self._num_cus = _num_cus(self.device)

        # Reserve the signal pad before the first symm_mem.empty (the pad size is
        # baked into the allocation). The whole-grid barrier kernel launches at
        # most `num_cus` blocks (recommended_grid caps grid_sms at num_cus) and
        # indexes the pad at block_id * ws + rank, so num_cus * ws uint32 slots
        # suffice for any grid we launch. max() never shrinks another module's pad.
        pad_bytes = self._num_cus * self.world_size * 4
        symm_mem.set_signal_pad_size(max(symm_mem.get_signal_pad_size(), pad_bytes))

        shape = (max_token_num, hidden_dim)
        itemsize = torch.empty((), dtype=dtype).element_size()
        buf_bytes = max_token_num * hidden_dim * itemsize

        # Input is always pulled by peers -> must be symmetric.
        self._x, x_hdl = _alloc_symm(shape, dtype, self.device, group)
        self._signal_pad = x_hdl.signal_pad_ptrs_dev
        self.my_pe = x_hdl.rank
        assert self.my_pe == rank_in_group, (
            f"rank mismatch: rank_in_group={rank_in_group}, symm_mem rank={self.my_pe}"
        )
        assert x_hdl.world_size == self.world_size, (
            f"symm_mem world {x_hdl.world_size} != group size {self.world_size}"
        )
        self._input_bases = _peer_ptrs_dev(
            x_hdl, shape, dtype, self.world_size, self.device
        )

        n_symm = 1
        if self._is_twoshot:
            # Two-shot pushes normalized output and pre-norm residual_out into
            # peers, so both must be symmetric and each needs its OWN pointer
            # table (symm_mem does not share offsets across allocations, §4).
            self._y, y_hdl = _alloc_symm(shape, dtype, self.device, group)
            self._residual_out, r_hdl = _alloc_symm(shape, dtype, self.device, group)
            self._output_bases = _peer_ptrs_dev(
                y_hdl, shape, dtype, self.world_size, self.device
            )
            self._residual_out_bases = _peer_ptrs_dev(
                r_hdl, shape, dtype, self.world_size, self.device
            )
            n_symm = 3
        else:
            self._y = None
            self._residual_out = None
            self._output_bases = None
            self._residual_out_bases = None

        # Local fp32 scratch: two-shot owns cdiv(M, ws) shard rows; the one-shot
        # blocked kernel reuses one row-slot per persistent program (<= #CUs).
        # oneshot_wholerow needs no scratch.
        if self._is_twoshot:
            m_shard_max = triton.cdiv(max_token_num, self.world_size)
            self._scratch = torch.empty(
                (m_shard_max, hidden_dim), dtype=torch.float32, device=self.device
            )
        elif self.kernel == "oneshot_blocked":
            self._scratch = torch.empty(
                (self._num_cus, hidden_dim), dtype=torch.float32, device=self.device
            )
        else:  # oneshot_wholerow
            self._scratch = None

        logger.info(
            "triton_shmem AR+RMSNorm state: kernel=%s ws=%d max_tokens=%d hidden=%d "
            "symm=%.1f MiB/rank",
            self.kernel,
            self.world_size,
            max_token_num,
            hidden_dim,
            n_symm * buf_bytes / 1024**2,
        )

    def _barrier(self) -> None:
        # Global signal-pad barrier at the kernel-launch boundary. A single
        # block per rank is sufficient: each rank's block signals every peer and
        # waits for every peer (all-to-all), and kernel-launch serialization does
        # the rest. Using 1 block (vs. the fused grid width) keeps the barrier
        # cheap -- it dominates the small-M decode regime otherwise.
        _k.symm_grid_barrier_kernel[(1,)](
            self._signal_pad,
            RANK=self.my_pe,
            WORLD_SIZE=self.world_size,
            num_warps=1,
        )

    def fused(
        self,
        input_tensor: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        norm_out: torch.Tensor | None = None,
        residual_out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert input_tensor.dtype == self.dtype
        assert input_tensor.dim() == 2 and input_tensor.shape == residual.shape
        assert input_tensor.shape[1] == self.hidden_dim
        assert weight.shape == (self.hidden_dim,)
        assert input_tensor.is_contiguous() and residual.is_contiguous()

        m = input_tensor.shape[0]
        n = self.hidden_dim
        ws = self.world_size
        assert m <= self.max_token_num

        if norm_out is None:
            norm_out = torch.empty_like(input_tensor)
        if residual_out is None:
            residual_out = torch.empty_like(residual)

        x = self._x[:m]
        x.copy_(input_tensor)

        work_rows = triton.cdiv(m, ws) if self._is_twoshot else m
        grid_sms = _k.recommended_grid(self.kernel, ws, work_rows, self._num_cus)
        num_warps = _k.recommended_num_warps(self.kernel)
        grid = (grid_sms,)

        # Leading barrier: every rank's freshly-copied input visible before any pull.
        self._barrier()

        if self._is_twoshot:
            y = self._y[:m]
            res_out = self._residual_out[:m]
            _k.fused_ar_rmsnorm_twoshot_blocked_kernel[grid](
                x,
                y,
                self._scratch,
                eps,
                weight,
                self.my_pe,
                self._input_bases,
                self._output_bases,
                self._residual_out_bases,
                residual,
                res_out,
                y,  # add_in placeholder (HAS_ADD=False)
                m,
                n,
                BLOCK_N=_k.recommended_block_n(self.dtype, n),
                ws=ws,
                NUM_SMS=grid_sms,
                HAS_RESIDUAL=True,
                HAS_ADD=False,
                num_warps=num_warps,
            )
            # Trailing barrier: peers' pushes into our output/residual_out visible.
            self._barrier()
            norm_out.copy_(y)
            residual_out.copy_(res_out)
            return norm_out, residual_out

        # One-shot pull: writes only locally, so hand the caller's plain output
        # tensors directly. A trailing barrier is still issued so peers finish
        # reading our symmetric input before the next call overwrites it.
        if self.kernel == "oneshot_wholerow":
            _k.fused_ar_rmsnorm_oneshot_wholerow_kernel[grid](
                x,
                norm_out,
                eps,
                weight,
                self.my_pe,
                self._input_bases,
                residual,
                residual_out,
                norm_out,  # add_in placeholder (HAS_ADD=False)
                m,
                N=n,
                ws=ws,
                NUM_SMS=grid_sms,
                HAS_RESIDUAL=True,
                HAS_ADD=False,
                num_warps=num_warps,
            )
        else:  # oneshot_blocked
            _k.fused_ar_rmsnorm_oneshot_blocked_kernel[grid](
                x,
                norm_out,
                self._scratch,
                eps,
                weight,
                self.my_pe,
                self._input_bases,
                residual,
                residual_out,
                norm_out,  # add_in placeholder (HAS_ADD=False)
                m,
                n,
                BLOCK_N=_k.recommended_block_n(self.dtype, n),
                ws=ws,
                NUM_SMS=grid_sms,
                HAS_RESIDUAL=True,
                HAS_ADD=False,
                num_warps=num_warps,
            )
        self._barrier()
        return norm_out, residual_out


def create_triton_shmem_ar_rmsnorm_state(
    group: dist.ProcessGroup,
    rank_in_group: int,
    max_token_num: int,
    hidden_dim: int,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | None = None,
) -> "TritonShmemAllReduceResidualRMSNorm | None":
    """Create a triton_shmem (symm_mem) fused AR+RMSNorm state, or ``None``.

    Returns ``None`` (so the caller can fall back) when the backend can't run on
    this platform or state construction fails.
    """
    if not is_available():
        return None
    try:
        return TritonShmemAllReduceResidualRMSNorm(
            group=group,
            rank_in_group=rank_in_group,
            max_token_num=max_token_num,
            hidden_dim=hidden_dim,
            dtype=dtype,
            device=device,
        )
    except Exception as exc:  # noqa: BLE001 - decline rather than crash forward
        logger.warning("triton_shmem AR+RMSNorm state creation failed: %s", exc)
        return None


def triton_shmem_allreduce_residual_rmsnorm(
    state: "TritonShmemAllReduceResidualRMSNorm",
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    norm_out: torch.Tensor | None = None,
    residual_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return state.fused(
        input_tensor=input_tensor,
        residual=residual,
        weight=weight,
        eps=eps,
        norm_out=norm_out,
        residual_out=residual_out,
    )
