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

PERFORMANCE (migration doc §8/§8.1): torch symm_mem on ROCm allocates
**fine-grained** memory (HIP VMM, no coherence knob) which bypasses L2 and
delivers only ~105 GB/s for bulk local access vs. ~3200 GB/s coarse-grained --
a ~30x penalty on copy-in/out and the kernel's local reads/writes. This is fixed
by ``_coarse_shmem.py``: the *data* buffers are coarse-grained ``torch.empty``
tensors shared peer-to-peer via HIP IPC (like the rocSHMEM heap), leaving only
the signal pad fine-grained. On by default (``TS_TRITON_SHMEM_COARSE``; set ``0``
for the legacy fine-grained path). Coarse is 5-38x faster (ws=8) and lands at
0.5-0.83x RCCL, competitive with the rocSHMEM reference. Remote (xGMI) access is
fabric-bound either way; the win is entirely on local bandwidth.

Scope (matches the Iris/native contract): AMD ROCm, bf16, 2-D
``(num_tokens, hidden)`` input, ``weight`` of shape ``(hidden,)``. The reduction
spans ``group.size()`` ranks; symm_mem rendezvous accepts any process group, so
this is not restricted to the whole world, but the current dispatch only
exercises whole-world TP.
"""

import logging
import os

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from tokenspeed_kernel._triton import triton
from tokenspeed_kernel.platform import current_platform

from . import _triton_shmem_kernels as _k
from ._coarse_shmem import alloc_coarse_symm
from .triton import _alloc_symm, _peer_ptrs_dev

logger = logging.getLogger(__file__)

_platform = current_platform()


def _coarse_enabled() -> bool:
    """Whether to back the *data* buffers with coarse-grained HBM + HIP IPC
    instead of fine-grained symm_mem (migration doc §8). Default ON: the
    fine-grained substrate is ~30x slower for all local access. Set
    ``TS_TRITON_SHMEM_COARSE=0`` to force the legacy symm_mem-only path."""
    return os.environ.get("TS_TRITON_SHMEM_COARSE", "1") not in ("0", "false", "False")


def _inkernel_barrier_enabled() -> bool:
    """Fold the leading+trailing signal-pad barriers into the fused kernels
    instead of launching two separate barrier kernels — removes the barrier-launch
    staging overhead that dominates small-M/decode latency (companion doc §6).
    **Default ON (Jul 2026):** validated correct + graph-safe + faster than the
    separate-barrier path e2e (ws=4 pure-TP gpt-oss-120b serve, conc 8–128 + mixed;
    companion §6). Set ``TS_TRITON_SHMEM_INKERNEL_BARRIER=0`` to force the legacy
    separate-barrier path.

    CAVEAT — set to 0 under configs that can diverge M across TP ranks (DP,
    ``overlap_schedule_depth>1``, speculative decode) until the M-independent
    barrier lands: the in-kernel barrier is issued by all ``grid_sms`` blocks at
    ``block_id=pid`` (M-dependent slot range), so it DEADLOCKS if two ranks run the
    barrier over different M simultaneously (the separate 1-block barrier is
    grid=(1,), block_id=0 -> M-independent -> robust). Pure TP (dp=1,
    overlap_schedule_depth=1) never diverges M, so the default is safe there. Repro:
    ``benchmark/probe_inkernel_barrier_graph.py`` (PROBE_MODE=multigraph). Fix plan:
    companion doc §6 plan."""
    return os.environ.get("TS_TRITON_SHMEM_INKERNEL_BARRIER", "1") not in (
        "0", "false", "False"
    )


def _oneshot_max_m() -> int:
    """At ws>=4 (two-shot state), route calls with ``M <= this`` through the
    one-shot pull kernel instead of two-shot. One-shot writes the output locally,
    so it skips the two-shot symmetric copy-out, and with in-kernel barriers has
    minimal fixed overhead -- decisive in the small-M/decode regime where
    two-shot's bandwidth advantage does not apply (companion doc §6). Two-shot's
    bandwidth-optimality still wins at large M. Default 256 (measured one-shot vs
    two-shot crossover on MI350X gfx950 ws=4: one-shot wins M<=256, two-shot wins
    M>=512); override with ``TS_TRITON_SHMEM_ONESHOT_MAX_M`` (0 disables). NOTE:
    the crossover is lower at ws=8 (one-shot pulls ws x bytes) -- validate/retune
    for ws=8."""
    return int(os.environ.get("TS_TRITON_SHMEM_ONESHOT_MAX_M", "256"))

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

        # Data buffers can be coarse-grained HBM (full bandwidth) shared via HIP
        # IPC, with only the signal pad left fine-grained (migration doc §8).
        # This recovers the ~30x local-bandwidth penalty of fine-grained symm_mem.
        self._coarse = _coarse_enabled()
        self._coarse_buffers: list = []
        self._opened_cache: dict = {}

        if self._coarse:
            # Tiny dedicated symm_mem allocation carries the fine-grained signal
            # pad and provides the rank/world identity used by the barrier.
            self._pad_tensor, pad_hdl = _alloc_symm((1,), dtype, self.device, group)
            self._signal_pad = pad_hdl.signal_pad_ptrs_dev
            self.my_pe = pad_hdl.rank
            ws_check = pad_hdl.world_size
            xb = self._alloc_data(shape, dtype, group)
            self._x, self._input_bases = xb.tensor, xb.peer_ptrs_dev
        else:
            self._pad_tensor = None
            # Input is always pulled by peers -> must be symmetric.
            self._x, x_hdl = _alloc_symm(shape, dtype, self.device, group)
            self._signal_pad = x_hdl.signal_pad_ptrs_dev
            self.my_pe = x_hdl.rank
            ws_check = x_hdl.world_size
            self._input_bases = _peer_ptrs_dev(
                x_hdl, shape, dtype, self.world_size, self.device
            )
        assert self.my_pe == rank_in_group, (
            f"rank mismatch: rank_in_group={rank_in_group}, symm_mem rank={self.my_pe}"
        )
        assert ws_check == self.world_size, (
            f"symm_mem world {ws_check} != group size {self.world_size}"
        )

        n_symm = 1
        if self._is_twoshot:
            # Two-shot pushes normalized output and pre-norm residual_out into
            # peers, so both must be peer-accessible and each needs its OWN
            # pointer table (offsets are not shared across allocations, §4).
            if self._coarse:
                yb = self._alloc_data(shape, dtype, group)
                rb = self._alloc_data(shape, dtype, group)
                self._y, self._output_bases = yb.tensor, yb.peer_ptrs_dev
                self._residual_out, self._residual_out_bases = rb.tensor, rb.peer_ptrs_dev
            else:
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

        # Small-M one-shot path (companion doc §6): at ws>=4 the dispatched kernel
        # is two-shot, but small-M/decode calls route through one-shot pull, which
        # writes output locally (skips the two-shot symmetric copy-out) and folds
        # its barriers in-kernel -- both dominate small-M latency. Reuses the
        # (already symmetric) input buffer; needs its own fp32 scratch.
        self._inkernel = _inkernel_barrier_enabled()
        self._oneshot_max_m = _oneshot_max_m()
        if self._is_twoshot:
            self._oneshot_kernel = (
                "oneshot_wholerow"
                if (hidden_dim & (hidden_dim - 1)) == 0
                else "oneshot_blocked"
            )
            self._oneshot_scratch = torch.empty(
                (self._num_cus, hidden_dim), dtype=torch.float32, device=self.device
            )
        else:
            self._oneshot_kernel = self.kernel  # ws<=2 is already one-shot
            self._oneshot_scratch = self._scratch

        logger.info(
            "triton_shmem AR+RMSNorm state: kernel=%s ws=%d max_tokens=%d hidden=%d "
            "substrate=%s data=%.1f MiB/rank inkernel_barrier=%s small_m_oneshot=%s(<=%d)",
            self.kernel,
            self.world_size,
            max_token_num,
            hidden_dim,
            "coarse+ipc" if self._coarse else "symm_mem(fine)",
            n_symm * buf_bytes / 1024**2,
            self._inkernel,
            self._oneshot_kernel if self._is_twoshot else "n/a",
            self._oneshot_max_m if self._is_twoshot else 0,
        )

    def _alloc_data(self, shape, dtype, group):
        """Allocate one coarse-grained peer-accessible data buffer (IPC-shared)."""
        buf = alloc_coarse_symm(
            shape, dtype, self.device, group, _opened_cache=self._opened_cache
        )
        self._coarse_buffers.append(buf)
        return buf

    def __del__(self) -> None:
        for buf in getattr(self, "_coarse_buffers", []):
            try:
                buf.close()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass

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

    def _run_oneshot(self, x, residual, weight, eps, m, n, ws, norm_out, residual_out):
        """One-shot pull into the caller's *local* output (no copy-out). Barriers
        are folded into the kernel when ``self._inkernel`` (default), else issued
        as the legacy separate launches. Reads the symmetric input ``x``; writes
        ``norm_out``/``residual_out`` directly."""
        kern = self._oneshot_kernel
        grid_sms = _k.recommended_grid(kern, ws, m, self._num_cus)
        grid = (grid_sms,)
        num_warps = _k.recommended_num_warps(kern)
        inkernel = self._inkernel
        if not inkernel:
            self._barrier()  # leading (legacy separate-barrier path)
        if kern == "oneshot_wholerow":
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
                self._signal_pad,
                N=n,
                ws=ws,
                NUM_SMS=grid_sms,
                HAS_RESIDUAL=True,
                HAS_ADD=False,
                RANK=self.my_pe,
                INKERNEL_BARRIER=inkernel,
                num_warps=num_warps,
            )
        else:  # oneshot_blocked
            _k.fused_ar_rmsnorm_oneshot_blocked_kernel[grid](
                x,
                norm_out,
                self._oneshot_scratch,
                eps,
                weight,
                self.my_pe,
                self._input_bases,
                residual,
                residual_out,
                norm_out,  # add_in placeholder (HAS_ADD=False)
                m,
                n,
                self._signal_pad,
                BLOCK_N=_k.recommended_block_n(self.dtype, n),
                ws=ws,
                NUM_SMS=grid_sms,
                HAS_RESIDUAL=True,
                HAS_ADD=False,
                RANK=self.my_pe,
                INKERNEL_BARRIER=inkernel,
                num_warps=num_warps,
            )
        if not inkernel:
            self._barrier()  # trailing

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

        # Dispatch: one-shot pull for ws<=2 (always) and small-M/decode at ws>=4
        # (companion doc §6). One-shot writes output locally (no two-shot copy-out)
        # and folds its barriers in-kernel; two-shot stays bandwidth-optimal for
        # large M at ws>=4 (separate barriers + copy-out, unchanged).
        use_oneshot = (not self._is_twoshot) or (
            self._oneshot_max_m > 0 and m <= self._oneshot_max_m
        )

        if use_oneshot:
            self._run_oneshot(x, residual, weight, eps, m, n, ws, norm_out, residual_out)
            return norm_out, residual_out

        # Two-shot push (ws>=4, large M).
        work_rows = triton.cdiv(m, ws)
        grid_sms = _k.recommended_grid("twoshot_blocked", ws, work_rows, self._num_cus)
        num_warps = _k.recommended_num_warps("twoshot_blocked")
        grid = (grid_sms,)
        y = self._y[:m]
        res_out = self._residual_out[:m]
        # Two-shot uses SEPARATE barriers even when in-kernel is enabled globally.
        # The in-kernel barrier's cost scales with grid width (all NUM_SMS blocks
        # spin-sync); two-shot's grid is large (up to num_cus), so folding it in
        # costs MORE than the two 1-block launches it removes (measured: reclaim
        # -0.005..-0.048 ms at M=512..1024, decomp probe). It is a win only for the
        # small-grid one-shot decode path. The kernel keeps the INKERNEL_BARRIER
        # param so the planned fixed-participant barrier (companion doc §6 plan) can
        # re-enable it once the barrier cost is decoupled from grid width.
        inkernel = False
        self._barrier()  # leading: peers' copy-in visible before any pull
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
            self._signal_pad,
            BLOCK_N=_k.recommended_block_n(self.dtype, n),
            ws=ws,
            NUM_SMS=grid_sms,
            HAS_RESIDUAL=True,
            HAS_ADD=False,
            RANK=self.my_pe,
            INKERNEL_BARRIER=inkernel,
            num_warps=num_warps,
        )
        self._barrier()  # trailing: peers' pushes into our output/residual_out visible
        norm_out.copy_(y)
        residual_out.copy_(res_out)
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
