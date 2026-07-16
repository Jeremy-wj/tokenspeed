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

"""Vendored fused all-reduce (+ add + residual) + RMSNorm Triton kernels.

These are copy-pasted (per the migration's "vendor, do not import" guardrail)
from the external ``triton-shmem`` repo
(``triton_shmem/ccl/fused_ar_rmsnorm.py``), then re-backed by **PyTorch
symmetric memory** instead of rocSHMEM. The only device-code change vs. upstream
is that the two-shot kernel takes **three** per-tensor peer-pointer tables
(input / output / residual_out) instead of one shared rocSHMEM ``heap_bases``
array, because symm_mem hands out an independent ``buffer_ptrs_dev`` per
allocation (see the migration doc §4). The one-shot kernels are unchanged apart
from being vendored: they only translate ``input``, so a single table suffices.

``triton``/``tl`` are imported from ``tokenspeed_kernel._triton`` (the vendored
``tokenspeed_triton`` distribution) so these run under the same Triton as the
rest of the package. There are **no** ``rocshmem4py`` / ``triton_shmem`` imports.

Host-side barriers (rocSHMEM ``barrier_all``) are replaced by the symm_mem
signal-pad CAS barrier (``symm_mem_barrier`` from ``triton.py``), issued from a
dedicated single-block barrier kernel by the caller (see ``triton_shmem.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tokenspeed_kernel._triton import tl, triton

from .triton import symm_mem_barrier

KERNELS = ("twoshot_blocked", "oneshot_blocked", "oneshot_wholerow")


# ---------------------------------------------------------------------------
# Device-side symmetric pointer translation (vendored from
# triton_shmem/utils/symmetric.py). Given a per-tensor peer-pointer table
# ``bases`` (rocSHMEM ``heap_bases`` OR symm_mem ``buffer_ptrs_dev`` -- the math
# is identical, see migration doc §4), translate ``local_ptr`` from my rank's
# address space into ``peer``'s.
# ---------------------------------------------------------------------------
@triton.jit
def symmetric_ptr(local_ptr, my_pe, peer, bases):
    local_int = tl.cast(local_ptr, tl.uint64)
    my_base = tl.load(bases + my_pe)
    peer_base = tl.load(bases + peer)
    offset = local_int - my_base
    peer_byte = tl.cast(peer_base, tl.pointer_type(tl.int8))
    return tl.cast(peer_byte + offset, local_ptr.dtype)


# ---------------------------------------------------------------------------
# Whole-grid symm_mem barrier kernel. Each of the ``grid_sms`` blocks performs a
# signal-pad CAS barrier against the same block index on every peer, so the
# kernel-launch boundary is a global barrier (all peers reached it). Requires
# ``grid_sms <= num_cus`` so every block is resident (the spin-wait would
# otherwise deadlock); the caller guarantees this via ``recommended_grid``.
# ---------------------------------------------------------------------------
@triton.jit
def symm_grid_barrier_kernel(
    signal_pad_ptrs_dev,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    symm_mem_barrier(signal_pad_ptrs_dev, tl.program_id(0), RANK, WORLD_SIZE)


# ===========================================================================
# Host-side launch tuning (vendored verbatim from triton-shmem).
# ===========================================================================
@dataclass(frozen=True)
class ArchProfile:
    """Host-side launch tuning for one GPU architecture (perf-only knobs)."""

    name: str
    grid_caps: dict
    num_warps: dict
    oneshot_max_ws: int
    block_n_bytes: int = 1024
    block_n_min: int = 128
    default_num_warps: int = 4
    tuned: bool = True

    def grid_cap(self, kernel: str, ws: int) -> int:
        caps = self.grid_caps.get(kernel) or self.grid_caps["twoshot_blocked"]
        return caps.get(ws) or caps[min(caps, key=lambda w: abs(w - ws))]

    def warps(self, kernel: str) -> int:
        return self.num_warps.get(kernel, self.default_num_warps)


# Empirically tuned on MI300X (gfx942, 304 CU / 8 XCD), bf16, single node.
_MI300X = ArchProfile(
    name="MI300X (gfx942)",
    grid_caps={
        "twoshot_blocked": {2: 256, 4: 152, 8: 128},
        "oneshot_blocked": {2: 128, 4: 128, 8: 192},
        "oneshot_wholerow": {2: 32, 4: 32, 8: 32},
    },
    num_warps={"twoshot_blocked": 8, "oneshot_blocked": 4, "oneshot_wholerow": 8},
    oneshot_max_ws=2,
)

# Measured on MI350X (gfx950, CDNA4, 256 CU / 8 XCD), same bf16 workload.
_MI350X = ArchProfile(
    name="MI350X (gfx950)",
    grid_caps={
        "twoshot_blocked": {2: 256, 4: 256, 8: 256},
        "oneshot_blocked": {2: 128, 4: 256, 8: 256},
        "oneshot_wholerow": {2: 64, 4: 64, 8: 64},
    },
    num_warps={"twoshot_blocked": 4, "oneshot_blocked": 4, "oneshot_wholerow": 8},
    oneshot_max_ws=2,
)

# Fallback for un-tuned architectures (MI300X prior).
_DEFAULT_PROFILE = ArchProfile(
    name="generic (untuned; MI300X prior)",
    grid_caps=_MI300X.grid_caps,
    num_warps=_MI300X.num_warps,
    oneshot_max_ws=_MI300X.oneshot_max_ws,
    tuned=False,
)

_PROFILES: dict = {
    "gfx942": _MI300X,
    "gfx950": _MI350X,
}


def detect_arch(device: int | None = None) -> str:
    """Base ``gfxNNN`` token for a CUDA/HIP device (e.g. ``"gfx942"``)."""
    idx = torch.cuda.current_device() if device is None else device
    return torch.cuda.get_device_properties(idx).gcnArchName.split(":")[0]


def get_arch_profile(arch=None) -> ArchProfile:
    """Resolve the active :class:`ArchProfile` (auto-detect on ``None``)."""
    if isinstance(arch, ArchProfile):
        return arch
    if arch is None:
        try:
            arch = detect_arch()
        except Exception:
            return _DEFAULT_PROFILE
    return _PROFILES.get(arch, _DEFAULT_PROFILE)


def recommended_grid(kernel: str, ws: int, work_rows: int, num_cus: int, *,
                     profile=None) -> int:
    """Tuned persistent-grid width, capped by fabric limit, work, and CU count."""
    cap = get_arch_profile(profile).grid_cap(kernel, ws)
    return max(1, min(cap, work_rows, num_cus))


def recommended_num_warps(kernel: str, *, profile=None) -> int:
    """Tuned launch ``num_warps`` for a fused AR+RMSNorm kernel (default 4)."""
    return get_arch_profile(profile).warps(kernel)


def recommended_block_n(dtype: torch.dtype, N: int, *, profile=None) -> int:
    """Tuned ``BLOCK_N`` for the N-blocked kernels (~1 KiB/block on MI300X)."""
    p = get_arch_profile(profile)
    return min(N, max(p.block_n_min, p.block_n_bytes // dtype.itemsize))


def recommended_kernel(ws: int, N: int, *, profile=None) -> str:
    """Best fused variant for ``(ws, N)`` (see upstream docstring)."""
    if ws <= get_arch_profile(profile).oneshot_max_ws:
        return "oneshot_wholerow" if (N & (N - 1)) == 0 else "oneshot_blocked"
    return "twoshot_blocked"


# ===========================================================================
# Kernels. HAS_ADD / HAS_RESIDUAL are constexpr flags; the fused ordering is:
#   x = all_reduce_sum(input); if HAS_ADD: x += add_in;
#   if HAS_RESIDUAL: x += residual; residual_out = x
#   norm_out = x * rsqrt(mean(x**2) + eps) * gamma
# ===========================================================================
@triton.jit
def fused_ar_rmsnorm_oneshot_wholerow_kernel(
    input,
    output,
    epsilon,
    gamma,
    my_pe,
    heap_bases,
    residual,
    residual_out,
    add_in,
    M,
    N: tl.constexpr,
    ws: tl.constexpr,
    NUM_SMS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_ADD: tl.constexpr,
):
    """One-shot pull, whole-row. Every PE reduces all rows by pulling each peer's
    ``input`` and writes only its own local ``output`` / ``residual_out`` (no
    push). ``N`` must be a power of two. Only ``input`` is symmetric, so a single
    peer-pointer table (``heap_bases``) is used."""
    tl.static_assert(
        (N & (N - 1)) == 0,
        "fused_ar_rmsnorm_oneshot_wholerow_kernel requires N to be a power of two; "
        "use fused_ar_rmsnorm_oneshot_blocked_kernel for arbitrary N",
    )
    pid = tl.program_id(0)

    offsets_n = tl.max_contiguous(tl.multiple_of(tl.arange(0, N), N), N)
    gamma_row = tl.load(gamma + offsets_n).to(tl.float32)

    for row_id in range(pid, M, NUM_SMS):
        offsets_io = offsets_n + (N * row_id)
        acc = tl.zeros((N, ), tl.float32)
        for peer in tl.static_range(0, ws):
            peer_ptr = symmetric_ptr(input, my_pe, peer, heap_bases)
            acc += tl.load(peer_ptr + offsets_io).to(tl.float32)

        if HAS_ADD:
            acc += tl.load(add_in + offsets_io).to(tl.float32)
        if HAS_RESIDUAL:
            acc += tl.load(residual + offsets_io).to(tl.float32)
            tl.store(residual_out + offsets_io, acc.to(residual_out.dtype.element_ty))

        sum_squares = tl.sum(acc * acc)
        norm_factor = tl.rsqrt((sum_squares / N) + epsilon)
        rms_norm = (acc * norm_factor * gamma_row).to(output.dtype.element_ty)
        tl.store(output + offsets_io, rms_norm)


@triton.jit
def fused_ar_rmsnorm_twoshot_blocked_kernel(
    input,
    output,
    scratch,
    epsilon,
    gamma,
    my_pe,
    input_bases,
    output_bases,
    residual_out_bases,
    residual,
    residual_out,
    add_in,
    M,
    N,
    BLOCK_N: tl.constexpr,
    ws: tl.constexpr,
    NUM_SMS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_ADD: tl.constexpr,
):
    """Two-pass, N-blocked. Contiguous row ownership: PE ``my_pe`` owns rows
    ``[my_pe*M_shard, (my_pe+1)*M_shard)``, reduces them by pulling every peer's
    ``input``, and pushes the pre-norm ``residual_out`` and normalized ``output``
    into every peer. ``input``, ``output`` and ``residual_out`` are all symmetric
    and each is translated with **its own** peer-pointer table (the sole device
    change vs. the rocSHMEM upstream, which shared one ``heap_bases``). A trailing
    barrier (issued by the caller) makes the peer pushes visible before copy-out."""
    tl.static_assert(
        (BLOCK_N & (BLOCK_N - 1)) == 0,
        "fused_ar_rmsnorm_twoshot_blocked_kernel requires BLOCK_N to be a power of two",
    )
    pid = tl.program_id(0)
    M_shard = tl.cdiv(M, ws)
    shard_row_offset = M_shard * my_pe
    n_blocks = tl.cdiv(N, BLOCK_N)
    col = tl.arange(0, BLOCK_N)

    for shard_row_id in range(pid, M_shard, NUM_SMS):
        global_row = shard_row_id + shard_row_offset
        if global_row < M:
            row_io_off = global_row * N  # into the symmetric (M, N) tensors
            scratch_off = shard_row_id * N  # into the local (M_shard, N) scratch

            sum_squares = tl.zeros((), tl.float32)
            for blk in range(0, n_blocks):
                cols = blk * BLOCK_N + col
                mask = cols < N
                offs = row_io_off + cols
                acc = tl.zeros((BLOCK_N, ), tl.float32)
                for peer in tl.static_range(0, ws):
                    peer_ptr = symmetric_ptr(input, my_pe, peer, input_bases)
                    acc += tl.load(peer_ptr + offs, mask=mask, other=0.0).to(tl.float32)
                if HAS_ADD:
                    acc += tl.load(add_in + offs, mask=mask, other=0.0).to(tl.float32)
                if HAS_RESIDUAL:
                    acc += tl.load(residual + offs, mask=mask, other=0.0).to(tl.float32)
                    res = acc.to(residual_out.dtype.element_ty)
                    tl.store(residual_out + offs, res, mask=mask)
                    for peer in tl.static_range(0, ws):
                        if peer != my_pe:
                            peer_ptr = symmetric_ptr(residual_out, my_pe, peer, residual_out_bases)
                            tl.store(peer_ptr + offs, res, mask=mask)
                sum_squares += tl.sum(acc * acc, axis=0)
                tl.store(scratch + scratch_off + cols, acc, mask=mask)

            norm_factor = tl.rsqrt((sum_squares / N) + epsilon)

            for blk in range(0, n_blocks):
                cols = blk * BLOCK_N + col
                mask = cols < N
                offs = row_io_off + cols
                reduced = tl.load(scratch + scratch_off + cols, mask=mask, other=0.0)
                block_g = tl.load(gamma + cols, mask=mask, other=0.0).to(tl.float32)
                rms_norm = (reduced * norm_factor * block_g).to(output.dtype.element_ty)
                tl.store(output + offs, rms_norm, mask=mask)
                for peer in tl.static_range(0, ws):
                    if peer != my_pe:
                        peer_ptr = symmetric_ptr(output, my_pe, peer, output_bases)
                        tl.store(peer_ptr + offs, rms_norm, mask=mask)


@triton.jit
def fused_ar_rmsnorm_oneshot_blocked_kernel(
    input,
    output,
    scratch,
    epsilon,
    gamma,
    my_pe,
    heap_bases,
    residual,
    residual_out,
    add_in,
    M,
    N,
    BLOCK_N: tl.constexpr,
    ws: tl.constexpr,
    NUM_SMS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_ADD: tl.constexpr,
):
    """One-shot pull, two-pass, N-blocked (arbitrary ``N``). No row ownership, no
    peer push. Every PE reduces all rows by pulling each peer's ``input`` and
    writes the full result to its own local ``output`` / ``residual_out``.
    ``scratch`` is a local fp32 ``(NUM_SMS, N)`` buffer (one slot per program).
    Only ``input`` is symmetric, so a single peer-pointer table is used."""
    tl.static_assert(
        (BLOCK_N & (BLOCK_N - 1)) == 0,
        "fused_ar_rmsnorm_oneshot_blocked_kernel requires BLOCK_N to be a power of two",
    )
    pid = tl.program_id(0)
    n_blocks = tl.cdiv(N, BLOCK_N)
    col = tl.arange(0, BLOCK_N)
    scratch_off = pid * N

    for row_id in range(pid, M, NUM_SMS):
        row_io_off = row_id * N

        sum_squares = tl.zeros((), tl.float32)
        for blk in range(0, n_blocks):
            cols = blk * BLOCK_N + col
            mask = cols < N
            offs = row_io_off + cols
            acc = tl.zeros((BLOCK_N, ), tl.float32)
            for peer in tl.static_range(0, ws):
                peer_ptr = symmetric_ptr(input, my_pe, peer, heap_bases)
                acc += tl.load(peer_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            if HAS_ADD:
                acc += tl.load(add_in + offs, mask=mask, other=0.0).to(tl.float32)
            if HAS_RESIDUAL:
                acc += tl.load(residual + offs, mask=mask, other=0.0).to(tl.float32)
                tl.store(residual_out + offs, acc.to(residual_out.dtype.element_ty), mask=mask)
            sum_squares += tl.sum(acc * acc, axis=0)
            tl.store(scratch + scratch_off + cols, acc, mask=mask)

        norm_factor = tl.rsqrt((sum_squares / N) + epsilon)

        for blk in range(0, n_blocks):
            cols = blk * BLOCK_N + col
            mask = cols < N
            offs = row_io_off + cols
            reduced = tl.load(scratch + scratch_off + cols, mask=mask, other=0.0)
            block_g = tl.load(gamma + cols, mask=mask, other=0.0).to(tl.float32)
            rms_norm = (reduced * norm_factor * block_g).to(output.dtype.element_ty)
            tl.store(output + offs, rms_norm, mask=mask)
