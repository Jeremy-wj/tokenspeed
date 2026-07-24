"""Minimal HIP-graph capture/replay repro for the in-kernel signal-pad barrier.

See benchmark/results/ar_rmsnorm/docs/backend-design-and-safety.md: the
one-shot fused kernels fold their leading+trailing barriers in-kernel
(``TS_TRITON_SHMEM_INKERNEL_BARRIER=1``) and reach parity in eager, but fault
under CUDA/HIP graph replay in the served decode path.

This probe reproduces the fault OUTSIDE the serve so it can be root-caused: it
captures a HIP graph around a single fused decode call and replays it, for a
grid of decode-sized M, under several barrier variants:

  * ``sep``            : separate 1-block barrier kernels
  * ``inkernel``       : in-kernel barrier with separate copy-in
  * ``inkernel_fold``  : in-kernel barrier with folded copy-in
  * ``inkernel_fold_nosync``: pre-fix folded path (diagnostic only)

KEY RESULT (Jul 2026): a single-op / fixed-M graph NEVER faults with ``inkernel``,
even across many layers/replays/skew. The in-kernel barrier only DEADLOCKS in
``PROBE_MODE=multigraph`` with ``PROBE_RNG_SHARED=0`` -- i.e. when TP ranks replay
DIFFERENT-M graphs at the same time (its signal-pad slot range is M-dependent).
With ``PROBE_RNG_SHARED=1`` (serve-faithful: all TP ranks pick the same M each
step) it PASSES. ``sep`` passes in all modes (block_id=0, M-independent). See
the canonical backend design and historical incident record.

Run (inside container, ws=4 avoids GPU3=HIP0):
    # single-op graph (both pass):
    HIP_VISIBLE_DEVICES=1,2,3,5 BENCH_WS=4 \
        python3 -m benchmark.probe_inkernel_barrier_graph
    # multigraph divergence repro (inkernel HANGS, sep PASSES):
    HIP_VISIBLE_DEVICES=1,2,3,5 BENCH_WS=4 PROBE_MODE=multigraph \
        PROBE_RNG_SHARED=0 python3 -m benchmark.probe_inkernel_barrier_graph
Optional: BENCH_N=2880 PROBE_MS="1 4 8 64 256"
          PROBE_VARIANTS="sep inkernel inkernel_fold"
          PROBE_REPLAYS=300  PROBE_BS="1 8 64 160"  PROBE_TIMEOUT=90
"""
from __future__ import annotations

import os
import socket
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from benchmark.shape_axes import default_hidden_size

_N = default_hidden_size()
_EPS = 1e-6


def _port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _reference(residual, weight, ws, hidden, eps, scale):
    reduced = torch.full(
        (residual.shape[0], hidden),
        ws * (ws + 1) // 2 * scale,
        dtype=torch.float32,
        device=residual.device,
    )
    ref_res = reduced + residual.float()
    ref_norm = ref_res * torch.rsqrt(ref_res.pow(2).mean(-1, keepdim=True) + eps)
    return ref_res, ref_norm * weight.float()


def _multigraph_worker(rank, ws, variant, replays, port, err, n_layers):
    try:
        _multigraph_main(rank, ws, variant, replays, port, n_layers)
    except Exception:
        err[rank] = traceback.format_exc()


def _multigraph_main(rank, ws, variant, replays, port, n_layers):
    """Serve-like: ONE shared state, MANY graphs at different M sharing a memory
    pool captured on a side stream, replayed interleaved back-to-back."""
    import random
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://localhost:{port}", rank=rank, world_size=ws
    )
    dev = torch.device(f"cuda:{rank}")
    try:
        os.environ["TS_TRITON_SHMEM_INKERNEL_BARRIER"] = (
            "1" if variant.startswith("inkernel") else "0"
        )
        os.environ["TS_TRITON_SHMEM_FOLD_COPYIN"] = (
            "1" if "fold" in variant else "0"
        )
        os.environ["TS_TRITON_SHMEM_WORKGROUP_SYNC"] = (
            "0" if "nosync" in variant else "1"
        )
        os.environ["TS_TRITON_SHMEM_BARRIER_GRID"] = (
            os.environ.get("PROBE_FIXED_GRID", "64")
            if "fixed" in variant
            else "0"
        )
        from tokenspeed_kernel.ops.communication import triton_shmem as ts

        bss = [int(v) for v in os.environ.get(
            "PROBE_BS", "1 2 4 8 16 32 64 128 160").split()]
        max_m = max(bss)
        state = ts.create_triton_shmem_ar_rmsnorm_state(
            group=dist.group.WORLD, rank_in_group=rank,
            max_token_num=max_m, hidden_dim=_N, dtype=torch.bfloat16)
        assert state is not None
        weight = torch.linspace(0.5, 1.5, _N, dtype=torch.bfloat16, device=dev)

        # Persistent per-bs input buffers (like serve input buffers).
        bufs = {}
        for m in bss:
            x = torch.empty((m, _N), dtype=torch.bfloat16, device=dev)
            res = (torch.arange(m * _N, dtype=torch.float32, device=dev)
                   .reshape(m, _N).mul_(0.001).to(torch.bfloat16))
            no = torch.empty_like(x); ro = torch.empty_like(x)
            bufs[m] = (x, res, no, ro)

        def launch(m):
            x, res, no, ro = bufs[m]
            for _ in range(n_layers):
                ts.triton_shmem_allreduce_residual_rmsnorm(
                    state, input_tensor=x, residual=res, weight=weight,
                    eps=_EPS, norm_out=no, residual_out=ro)

        cap_stream = torch.cuda.Stream()
        # Warmup all sizes on side stream.
        cap_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(cap_stream):
            for m in bss:
                bufs[m][0].fill_(rank + 1)
                launch(m)
        torch.cuda.current_stream().wait_stream(cap_stream)
        dist.barrier()

        # Capture one graph per bs, sharing a memory pool, on the side stream.
        graphs = {}
        pool = None
        for m in bss:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool, stream=cap_stream):
                launch(m)
            if pool is None:
                pool = g.pool()
            graphs[m] = g
        dist.barrier()

        # Interleaved back-to-back replay, no inter-replay cross-rank sync.
        # PROBE_RNG_SHARED=1 (default): all TP ranks pick the SAME M each step
        # (serve-faithful -- TP ranks share one global batch). =0: per-rank M.
        shared = os.environ.get("PROBE_RNG_SHARED", "1") not in ("0", "false", "False")
        rng = random.Random(1234 if shared else 1234 + rank)
        for m in bss:
            bufs[m][0].fill_(rank + 1)
        import time as _t
        skew_us = float(os.environ.get("PROBE_SKEW_US", "0")) * rank
        for _ in range(replays):
            m = rng.choice(bss)
            if skew_us:
                _t.sleep(skew_us / 1e6)
            graphs[m].replay()
        torch.cuda.synchronize()
        # Correctness spot-check on the last size.
        for m in bss:
            bufs[m][0].fill_(rank + 1)
            graphs[m].replay()
        torch.cuda.synchronize()
        ref_res, ref_norm = _reference(bufs[bss[-1]][1], weight, ws, _N, _EPS, 1)
        torch.testing.assert_close(bufs[bss[-1]][3].float(), ref_res, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(bufs[bss[-1]][2].float(), ref_norm, atol=2e-2, rtol=2e-2)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _worker(rank, ws, m, variant, replays, port, err, n_layers):
    try:
        _worker_main(rank, ws, m, variant, replays, port, n_layers)
    except Exception:
        err[rank] = traceback.format_exc()


def _worker_main(rank, ws, m, variant, replays, port, n_layers):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://localhost:{port}", rank=rank, world_size=ws
    )
    dev = torch.device(f"cuda:{rank}")
    try:
        # Force the state to build a two-shot dispatcher (ws>=4) so the small-M
        # one-shot path -- the one that carries the in-kernel barrier -- is used.
        os.environ["TS_TRITON_SHMEM_INKERNEL_BARRIER"] = (
            "1" if variant.startswith("inkernel") else "0"
        )
        os.environ["TS_TRITON_SHMEM_FOLD_COPYIN"] = (
            "1" if "fold" in variant else "0"
        )
        os.environ["TS_TRITON_SHMEM_WORKGROUP_SYNC"] = (
            "0" if "nosync" in variant else "1"
        )
        os.environ["TS_TRITON_SHMEM_BARRIER_GRID"] = (
            os.environ.get("PROBE_FIXED_GRID", "64")
            if "fixed" in variant
            else "0"
        )
        from tokenspeed_kernel.ops.communication import triton_shmem as ts

        state = ts.create_triton_shmem_ar_rmsnorm_state(
            group=dist.group.WORLD,
            rank_in_group=rank,
            max_token_num=max(256, m),
            hidden_dim=_N,
            dtype=torch.bfloat16,
        )
        assert state is not None

        weight = torch.linspace(0.5, 1.5, _N, dtype=torch.bfloat16, device=dev)
        x = torch.empty((m, _N), dtype=torch.bfloat16, device=dev)
        residual = (
            torch.arange(m * _N, dtype=torch.float32, device=dev)
            .reshape(m, _N)
            .mul_(0.001)
            .to(torch.bfloat16)
        )
        norm_out = torch.empty_like(x)
        res_out = torch.empty_like(x)

        def launch():
            # Mimic the serve: N_LAYERS fused calls in one graph, all reusing the
            # same state (same signal pad + symmetric input buffer). copy-in each
            # call overwrites state._x, exactly like the per-layer residual stream.
            for _ in range(n_layers):
                ts.triton_shmem_allreduce_residual_rmsnorm(
                    state, input_tensor=x, residual=residual, weight=weight,
                    eps=_EPS, norm_out=norm_out, residual_out=res_out,
                )

        # Warmup on side stream (required before capture).
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

        # Serve-like replay: back-to-back replays with NO inter-replay cross-rank
        # sync, plus injected per-rank skew, so cross-rank skew accumulates the
        # way it does across independent per-rank decode loops (unlike a probe
        # that dist.barrier()s every step and hides skew).
        no_sync = os.environ.get("PROBE_NOSYNC", "1") not in ("0", "false", "False")
        skew_us = float(os.environ.get("PROBE_SKEW_US", "50")) * rank
        x.fill_(rank + 1)
        if no_sync:
            import time as _t
            for _ in range(replays):
                if skew_us:
                    _t.sleep(skew_us / 1e6)
                g.replay()
            torch.cuda.synchronize()
            ref_res, ref_norm = _reference(residual, weight, ws, _N, _EPS, 1)
            torch.testing.assert_close(res_out.float(), ref_res, atol=2e-2, rtol=2e-2)
            torch.testing.assert_close(norm_out.float(), ref_norm, atol=2e-2, rtol=2e-2)
            dist.barrier()
        else:
            for it in range(1, replays + 1):
                x.fill_((rank + 1) * it)
                g.replay()
                torch.cuda.synchronize()
                ref_res, ref_norm = _reference(residual, weight, ws, _N, _EPS, it)
                torch.testing.assert_close(res_out.float(), ref_res, atol=2e-2, rtol=2e-2)
                torch.testing.assert_close(norm_out.float(), ref_norm, atol=2e-2, rtol=2e-2)
                dist.barrier()
    finally:
        dist.destroy_process_group()


def main():
    ws = int(os.environ.get("BENCH_WS", "4"))
    ms = [int(v) for v in os.environ.get("PROBE_MS", "1 4 8 64 256").split()]
    variants = os.environ.get(
        "PROBE_VARIANTS", "sep inkernel inkernel_fold"
    ).split()
    replays = int(os.environ.get("PROBE_REPLAYS", "50"))
    n_layers = int(os.environ.get("PROBE_LAYERS", "36"))
    print(f"ws={ws} N={_N} replays={replays} layers/graph={n_layers}  "
          f"(kernel=oneshot small-M path)")

    if os.environ.get("PROBE_MODE", "single") == "multigraph":
        import time as _t
        timeout = float(os.environ.get("PROBE_TIMEOUT", "120"))
        for v in variants:
            err = mp.Manager().dict()
            proc = mp.spawn(_multigraph_worker,
                            args=(ws, v, replays, _port(), err, n_layers),
                            nprocs=ws, join=False)
            deadline = _t.time() + timeout
            done = False
            while _t.time() < deadline:
                if proc.join(timeout=2):
                    done = True
                    break
            if not done:
                res = f"HANG/DEADLOCK (timeout {timeout:.0f}s)"
                for pr in proc.processes:
                    if pr.is_alive():
                        pr.terminate()
                _t.sleep(3)
                for pr in proc.processes:
                    if pr.is_alive():
                        pr.kill()
            elif err:
                first = sorted(err)[0]
                res = "FAIL:" + err[first].strip().splitlines()[-1][:70]
            else:
                res = "PASS"
            print(f"multigraph variant={v:>18}: {res}", flush=True)
        return
    print(f"{'M':>6} " + " ".join(f"{v:>18}" for v in variants))
    for m in ms:
        cells = []
        for v in variants:
            err = mp.Manager().dict()
            try:
                mp.spawn(_worker, args=(ws, m, v, replays, _port(), err, n_layers),
                         nprocs=ws, join=True)
                cells.append("PASS" if not err else "FAIL")
            except Exception as e:  # noqa: BLE001
                msg = str(e).splitlines()[-1][:16] if str(e) else type(e).__name__
                cells.append(f"FAULT:{msg}")
            if err:
                # Print first rank's traceback tail for diagnosis.
                first = sorted(err)[0]
                tail = err[first].strip().splitlines()[-1][:60]
                cells[-1] = f"FAIL:{tail}"
        print(f"{m:>6} " + " ".join(f"{c:>18}" for c in cells))


if __name__ == "__main__":
    main()
