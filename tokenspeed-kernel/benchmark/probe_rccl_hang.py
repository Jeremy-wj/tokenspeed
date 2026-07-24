"""Isolate the RCCL all-reduce hang seen under serve (ws=4, no GPU3).

Standalone multi-proc RCCL all-reduce stress at configurable unfused
large-prefill sizes. Optionally
injects per-rank timing jitter (RANK_JITTER=1) to mimic scheduler desync, since
the hang showed only 2/4 ranks spinning. If this hangs standalone -> RCCL/driver
issue; if not -> the serve's mixed symm_mem/custom-AR + RCCL usage or scheduler
desync is the trigger.

Run (avoid GPU3=HIP0):
    HIP_VISIBLE_DEVICES=1,2,3,5 WS=4 BENCH_N=2880 ITERS=300 \
      python3 -m benchmark.probe_rccl_hang
    HIP_VISIBLE_DEVICES=1,2,3,5 WS=4 BENCH_N=7168 \
      M_VALUES=2048,4096 RANK_JITTER=1 python3 -m benchmark.probe_rccl_hang
"""
import os
import socket
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from benchmark.shape_axes import default_hidden_size

N = default_hidden_size()
MS = [
    int(value)
    for value in os.environ.get("M_VALUES", "2048,4096,6656,8192").split(",")
    if value
]
ITERS = int(os.environ.get("ITERS", "300"))
JITTER = os.environ.get("RANK_JITTER", "0") == "1"


def _port():
    with socket.socket() as s:
        s.bind(("", 0)); return s.getsockname()[1]


def _worker(rank, ws, port):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"tcp://localhost:{port}", rank=rank, world_size=ws)
    g = dist.group.WORLD
    for m in MS:
        x = torch.randn((m, N), dtype=torch.bfloat16, device=f"cuda:{rank}")
        t0 = time.time()
        for i in range(ITERS):
            if JITTER and rank == 0 and i % 17 == 0:
                time.sleep(0.002)  # perturb one rank's arrival time
            dist.all_reduce(x, group=g)
        torch.cuda.synchronize()
        dist.barrier(g)
        if rank == 0:
            dt = (time.time() - t0) / ITERS * 1e3
            print(f"  M={m:>5} x {N}: {ITERS} all-reduces OK, {dt:.3f} ms/iter", flush=True)
    if rank == 0:
        print("RCCL stress PASSED (no hang)", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    ws = int(os.environ.get("WS", "4"))
    print(f"RCCL all-reduce stress: ws={ws} N={N} iters={ITERS} jitter={JITTER}", flush=True)
    mp.spawn(_worker, args=(ws, _port()), nprocs=ws, join=True)
