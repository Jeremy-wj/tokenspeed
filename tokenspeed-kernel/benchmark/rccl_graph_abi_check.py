"""Validate eager and captured RCCL on the loaded ROCm userspace."""
import os
import sys
import gc
import torch
import torch.distributed as dist


def log(*a):
    print(f"[rank{os.environ.get('RANK','?')}]", *a, file=sys.stderr, flush=True)


def main() -> None:
    log("init_process_group...")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    log("init done, device set")

    # 1. Eager RCCL all_reduce
    x = torch.ones(1024, 1024, device=dev) * (rank + 1)
    dist.all_reduce(x)
    torch.cuda.synchronize()
    expected = sum(range(1, dist.get_world_size() + 1))
    assert torch.allclose(x[0, 0], torch.tensor(float(expected), device=dev)), x[0, 0]
    log("STEP1 eager all_reduce OK")

    # 2. Capture RCCL itself. Resetting x is part of the graph so every replay
    # has the same expected reduction and exercises the real collective node.
    graph_x = torch.empty((1024, 1024), device=dev)
    for _ in range(3):
        graph_x.fill_(rank + 1)
        dist.all_reduce(graph_x)
    torch.cuda.synchronize()
    dist.barrier()

    collective_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(collective_graph):
        graph_x.fill_(rank + 1)
        dist.all_reduce(graph_x)
    dist.barrier()
    for _ in range(10):
        collective_graph.replay()
        torch.cuda.synchronize()
        assert torch.allclose(
            graph_x[0, 0], torch.tensor(float(expected), device=dev)
        ), graph_x[0, 0]
        dist.barrier()
    log("STEP2 captured all_reduce replay OK")

    # 3. Plain compute graph capture/replay (baseline HIP graph health).
    g2 = torch.cuda.CUDAGraph()
    z = torch.zeros(1, device=dev)
    with torch.cuda.graph(g2):
        z.add_(1)
    for _ in range(3):
        g2.replay()
    torch.cuda.synchronize()
    log("STEP3 compute graph replay OK")

    dist.barrier()
    if rank == 0:
        print(
            f"OK torch={torch.__version__} baked_hip={torch.version.hip} "
            f"ws={dist.get_world_size()} eager+captured_allreduce+compute_graph PASSED",
            flush=True,
        )
    # Destroy captured Work/graph objects before ProcessGroupNCCL. Keeping a
    # captured collective graph alive while tearing down its communicator can
    # leave the watchdog/heartbeat threads waiting on graph-owned events.
    del collective_graph, g2
    gc.collect()
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
