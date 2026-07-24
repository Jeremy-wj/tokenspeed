#!/usr/bin/env python3
"""Minimal eager/graph probes for torch profiler and Triton Proton on ROCm."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark.shape_axes import default_hidden_size


def _torch_probe(args: argparse.Namespace) -> None:
    import torch

    activities = [torch.profiler.ProfilerActivity.CPU]
    if args.activity == "cpu-gpu":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    x = torch.randn(args.size, args.size, device="cuda")
    y = torch.randn(args.size, args.size, device="cuda")

    graph = None
    if args.graph:
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                torch.mm(x, y)
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(graph):
            torch.mm(x, y)

    with torch.profiler.profile(
        activities=activities,
        with_stack=False,
        record_shapes=args.record_shapes,
    ) as profiler:
        for _ in range(args.repeats):
            if graph is None:
                torch.mm(x, y)
            else:
                graph.replay()
        torch.cuda.synchronize()

    profiler.export_chrome_trace(str(args.output))
    print(
        f"PASS profiler=torch activity={args.activity} graph={args.graph} "
        f"output={args.output}"
    )


def _proton_probe(args: argparse.Namespace) -> None:
    import torch

    from tokenspeed_kernel._triton import proton
    from tokenspeed_kernel.ops.layernorm.triton import rmsnorm
    from tokenspeed_kernel.profiling import (
        ProfilingConfig,
        start_profiling,
        stop_profiling,
    )

    config = ProfilingConfig(
        output=str(args.output),
        data=args.proton_data,
        backend=args.proton_backend,
        mode=args.proton_mode,
        hook=None if args.proton_hook == "none" else args.proton_hook,
        output_format=args.proton_output_format,
    )

    session = None
    if args.profile_before_runtime:
        session = start_profiling(config)

    x = torch.randn(
        args.size, args.hidden_size, device="cuda", dtype=torch.bfloat16
    )
    weight = torch.ones(args.hidden_size, device="cuda", dtype=torch.bfloat16)

    if not args.profile_before_runtime:
        rmsnorm(x, weight, 1e-5)
        torch.cuda.synchronize()

    graph = None
    if args.dedicated_stream:
        torch.cuda.set_stream(torch.cuda.Stream())
    if args.graph and not args.profile_before_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            if args.graph_scopes:
                with proton.scope("captured_region"):
                    rmsnorm(x, weight, 1e-5)
            else:
                rmsnorm(x, weight, 1e-5)

    if not args.profile_before_runtime:
        session = start_profiling(config)

    if args.graph and args.profile_before_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            if args.graph_scopes:
                with proton.scope("captured_region"):
                    rmsnorm(x, weight, 1e-5)
            else:
                rmsnorm(x, weight, 1e-5)

    if args.cycle_session_after_capture:
        proton.deactivate(session)
        proton.activate(session)

    for _ in range(args.repeats):
        if graph is None:
            rmsnorm(x, weight, 1e-5)
        elif args.graph_scopes or args.replay_scope:
            with proton.scope("graph_replay"):
                graph.replay()
        else:
            graph.replay()
    if graph is not None and args.reset_graph:
        graph.reset()
    torch.cuda.synchronize()
    stop_profiling()
    print(
        f"PASS profiler=proton backend={args.proton_backend} graph={args.graph} "
        f"profile_before_runtime={args.profile_before_runtime} "
        f"profile_before_graph={args.profile_before_graph} output={args.output}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiler", choices=["torch", "proton"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--activity", choices=["cpu", "cpu-gpu"], default="cpu-gpu")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--record-shapes", action="store_true")
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=default_hidden_size())
    parser.add_argument(
        "--proton-backend",
        choices=["roctracer", "rocprofiler", "instrumentation"],
        default="roctracer",
    )
    parser.add_argument("--proton-mode")
    parser.add_argument("--proton-data", choices=["tree", "trace"], default="trace")
    parser.add_argument(
        "--proton-output-format",
        choices=["hatchet", "hatchet_msgpack", "chrome_trace"],
        default="chrome_trace",
    )
    parser.add_argument("--proton-hook", choices=["triton", "none"], default="triton")
    parser.add_argument("--profile-before-runtime", action="store_true")
    parser.add_argument("--profile-before-graph", action="store_true")
    parser.add_argument("--graph-scopes", action="store_true")
    parser.add_argument("--replay-scope", action="store_true")
    parser.add_argument("--cycle-session-after-capture", action="store_true")
    parser.add_argument("--dedicated-stream", action="store_true")
    parser.add_argument("--reset-graph", action="store_true")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.profiler == "torch":
        _torch_probe(args)
    else:
        _proton_probe(args)


if __name__ == "__main__":
    main()
