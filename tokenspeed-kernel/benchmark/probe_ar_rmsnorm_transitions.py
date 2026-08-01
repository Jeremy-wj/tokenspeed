"""AMD post-rebase AR+RMSNorm graph/eager transition correctness probe.

This is a safety probe, not a benchmark: it records no latency and supports no
performance conclusions.  It precreates the selected production state, creates
all buffers, and captures exactly one HIP graph per M before running a
deterministic, rank-shared sequence.  The sequence covers repeated and
interleaved graph replay, odd and even calls per graph, eager->graph and
graph->eager transitions, changing inputs, and both returned tensors.

Supported ``BENCH_IMPL`` values are ``production_unfused``, ``auto``, ``iris``,
``symm_mem``, and ``triton_shmem``.  ``production_unfused`` follows the serving
transport gate exactly: ordinary Iris all-reduce for payloads <=512 KiB, RCCL
above 512 KiB, then TokenSpeed residual RMSNorm.  Every fused arm goes through
the production dispatcher and treats a decline as an error.

Run from the repository root inside an AMD ROCm environment::

    HIP_VISIBLE_DEVICES=1,2,3,5 BENCH_IMPL=auto \
      python3 -m benchmark.probe_ar_rmsnorm_transitions
    HIP_VISIBLE_DEVICES=1,2,3,5 BENCH_IMPL=triton_shmem \
      PROBE_REPLAYS=1000 PROBE_JSON=/tmp/transitions.json \
      python3 -m benchmark.probe_ar_rmsnorm_transitions

Important environment variables:

* ``BENCH_WS`` (default 4), ``BENCH_N`` (default 2880)
* ``PROBE_MS`` (default ``1,32,255,256,257,512,2048``)
* ``PROBE_REPLAYS`` (default 1000): total graph replays across both variants
* ``PROBE_MAX_REPLAYS`` (default 1000, absolute ceiling 10000)
* ``PROBE_CALLS_PER_GRAPH`` (default ``1,2``; must cover odd and even)
* ``PROBE_SEED`` (default 20260729), ``PROBE_TIMEOUT_S`` (default 3600)
* ``PROBE_JSON``: optional JSON output path; JSON is always printed

The probe deliberately makes no signal-pad pointer or layout assumptions.
TokenSpeed does not expose a safe signal-zero query for ``triton_shmem``, so
that arm reports ``not_exposed_by_safe_public_api`` rather than peeking at
private signal storage.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import socket
import subprocess
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


_REPO_ROOT = Path(__file__).resolve().parents[1]
_EPS = 1e-6
_ATOL = 2e-2
_RTOL = 2e-2
_ORDINARY_AR_MAX_BYTES = 512 * 1024
_HARD_MAX_REPLAYS = 10_000
_SUPPORTED_IMPLS = {
    "production_unfused",
    "auto",
    "iris",
    "symm_mem",
    "triton_shmem",
}
_IDENTITY_FILES = (
    "benchmark/probe_ar_rmsnorm_transitions.py",
    "python/tokenspeed_kernel/ops/communication/triton.py",
    "python/tokenspeed_kernel/ops/communication/iris.py",
    "python/tokenspeed_kernel/ops/communication/triton_shmem.py",
    "python/tokenspeed_kernel/ops/layernorm/triton.py",
)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def _env_numbers(name: str, default: str) -> list[int]:
    raw = os.environ.get(name, default).replace(",", " ")
    values = [int(value) for value in raw.split()]
    if not values:
        raise ValueError(f"{name} must contain at least one integer")
    return values


def _open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def _code_identity() -> dict[str, Any]:
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "repository": str(_REPO_ROOT),
        "git_head": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "git_dirty": bool(status) if status is not None else None,
        "git_status_porcelain": status.splitlines() if status else [],
        "source_sha256": {
            relative: _sha256(_REPO_ROOT / relative) for relative in _IDENTITY_FILES
        },
    }


def _relevant_environment() -> dict[str, str]:
    exact = {
        "BENCH_IMPL",
        "BENCH_N",
        "BENCH_WS",
        "HIP_VISIBLE_DEVICES",
        "PROBE_CALLS_PER_GRAPH",
        "PROBE_MAX_REPLAYS",
        "PROBE_MS",
        "PROBE_REPLAYS",
        "PROBE_SEED",
        "PROBE_TIMEOUT_S",
        "TS_ARNORM_BACKEND",
    }
    prefixes = ("TS_TRITON_SHMEM_",)
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if key in exact or key.startswith(prefixes)
    }


def _validate_config() -> dict[str, Any]:
    impl = os.environ.get("BENCH_IMPL", "auto").strip().lower()
    if impl not in _SUPPORTED_IMPLS:
        raise ValueError(
            f"unsupported BENCH_IMPL={impl!r}; expected {sorted(_SUPPORTED_IMPLS)}"
        )
    world_size = _env_int("BENCH_WS", 4)
    hidden = _env_int("BENCH_N", 2880)
    ms = _env_numbers("PROBE_MS", "1,32,255,256,257,512,2048")
    if world_size < 2 or hidden < 1 or any(m < 1 for m in ms):
        raise ValueError("BENCH_WS must be >=2 and BENCH_N/PROBE_MS positive")
    if len(set(ms)) != len(ms):
        raise ValueError("PROBE_MS must not contain duplicates")

    call_counts = _env_numbers("PROBE_CALLS_PER_GRAPH", "1,2")
    if any(count < 1 for count in call_counts):
        raise ValueError("PROBE_CALLS_PER_GRAPH values must be positive")
    if not any(count % 2 for count in call_counts) or not any(
        count % 2 == 0 for count in call_counts
    ):
        raise ValueError("PROBE_CALLS_PER_GRAPH must cover odd and even counts")
    calls_by_m = {
        m: call_counts[index % len(call_counts)] for index, m in enumerate(ms)
    }
    if not any(count % 2 for count in calls_by_m.values()) or not any(
        count % 2 == 0 for count in calls_by_m.values()
    ):
        raise ValueError("the M grid must exercise both odd and even graph sizes")

    replay_limit = _env_int("PROBE_MAX_REPLAYS", 1000)
    replays = _env_int("PROBE_REPLAYS", 1000)
    if replay_limit < 1 or replay_limit > _HARD_MAX_REPLAYS:
        raise ValueError(f"PROBE_MAX_REPLAYS must be in [1, {_HARD_MAX_REPLAYS}]")
    minimum_replays = 3 * len(ms)
    if replays < minimum_replays:
        raise ValueError(
            f"PROBE_REPLAYS must be >= {minimum_replays} so every M is "
            "covered repeatedly and in the interleaved variant"
        )
    if replays > replay_limit:
        raise ValueError(
            f"PROBE_REPLAYS={replays} exceeds PROBE_MAX_REPLAYS={replay_limit}"
        )
    timeout_s = _env_int("PROBE_TIMEOUT_S", 3600)
    if timeout_s < 1:
        raise ValueError("PROBE_TIMEOUT_S must be positive")

    return {
        "impl": impl,
        "world_size": world_size,
        "hidden": hidden,
        "ms": ms,
        "max_token_num": max(ms),
        "calls_by_m": calls_by_m,
        "call_count_candidates": call_counts,
        "graph_replays": replays,
        "replay_limit": replay_limit,
        "absolute_replay_limit": _HARD_MAX_REPLAYS,
        "seed": _env_int("PROBE_SEED", 20260729),
        "timeout_s": timeout_s,
        "eps": _EPS,
        "atol": _ATOL,
        "rtol": _RTOL,
        "ordinary_all_reduce_max_bytes": _ORDINARY_AR_MAX_BYTES,
    }


def _build_sequence(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Build one deterministic sequence shared verbatim by every rank."""
    ms = list(config["ms"])
    graph_budget = int(config["graph_replays"])
    sequence: list[dict[str, Any]] = []
    graph_replay_index = 0

    def append(variant: str, mode: str, m: int) -> None:
        nonlocal graph_replay_index
        if mode == "graph":
            graph_replay_index += 1
            replay_index: int | None = graph_replay_index
        else:
            replay_index = None
        previous_mode = sequence[-1]["mode"] if sequence else None
        sequence.append(
            {
                "index": len(sequence),
                "variant": variant,
                "mode": mode,
                "M": m,
                "calls": config["calls_by_m"][m],
                "epoch": len(sequence) + 1,
                "transition": (
                    f"{previous_mode}->{mode}" if previous_mode else f"start->{mode}"
                ),
                "graph_replay_index": replay_index,
            }
        )

    # Consecutive same-M graph replays are bracketed by eager calls, providing
    # both transition directions for every captured graph.
    for m in ms:
        append("repeated", "eager", m)
        append("repeated", "graph", m)
        append("repeated", "graph", m)
        append("repeated", "eager", m)

    remaining = graph_budget - graph_replay_index
    rng = random.Random(config["seed"])
    interleaved_graphs = 0
    cycle = 0
    while interleaved_graphs < remaining:
        order = list(ms)
        rng.shuffle(order)
        for m in order:
            if interleaved_graphs >= remaining:
                break
            append("interleaved", "graph", m)
            interleaved_graphs += 1
        # Insert a changing-M eager call between graph cycles.  It is not part
        # of the bounded graph replay count.
        if interleaved_graphs < remaining:
            append("interleaved", "eager", ms[(cycle * 3 + 1) % len(ms)])
        cycle += 1

    if graph_replay_index != graph_budget:
        raise AssertionError("internal graph replay budget mismatch")
    return sequence


@dataclass
class ProbeCase:
    m: int
    calls: int
    xs: list[torch.Tensor]
    residuals: list[torch.Tensor]
    residual_templates: list[torch.Tensor]
    scratches: list[torch.Tensor]
    graph: torch.cuda.CUDAGraph | None = None
    graph_outputs: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)


def _residual_template(m: int, n: int, call: int, device: torch.device) -> torch.Tensor:
    rows = torch.arange(m, dtype=torch.int32, device=device).remainder(31)
    cols = torch.arange(n, dtype=torch.int32, device=device).remainder(257)
    values = (
        rows[:, None].float().mul_(1.0 / 32.0)
        + cols[None, :].float().sub_(128).mul_(1.0 / 128.0)
        + call * (1.0 / 16.0)
    )
    return values.to(torch.bfloat16)


def _create_cases(config: dict[str, Any], device: torch.device) -> dict[int, ProbeCase]:
    cases: dict[int, ProbeCase] = {}
    for m in config["ms"]:
        calls = config["calls_by_m"][m]
        xs = [
            torch.empty((m, config["hidden"]), dtype=torch.bfloat16, device=device)
            for _ in range(calls)
        ]
        templates = [
            _residual_template(m, config["hidden"], call, device)
            for call in range(calls)
        ]
        residuals = [template.clone() for template in templates]
        scratches = [torch.empty_like(x) for x in xs]
        cases[m] = ProbeCase(
            m=m,
            calls=calls,
            xs=xs,
            residuals=residuals,
            residual_templates=templates,
            scratches=scratches,
        )
    return cases


def _input_value(rank: int, call: int, epoch: int, m: int) -> float:
    phase = (epoch * 13 + m * 7 + call * 5) % 29
    sign = -1.0 if (epoch + call) % 2 else 1.0
    return sign * (0.5 * (rank + 1) + phase * 0.125 + call * 0.25)


def _set_case_inputs(case: ProbeCase, rank: int, epoch: int) -> None:
    residual_phase = ((epoch * 11 + case.m) % 17 - 8) * (1.0 / 64.0)
    for call, (x, residual, template) in enumerate(
        zip(case.xs, case.residuals, case.residual_templates)
    ):
        x.fill_(_input_value(rank, call, epoch, case.m))
        residual.copy_(template)
        residual.add_(residual_phase)


def _presize_iris_heap(max_token_num: int, hidden: int) -> None:
    from tokenspeed_kernel.ops.communication import iris as iris_mod

    fused_bytes = max_token_num * hidden * torch.bfloat16.itemsize
    heap_size = max(
        1 << 28,
        4 * (fused_bytes + _ORDINARY_AR_MAX_BYTES) + (64 << 20),
    )
    iris_mod._get_or_create_iris_context(heap_size)


def _precreate_backend(
    config: dict[str, Any],
    rank: int,
    group: dist.ProcessGroup,
    device: torch.device,
) -> tuple[Any, dict[str, Any]]:
    """Rendezvous the one shared max-M state before warmup or capture."""
    from tokenspeed_kernel.ops.communication import triton as tri

    impl = config["impl"]
    max_m = config["max_token_num"]
    hidden = config["hidden"]
    dtype = torch.bfloat16

    if impl == "production_unfused":
        from tokenspeed_kernel.ops.communication import iris as iris_mod

        _presize_iris_heap(max_m, hidden)
        ordinary_state = tri.create_state(
            group=group,
            rank_in_group=rank,
            device=device,
            max_numel=_ORDINARY_AR_MAX_BYTES // dtype.itemsize,
        )
        iris_key = (id(group), ordinary_state.max_numel, dtype)
        iris_state = iris_mod.create_iris_state(
            group=group,
            rank_in_group=rank,
            max_numel=ordinary_state.max_numel,
            dtype=dtype,
            device=device,
        )
        iris_mod.IRIS_AR_STATES[iris_key] = iris_state
        # Materialize the RCCL communicator before graph capture as well.
        rccl_probe = torch.ones(1, dtype=dtype, device=device)
        dist.all_reduce(rccl_probe, group=group)
        return ordinary_state, {
            str(m): (
                "ordinary_iris_all_reduce+tokenspeed_residual_rmsnorm"
                if m * hidden * dtype.itemsize <= _ORDINARY_AR_MAX_BYTES
                else "rccl_all_reduce+tokenspeed_residual_rmsnorm"
            )
            for m in config["ms"]
        }

    key = (id(group), max_m, hidden, dtype)
    if impl in ("auto", "iris"):
        from tokenspeed_kernel.ops.communication import iris as iris_mod

        _presize_iris_heap(max_m, hidden)
        state = iris_mod.create_iris_ar_rmsnorm_state(
            group=group,
            rank_in_group=rank,
            max_token_num=max_m,
            hidden_dim=hidden,
            dtype=dtype,
            device=device,
        )
        iris_mod.IRIS_AR_RMSNORM_STATES[key] = state
        return None, {
            str(m): "fused_iris_allreduce_residual_rmsnorm" for m in config["ms"]
        }

    if impl == "symm_mem":
        state = tri.allreduce_residual_rmsnorm_get_state(
            group=group,
            rank_in_group=rank,
            max_token_num=max_m,
            hidden_dim=hidden,
            device=device,
        )
        return None, {str(m): "fused_native_symm_mem" for m in config["ms"]}

    from tokenspeed_kernel.ops.communication import triton_shmem as ts

    state = ts.create_triton_shmem_ar_rmsnorm_state(
        group=group,
        rank_in_group=rank,
        max_token_num=max_m,
        hidden_dim=hidden,
        dtype=dtype,
        device=device,
    )
    if state is None:
        raise RuntimeError("triton_shmem state creation declined")
    ts.TRITON_SHMEM_AR_RMSNORM_STATES[key] = state
    paths: dict[str, str] = {}
    for m in config["ms"]:
        oneshot = (not state._is_twoshot) or (
            state._oneshot_max_m > 0 and m <= state._oneshot_max_m
        )
        paths[str(m)] = state._oneshot_kernel_for_m(m) if oneshot else "twoshot_blocked"
    return None, paths


def _launch(
    case: ProbeCase,
    config: dict[str, Any],
    rank: int,
    group: dist.ProcessGroup,
    weight: torch.Tensor,
    ordinary_state: Any,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    from tokenspeed_kernel.ops.communication import triton as tri
    from tokenspeed_kernel.ops.layernorm.triton import rmsnorm

    outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
    if config["impl"] == "production_unfused":
        use_iris = (
            case.m * config["hidden"] * torch.bfloat16.itemsize
            <= _ORDINARY_AR_MAX_BYTES
        )
        for x, residual, scratch in zip(case.xs, case.residuals, case.scratches):
            scratch.copy_(x)
            if use_iris:
                if not tri.all_reduce_can_run(ordinary_state, scratch):
                    raise RuntimeError(
                        f"ordinary Iris unexpectedly declined M={case.m}"
                    )
                tri.all_reduce(ordinary_state, scratch)
            else:
                dist.all_reduce(scratch, group=group)
            result = rmsnorm(
                scratch,
                weight,
                config["eps"],
                residual=residual,
            )
            if not isinstance(result, tuple) or len(result) != 2:
                raise RuntimeError("TokenSpeed residual RMSNorm lost an output")
            outputs.append(result)
        return outputs

    for x, residual in zip(case.xs, case.residuals):
        norm_out, residual_out, _, _ = tri.allreduce_residual_rmsnorm(
            input_tensor=x,
            residual=residual,
            weight=weight,
            rank=rank,
            group=group,
            eps=config["eps"],
            max_token_num=config["max_token_num"],
        )
        if norm_out is None or residual_out is None:
            raise RuntimeError(
                f"production fused dispatcher declined BENCH_IMPL="
                f"{config['impl']} M={case.m}"
            )
        outputs.append((norm_out, residual_out))
    return outputs


def _check_outputs(
    case: ProbeCase,
    outputs: list[tuple[torch.Tensor, torch.Tensor]],
    config: dict[str, Any],
    epoch: int,
) -> dict[str, Any]:
    if len(outputs) != case.calls:
        return {
            "pass": False,
            "error": f"returned {len(outputs)} outputs for {case.calls} calls",
            "calls": [],
        }

    call_results: list[dict[str, Any]] = []
    all_pass = True
    for call, (norm_out, residual_out) in enumerate(outputs):
        reduced_value = sum(
            float(
                torch.tensor(
                    _input_value(peer, call, epoch, case.m),
                    dtype=torch.bfloat16,
                ).item()
            )
            for peer in range(config["world_size"])
        )
        reference_residual = case.residuals[call].float() + reduced_value
        reference_norm = reference_residual * torch.rsqrt(
            reference_residual.pow(2).mean(dim=-1, keepdim=True) + config["eps"]
        )
        reference_norm *= config["_weight"].float()
        residual_error = (residual_out.float() - reference_residual).abs().max().item()
        norm_error = (norm_out.float() - reference_norm).abs().max().item()
        residual_ok = torch.allclose(
            residual_out.float(),
            reference_residual,
            atol=config["atol"],
            rtol=config["rtol"],
        )
        norm_ok = torch.allclose(
            norm_out.float(),
            reference_norm,
            atol=config["atol"],
            rtol=config["rtol"],
        )
        call_pass = bool(residual_ok and norm_ok)
        all_pass = all_pass and call_pass
        call_results.append(
            {
                "call": call,
                "pass": call_pass,
                "max_abs_residual_error": residual_error,
                "max_abs_norm_error": norm_error,
            }
        )
    return {"pass": all_pass, "calls": call_results}


def _capture_all_graphs(
    cases: dict[int, ProbeCase],
    config: dict[str, Any],
    rank: int,
    group: dist.ProcessGroup,
    weight: torch.Tensor,
    ordinary_state: Any,
) -> None:
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for case in cases.values():
            _launch(case, config, rank, group, weight, ordinary_state)
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()
    dist.barrier(group=group)

    pool = None
    for case in cases.values():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool, stream=capture_stream):
            graph_outputs = _launch(case, config, rank, group, weight, ordinary_state)
        if pool is None:
            pool = graph.pool()
        case.graph = graph
        # These references intentionally live for the entire probe.
        case.graph_outputs = graph_outputs
    torch.cuda.synchronize()
    dist.barrier(group=group)


def _rank_main(
    rank: int,
    config: dict[str, Any],
    sequence: list[dict[str, Any]],
    port: int,
) -> dict[str, Any]:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=config["world_size"],
    )
    group = dist.group.WORLD
    device = torch.device(f"cuda:{rank}")
    result: dict[str, Any] = {
        "rank": rank,
        "status": "running",
        "device": torch.cuda.get_device_name(device),
        "steps": [],
        "preflight": [],
    }
    try:
        os.environ["TS_ARNORM_BACKEND"] = config["impl"]
        cases = _create_cases(config, device)
        weight = torch.linspace(
            0.5,
            1.5,
            config["hidden"],
            dtype=torch.bfloat16,
            device=device,
        )
        config["_weight"] = weight

        ordinary_state, paths = _precreate_backend(config, rank, group, device)
        result["paths"] = paths
        if config["impl"] == "triton_shmem":
            result["signal_zero_status"] = "not_exposed_by_safe_public_api"

        # Exercise every state/path eagerly after all rendezvous work and before
        # graph capture.  Failures are retained without changing rank control flow.
        for index, case in enumerate(cases.values()):
            epoch = -(index + 1)
            _set_case_inputs(case, rank, epoch)
            outputs = _launch(case, config, rank, group, weight, ordinary_state)
            torch.cuda.synchronize()
            checked = _check_outputs(case, outputs, config, epoch)
            result["preflight"].append({"M": case.m, "epoch": epoch, **checked})

        gathered_paths: list[Any] = [None] * config["world_size"]
        dist.all_gather_object(gathered_paths, paths, group=group)
        path_agreement = all(item == gathered_paths[0] for item in gathered_paths)
        result["rank_path_agreement"] = path_agreement
        result["all_rank_paths"] = gathered_paths

        # Every graph is captured before the recorded safety sequence begins.
        for case in cases.values():
            _set_case_inputs(case, rank, epoch=0)
        _capture_all_graphs(cases, config, rank, group, weight, ordinary_state)
        result["captured_graphs"] = len(cases)
        result["graph_outputs_retained"] = True
        result["retained_graph_output_pairs"] = sum(
            len(case.graph_outputs) for case in cases.values()
        )

        failed_steps = 0
        checked_ops = 0
        for step in sequence:
            case = cases[step["M"]]
            _set_case_inputs(case, rank, step["epoch"])
            launch_error = None
            try:
                if step["mode"] == "graph":
                    if case.graph is None:
                        raise RuntimeError(f"M={case.m} graph is missing")
                    case.graph.replay()
                    outputs = case.graph_outputs
                else:
                    outputs = _launch(case, config, rank, group, weight, ordinary_state)
                torch.cuda.synchronize()
                checked = _check_outputs(case, outputs, config, step["epoch"])
            except Exception:
                launch_error = traceback.format_exc()
                checked = {"pass": False, "calls": []}
            checked_ops += len(checked["calls"])
            if not checked["pass"]:
                failed_steps += 1
            result["steps"].append(
                {
                    "index": step["index"],
                    "pass": checked["pass"],
                    "calls": checked["calls"],
                    "error": launch_error or checked.get("error"),
                }
            )

        preflight_pass = all(item["pass"] for item in result["preflight"])
        result.update(
            {
                "status": (
                    "pass"
                    if preflight_pass and path_agreement and failed_steps == 0
                    else "fail"
                ),
                "preflight_pass": preflight_pass,
                "steps_checked": len(sequence),
                "ops_checked": checked_ops,
                "failed_steps": failed_steps,
            }
        )
        return result
    finally:
        config.pop("_weight", None)
        os.environ.pop("TS_ARNORM_BACKEND", None)
        if dist.is_initialized():
            dist.destroy_process_group()


def _worker(
    rank: int,
    config: dict[str, Any],
    sequence: list[dict[str, Any]],
    port: int,
    rank_results: Any,
    rank_errors: Any,
) -> None:
    try:
        rank_results[rank] = _rank_main(rank, config, sequence, port)
    except BaseException as exc:  # Preserve rank-local evidence for parent JSON.
        rank_errors[rank] = {
            "rank": rank,
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        rank_results[rank] = {
            "rank": rank,
            "status": "error",
            "steps": [],
        }


def _run_spawn(
    config: dict[str, Any], sequence: list[dict[str, Any]]
) -> tuple[dict[int, Any], dict[int, Any], str | None]:
    manager = mp.Manager()
    rank_results = manager.dict()
    rank_errors = manager.dict()
    context = mp.spawn(
        _worker,
        args=(
            config,
            sequence,
            _open_port(),
            rank_results,
            rank_errors,
        ),
        nprocs=config["world_size"],
        join=False,
    )
    deadline = time.monotonic() + config["timeout_s"]
    spawn_error: str | None = None
    complete = False
    while time.monotonic() < deadline:
        try:
            if context.join(timeout=1):
                complete = True
                break
        except Exception:
            spawn_error = traceback.format_exc()
            complete = True
            break
    if not complete:
        spawn_error = f"probe exceeded hard timeout of {config['timeout_s']} seconds"
        for child in context.processes:
            if child.is_alive():
                child.terminate()
        for child in context.processes:
            child.join(timeout=5)
            if child.is_alive():
                child.kill()
        for child in context.processes:
            child.join(timeout=5)
        for rank in range(config["world_size"]):
            if rank not in rank_results:
                rank_errors[rank] = {
                    "rank": rank,
                    "type": "Timeout",
                    "message": spawn_error,
                    "traceback": None,
                }
                rank_results[rank] = {
                    "rank": rank,
                    "status": "timeout",
                    "steps": [],
                }
    return dict(rank_results), dict(rank_errors), spawn_error


def _write_json(payload: dict[str, Any], output: str | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(text, end="", flush=True)


def main() -> int:
    generated_at = datetime.now(timezone.utc).isoformat()
    try:
        config = _validate_config()
        sequence = _build_sequence(config)
    except Exception:
        payload = {
            "schema_version": 1,
            "status": "configuration_error",
            "generated_at": generated_at,
            "error": traceback.format_exc(),
            "code_identity": _code_identity(),
            "environment": _relevant_environment(),
            "safety_evidence_only": True,
            "performance_conclusions_allowed": False,
        }
        _write_json(payload, os.environ.get("PROBE_JSON"))
        return 2

    public_config = dict(config)
    if not torch.cuda.is_available() or torch.version.hip is None:
        runtime_error = "AMD ROCm CUDA-compatible runtime is required"
        rank_results: dict[int, Any] = {}
        rank_errors = {
            rank: {
                "rank": rank,
                "type": "RuntimeUnavailable",
                "message": runtime_error,
                "traceback": None,
            }
            for rank in range(config["world_size"])
        }
        spawn_error = runtime_error
    elif torch.cuda.device_count() < config["world_size"]:
        runtime_error = (
            f"BENCH_WS={config['world_size']} but only "
            f"{torch.cuda.device_count()} visible devices"
        )
        rank_results = {}
        rank_errors = {
            rank: {
                "rank": rank,
                "type": "InsufficientDevices",
                "message": runtime_error,
                "traceback": None,
            }
            for rank in range(config["world_size"])
        }
        spawn_error = runtime_error
    else:
        rank_results, rank_errors, spawn_error = _run_spawn(config, sequence)

    ordered_results = [
        rank_results.get(
            rank,
            {"rank": rank, "status": "missing", "steps": []},
        )
        for rank in range(config["world_size"])
    ]
    status = (
        "pass"
        if not rank_errors
        and spawn_error is None
        and len(rank_results) == config["world_size"]
        and all(result.get("status") == "pass" for result in ordered_results)
        else "fail"
    )
    payload = {
        "schema_version": 1,
        "probe": "amd_post_rebase_ar_rmsnorm_graph_eager_transitions",
        "status": status,
        "generated_at": generated_at,
        "safety_evidence_only": True,
        "performance_conclusions_allowed": False,
        "method": {
            "one_graph_per_M": True,
            "all_states_precreated_before_capture": True,
            "all_graphs_captured_before_sequence": True,
            "changing_inputs_every_step": True,
            "correctness_checked_every_step": True,
            "graph_outputs_retained": True,
            "rank_shared_sequence": True,
            "signal_pad_pointer_assumptions": False,
        },
        "configuration": public_config,
        "environment": _relevant_environment(),
        "runtime_identity": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_hip": torch.version.hip,
            "visible_device_count": (
                torch.cuda.device_count() if torch.cuda.is_available() else 0
            ),
        },
        "code_identity": _code_identity(),
        "sequence_sha256": hashlib.sha256(
            json.dumps(sequence, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "sequence": sequence,
        "per_rank_results": ordered_results,
        "per_rank_errors": [rank_errors[rank] for rank in sorted(rank_errors)],
        "spawn_error": spawn_error,
    }
    _write_json(payload, os.environ.get("PROBE_JSON"))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
