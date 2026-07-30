"""Benchmark post-rebase AMD ``all_reduce_two`` correctness and latency.

The paired arm uses one Iris launch when the combined BF16 payload is eligible;
otherwise it falls back to the same two ordinary production reductions used by
the control arm (each ordinary reduction selects Iris up to 512 KiB, RCCL
otherwise). Timings are reduced to the maximum rank for each iteration before
percentiles are computed.

Run from the repository root, for example::

    HIP_VISIBLE_DEVICES=1,2,3,5 \
      BENCH_WORLD_SIZES=2,4 \
      BENCH_SHAPE_PAIRS='1x7168+1x3584;64x7168+64x3584' \
      BENCH_N_WARMUP=30 BENCH_N_REPEAT=150 \
      BENCH_CSV=results/all_reduce_two.csv \
      BENCH_JSON=results/all_reduce_two.json \
      python -m benchmark.bench_all_reduce_two
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import socket
import traceback
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


_MAX_IRIS_BYTES = 512 * 1024
_DTYPE = torch.bfloat16
_DEFAULT_SHAPE_PAIRS = "1x7168+1x3584;64x7168+64x3584"


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    result = int(value) if value else default
    if result < 0:
        raise ValueError(f"{name} must be non-negative, got {result}")
    return result


def _parse_world_sizes() -> list[int]:
    raw = os.environ.get("BENCH_WORLD_SIZES", "2,4")
    sizes = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not sizes or any(size < 1 for size in sizes):
        raise ValueError("BENCH_WORLD_SIZES must contain positive integers")
    return sizes


def _parse_shape(text: str) -> tuple[int, ...]:
    dimensions = tuple(
        int(value.strip()) for value in text.strip().lower().split("x") if value.strip()
    )
    if not dimensions or any(dimension < 1 for dimension in dimensions):
        raise ValueError(f"invalid non-empty shape: {text!r}")
    return dimensions


def _parse_shape_pairs() -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    raw = os.environ.get("BENCH_SHAPE_PAIRS", _DEFAULT_SHAPE_PAIRS)
    pairs = []
    for entry in raw.split(";"):
        if not entry.strip():
            continue
        parts = entry.split("+")
        if len(parts) != 2:
            raise ValueError(
                "each BENCH_SHAPE_PAIRS entry must be FIRST+SECOND, "
                f"got {entry!r}"
            )
        pairs.append((_parse_shape(parts[0]), _parse_shape(parts[1])))
    if not pairs:
        raise ValueError("BENCH_SHAPE_PAIRS must contain at least one shape pair")
    return pairs


def _open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def _numel(shape: tuple[int, ...]) -> int:
    return math.prod(shape)


def _percentile(samples: list[float], percentile: float) -> float | None:
    if not samples:
        return None
    ordered = sorted(samples)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _make_inputs(
    first_shape: tuple[int, ...],
    second_shape: tuple[int, ...],
    rank: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    contribution = rank + 1
    first = torch.full(
        first_shape, contribution, dtype=_DTYPE, device=device
    )
    second = torch.full(
        second_shape, 3 * contribution, dtype=_DTYPE, device=device
    )
    return first, second


def _ordinary_path(
    triton_comm: Any,
    state: Any,
    tensor: torch.Tensor,
    iris_available: bool,
) -> str:
    if iris_available and triton_comm.all_reduce_can_run(state, tensor):
        return "iris"
    return "rccl"


def _ordinary_reduce(
    triton_comm: Any,
    state: Any,
    tensor: torch.Tensor,
    path: str,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    if path == "iris":
        return triton_comm.all_reduce(state, tensor)
    dist.all_reduce(tensor, group=group)
    return tensor


def _check_outputs(
    first: torch.Tensor,
    second: torch.Tensor,
    world_size: int,
    rank: int,
) -> dict[str, Any]:
    rank_sum = world_size * (world_size + 1) // 2
    expected_values = (rank_sum, 3 * rank_sum)
    metrics: dict[str, Any] = {"rank": rank, "errors": []}

    for name, output, expected_value in (
        ("first", first, expected_values[0]),
        ("second", second, expected_values[1]),
    ):
        expected_fp32 = torch.full(
            output.shape,
            expected_value,
            dtype=torch.float32,
            device=output.device,
        )
        expected_bf16 = expected_fp32.to(_DTYPE)
        fp32_error = float(
            (output.float() - expected_fp32).abs().max().item()
        )
        bf16_error = float(
            (output - expected_bf16).abs().float().max().item()
        )
        metrics[f"{name}_max_abs_error_fp32"] = fp32_error
        metrics[f"{name}_max_abs_error_bf16"] = bf16_error
        if not torch.equal(output, expected_bf16):
            metrics["errors"].append(
                f"{name} differs from the BF16 expected sum "
                f"(max_abs_error={bf16_error})"
            )
        if fp32_error != 0.0:
            metrics["errors"].append(
                f"{name} differs from the FP32 expected sum "
                f"(max_abs_error={fp32_error})"
            )
    return metrics


def _gather_objects(local: Any, world_size: int) -> list[Any]:
    gathered: list[Any] = [None] * world_size
    dist.all_gather_object(gathered, local)
    return gathered


def _time_iterations(
    operation: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    reset: Callable[[], None],
    warmup: int,
    repeat: int,
    world_size: int,
) -> tuple[list[list[float]], list[float]]:
    for _ in range(warmup):
        reset()
        operation()
    torch.cuda.synchronize()
    dist.barrier()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    for index in range(repeat):
        reset()
        starts[index].record()
        operation()
        ends[index].record()
    torch.cuda.synchronize()

    local_samples = [
        float(start.elapsed_time(end)) for start, end in zip(starts, ends)
    ]
    local_tensor = torch.tensor(local_samples, dtype=torch.float64, device="cuda")
    gathered_tensors = [torch.empty_like(local_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, local_tensor)
    rank_samples = [tensor.cpu().tolist() for tensor in gathered_tensors]
    max_rank_samples = [
        max(rank_samples[rank][iteration] for rank in range(world_size))
        for iteration in range(repeat)
    ]
    return rank_samples, max_rank_samples


def _run_arm(
    *,
    arm: str,
    rank: int,
    world_size: int,
    first_shape: tuple[int, ...],
    second_shape: tuple[int, ...],
    state: Any,
    triton_comm: Any,
    iris_available: bool,
    combined_eligible: bool,
    ordinary_paths: tuple[str, str],
    warmup: int,
    repeat: int,
) -> dict[str, Any] | None:
    device = torch.device(f"cuda:{rank}")
    group = dist.group.WORLD
    first_source, second_source = _make_inputs(
        first_shape, second_shape, rank, device
    )
    first = torch.empty_like(first_source)
    second = torch.empty_like(second_source)

    def reset() -> None:
        first.copy_(first_source)
        second.copy_(second_source)

    def ordinary_operation() -> tuple[torch.Tensor, torch.Tensor]:
        first_out = _ordinary_reduce(
            triton_comm, state, first, ordinary_paths[0], group
        )
        second_out = _ordinary_reduce(
            triton_comm, state, second, ordinary_paths[1], group
        )
        return first_out, second_out

    if arm == "all_reduce_two_dispatch" and combined_eligible:
        selected_path = "iris_all_reduce_two"

        def operation() -> tuple[torch.Tensor, torch.Tensor]:
            return triton_comm.all_reduce_two(state, first, second)

    else:
        prefix = "fallback:" if arm == "all_reduce_two_dispatch" else ""
        selected_path = prefix + "+".join(ordinary_paths)
        operation = ordinary_operation

    selected_paths = _gather_objects(selected_path, world_size)
    path_errors = []
    if len(set(selected_paths)) != 1:
        path_errors.append(f"selected-path rank disagreement: {selected_paths}")

    reset()
    first_out, second_out = operation()
    torch.cuda.synchronize()
    local_correctness = _check_outputs(
        first_out, second_out, world_size, rank
    )
    correctness = _gather_objects(local_correctness, world_size)
    correctness_errors = [
        f"rank {item['rank']}: {error}"
        for item in correctness
        for error in item["errors"]
    ]
    errors = path_errors + correctness_errors

    if errors:
        rank_samples: list[list[float]] = []
        max_rank_samples: list[float] = []
    else:
        rank_samples, max_rank_samples = _time_iterations(
            operation, reset, warmup, repeat, world_size
        )

    if rank != 0:
        return None

    return {
        "arm": arm,
        "selected_path": selected_paths[0]
        if len(set(selected_paths)) == 1
        else "rank_disagreement",
        "selected_paths_by_rank": selected_paths,
        "combined_eligible": combined_eligible,
        "ordinary_first_eligible": ordinary_paths[0] == "iris",
        "ordinary_second_eligible": ordinary_paths[1] == "iris",
        "world_size": world_size,
        "first_shape": list(first_shape),
        "second_shape": list(second_shape),
        "first_numel": _numel(first_shape),
        "second_numel": _numel(second_shape),
        "combined_numel": _numel(first_shape) + _numel(second_shape),
        "combined_bytes": (
            _numel(first_shape) + _numel(second_shape)
        )
        * torch.empty((), dtype=_DTYPE).element_size(),
        "dtype": str(_DTYPE),
        "warmup": warmup,
        "repeat": repeat,
        "rank_samples_ms": rank_samples,
        "max_rank_samples_ms": max_rank_samples,
        "p50_max_rank_ms": _percentile(max_rank_samples, 50),
        "p95_max_rank_ms": _percentile(max_rank_samples, 95),
        "p99_max_rank_ms": _percentile(max_rank_samples, 99),
        "correctness_by_rank": correctness,
        "errors": errors,
    }


def _worker_main(
    rank: int,
    world_size: int,
    port: int,
    shape_pairs: list[tuple[tuple[int, ...], tuple[int, ...]]],
    warmup: int,
    repeat: int,
    output_rows: Any,
) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        from tokenspeed_kernel.ops.communication import triton as triton_comm

        device = torch.device(f"cuda:{rank}")
        max_numel = _MAX_IRIS_BYTES // torch.empty(
            (), dtype=_DTYPE
        ).element_size()
        state = triton_comm.create_state(
            group=dist.group.WORLD,
            rank_in_group=dist.get_rank(),
            device=device,
            max_numel=max_numel,
        )
        iris_available = importlib.util.find_spec("iris") is not None

        for first_shape, second_shape in shape_pairs:
            first_probe, second_probe = _make_inputs(
                first_shape, second_shape, rank, device
            )
            combined_eligible = bool(
                iris_available
                and triton_comm.all_reduce_two_can_run(
                    state, first_probe, second_probe
                )
            )
            ordinary_paths = (
                _ordinary_path(
                    triton_comm, state, first_probe, iris_available
                ),
                _ordinary_path(
                    triton_comm, state, second_probe, iris_available
                ),
            )
            eligibility = _gather_objects(
                {
                    "combined": combined_eligible,
                    "ordinary_paths": ordinary_paths,
                },
                world_size,
            )
            if len({json.dumps(item, sort_keys=True) for item in eligibility}) != 1:
                if rank == 0:
                    output_rows.append(
                        {
                            "arm": "eligibility",
                            "selected_path": "rank_disagreement",
                            "world_size": world_size,
                            "first_shape": list(first_shape),
                            "second_shape": list(second_shape),
                            "rank_samples_ms": [],
                            "max_rank_samples_ms": [],
                            "errors": [
                                f"eligibility rank disagreement: {eligibility}"
                            ],
                        }
                    )
                continue

            for arm in ("all_reduce_two_dispatch", "two_ordinary"):
                row = _run_arm(
                    arm=arm,
                    rank=rank,
                    world_size=world_size,
                    first_shape=first_shape,
                    second_shape=second_shape,
                    state=state,
                    triton_comm=triton_comm,
                    iris_available=iris_available,
                    combined_eligible=combined_eligible,
                    ordinary_paths=ordinary_paths,
                    warmup=warmup,
                    repeat=repeat,
                )
                if row is not None:
                    row["iris_available"] = iris_available
                    output_rows.append(row)
    finally:
        dist.destroy_process_group()


def _worker(
    rank: int,
    world_size: int,
    port: int,
    shape_pairs: list[tuple[tuple[int, ...], tuple[int, ...]]],
    warmup: int,
    repeat: int,
    output_rows: Any,
    worker_errors: Any,
) -> None:
    try:
        _worker_main(
            rank,
            world_size,
            port,
            shape_pairs,
            warmup,
            repeat,
            output_rows,
        )
    except Exception:
        worker_errors.append(
            {
                "world_size": world_size,
                "rank": rank,
                "error": traceback.format_exc(),
            }
        )
        raise


def _write_json(
    path: Path,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    worker_errors: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"config": config, "results": rows, "worker_errors": worker_errors},
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    worker_errors: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "world_size",
        "first_shape",
        "second_shape",
        "arm",
        "selected_path",
        "combined_eligible",
        "ordinary_first_eligible",
        "ordinary_second_eligible",
        "iteration",
        "rank",
        "rank_sample_ms",
        "iteration_max_rank_ms",
        "p50_max_rank_ms",
        "p95_max_rank_ms",
        "p99_max_rank_ms",
        "errors",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            rank_samples = row.get("rank_samples_ms", [])
            max_samples = row.get("max_rank_samples_ms", [])
            base = {
                "world_size": row.get("world_size"),
                "first_shape": "x".join(map(str, row.get("first_shape", []))),
                "second_shape": "x".join(map(str, row.get("second_shape", []))),
                "arm": row.get("arm"),
                "selected_path": row.get("selected_path"),
                "combined_eligible": row.get("combined_eligible"),
                "ordinary_first_eligible": row.get("ordinary_first_eligible"),
                "ordinary_second_eligible": row.get("ordinary_second_eligible"),
                "p50_max_rank_ms": row.get("p50_max_rank_ms"),
                "p95_max_rank_ms": row.get("p95_max_rank_ms"),
                "p99_max_rank_ms": row.get("p99_max_rank_ms"),
                "errors": json.dumps(row.get("errors", [])),
            }
            if not rank_samples:
                writer.writerow(base)
                continue
            for rank, samples in enumerate(rank_samples):
                for iteration, sample in enumerate(samples):
                    writer.writerow(
                        {
                            **base,
                            "iteration": iteration,
                            "rank": rank,
                            "rank_sample_ms": sample,
                            "iteration_max_rank_ms": max_samples[iteration],
                        }
                    )
        for error in worker_errors:
            writer.writerow(
                {
                    "world_size": error["world_size"],
                    "rank": error["rank"],
                    "arm": "worker",
                    "errors": json.dumps([error["error"]]),
                }
            )


def main() -> None:
    world_sizes = _parse_world_sizes()
    shape_pairs = _parse_shape_pairs()
    warmup = _env_int("BENCH_N_WARMUP", 30)
    repeat = _env_int("BENCH_N_REPEAT", 150)
    if repeat < 1:
        raise ValueError("BENCH_N_REPEAT must be at least 1")

    csv_path = Path(
        os.environ.get("BENCH_CSV", "benchmark/results/all_reduce_two.csv")
    )
    json_path = Path(
        os.environ.get("BENCH_JSON", "benchmark/results/all_reduce_two.json")
    )
    config = {
        "world_sizes": world_sizes,
        "shape_pairs": [
            [list(first), list(second)] for first, second in shape_pairs
        ],
        "warmup": warmup,
        "repeat": repeat,
        "iris_max_bytes": _MAX_IRIS_BYTES,
        "csv": str(csv_path),
        "json": str(json_path),
    }

    with mp.Manager() as manager:
        output_rows = manager.list()
        worker_errors = manager.list()
        for world_size in world_sizes:
            if world_size > torch.cuda.device_count():
                worker_errors.append(
                    {
                        "world_size": world_size,
                        "rank": None,
                        "error": (
                            f"requested {world_size} ranks but only "
                            f"{torch.cuda.device_count()} CUDA/ROCm devices are visible"
                        ),
                    }
                )
                continue
            try:
                mp.spawn(
                    _worker,
                    args=(
                        world_size,
                        _open_port(),
                        shape_pairs,
                        warmup,
                        repeat,
                        output_rows,
                        worker_errors,
                    ),
                    nprocs=world_size,
                    join=True,
                )
            except Exception as error:
                worker_errors.append(
                    {
                        "world_size": world_size,
                        "rank": None,
                        "error": f"mp.spawn failed: {error}",
                    }
                )

        rows = list(output_rows)
        errors = list(worker_errors)

    rows.sort(
        key=lambda row: (
            row.get("world_size", -1),
            row.get("first_shape", []),
            row.get("second_shape", []),
            row.get("arm", ""),
        )
    )
    _write_csv(csv_path, rows, errors)
    _write_json(json_path, config, rows, errors)

    for row in rows:
        percentiles = [
            (
                f"{value:.6f}"
                if (value := row.get(f"p{percentile}_max_rank_ms")) is not None
                else "n/a"
            )
            for percentile in (50, 95, 99)
        ]
        print(
            f"ws={row.get('world_size')} "
            f"shapes={row.get('first_shape')}+{row.get('second_shape')} "
            f"arm={row.get('arm')} path={row.get('selected_path')} "
            f"p50/p95/p99(ms)={'/'.join(percentiles)} "
            f"errors={len(row.get('errors', []))}"
        )
    if errors:
        print(f"worker_errors={len(errors)} (see {json_path})")
    print(f"wrote {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
