"""Analyze AR+RMSNorm by authoritative TokenSpeed model-forward markers."""
from __future__ import annotations

import argparse
import gzip
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

_MARKER_PREFIX = "tokenspeed.model_forward.v1|"


def _load(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _rank(path: Path, payload: dict[str, Any]) -> int:
    distributed = payload.get("distributedInfo", {})
    if isinstance(distributed.get("rank"), int):
        return int(distributed["rank"])
    match = re.search(r"rank(\d+)", path.name, re.IGNORECASE)
    if match is None:
        match = re.search(r"TP(\d+)", path.name, re.IGNORECASE)
    if match is None:
        raise ValueError(f"cannot determine rank from {path}")
    return int(match.group(1))


def _parse_marker(name: str) -> dict[str, Any]:
    if not name.startswith(_MARKER_PREFIX):
        raise ValueError(f"not a forward marker: {name}")
    values = {}
    for item in name[len(_MARKER_PREFIX) :].split("|"):
        key, value = item.split("=", 1)
        values[key] = value
    required = {
        "id",
        "mode",
        "actual_m",
        "executed_m",
        "bs",
        "padded_bs",
        "num_extends",
        "execution",
    }
    missing = required - values.keys()
    if missing:
        raise ValueError(f"forward marker missing {sorted(missing)}")
    for key in ("id", "actual_m", "executed_m", "bs", "padded_bs", "num_extends"):
        values[key] = int(values[key])
    return values


def _kernel_breakdown(events: list[dict[str, Any]]) -> dict[str, Any]:
    names = [str(event.get("name", "")) for event in events]

    def count(token: str) -> int:
        return sum(token in name for name in names)

    def duration(token: str) -> float:
        return sum(
            float(event["dur"])
            for event, name in zip(events, names)
            if token in name
        )

    counts = {
        "fused_oneshot": count("fused_ar_rmsnorm_oneshot"),
        "fused_twoshot": count("fused_ar_rmsnorm_twoshot"),
        "unfused_native_ar": count("amd_all_reduce_kernel"),
        "rmsnorm": count("_rmsnorm_kernel"),
        "barrier": count("symm_grid_barrier_kernel"),
        "rccl_family": count("ncclDevKernel"),
    }
    if counts["fused_oneshot"]:
        primary = "fused_oneshot"
    elif counts["fused_twoshot"]:
        primary = "fused_twoshot"
    elif counts["unfused_native_ar"] and counts["rmsnorm"]:
        primary = "unfused_native"
    elif counts["rccl_family"] and counts["rmsnorm"]:
        primary = "unfused_rccl_candidate"
    else:
        primary = "unknown"
    target_duration = sum(
        duration(token)
        for token in (
            "fused_ar_rmsnorm_",
            "amd_all_reduce_kernel",
            "_rmsnorm_kernel",
            "symm_grid_barrier_kernel",
        )
    )
    return {
        "primary": primary,
        "counts": counts,
        "target_kernel_sum_us": target_duration,
        "kernel_count": len(events),
    }


def _analyze_trace(path: Path) -> dict[str, Any]:
    payload = _load(path)
    trace_events = payload.get("traceEvents", [])
    rank = _rank(path, payload)
    markers = [
        event
        for event in trace_events
        if event.get("cat") == "user_annotation"
        and event.get("ph") == "X"
        and str(event.get("name", "")).startswith(_MARKER_PREFIX)
    ]
    if not markers:
        raise ValueError(f"exact forward markers not found in {path}")

    runtime = [
        event
        for event in trace_events
        if event.get("cat") == "cuda_runtime"
        and isinstance(event.get("args", {}).get("correlation"), int)
        and isinstance(event.get("ts"), (int, float))
        and isinstance(event.get("dur"), (int, float))
    ]
    gpu_by_correlation: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in trace_events:
        correlation = event.get("args", {}).get("correlation")
        if (
            event.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}
            and isinstance(correlation, int)
            and isinstance(event.get("dur"), (int, float))
        ):
            gpu_by_correlation[correlation].append(event)

    forwards = []
    seen_ids = set()
    for marker in sorted(markers, key=lambda event: float(event["ts"])):
        metadata = _parse_marker(str(marker["name"]))
        if metadata["id"] in seen_ids:
            raise ValueError(f"duplicate forward id {metadata['id']} in {path}")
        seen_ids.add(metadata["id"])
        start = float(marker["ts"])
        end = start + float(marker["dur"])
        correlations = {
            int(event["args"]["correlation"])
            for event in runtime
            if event.get("pid") == marker.get("pid")
            and event.get("tid") == marker.get("tid")
            and float(event["ts"]) >= start
            and float(event["ts"]) + float(event["dur"]) <= end
        }
        gpu_events = [
            event
            for correlation in correlations
            for event in gpu_by_correlation.get(correlation, [])
        ]
        if not gpu_events:
            raise ValueError(
                f"forward {metadata['id']} rank {rank} has no GPU activities"
            )
        gpu_start = min(float(event["ts"]) for event in gpu_events)
        gpu_end = max(
            float(event["ts"]) + float(event["dur"]) for event in gpu_events
        )
        forwards.append(
            {
                **metadata,
                "rank": rank,
                "gpu_period_us": gpu_end - gpu_start,
                "gpu_kernel_sum_us": sum(
                    float(event["dur"])
                    for event in gpu_events
                    if event.get("cat") == "kernel"
                ),
                "runtime_correlation_count": len(correlations),
                "path": _kernel_breakdown(
                    [
                        event
                        for event in gpu_events
                        if event.get("cat") == "kernel"
                    ]
                ),
            }
        )
    return {"path": str(path), "rank": rank, "forwards": forwards}


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "p95": _percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def analyze(
    paths: list[Path],
    *,
    expected_world_size: int | None,
    mode: str,
) -> dict[str, Any]:
    traces = [_analyze_trace(path) for path in paths]
    ranks = [trace["rank"] for trace in traces]
    if len(ranks) != len(set(ranks)):
        raise ValueError("duplicate ranks in input traces")
    if expected_world_size is not None and sorted(ranks) != list(
        range(expected_world_size)
    ):
        raise ValueError(
            f"expected ranks 0..{expected_world_size - 1}, got {sorted(ranks)}"
        )

    by_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        for forward in trace["forwards"]:
            if mode == "all" or forward["mode"] == mode:
                by_id[forward["id"]].append(forward)

    forwards = []
    for forward_id, records in sorted(by_id.items()):
        if len(records) != len(traces):
            raise ValueError(f"forward {forward_id} is missing one or more ranks")
        identity = {
            (
                record["mode"],
                record["actual_m"],
                record["executed_m"],
                record["bs"],
                record["padded_bs"],
                record["execution"],
            )
            for record in records
        }
        if len(identity) != 1:
            raise ValueError(f"forward {forward_id} metadata differs across ranks")
        max_period = max(records, key=lambda record: record["gpu_period_us"])
        max_comm = max(
            records,
            key=lambda record: record["path"]["target_kernel_sum_us"],
        )
        forwards.append(
            {
                "forward_id": forward_id,
                "mode": records[0]["mode"],
                "actual_m": records[0]["actual_m"],
                "executed_m": records[0]["executed_m"],
                "batch_size": records[0]["bs"],
                "padded_batch_size": records[0]["padded_bs"],
                "execution": records[0]["execution"],
                "primary_paths": sorted(
                    {record["path"]["primary"] for record in records}
                ),
                "ranks": records,
                "max_rank": {
                    "gpu_period_us": {
                        "value": max_period["gpu_period_us"],
                        "rank": max_period["rank"],
                    },
                    "target_kernel_sum_us": {
                        "value": max_comm["path"]["target_kernel_sum_us"],
                        "rank": max_comm["rank"],
                    },
                },
            }
        )

    cohorts: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for forward in forwards:
        key = (
            forward["mode"],
            forward["executed_m"],
            tuple(forward["primary_paths"]),
        )
        cohorts[key].append(forward)
    cohort_rows = []
    for key, records in cohorts.items():
        cohort_rows.append(
            {
                "mode": key[0],
                "executed_m": key[1],
                "primary_paths": list(key[2]),
                "count": len(records),
                "max_rank_gpu_period_us": _stats(
                    [
                        record["max_rank"]["gpu_period_us"]["value"]
                        for record in records
                    ]
                ),
                "max_rank_target_kernel_sum_us": _stats(
                    [
                        record["max_rank"]["target_kernel_sum_us"]["value"]
                        for record in records
                    ]
                ),
            }
        )
    return {
        "schema_version": 1,
        "analysis": "tokenspeed.ar_rmsnorm_model_forwards",
        "validation": {"status": "ok", "warnings": []},
        "inputs": [
            {"path": trace["path"], "rank": trace["rank"]} for trace in traces
        ],
        "forwards": forwards,
        "cohorts": cohort_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--mode",
        choices=("all", "decode", "extend", "mixed", "idle"),
        default="all",
    )
    parser.add_argument("--expected-world-size", type=int)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    result = analyze(
        args.traces,
        expected_world_size=args.expected_world_size,
        mode=args.mode,
    )
    if args.summary_only:
        result.pop("forwards", None)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
