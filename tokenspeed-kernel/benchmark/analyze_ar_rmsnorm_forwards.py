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

    standalone_rmsnorm = sum(
        "_rmsnorm_kernel" in name
        and "iris_allreduce_residual_rmsnorm_kernel" not in name
        for name in names
    )
    counts = {
        "triton_shmem_fused_oneshot": count("fused_ar_rmsnorm_oneshot"),
        "triton_shmem_fused_twoshot": count("fused_ar_rmsnorm_twoshot"),
        "iris_fused": count("iris_allreduce_residual_rmsnorm_kernel"),
        "symm_mem_fused": count("amd_allreduce_residual_rmsnorm_kernel"),
        "ordinary_iris": count("iris_stage_one_shot_allreduce_kernel"),
        "unfused_native_ar": count("amd_all_reduce_kernel"),
        "standalone_rmsnorm": standalone_rmsnorm,
        "barrier": count("symm_grid_barrier_kernel"),
        "rccl_family": count("ncclDevKernel"),
    }
    if counts["iris_fused"]:
        primary = "iris_fused"
    elif counts["triton_shmem_fused_oneshot"]:
        primary = "triton_shmem_fused_oneshot"
    elif counts["triton_shmem_fused_twoshot"]:
        primary = "triton_shmem_fused_twoshot"
    elif counts["symm_mem_fused"]:
        primary = "symm_mem_fused"
    elif counts["ordinary_iris"] and counts["standalone_rmsnorm"]:
        primary = "unfused_iris"
    elif counts["unfused_native_ar"] and counts["standalone_rmsnorm"]:
        primary = "unfused_native"
    elif counts["rccl_family"] and counts["standalone_rmsnorm"]:
        primary = "unfused_rccl"
    else:
        primary = "unknown"
    target_tokens = (
        "fused_ar_rmsnorm_",
        "iris_allreduce_residual_rmsnorm_kernel",
        "amd_allreduce_residual_rmsnorm_kernel",
        "iris_stage_one_shot_allreduce_kernel",
        "amd_all_reduce_kernel",
        "_rmsnorm_kernel",
        "symm_grid_barrier_kernel",
    )
    target_duration = sum(
        float(event["dur"])
        for event, name in zip(events, names)
        if any(token in name for token in target_tokens)
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
        gpu_end = max(float(event["ts"]) + float(event["dur"]) for event in gpu_events)
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
                    [event for event in gpu_events if event.get("cat") == "kernel"]
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


def _micro_link(
    summary: dict[str, Any] | None,
    *,
    world_size: int | None,
    executed_m: int,
    arm: str | None,
) -> dict[str, Any] | None:
    if summary is None or world_size is None or arm is None:
        return None
    row = next(
        (
            value
            for value in summary.get("rows", [])
            if int(value["world_size"]) == world_size
            and int(value["M"]) == executed_m
            and value["arm"] == arm
        ),
        None,
    )
    comparison = next(
        (
            value
            for value in summary.get("comparisons", [])
            if int(value["world_size"]) == world_size and int(value["M"]) == executed_m
        ),
        None,
    )
    if row is None:
        return None
    p50 = row.get("p50_us_per_site", row.get("p50_us"))
    return {
        "mode": summary.get("mode", "graph"),
        "world_size": world_size,
        "M": executed_m,
        "arm": arm,
        "expected_path": row.get("expected_path"),
        "p50_us_per_site": p50,
        "comparison": comparison,
    }


def analyze(
    paths: list[Path],
    *,
    expected_world_size: int | None,
    mode: str,
    graph_summary: dict[str, Any] | None = None,
    eager_summary: dict[str, Any] | None = None,
    arm: str | None = None,
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
        primary_paths = sorted({record["path"]["primary"] for record in records})
        if len(primary_paths) != 1:
            raise ValueError(
                f"forward {forward_id} backend path differs across ranks: "
                f"{primary_paths}"
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
                "primary_paths": primary_paths,
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
    weighted_micro_changes = []
    for key, records in cohorts.items():
        execution = records[0]["execution"]
        micro_summary = graph_summary if "graph" in execution else eager_summary
        micro = _micro_link(
            micro_summary,
            world_size=expected_world_size,
            executed_m=key[1],
            arm=arm,
        )
        cohort = {
            "mode": key[0],
            "execution": execution,
            "executed_m": key[1],
            "primary_paths": list(key[2]),
            "count": len(records),
            "max_rank_gpu_period_us": _stats(
                [record["max_rank"]["gpu_period_us"]["value"] for record in records]
            ),
            "max_rank_target_kernel_sum_us": _stats(
                [
                    record["max_rank"]["target_kernel_sum_us"]["value"]
                    for record in records
                ]
            ),
            "microbenchmark": micro,
        }
        cohort_rows.append(cohort)
        if micro and micro["comparison"]:
            field = (
                "iris_vs_unfused_pct"
                if arm == "iris_fused"
                else "triton_vs_unfused_adjusted_pct"
                if "graph" in execution
                else "triton_vs_unfused_pct"
            )
            change = micro["comparison"].get(field)
            if change is not None:
                weighted_micro_changes.extend([float(change)] * len(records))
    return {
        "schema_version": 1,
        "analysis": "tokenspeed.ar_rmsnorm_model_forwards",
        "validation": {"status": "ok", "warnings": []},
        "inputs": [{"path": trace["path"], "rank": trace["rank"]} for trace in traces],
        "forwards": forwards,
        "cohorts": cohort_rows,
        "microbenchmark_linkage": {
            "arm": arm,
            "linked_cohorts": sum(
                cohort["microbenchmark"] is not None for cohort in cohort_rows
            ),
            "total_cohorts": len(cohort_rows),
            "forward_weighted_candidate_vs_unfused_pct": (
                statistics.fmean(weighted_micro_changes)
                if weighted_micro_changes
                else None
            ),
        },
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
    parser.add_argument(
        "--arm",
        choices=("upstream_unfused", "iris_fused", "triton_profile"),
    )
    parser.add_argument("--graph-summary", type=Path)
    parser.add_argument("--eager-summary", type=Path)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    result = analyze(
        args.traces,
        expected_world_size=args.expected_world_size,
        mode=args.mode,
        graph_summary=_load(args.graph_summary) if args.graph_summary else None,
        eager_summary=_load(args.eager_summary) if args.eager_summary else None,
        arm=args.arm,
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
