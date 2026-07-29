"""Heuristically reconstruct model forwards in legacy TokenSpeed traces.

Correlation IDs are HIP graph-launch units, not model-forward units. A single
model forward can appear as one 72-site graph or as 36 two-site breakable graph
segments. Legacy traces contain no authoritative forward ID, mode, or M marker,
so site-count accumulation remains heuristic and must not be labeled exact.
New traces should use ``tokenspeed.model_forward.v1`` markers and the dedicated
forward analyzer.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _stats(values: list[float]) -> dict:
    if not values:
        return {}
    return {
        "median_us": statistics.median(values),
        "p95_us": _percentile(values, 0.95),
        "min_us": min(values),
        "max_us": max(values),
    }


def _is_comm_kernel(name: str) -> bool:
    return any(
        token in name
        for token in (
            "fused_ar_rmsnorm_",
            "amd_all_reduce_kernel",
            "_rmsnorm_kernel",
            "symm_grid_barrier_kernel",
        )
    )


def _rank_from_path(path: Path) -> int | None:
    match = re.search(r"rank(\d+)", path.name, re.IGNORECASE)
    if match is None:
        match = re.search(r"TP(\d+)", path.name, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _correlation_groups(path: Path) -> list[dict[str, Any]]:
    groups: dict[int, list[dict]] = defaultdict(list)
    for event in _load(path).get("traceEvents", []):
        args = event.get("args", {})
        correlation = args.get("correlation")
        if (
            event.get("cat") == "kernel"
            and isinstance(correlation, int)
            and isinstance(event.get("dur"), (int, float))
        ):
            groups[correlation].append(event)

    records = []
    for correlation, events in groups.items():
        names = [str(event.get("name", "")) for event in events]
        start = min(float(event["ts"]) for event in events)
        end = max(float(event["ts"]) + float(event["dur"]) for event in events)
        target_sum = sum(
            float(event["dur"])
            for event, name in zip(events, names)
            if _is_comm_kernel(name)
        )
        rccl_sum = sum(
            float(event["dur"])
            for event, name in zip(events, names)
            if "ncclDevKernel" in name
        )
        fused_sum = sum(
            float(event["dur"])
            for event, name in zip(events, names)
            if "fused_ar_rmsnorm_" in name
        )
        triton_ar_sum = sum(
            float(event["dur"])
            for event, name in zip(events, names)
            if "amd_all_reduce_kernel" in name
        )
        rmsnorm_sum = sum(
            float(event["dur"])
            for event, name in zip(events, names)
            if "_rmsnorm_kernel" in name
        )
        records.append(
            {
                "correlation": correlation,
                "start_us": start,
                "end_us": end,
                "gpu_span_us": end - start,
                "kernel_sum_us": sum(float(event["dur"]) for event in events),
                "comm_sum_us": target_sum,
                "rccl_sum_us": rccl_sum,
                "collective_sum_us": target_sum + rccl_sum,
                "fused_sum_us": fused_sum,
                "triton_ar_sum_us": triton_ar_sum,
                "rmsnorm_sum_us": rmsnorm_sum,
                "kernel_count": len(events),
                "oneshot_count": sum(
                    "fused_ar_rmsnorm_oneshot" in name for name in names
                ),
                "twoshot_count": sum(
                    "fused_ar_rmsnorm_twoshot" in name for name in names
                ),
                "all_reduce_count": sum(
                    "amd_all_reduce_kernel" in name for name in names
                ),
                "rmsnorm_count": sum("_rmsnorm_kernel" in name for name in names),
                "rccl_count": sum("ncclDevKernel" in name for name in names),
            }
        )
    return sorted(records, key=lambda record: record["start_us"])


def _forward(
    groups: list[dict[str, Any]],
    *,
    mode: str,
    path: str,
    m: int | None,
) -> dict[str, Any]:
    start = min(group["start_us"] for group in groups)
    end = max(group["end_us"] for group in groups)
    return {
        "mode": mode,
        "path": path,
        "M": m,
        "M_source": "cli_hint" if m is not None else "unavailable_in_kernel_trace",
        "start_us": start,
        "end_us": end,
        "gpu_span_us": end - start,
        "period_us": None,
        "kernel_sum_us": sum(group["kernel_sum_us"] for group in groups),
        "comm_sum_us": sum(group["comm_sum_us"] for group in groups),
        "rccl_sum_us": sum(group["rccl_sum_us"] for group in groups),
        "collective_sum_us": sum(
            group["collective_sum_us"] for group in groups
        ),
        "fused_sum_us": sum(group["fused_sum_us"] for group in groups),
        "triton_ar_sum_us": sum(
            group["triton_ar_sum_us"] for group in groups
        ),
        "rmsnorm_sum_us": sum(
            group["rmsnorm_sum_us"] for group in groups
        ),
        "correlations": [group["correlation"] for group in groups],
        "kernel_count": sum(group["kernel_count"] for group in groups),
        "oneshot_count": sum(group["oneshot_count"] for group in groups),
        "twoshot_count": sum(group["twoshot_count"] for group in groups),
        "all_reduce_count": sum(
            group["all_reduce_count"] for group in groups
        ),
        "rmsnorm_count": sum(group["rmsnorm_count"] for group in groups),
        "rccl_count": sum(group["rccl_count"] for group in groups),
    }


def _merge_counted_groups(
    groups: list[dict[str, Any]],
    *,
    count_key: str,
    target: int,
    mode: str,
    path: str,
    m: int | None,
) -> tuple[list[dict[str, Any]], int]:
    selected = [group for group in groups if group[count_key] > 0]
    forwards = []
    current: list[dict[str, Any]] = []
    count = 0
    for group in selected:
        if count + group[count_key] > target:
            raise ValueError(
                f"{path} group crosses {target}-site boundary: "
                f"current={count}, next={group[count_key]}, "
                f"correlation={group['correlation']}"
            )
        current.append(group)
        count += group[count_key]
        if count == target:
            forwards.append(_forward(current, mode=mode, path=path, m=m))
            current = []
            count = 0
    return forwards, count


def _merge_unfused_prefill(
    groups: list[dict[str, Any]],
    *,
    target: int,
    m: int | None,
) -> tuple[list[dict[str, Any]], int]:
    """Merge RCCL/RMSNorm groups while ignoring fused/final-pair leftovers."""
    selected = [
        group
        for group in groups
        if group["oneshot_count"] == 0
        and group["twoshot_count"] == 0
        and group["all_reduce_count"] == 0
        and (group["rmsnorm_count"] > 0 or group["rccl_count"] > 0)
    ]
    forwards = []
    current: list[dict[str, Any]] = []
    rmsnorm_count = 0
    for group in selected:
        # RCCL-only groups following the 73rd norm still belong to the current
        # prefill. The next RMSNorm-bearing group starts the next forward.
        if rmsnorm_count == target and group["rmsnorm_count"] > 0:
            forwards.append(
                _forward(
                    current,
                    mode="prefill",
                    path="unfused_rccl_rmsnorm",
                    m=m,
                )
            )
            current = []
            rmsnorm_count = 0
        if rmsnorm_count + group["rmsnorm_count"] > target:
            raise ValueError(
                "unfused prefill RMSNorm group crosses site boundary"
            )
        current.append(group)
        rmsnorm_count += group["rmsnorm_count"]
    if rmsnorm_count == target:
        forwards.append(
            _forward(
                current,
                mode="prefill",
                path="unfused_rccl_rmsnorm",
                m=m,
            )
        )
        rmsnorm_count = 0
    return forwards, rmsnorm_count


def _attach_periods(forwards: list[dict[str, Any]]) -> None:
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for forward in forwards:
        by_mode[forward["mode"]].append(forward)
    for mode_forwards in by_mode.values():
        mode_forwards.sort(key=lambda record: record["start_us"])
        for current, following in zip(mode_forwards, mode_forwards[1:]):
            current["period_us"] = following["start_us"] - current["start_us"]


def _forward_summary(forwards: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for mode in ("prefill", "decode"):
        records = [record for record in forwards if record["mode"] == mode]
        if not records:
            continue
        paths = sorted({record["path"] for record in records})
        periods = [
            record["period_us"]
            for record in records
            if record["period_us"] is not None
        ]
        summary[mode] = {
            "count": len(records),
            "paths": paths,
            "M_values": sorted(
                {
                    record["M"]
                    for record in records
                    if record["M"] is not None
                }
            ),
            "period_us": _stats(periods),
            "gpu_span_us": _stats(
                [record["gpu_span_us"] for record in records]
            ),
            "comm_sum_us": _stats(
                [record["comm_sum_us"] for record in records]
            ),
            "rccl_sum_us": _stats(
                [record["rccl_sum_us"] for record in records]
            ),
            "collective_sum_us": _stats(
                [record["collective_sum_us"] for record in records]
            ),
            "fused_sum_us": _stats(
                [record["fused_sum_us"] for record in records]
            ),
            "triton_ar_sum_us": _stats(
                [record["triton_ar_sum_us"] for record in records]
            ),
            "rmsnorm_sum_us": _stats(
                [record["rmsnorm_sum_us"] for record in records]
            ),
            "kernel_sum_us": _stats(
                [record["kernel_sum_us"] for record in records]
            ),
        }
    return summary


def _summarize(
    path: Path,
    *,
    eligible_sites: int,
    unfused_sites: int,
    decode_m: int | None,
    prefill_m: int | None,
) -> dict[str, Any]:
    groups = _correlation_groups(path)
    warnings = []
    forwards = []

    fused_decode, leftover = _merge_counted_groups(
        groups,
        count_key="oneshot_count",
        target=eligible_sites,
        mode="decode",
        path="fused_oneshot",
        m=decode_m,
    )
    forwards.extend(fused_decode)
    if leftover:
        warnings.append(f"incomplete fused decode: {leftover}/{eligible_sites} sites")

    fused_prefill, leftover = _merge_counted_groups(
        groups,
        count_key="twoshot_count",
        target=eligible_sites,
        mode="prefill",
        path="fused_twoshot",
        m=prefill_m,
    )
    forwards.extend(fused_prefill)
    if leftover:
        warnings.append(
            f"incomplete fused prefill: {leftover}/{eligible_sites} sites"
        )

    unfused_decode_groups = [
        group
        for group in groups
        if group["oneshot_count"] == 0
        and group["twoshot_count"] == 0
        and group["all_reduce_count"] > 0
    ]
    unfused_decode, leftover = _merge_counted_groups(
        unfused_decode_groups,
        count_key="all_reduce_count",
        target=unfused_sites,
        mode="decode",
        path="unfused_triton_ar_rmsnorm",
        m=decode_m,
    )
    forwards.extend(unfused_decode)
    # Fused forwards can leave one explicit final pair per step. Ignore those
    # only when a fused decode was identified; otherwise surface the mismatch.
    if leftover and not fused_decode:
        warnings.append(
            f"incomplete unfused decode: {leftover}/{unfused_sites} sites"
        )

    unfused_prefill, leftover = _merge_unfused_prefill(
        groups,
        target=unfused_sites,
        m=prefill_m,
    )
    forwards.extend(unfused_prefill)
    if leftover >= unfused_sites:
        warnings.append(
            f"incomplete unfused prefill: {leftover}/{unfused_sites} RMSNorm sites"
        )

    forwards.sort(key=lambda record: record["start_us"])
    _attach_periods(forwards)
    for index, forward in enumerate(forwards):
        forward["index"] = index
    return {
        "trace": str(path),
        "rank": _rank_from_path(path),
        "forwards": forwards,
        "summary": _forward_summary(forwards),
        "warnings": warnings,
        "correlation_group_count": len(groups),
    }


def _aligned_aggregate(traces: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = {}
    for mode in ("prefill", "decode"):
        per_rank = [
            [record for record in trace["forwards"] if record["mode"] == mode]
            for trace in traces
        ]
        if not per_rank or not any(per_rank):
            continue
        counts = [len(records) for records in per_rank]
        if len(set(counts)) != 1:
            aggregate[mode] = {
                "status": "unaligned",
                "forward_counts_by_rank": counts,
            }
            continue
        aligned = []
        for index, records in enumerate(zip(*per_rank)):
            periods = [
                record["period_us"]
                for record in records
                if record["period_us"] is not None
            ]
            aligned.append(
                {
                    "index": index,
                    "max_rank_period_us": max(periods) if periods else None,
                    "max_rank_gpu_span_us": max(
                        record["gpu_span_us"] for record in records
                    ),
                    "max_rank_comm_sum_us": max(
                        record["comm_sum_us"] for record in records
                    ),
                    "max_rank_collective_sum_us": max(
                        record["collective_sum_us"] for record in records
                    ),
                    "max_rank_fused_sum_us": max(
                        record["fused_sum_us"] for record in records
                    ),
                    "max_rank_triton_ar_sum_us": max(
                        record["triton_ar_sum_us"] for record in records
                    ),
                    "max_rank_rmsnorm_sum_us": max(
                        record["rmsnorm_sum_us"] for record in records
                    ),
                    "paths": sorted({record["path"] for record in records}),
                    "M_values": sorted(
                        {
                            record["M"]
                            for record in records
                            if record["M"] is not None
                        }
                    ),
                }
            )
        period_values = [
            record["max_rank_period_us"]
            for record in aligned
            if record["max_rank_period_us"] is not None
        ]
        rank_period_medians = [
            value
            for trace in traces
            if (
                value := trace["summary"]
                .get(mode, {})
                .get("period_us", {})
                .get("median_us")
            )
            is not None
        ]
        rank_comm_medians = [
            value
            for trace in traces
            if (
                value := trace["summary"]
                .get(mode, {})
                .get("comm_sum_us", {})
                .get("median_us")
            )
            is not None
        ]
        rank_fused_medians = [
            value
            for trace in traces
            if (
                value := trace["summary"]
                .get(mode, {})
                .get("fused_sum_us", {})
                .get("median_us")
            )
            is not None
        ]
        aggregate[mode] = {
            "status": "aligned",
            "forward_count_per_rank": counts[0],
            "aligned_forwards": aligned,
            "max_rank_period_us": _stats(period_values),
            "max_rank_gpu_span_us": _stats(
                [record["max_rank_gpu_span_us"] for record in aligned]
            ),
            "max_rank_comm_sum_us": _stats(
                [record["max_rank_comm_sum_us"] for record in aligned]
            ),
            "max_rank_collective_sum_us": _stats(
                [
                    record["max_rank_collective_sum_us"]
                    for record in aligned
                ]
            ),
            "max_rank_fused_sum_us": _stats(
                [record["max_rank_fused_sum_us"] for record in aligned]
            ),
            "max_rank_triton_ar_sum_us": _stats(
                [
                    record["max_rank_triton_ar_sum_us"]
                    for record in aligned
                ]
            ),
            "max_rank_rmsnorm_sum_us": _stats(
                [record["max_rank_rmsnorm_sum_us"] for record in aligned]
            ),
            "max_of_rank_median_period_us": (
                max(rank_period_medians) if rank_period_medians else None
            ),
            "max_of_rank_median_comm_sum_us": (
                max(rank_comm_medians) if rank_comm_medians else None
            ),
            "max_of_rank_median_fused_sum_us": (
                max(rank_fused_medians) if rank_fused_medians else None
            ),
        }
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--eligible-sites", type=int, default=72)
    parser.add_argument("--unfused-sites", type=int, default=73)
    parser.add_argument("--decode-m", type=int)
    parser.add_argument("--prefill-m", type=int)
    args = parser.parse_args()
    traces = [
        _summarize(
            path,
            eligible_sites=args.eligible_sites,
            unfused_sites=args.unfused_sites,
            decode_m=args.decode_m,
            prefill_m=args.prefill_m,
        )
        for path in args.traces
    ]
    result = {
        "analysis_quality": "heuristic_legacy",
        "configuration": {
            "eligible_sites": args.eligible_sites,
            "unfused_sites": args.unfused_sites,
            "decode_m": args.decode_m,
            "prefill_m": args.prefill_m,
        },
        "traces": traces,
        "aggregate": _aligned_aggregate(traces),
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
