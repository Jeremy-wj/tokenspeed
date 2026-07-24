"""Summarize graph-replay critical paths from torch Chrome traces."""
from __future__ import annotations

import argparse
import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path


def _load(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _stats(values: list[float]) -> dict:
    return {
        "median_us": statistics.median(values),
        "p95_us": _percentile(values, 0.95),
        "min_us": min(values),
        "max_us": max(values),
    }


def _signature(names: list[str]) -> str | None:
    if any("fused_ar_rmsnorm_oneshot" in name for name in names):
        return "fused_oneshot"
    if any("fused_ar_rmsnorm_twoshot" in name for name in names):
        return "fused_twoshot"
    if any("amd_all_reduce_kernel" in name for name in names) and any(
        "_rmsnorm_kernel" in name for name in names
    ):
        return "unfused_ar_rmsnorm"
    return None


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


def _summarize(path: Path) -> dict:
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

    by_signature: dict[str, list[dict]] = defaultdict(list)
    for correlation, events in groups.items():
        names = [str(event.get("name", "")) for event in events]
        signature = _signature(names)
        if signature is None:
            continue
        start = min(float(event["ts"]) for event in events)
        end = max(float(event["ts"]) + float(event["dur"]) for event in events)
        by_signature[signature].append(
            {
                "correlation": correlation,
                "gpu_span_us": end - start,
                "kernel_sum_us": sum(float(event["dur"]) for event in events),
                "comm_sum_us": sum(
                    float(event["dur"])
                    for event, name in zip(events, names)
                    if _is_comm_kernel(name)
                ),
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
            }
        )

    summaries = {}
    for signature, records in by_signature.items():
        summaries[signature] = {
            "count": len(records),
            "gpu_span_us": _stats([record["gpu_span_us"] for record in records]),
            "kernel_sum_us": _stats(
                [record["kernel_sum_us"] for record in records]
            ),
            "comm_sum_us": _stats([record["comm_sum_us"] for record in records]),
            "median_kernel_count": statistics.median(
                record["kernel_count"] for record in records
            ),
            "median_oneshot_count": statistics.median(
                record["oneshot_count"] for record in records
            ),
            "median_twoshot_count": statistics.median(
                record["twoshot_count"] for record in records
            ),
            "median_all_reduce_count": statistics.median(
                record["all_reduce_count"] for record in records
            ),
            "median_rmsnorm_count": statistics.median(
                record["rmsnorm_count"] for record in records
            ),
        }
    return {"trace": str(path), "graphs": summaries}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {"traces": [_summarize(path) for path in args.traces]}
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
