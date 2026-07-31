#!/usr/bin/env python3
"""Summarize AR+RMSNorm events from Proton Chrome traces."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

COMM_KERNELS = {
    "amd_all_reduce_kernel",
    "_rmsnorm_kernel",
    "fused_ar_rmsnorm_oneshot_wholerow_kernel",
    "fused_ar_rmsnorm_oneshot_wholerow_padded_kernel",
    "fused_ar_rmsnorm_oneshot_blocked_kernel",
    "fused_ar_rmsnorm_twoshot_blocked_kernel",
    "symm_grid_barrier_kernel",
}
COMM_SCOPE_PREFIX = "communication.allreduce_residual_rmsnorm"


def _duration_stats(durations: list[float]) -> dict[str, float | int]:
    ordered = sorted(durations)
    return {
        "count": len(ordered),
        "total_us": round(sum(ordered), 3),
        "median_us": round(statistics.median(ordered), 3),
        "p95_us": round(ordered[int(0.95 * (len(ordered) - 1))], 3),
        "max_us": round(ordered[-1], 3),
    }


def analyze_trace(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    events = payload.get("traceEvents", [])

    kernel_durations: dict[str, list[float]] = defaultdict(list)
    scopes: dict[tuple[tuple[str, str], ...], list[float]] = defaultdict(list)
    total_kernel_us = 0.0
    total_kernel_count = 0

    for event in events:
        if event.get("ph") != "X":
            continue
        category = event.get("cat")
        name = event.get("name", "")
        duration = float(event.get("dur", 0.0))
        if category == "kernel":
            total_kernel_count += 1
            total_kernel_us += duration
            if name in COMM_KERNELS:
                kernel_durations[name].append(duration)
        elif category == "metric" and name.startswith(COMM_SCOPE_PREFIX):
            metrics = event.get("args", {}).get("metrics", {})
            key = tuple(sorted((str(k), str(v)) for k, v in metrics.items()))
            scopes[key].append(duration)

    return {
        "trace": str(path),
        "kernel_count": total_kernel_count,
        "total_kernel_us": round(total_kernel_us, 3),
        "communication_kernels": {
            name: _duration_stats(durations)
            for name, durations in sorted(kernel_durations.items())
        },
        "communication_scopes": [
            {
                "metrics": dict(key),
                **_duration_stats(durations),
            }
            for key, durations in sorted(scopes.items())
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Chrome trace files or directories containing *.chrome_trace files",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()

    trace_paths: list[Path] = []
    for path in args.paths:
        if path.is_dir():
            trace_paths.extend(sorted(path.glob("*.chrome_trace")))
        else:
            trace_paths.append(path)
    if not trace_paths:
        parser.error("no Chrome trace files found")

    result = {"traces": [analyze_trace(path) for path in trace_paths]}
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
