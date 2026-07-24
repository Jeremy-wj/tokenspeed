"""Summarize AR+RMSNorm GPU kernels in torch Chrome traces."""
from __future__ import annotations

import argparse
import gzip
import json
import statistics
from pathlib import Path


_KERNELS = (
    "fused_ar_rmsnorm_oneshot_blocked_kernel",
    "fused_ar_rmsnorm_oneshot_wholerow_kernel",
    "fused_ar_rmsnorm_twoshot_blocked_kernel",
    "amd_all_reduce_kernel",
    "_rmsnorm_kernel",
)


def _load(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _summarize(path: Path) -> dict:
    durations: dict[str, list[float]] = {}
    for event in _load(path).get("traceEvents", []):
        name = str(event.get("name", ""))
        duration = event.get("dur")
        if not isinstance(duration, (int, float)):
            continue
        for kernel in _KERNELS:
            if kernel in name:
                durations.setdefault(kernel, []).append(float(duration))
                break

    kernels = {}
    for kernel, values in durations.items():
        kernels[kernel] = {
            "count": len(values),
            "median_us": statistics.median(values),
            "p95_us": _percentile(values, 0.95),
            "total_us": sum(values),
        }
    return {"trace": str(path), "kernels": kernels}


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
