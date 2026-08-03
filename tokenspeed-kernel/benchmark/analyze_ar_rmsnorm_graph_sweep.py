"""Consolidate repeated AR+RMSNorm graph-sweep JSON artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ARMS = ("upstream_unfused", "iris_fused", "triton_forced")


def _percent_change(candidate: float, control: float) -> float:
    return (candidate / control - 1.0) * 100.0


def _ancestor_value(path: Path, prefix: str) -> int | None:
    for parent in path.parents:
        if parent.name.startswith(prefix):
            return int(parent.name.removeprefix(prefix))
    return None


def _pass_name(path: Path) -> str:
    for parent in path.parents:
        if parent.name.startswith("pass"):
            return parent.name
    raise ValueError(f"missing pass directory in {path}")


def _artifact_paths(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("m*.json")):
        if path.parent.name in ARMS and _ancestor_value(path, "calls-") is not None:
            yield path


def _frontier(
    comparisons: list[dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    winning = [row["M"] for row in comparisons if row[field] < 0.0]
    return {
        "profitable_m_values": winning,
        "first_profitable_m": min(winning) if winning else None,
        "last_profitable_m": max(winning) if winning else None,
        "first_measured_loss_after_profit": next(
            (
                row["M"]
                for row in comparisons
                if winning and row["M"] > max(winning) and row[field] >= 0.0
            ),
            None,
        ),
    }


def collect(
    root: Path,
    max_m: int,
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    samples: dict[tuple[int, int, int, int, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    pass_arms: dict[tuple[str, int, int, int, int], set[str]] = defaultdict(set)
    for path in _artifact_paths(root):
        arm = path.parent.name
        payload = json.loads(path.read_text(encoding="utf-8"))
        ws = int(payload["world_size"])
        n = int(payload["N"])
        calls = int(payload["calls_per_graph"])
        m = int(payload["M"])
        if m > max_m:
            continue

        path_ws = _ancestor_value(path, "ws-")
        path_n = _ancestor_value(path, "n-")
        path_calls = _ancestor_value(path, "calls-")
        if path_ws is not None and path_ws != ws:
            raise ValueError(f"world-size mismatch in {path}")
        if path_n is not None and path_n != n:
            raise ValueError(f"hidden-size mismatch in {path}")
        if path_calls != calls:
            raise ValueError(f"calls mismatch in {path}")
        if int(payload["repeat"]) < 1000:
            raise ValueError(f"insufficient replay count in {path}")

        per_site = payload["max_rank_samples_per_call_stats_us"]
        per_graph = payload["max_rank_samples_stats_us"]
        pass_name = _pass_name(path)
        pass_arms[(pass_name, ws, n, calls, m)].add(arm)
        samples[(ws, n, calls, m, arm)].append(
            {
                "pass": pass_name,
                "path": str(path),
                "expected_backend": payload["expected_backend"],
                "expected_path": payload["expected_path"],
                "max_token_num": int(payload.get("max_token_num", m)),
                "payload_bytes": int(payload["payload_bytes"]),
                "replays": int(payload["repeat"]),
                "p50_us_per_site": float(per_site["p50_us"]),
                "p95_us_per_site": float(per_site["p95_us"]),
                "p99_us_per_site": float(per_site["p99_us"]),
                "mean_us_per_site": float(per_site["mean_us"]),
                "p50_us_per_graph": float(per_graph["p50_us"]),
            }
        )

    rows = []
    for (ws, n, calls, m, arm), values in sorted(samples.items()):
        paths = {value["expected_path"] for value in values}
        backends = {value["expected_backend"] for value in values}
        max_tokens = {value["max_token_num"] for value in values}
        if len(paths) != 1 or len(backends) != 1 or len(max_tokens) != 1:
            raise ValueError(
                f"identity mismatch for WS={ws} N={n} calls={calls} M={m} arm={arm}"
            )

        def mean(field: str, pass_values=values) -> float:
            return statistics.fmean(value[field] for value in pass_values)

        p50_values = [value["p50_us_per_site"] for value in values]
        rows.append(
            {
                "world_size": ws,
                "N": n,
                "calls_per_graph": calls,
                "M": m,
                "arm": arm,
                "policy": (
                    "forced_diagnostic" if arm == "triton_forced" else "control"
                ),
                "expected_backend": next(iter(backends)),
                "expected_path": next(iter(paths)),
                "max_token_num": next(iter(max_tokens)),
                "payload_bytes": values[0]["payload_bytes"],
                "replays_per_pass": min(value["replays"] for value in values),
                "passes": len(values),
                "pass_names": [value["pass"] for value in values],
                "pass_p50_us_per_site": p50_values,
                "p50_us_per_site": mean("p50_us_per_site"),
                "p95_us_per_site": mean("p95_us_per_site"),
                "p99_us_per_site": mean("p99_us_per_site"),
                "mean_us_per_site": mean("mean_us_per_site"),
                "p50_us_per_graph": mean("p50_us_per_graph"),
                "p50_pass_spread_pct": (
                    _percent_change(max(p50_values), min(p50_values))
                    if len(p50_values) > 1
                    else 0.0
                ),
            }
        )

    by_case = {
        (
            row["world_size"],
            row["N"],
            row["calls_per_graph"],
            row["M"],
            row["arm"],
        ): row
        for row in rows
    }
    case_ids = sorted(
        {
            (row["world_size"], row["N"], row["calls_per_graph"], row["M"])
            for row in rows
        }
    )
    incomplete = []
    for (pass_name, ws, n, calls, m), observed in sorted(pass_arms.items()):
        missing = [arm for arm in ARMS if arm not in observed]
        if missing:
            incomplete.append(
                {
                    "pass": pass_name,
                    "world_size": ws,
                    "N": n,
                    "calls_per_graph": calls,
                    "M": m,
                    "missing_arms": missing,
                }
            )
    comparisons = []
    for ws, n, calls, m in case_ids:
        missing = [arm for arm in ARMS if (ws, n, calls, m, arm) not in by_case]
        if missing:
            continue
        unfused = by_case[(ws, n, calls, m, "upstream_unfused")]
        iris = by_case[(ws, n, calls, m, "iris_fused")]
        triton = by_case[(ws, n, calls, m, "triton_forced")]
        comparisons.append(
            {
                "world_size": ws,
                "N": n,
                "calls_per_graph": calls,
                "M": m,
                "unfused_path": unfused["expected_path"],
                "iris_path": iris["expected_path"],
                "triton_path": triton["expected_path"],
                "triton_vs_unfused_pct": _percent_change(
                    triton["p50_us_per_site"], unfused["p50_us_per_site"]
                ),
                "triton_vs_iris_pct": _percent_change(
                    triton["p50_us_per_site"], iris["p50_us_per_site"]
                ),
                "iris_vs_unfused_pct": _percent_change(
                    iris["p50_us_per_site"], unfused["p50_us_per_site"]
                ),
                "triton_vs_unfused_forward_delta_ms": (
                    triton["p50_us_per_graph"] - unfused["p50_us_per_graph"]
                )
                / 1000.0,
            }
        )

    if require_complete and incomplete:
        raise ValueError(f"incomplete arm triples: {incomplete}")

    frontiers = []
    frontier_ids = sorted(
        {(row["world_size"], row["N"], row["calls_per_graph"]) for row in comparisons}
    )
    for ws, n, calls in frontier_ids:
        group = [
            row
            for row in comparisons
            if (row["world_size"], row["N"], row["calls_per_graph"]) == (ws, n, calls)
        ]
        group.sort(key=lambda row: row["M"])
        frontiers.append(
            {
                "world_size": ws,
                "N": n,
                "calls_per_graph": calls,
                "triton_vs_unfused": _frontier(group, "triton_vs_unfused_pct"),
                "triton_vs_iris": _frontier(group, "triton_vs_iris_pct"),
            }
        )

    return {
        "schema_version": 2,
        "source_root": str(root),
        "aggregation": (
            "Arithmetic mean of pass-level max-rank-per-iteration p50 values; "
            "single pass where no confirmation pass exists."
        ),
        "arms": list(ARMS),
        "rows": rows,
        "comparisons": comparisons,
        "frontiers": frontiers,
        "incomplete_cases": incomplete,
    }


def write_csv(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "world_size",
        "N",
        "calls_per_graph",
        "M",
        "arm",
        "policy",
        "expected_backend",
        "expected_path",
        "max_token_num",
        "payload_bytes",
        "replays_per_pass",
        "passes",
        "p50_us_per_site",
        "p95_us_per_site",
        "p99_us_per_site",
        "mean_us_per_site",
        "p50_us_per_graph",
        "p50_pass_spread_pct",
        "pass_names",
        "pass_p50_us_per_site",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary["rows"]:
            writer.writerow(
                {
                    **row,
                    "pass_names": json.dumps(row["pass_names"], separators=(",", ":")),
                    "pass_p50_us_per_site": json.dumps(
                        row["pass_p50_us_per_site"], separators=(",", ":")
                    ),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--max-m", type=int, default=512)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()

    summary = collect(
        args.root.resolve(),
        args.max_m,
        require_complete=not args.allow_incomplete,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_csv, summary)


if __name__ == "__main__":
    main()
