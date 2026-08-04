"""Consolidate fresh-process eager AR+RMSNorm sweep artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

CONTROL_ARMS = ("upstream_unfused", "iris_fused")
CANDIDATE_ARMS = ("triton_forced", "triton_profile")
ARM_NAMES = {*CONTROL_ARMS, *CANDIDATE_ARMS}


def _percent_change(candidate: float, control: float) -> float:
    return (candidate / control - 1.0) * 100.0


def collect(root: Path, *, require_complete: bool = True) -> dict[str, Any]:
    samples: dict[tuple[int, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    pass_arms: dict[tuple[str, int, int], set[str]] = defaultdict(set)
    candidates = set()
    for path in sorted(root.rglob("sweep.json")):
        arm = path.parent.name
        if arm not in ARM_NAMES:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("mode") != "eager":
            raise ValueError(f"non-eager artifact in {path}")
        ws = int(payload["world_size"])
        n = int(payload["N"])
        pass_name = path.parents[3].name
        path_ws = int(path.parents[2].name.removeprefix("ws-"))
        path_n = int(path.parents[1].name.removeprefix("n-"))
        if (path_ws, path_n) != (ws, n):
            raise ValueError(f"path identity mismatch in {path}")
        if arm in CANDIDATE_ARMS:
            candidates.add(arm)
        for row in payload["rows"]:
            m = int(row["M"])
            pass_arms[(pass_name, ws, m)].add(arm)
            stats = row["max_rank_samples_stats_us"]
            samples[(ws, n, m, arm)].append(
                {
                    "pass": pass_name,
                    "expected_backend": row["expected_backend"],
                    "expected_path": row["expected_path"],
                    "p50_us": float(stats["p50_us"]),
                    "p95_us": float(stats["p95_us"]),
                    "p99_us": float(stats["p99_us"]),
                    "mean_us": float(stats["mean_us"]),
                    "repeat": int(payload["repeat"]),
                }
            )

    if len(candidates) != 1:
        raise ValueError(f"expected one candidate arm, got {candidates}")
    candidate = next(iter(candidates))
    arms = (*CONTROL_ARMS, candidate)

    incomplete = []
    for (pass_name, ws, m), observed in sorted(pass_arms.items()):
        missing = [arm for arm in arms if arm not in observed]
        if missing:
            incomplete.append(
                {
                    "pass": pass_name,
                    "world_size": ws,
                    "M": m,
                    "missing_arms": missing,
                }
            )
    if require_complete and incomplete:
        raise ValueError(f"incomplete eager arm triples: {incomplete}")

    rows = []
    for (ws, n, m, arm), values in sorted(samples.items()):
        paths = {value["expected_path"] for value in values}
        backends = {value["expected_backend"] for value in values}
        if len(paths) != 1 or len(backends) != 1:
            raise ValueError(f"eager path mismatch for WS={ws} N={n} M={m} arm={arm}")

        def mean(field: str, pass_values=values) -> float:
            return statistics.fmean(value[field] for value in pass_values)

        pass_p50 = [value["p50_us"] for value in values]
        rows.append(
            {
                "world_size": ws,
                "N": n,
                "M": m,
                "arm": arm,
                "expected_backend": next(iter(backends)),
                "expected_path": next(iter(paths)),
                "passes": len(values),
                "pass_names": [value["pass"] for value in values],
                "iterations_per_pass": min(value["repeat"] for value in values),
                "pass_p50_us": pass_p50,
                "p50_us": mean("p50_us"),
                "p95_us": mean("p95_us"),
                "p99_us": mean("p99_us"),
                "mean_us": mean("mean_us"),
                "p50_pass_spread_pct": (
                    _percent_change(max(pass_p50), min(pass_p50))
                    if len(pass_p50) > 1
                    else 0.0
                ),
            }
        )

    by_case = {(row["world_size"], row["N"], row["M"], row["arm"]): row for row in rows}
    comparisons = []
    case_ids = sorted({(row["world_size"], row["N"], row["M"]) for row in rows})
    for ws, n, m in case_ids:
        if not all((ws, n, m, arm) in by_case for arm in arms):
            continue
        unfused = by_case[(ws, n, m, "upstream_unfused")]
        iris = by_case[(ws, n, m, "iris_fused")]
        triton = by_case[(ws, n, m, candidate)]
        comparisons.append(
            {
                "world_size": ws,
                "N": n,
                "M": m,
                "unfused_path": unfused["expected_path"],
                "iris_path": iris["expected_path"],
                "triton_path": triton["expected_path"],
                "triton_vs_unfused_pct": _percent_change(
                    triton["p50_us"], unfused["p50_us"]
                ),
                "triton_vs_iris_pct": _percent_change(triton["p50_us"], iris["p50_us"]),
                "iris_vs_unfused_pct": _percent_change(
                    iris["p50_us"], unfused["p50_us"]
                ),
            }
        )

    return {
        "schema_version": 1,
        "mode": "eager",
        "source_root": str(root),
        "arms": list(arms),
        "candidate_arm": candidate,
        "rows": rows,
        "comparisons": comparisons,
        "incomplete_cases": incomplete,
    }


def write_csv(path: Path, summary: dict[str, Any]) -> None:
    fields = [
        "world_size",
        "N",
        "M",
        "arm",
        "expected_backend",
        "expected_path",
        "passes",
        "iterations_per_pass",
        "p50_us",
        "p95_us",
        "p99_us",
        "mean_us",
        "p50_pass_spread_pct",
        "pass_names",
        "pass_p50_us",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary["rows"]:
            writer.writerow(
                {
                    **row,
                    "pass_names": json.dumps(row["pass_names"], separators=(",", ":")),
                    "pass_p50_us": json.dumps(
                        row["pass_p50_us"], separators=(",", ":")
                    ),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()

    summary = collect(
        args.root.resolve(),
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
