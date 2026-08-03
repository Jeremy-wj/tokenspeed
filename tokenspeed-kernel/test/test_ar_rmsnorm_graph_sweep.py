from __future__ import annotations

import argparse
import json

import pytest

from benchmark.analyze_ar_rmsnorm_graph_sweep import collect
from benchmark.run_ar_rmsnorm_graph_sweep import (
    _parse_devices,
    _result_path,
    build_schedule,
)


def _write_case(
    root,
    *,
    arm: str,
    per_site_us: float,
    ws: int = 8,
    n: int = 6144,
    m: int = 2,
    block: str = "pass1",
) -> None:
    path = root / block / "calls-156" / f"ws-{ws}" / f"n-{n}" / arm / f"m{m}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "min_us": per_site_us,
        "p50_us": per_site_us,
        "p95_us": per_site_us + 1,
        "p99_us": per_site_us + 2,
        "max_us": per_site_us + 3,
        "mean_us": per_site_us + 0.5,
    }
    backend = "triton_shmem" if arm == "triton_forced" else "iris"
    payload = {
        "impl": arm,
        "resolved_impl": arm,
        "expected_backend": backend,
        "expected_path": f"{arm}-path",
        "world_size": ws,
        "M": m,
        "N": n,
        "calls_per_graph": 156,
        "max_token_num": max(42, m),
        "payload_bytes": 2 * m * n,
        "repeat": 1000,
        "max_rank_samples_per_call_stats_us": stats,
        "max_rank_samples_stats_us": {key: value * 156 for key, value in stats.items()},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_triple(root, *, ws: int, m: int, values=(20.0, 22.0, 16.0)) -> None:
    for arm, value in zip(
        ("upstream_unfused", "iris_fused", "triton_forced"),
        values,
    ):
        _write_case(root, arm=arm, per_site_us=value, ws=ws, m=m)


def test_collect_keys_comparisons_by_world_size_and_hidden_size(tmp_path):
    _write_triple(tmp_path, ws=4, m=2)
    _write_triple(tmp_path, ws=8, m=2, values=(30.0, 31.0, 24.0))

    summary = collect(tmp_path, max_m=32)

    assert summary["schema_version"] == 2
    assert len(summary["rows"]) == 6
    assert len(summary["comparisons"]) == 2
    assert {row["world_size"] for row in summary["comparisons"]} == {4, 8}
    comparison = next(row for row in summary["comparisons"] if row["world_size"] == 4)
    assert comparison["triton_vs_unfused_pct"] == pytest.approx(-20.0)
    assert comparison["triton_vs_iris_pct"] == pytest.approx(-27.272727)
    assert comparison["triton_vs_unfused_forward_delta_ms"] == pytest.approx(-0.624)


def test_collect_reports_measured_profitability_frontier(tmp_path):
    _write_triple(tmp_path, ws=8, m=2, values=(20.0, 22.0, 16.0))
    _write_triple(tmp_path, ws=8, m=42, values=(40.0, 44.0, 39.0))
    _write_triple(tmp_path, ws=8, m=43, values=(30.0, 45.0, 38.0))

    summary = collect(tmp_path, max_m=64)

    frontier = summary["frontiers"][0]["triton_vs_unfused"]
    assert frontier["profitable_m_values"] == [2, 42]
    assert frontier["first_profitable_m"] == 2
    assert frontier["last_profitable_m"] == 42
    assert frontier["first_measured_loss_after_profit"] == 43


def test_collect_rejects_incomplete_arm_triples(tmp_path):
    _write_case(
        tmp_path,
        arm="upstream_unfused",
        per_site_us=20.0,
    )

    with pytest.raises(ValueError, match="incomplete arm triples"):
        collect(tmp_path, max_m=32)

    summary = collect(tmp_path, max_m=32, require_complete=False)
    assert summary["incomplete_cases"][0]["missing_arms"] == [
        "iris_fused",
        "triton_forced",
    ]


def test_collect_rejects_unbalanced_confirmation_pass(tmp_path):
    _write_triple(tmp_path, ws=8, m=42)
    _write_case(
        tmp_path,
        arm="upstream_unfused",
        per_site_us=40.0,
        ws=8,
        m=42,
        block="pass2",
    )

    with pytest.raises(ValueError, match="incomplete arm triples"):
        collect(tmp_path, max_m=42)


def test_definitive_schedule_has_315_fresh_processes(tmp_path):
    spec = {
        "hidden_size": 6144,
        "profile_cap": 42,
        "world_sizes": [2, 4, 8],
        "arms": {
            "upstream_unfused": {"bench_impl": "production_unfused"},
            "iris_fused": {"bench_impl": "auto"},
            "triton_forced": {"bench_impl": "triton_shmem"},
        },
        "blocks": [
            {
                "name": "pass1",
                "calls_per_graph": 156,
                "arm_order": [
                    "upstream_unfused",
                    "iris_fused",
                    "triton_forced",
                ],
                "m_values": list(range(18)),
            },
            {
                "name": "pass2",
                "calls_per_graph": 156,
                "arm_order": [
                    "triton_forced",
                    "iris_fused",
                    "upstream_unfused",
                ],
                "m_values": list(range(12)),
            },
            {
                "name": "pass-diagnostic",
                "calls_per_graph": 1,
                "arm_order": [
                    "iris_fused",
                    "triton_forced",
                    "upstream_unfused",
                ],
                "m_values": list(range(5)),
            },
        ],
    }

    schedule = build_schedule(spec)

    assert len(schedule) == 315
    assert _result_path(tmp_path, schedule[0]).parts[-6:] == (
        "pass1",
        "calls-156",
        "ws-2",
        "n-6144",
        "upstream_unfused",
        "m0.json",
    )


def test_device_map_requires_unique_count():
    assert _parse_devices(["2=4,5", "4=4,5,6,7"]) == {
        2: "4,5",
        4: "4,5,6,7",
    }
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_devices(["2=4"])
