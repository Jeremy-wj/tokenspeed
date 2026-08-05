from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from benchmark.analyze_ar_rmsnorm_graph_sweep import collect
from benchmark.run_ar_rmsnorm_graph_sweep import (
    Run,
    _expected_backend,
    _expected_path,
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
    calls: int = 156,
    reset_us: float | None = None,
) -> None:
    path = root / block / f"calls-{calls}" / f"ws-{ws}" / f"n-{n}" / arm / f"m{m}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "min_us": per_site_us,
        "p50_us": per_site_us,
        "p95_us": per_site_us + 1,
        "p99_us": per_site_us + 2,
        "max_us": per_site_us + 3,
        "mean_us": per_site_us + 0.5,
    }
    backend = "triton_shmem" if arm.startswith("triton_") else "iris"
    reset_stats = None if reset_us is None else {key: reset_us for key in stats}
    serving_stats = (
        None
        if reset_us is None
        else {key: value - reset_us for key, value in stats.items()}
    )
    payload = {
        "impl": arm,
        "resolved_impl": arm,
        "expected_backend": backend,
        "expected_path": f"{arm}-path",
        "world_size": ws,
        "M": m,
        "N": n,
        "calls_per_graph": calls,
        "max_token_num": max(42, m),
        "payload_bytes": 2 * m * n,
        "repeat": 1000,
        "max_rank_samples_per_call_stats_us": stats,
        "max_rank_samples_stats_us": {
            key: value * calls for key, value in stats.items()
        },
        "benchmark_reset_copy_per_call_stats_us": reset_stats,
        "serving_faithful_estimate_per_call_stats_us": serving_stats,
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


def test_collect_supports_profile_candidate_and_reset_adjustment(tmp_path):
    for arm, value in (
        ("upstream_unfused", 16.0),
        ("iris_fused", 19.0),
        ("triton_profile", 15.0),
    ):
        _write_case(
            tmp_path,
            arm=arm,
            per_site_us=value,
            ws=4,
            n=2880,
            m=32,
            block="graph-pass1",
            calls=72,
            reset_us=2.0 if arm == "upstream_unfused" else None,
        )

    summary = collect(tmp_path, max_m=64)

    assert summary["candidate_arm"] == "triton_profile"
    comparison = summary["comparisons"][0]
    assert comparison["triton_vs_unfused_pct"] == pytest.approx(-6.25)
    assert comparison["triton_vs_unfused_adjusted_pct"] == pytest.approx(7.142857)


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


def test_gpt_definitive_graph_schedule_has_162_processes():
    spec_path = (
        Path(__file__).parents[1]
        / "benchmark/results/ar_rmsnorm/studies/mi350x"
        / "2026-08-gpt-oss-120b-definitive-sweep/campaign.json"
    )
    spec = json.loads(spec_path.read_text(encoding="utf-8"))

    schedule = build_schedule(spec)

    assert len(schedule) == 162
    assert {run.max_token_num for run in schedule} == {2048}
    assert {run.calls_per_graph for run in schedule} == {72}
    assert {run.arm for run in schedule} == {
        "upstream_unfused",
        "iris_fused",
        "triton_profile",
    }


def test_gpt_graph_candidate_expects_complete_fallback_above_m384():
    spec = {"model": "gpt-oss-120b"}
    base = {
        "block": "graph-pass1",
        "calls_per_graph": 72,
        "world_size": 8,
        "hidden_size": 2880,
        "arm": "triton_profile",
        "bench_impl": "triton_shmem",
        "max_token_num": 2048,
    }

    assert _expected_backend(spec, Run(m=384, **base)) == "triton_shmem"
    assert _expected_backend(spec, Run(m=385, **base)) == "rccl"


def test_definitive_expected_paths_include_forced_variant():
    spec = {
        "arms": {
            "triton_forced": {
                "overrides": {
                    "TS_TRITON_SHMEM_ONESHOT_VARIANT": "padded",
                }
            }
        }
    }
    base = {
        "block": "pass1",
        "calls_per_graph": 156,
        "world_size": 8,
        "hidden_size": 6144,
        "bench_impl": "triton_shmem",
        "max_token_num": 42,
    }

    assert (
        _expected_path(spec, Run(m=42, arm="triton_forced", **base))
        == "oneshot_wholerow_padded"
    )
    assert (
        _expected_path(
            spec,
            Run(
                m=42,
                arm="upstream_unfused",
                **{**base, "bench_impl": "production_unfused"},
            ),
        )
        == "ordinary_iris_all_reduce+triton_residual_rmsnorm"
    )
    assert (
        _expected_path(
            spec,
            Run(
                m=43,
                arm="upstream_unfused",
                **{
                    **base,
                    "bench_impl": "production_unfused",
                    "max_token_num": 43,
                },
            ),
        )
        == "rccl_all_reduce+triton_residual_rmsnorm"
    )


def test_device_map_requires_unique_count():
    assert _parse_devices(["2=4,5", "4=4,5,6,7"]) == {
        2: "4,5",
        4: "4,5,6,7",
    }
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_devices(["2=4"])
