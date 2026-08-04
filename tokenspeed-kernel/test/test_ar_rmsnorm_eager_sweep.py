from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.analyze_ar_rmsnorm_eager_sweep import collect
from benchmark.run_ar_rmsnorm_eager_sweep import build_schedule


def _stats(value: float) -> dict[str, float]:
    return {
        "min_us": value,
        "p50_us": value,
        "p95_us": value + 1,
        "p99_us": value + 2,
        "max_us": value + 3,
        "mean_us": value + 0.5,
    }


def _write_artifact(
    root: Path,
    *,
    block: str,
    arm: str,
    value: float,
    ws: int = 4,
    m_values=(32, 64),
) -> None:
    path = root / block / f"ws-{ws}" / "n-2880" / arm / "sweep.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    backend = "triton_shmem" if arm == "triton_profile" else "iris"
    path.write_text(
        json.dumps(
            {
                "mode": "eager",
                "resolved_impl": arm,
                "world_size": ws,
                "N": 2880,
                "max_token_num": 2048,
                "warmup": 30,
                "repeat": 150,
                "M_values": list(m_values),
                "rows": [
                    {
                        "M": m,
                        "expected_backend": backend,
                        "expected_path": f"{arm}-path",
                        "max_rank_samples_stats_us": _stats(value + m / 1000),
                    }
                    for m in m_values
                ],
            }
        ),
        encoding="utf-8",
    )


def test_eager_schedule_has_18_fresh_arm_processes():
    spec_path = (
        Path(__file__).parents[1]
        / "benchmark/results/ar_rmsnorm/studies/mi350x"
        / "2026-08-gpt-oss-120b-definitive-sweep/campaign.json"
    )
    spec = json.loads(spec_path.read_text(encoding="utf-8"))

    schedule = build_schedule(spec)

    assert len(schedule) == 18
    assert {run.world_size for run in schedule} == {2, 4, 8}
    assert {run.arm for run in schedule} == {
        "upstream_unfused",
        "iris_fused",
        "triton_profile",
    }


def test_eager_analyzer_keeps_mode_and_passes_separate(tmp_path):
    for block in ("eager-pass1", "eager-pass2"):
        for arm, value in (
            ("upstream_unfused", 20.0),
            ("iris_fused", 22.0),
            ("triton_profile", 16.0),
        ):
            _write_artifact(
                tmp_path,
                block=block,
                arm=arm,
                value=value,
            )

    summary = collect(tmp_path)

    assert summary["mode"] == "eager"
    assert summary["candidate_arm"] == "triton_profile"
    assert len(summary["rows"]) == 6
    assert summary["comparisons"][0]["triton_vs_unfused_pct"] == pytest.approx(
        -19.968051
    )


def test_eager_analyzer_rejects_incomplete_pass(tmp_path):
    _write_artifact(
        tmp_path,
        block="eager-pass1",
        arm="upstream_unfused",
        value=20.0,
    )
    _write_artifact(
        tmp_path,
        block="eager-pass1",
        arm="triton_profile",
        value=16.0,
    )

    with pytest.raises(ValueError, match="incomplete eager arm triples"):
        collect(tmp_path)
