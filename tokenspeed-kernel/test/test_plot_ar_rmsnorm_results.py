from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.plot_ar_rmsnorm_results import (
    extract_e2e_blocks,
    extract_e2e_metrics,
    load_json,
    main,
    select_graph_comparisons,
)

ROOT = Path(__file__).parents[1]
STUDIES = ROOT / "benchmark/results/ar_rmsnorm/studies/mi350x"
GLM_SUMMARY = (
    STUDIES
    / "2026-08-glm-5.2-fp8-definitive-sweep"
    / "graph-sweep-summary.json"
)
GPT_E2E_SUMMARY = (
    STUDIES
    / "2026-08-gpt-oss-120b-definitive-sweep"
    / "end-to-end-summary.json"
)


def test_select_graph_comparisons_filters_site_count_and_preserves_glm_anchors():
    selected = select_graph_comparisons(
        load_json(GLM_SUMMARY),
        world_sizes=[4, 8],
        calls_per_graph=156,
    )

    assert set(selected) == {4, 8}
    assert all(row["calls_per_graph"] == 156 for rows in selected.values() for row in rows)
    assert all(
        row["triton_vs_iris_pct"] < 0 for rows in selected.values() for row in rows
    )
    ws8_m4 = next(row for row in selected[8] if row["M"] == 4)
    assert ws8_m4["triton_vs_unfused_pct"] == pytest.approx(-23.0401582114)


def test_glm_ws8_border_keeps_raw_adjusted_and_transport_distinct():
    rows = select_graph_comparisons(
        load_json(GLM_SUMMARY),
        world_sizes=[8],
        calls_per_graph=156,
        min_m=40,
        max_m=43,
    )[8]
    by_m = {row["M"]: row for row in rows}

    assert by_m[40]["triton_vs_unfused_adjusted_pct"] == pytest.approx(
        -2.9742424550
    )
    assert by_m[41]["triton_vs_unfused_adjusted_pct"] == pytest.approx(
        0.1728028621
    )
    assert by_m[42]["triton_vs_unfused_adjusted_pct"] == pytest.approx(
        2.1931280096
    )
    assert by_m[42]["triton_vs_unfused_pct"] == pytest.approx(-2.8738403302)
    assert by_m[43]["triton_vs_unfused_pct"] == pytest.approx(33.59287111696)
    assert by_m[43]["triton_vs_unfused_adjusted_pct"] == pytest.approx(
        43.6583756914
    )
    assert by_m[43]["unfused_path"].startswith("rccl_all_reduce")


def test_select_graph_comparisons_rejects_missing_requested_site_count():
    with pytest.raises(ValueError, match="no graph rows"):
        select_graph_comparisons(
            load_json(GLM_SUMMARY),
            world_sizes=[8],
            calls_per_graph=72,
        )


def test_extract_e2e_metrics_preserves_gpt_world_size_decisions():
    metrics = extract_e2e_metrics(
        load_json(GPT_E2E_SUMMARY),
        world_sizes=[2, 4, 8],
    )

    assert metrics[2]["output_throughput"]["mean"] == pytest.approx(1.10, abs=0.01)
    assert metrics[4]["output_throughput"]["mean"] == pytest.approx(
        1.2725641281
    )
    assert metrics[4]["output_throughput"]["ci95_low"] == pytest.approx(
        0.4938561472
    )
    assert metrics[4]["median_tpot_ms"]["mean"] == pytest.approx(-1.2953225629)
    assert metrics[8]["output_throughput"]["mean"] == pytest.approx(-2.48, abs=0.01)
    assert all(
        metric["n_blocks"] == 5 and metric["n_pairs"] == 5
        for world_size in metrics.values()
        for metric in world_size.values()
    )


def test_extract_e2e_blocks_shows_all_five_ws4_pairs_favorable():
    blocks = extract_e2e_blocks(load_json(GPT_E2E_SUMMARY), world_size=4)

    throughput = dict(blocks["output_throughput"])
    tpot = dict(blocks["median_tpot_ms"])
    assert throughput[0] == pytest.approx(1.7454867126)
    assert throughput[2] == pytest.approx(0.0195427972)
    assert throughput[3] == pytest.approx(2.3911095471)
    assert tpot[1] == pytest.approx(-2.34, abs=0.01)
    assert len(throughput) == len(tpot) == 5
    assert all(value > 0 for value in throughput.values())
    assert all(value < 0 for value in tpot.values())


def test_cli_renders_headless_png_when_matplotlib_is_available(tmp_path):
    pytest.importorskip("matplotlib")
    output = tmp_path / "e2e.png"

    assert (
        main(
            [
                "e2e-summary",
                str(GPT_E2E_SUMMARY),
                str(output),
                "--world-sizes",
                "2",
                "4",
                "8",
            ]
        )
        == 0
    )
    assert output.stat().st_size > 10_000
