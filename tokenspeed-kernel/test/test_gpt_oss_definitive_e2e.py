from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.run_gpt_oss_definitive_e2e import (
    STAGES,
    _command,
    _selected_caps,
)


def _spec() -> dict:
    path = (
        Path(__file__).parents[1]
        / "benchmark/results/ar_rmsnorm/studies/mi350x"
        / "2026-08-gpt-oss-120b-definitive-sweep/campaign.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_core_stage_is_27_fresh_server_lifecycles():
    spec = _spec()
    assert STAGES["core"]["blocks"] * 3 * len(spec["world_sizes"]) == 27
    assert STAGES["extension"]["blocks"] * 3 * len(spec["world_sizes"]) == 18
    assert sum(stage["blocks"] for stage in STAGES.values()) == 15


def test_pending_policy_is_dry_run_only():
    spec = _spec()
    pending = {
        "status": "pending_microbenchmark",
        "world_sizes": {
            "2": {"fusion_max_m": None},
            "4": {"fusion_max_m": None},
            "8": {"fusion_max_m": None},
        },
    }
    assert _selected_caps(spec, pending, dry_run=True) == {2: 0, 4: 0, 8: 0}
    with pytest.raises(ValueError, match="must be frozen"):
        _selected_caps(spec, pending, dry_run=False)


def test_e2e_command_uses_common_control_triplet_and_selected_cap(tmp_path):
    spec = _spec()
    command = _command(
        spec,
        ws=8,
        devices="0,1,2,3,4,5,6,7",
        cap=64,
        blocks=3,
        block_offset=0,
        stage="core",
        output_root=tmp_path,
        resume=True,
    )
    shell = command[-1]
    assert "--comparison three_backend" in shell
    assert "--blocks 3" in shell
    assert "--block-offset 0" in shell
    assert "--seeds 0" in shell
    assert "GPT_OSS_DEFINITIVE_FUSION_MAX_M=64" in shell
    assert "--definitive-diagnostics" in shell
    assert "--resume" in shell
