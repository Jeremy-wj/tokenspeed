from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from benchmark.analyze_ar_rmsnorm_forwards import (
    _kernel_breakdown,
    analyze as analyze_marked,
)
from benchmark.analyze_graph_replay import _aligned_aggregate, _summarize


def _event(name: str, correlation: int, ts: float) -> dict:
    return {
        "ph": "X",
        "cat": "kernel",
        "name": name,
        "ts": ts,
        "dur": 1.0,
        "args": {"correlation": correlation},
    }


def _write(path: Path, events: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump({"traceEvents": events}, handle)


def test_merges_split_and_whole_fused_decode_forwards(tmp_path):
    path = tmp_path / "fused-rank0.trace.json.gz"
    events = []
    # First four-site forward is split into two correlation groups.
    events.extend(
        _event("fused_ar_rmsnorm_oneshot_blocked_kernel", 10, index)
        for index in range(2)
    )
    events.extend(
        _event(
            "fused_ar_rmsnorm_oneshot_blocked_kernel",
            11,
            10 + index,
        )
        for index in range(2)
    )
    # Second forward is a single graph correlation.
    events.extend(
        _event(
            "fused_ar_rmsnorm_oneshot_blocked_kernel",
            12,
            100 + index,
        )
        for index in range(4)
    )
    _write(path, events)

    result = _summarize(
        path,
        eligible_sites=4,
        unfused_sites=5,
        decode_m=32,
        prefill_m=None,
    )
    decode = [
        forward for forward in result["forwards"] if forward["mode"] == "decode"
    ]
    assert len(decode) == 2
    assert decode[0]["correlations"] == [10, 11]
    assert decode[0]["oneshot_count"] == 4
    assert decode[0]["period_us"] == 100
    assert decode[1]["correlations"] == [12]
    assert result["rank"] == 0


def test_segments_unfused_decode_and_prefill(tmp_path):
    path = tmp_path / "unfused-TP1.trace.json.gz"
    events = []
    for correlation, start in ((20, 100), (21, 200)):
        events.extend(
            _event("amd_all_reduce_kernel", correlation, start + index)
            for index in range(5)
        )
        events.extend(
            _event("_rmsnorm_kernel", correlation, start + 10 + index)
            for index in range(5)
        )
    # One prefill forward: two RMS/RCCL groups followed by an RCCL-only tail.
    events.extend(
        [
            _event("_rmsnorm_kernel", 30, 10),
            _event("_rmsnorm_kernel", 30, 11),
            _event("ncclDevKernel_Generic_1", 30, 12),
            _event("_rmsnorm_kernel", 31, 20),
            _event("_rmsnorm_kernel", 31, 21),
            _event("_rmsnorm_kernel", 31, 22),
            _event("ncclDevKernel_Generic_1", 31, 23),
            _event("ncclDevKernel_Generic_1", 32, 24),
        ]
    )
    _write(path, events)

    result = _summarize(
        path,
        eligible_sites=4,
        unfused_sites=5,
        decode_m=32,
        prefill_m=512,
    )
    decode = [
        forward for forward in result["forwards"] if forward["mode"] == "decode"
    ]
    prefill = [
        forward for forward in result["forwards"] if forward["mode"] == "prefill"
    ]
    assert len(decode) == 2
    assert all(forward["all_reduce_count"] == 5 for forward in decode)
    assert len(prefill) == 1
    assert prefill[0]["rmsnorm_count"] == 5
    assert prefill[0]["rccl_count"] == 3
    assert prefill[0]["M"] == 512
    assert result["rank"] == 1


def test_aligned_aggregate_uses_max_rank_per_forward(tmp_path):
    traces = []
    for rank, offset in ((0, 0), (1, 10)):
        path = tmp_path / f"fused-rank{rank}.trace.json.gz"
        events = []
        for correlation, start in ((10, 100 + offset), (11, 200 + 2 * offset)):
            events.extend(
                _event(
                    "fused_ar_rmsnorm_oneshot_blocked_kernel",
                    correlation,
                    start + index,
                )
                for index in range(4)
            )
        _write(path, events)
        traces.append(
            _summarize(
                path,
                eligible_sites=4,
                unfused_sites=5,
                decode_m=32,
                prefill_m=None,
            )
        )

    aggregate = _aligned_aggregate(traces)["decode"]
    assert aggregate["status"] == "aligned"
    assert aggregate["forward_count_per_rank"] == 2
    assert aggregate["aligned_forwards"][0]["max_rank_period_us"] == 110


def _write_marked(path: Path, rank: int, duration: float) -> None:
    marker = (
        "tokenspeed.model_forward.v1|id=0|mode=decode|actual_m=32"
        "|executed_m=32|bs=32|padded_bs=32|num_extends=0"
        "|execution=decode_graph"
    )
    payload = {
        "distributedInfo": {"rank": rank},
        "traceEvents": [
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": marker,
                "pid": 1,
                "tid": 10,
                "ts": 0,
                "dur": 100,
                "args": {},
            },
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "hipGraphLaunch",
                "pid": 1,
                "tid": 10,
                "ts": 10,
                "dur": 1,
                "args": {"correlation": 100},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "fused_ar_rmsnorm_oneshot_blocked_kernel",
                "pid": 2,
                "tid": 20,
                "ts": 1000,
                "dur": duration,
                "args": {"correlation": 100},
            },
        ],
    }
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


def test_marker_analyzer_aligns_forward_ids_across_ranks(tmp_path):
    paths = []
    for rank, duration in ((0, 10.0), (1, 12.0)):
        path = tmp_path / f"trace-rank{rank}.json.gz"
        _write_marked(path, rank, duration)
        paths.append(path)
    result = analyze_marked(paths, expected_world_size=2, mode="decode")
    assert result["validation"]["status"] == "ok"
    assert len(result["forwards"]) == 1
    forward = result["forwards"][0]
    assert forward["executed_m"] == 32
    assert forward["max_rank"]["gpu_period_us"] == {
        "value": 12.0,
        "rank": 1,
    }
    assert result["cohorts"][0]["count"] == 1


def test_marker_analyzer_rejects_legacy_trace(tmp_path):
    path = tmp_path / "legacy-rank0.json.gz"
    _write(path, [_event("amd_all_reduce_kernel", 1, 0)])
    with pytest.raises(ValueError, match="exact forward markers not found"):
        analyze_marked([path], expected_world_size=1, mode="all")


def test_marker_analyzer_classifies_post_rebase_iris_paths():
    fused = _kernel_breakdown(
        [_event("iris_allreduce_residual_rmsnorm_kernel", 1, 0)]
    )
    assert fused["primary"] == "iris_fused"
    assert fused["counts"]["standalone_rmsnorm"] == 0

    unfused = _kernel_breakdown(
        [
            _event("iris_stage_one_shot_allreduce_kernel", 1, 0),
            _event("_rmsnorm_kernel", 2, 10),
        ]
    )
    assert unfused["primary"] == "unfused_iris"
    assert unfused["target_kernel_sum_us"] == 2
