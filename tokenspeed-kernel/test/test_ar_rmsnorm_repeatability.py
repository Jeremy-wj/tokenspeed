from __future__ import annotations

import gzip
import json
import sys
import time
from pathlib import Path

import pytest

import benchmark.run_ar_rmsnorm_repeatability as repeatability
from benchmark.run_ar_rmsnorm_repeatability import (
    ARMS,
    METRICS,
    Workload,
    _acquire_gpu_campaign_lock,
    _assert_gpu_isolation,
    _bench_command,
    _gpu_guard_history_is_clean,
    _model_profile_proof,
    _qualified_profile_proof,
    _run,
    _wait_for_health,
    analyze_campaign,
    analyze_stability_campaign,
    build_schedule,
    campaign_arms,
    comparison_arms,
    hierarchical_bootstrap,
    hip_to_physical_gpu_map,
    parse_amd_smi_processes,
    partition_live_gpu_processes,
    physical_to_kfd_id_map,
    read_kfd_gpu_processes,
    validate_trace_signatures,
)


def test_schedule_is_deterministic_and_balanced():
    first = build_schedule(5, [0, 1, 2, 3, 4], 1234)
    second = build_schedule(5, [0, 1, 2, 3, 4], 1234)
    assert first == second
    for block, spec in enumerate(first):
        assert spec["block"] == block
        assert sorted(spec["arm_order"]) == sorted(arm.name for arm in ARMS)
        assert sorted(spec["seed_order"]) == [0, 1, 2, 3, 4]


def test_schedule_block_offset_supports_additive_stages():
    schedule = build_schedule(2, [0], 1234, block_offset=3)
    assert [block["block"] for block in schedule] == [3, 4]


def test_gpu_campaign_lock_rejects_overlap(tmp_path):
    path = tmp_path / "gpu.lock"
    first = _acquire_gpu_campaign_lock(path)
    try:
        with pytest.raises(RuntimeError, match="GPU campaign lock"):
            _acquire_gpu_campaign_lock(path)
    finally:
        first.close()
    second = _acquire_gpu_campaign_lock(path)
    second.close()


def test_comparison_arms_are_unconfounded():
    unfused, fused = comparison_arms("unfused")
    assert not unfused.fusion_enabled
    assert fused.fusion_enabled
    control, iris = comparison_arms("iris")
    assert (control.backend, control.signature_family) == (
        "auto",
        "production_unfused",
    )
    assert (iris.backend, iris.signature_family) == ("auto", "iris_fused")
    control, candidate = comparison_arms("triton_shmem")
    assert control.name == "upstream_unfused"
    assert (candidate.backend, candidate.signature_family) == (
        "triton_shmem",
        "triton_shmem_fused",
    )
    triplet = comparison_arms("three_backend")
    assert [arm.name for arm in triplet] == [
        "upstream_unfused",
        "iris_fused",
        "triton_profile",
    ]
    single, double = comparison_arms("input_ring")
    assert single.double_buffer_input == 0
    assert double.double_buffer_input == 1


def test_stability_campaign_selects_only_unfused():
    arms = campaign_arms("iris", stability_only=True)
    assert len(arms) == 1
    assert arms[0].name == "upstream_unfused"
    assert not arms[0].fusion_enabled
    with pytest.raises(ValueError, match="upstream-unfused comparison"):
        campaign_arms("input_ring", stability_only=True)


def test_qualified_profile_proof_accepts_canonical_profile(tmp_path):
    serve_log = tmp_path / "serve.log"
    serve_log.write_text(
        "RUN_ENV PROFILE_ID=gpt-oss-120b-mi350x-triton-core-v3 "
        "PROFILE_PURE_TP=1 HVD=1,2,5,6 CAP=2048 "
        "DEEP_HEALTH_MODE=generate COARSE=1 FOLD_COPYIN=0 WORKGROUP_SYNC=1 "
        "SHMEM_OUTPUT_RING=72 "
        "INPUT_SITE_RING=72 BORROW_TWOSHOT_OUTPUT=1 "
        "INKERNEL=1 ONESHOT_MAX_M=384 GRID_CAP=128 GRID_CAP_MIN_M=256 "
        "ONESHOT_VARIANT=padded PADDED_MAX_M=64 ONESHOT_NUM_WARPS=4 "
        "DOUBLE_BUFFER_INPUT=0 BARRIER_GRID=0 FUSION_MAX_M=0 "
        "FORWARD_MARKERS=1\n"
        "ServerArgs(gpu_memory_utilization=0.95, "
        "cudagraph_capture_sizes=None, disable_prefill_graph=False, "
        "disable_overlap_schedule=False)\n"
        "prefill breakable graph: captured buckets [16, 32, 2048]\n",
        encoding="utf-8",
    )
    assert _qualified_profile_proof(serve_log)["status"] == "passed"
    serve_log.write_text(
        serve_log.read_text(encoding="utf-8").replace(
            "disable_overlap_schedule=False",
            "disable_overlap_schedule=True",
        ),
        encoding="utf-8",
    )
    assert (
        _qualified_profile_proof(serve_log, disable_overlap_schedule=True)["status"]
        == "passed"
    )


def test_qualified_profile_proof_rejects_memory_override(tmp_path):
    serve_log = tmp_path / "serve.log"
    serve_log.write_text(
        "RUN_ENV PROFILE_ID=gpt-oss-120b-mi350x-triton-core-v3 "
        "PROFILE_PURE_TP=1 HVD=1,2,5,6 CAP=2048 "
        "DEEP_HEALTH_MODE=generate COARSE=1 FOLD_COPYIN=0 WORKGROUP_SYNC=1 "
        "SHMEM_OUTPUT_RING=72 "
        "INPUT_SITE_RING=72 BORROW_TWOSHOT_OUTPUT=1 "
        "INKERNEL=1 ONESHOT_MAX_M=384 GRID_CAP=128 GRID_CAP_MIN_M=256 "
        "ONESHOT_VARIANT=padded PADDED_MAX_M=64 ONESHOT_NUM_WARPS=4 "
        "DOUBLE_BUFFER_INPUT=0 BARRIER_GRID=0 FUSION_MAX_M=0 "
        "FORWARD_MARKERS=1\n"
        "ServerArgs(gpu_memory_utilization=0.9, "
        "cudagraph_capture_sizes=None, disable_prefill_graph=False, "
        "disable_overlap_schedule=False)\n"
        "prefill breakable graph: captured buckets [16, 32, 2048]\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="gpu_memory_utilization=0.95"):
        _qualified_profile_proof(serve_log)


def test_model_profile_proof_accepts_generic_profile(tmp_path, monkeypatch):
    model_path = "/models/glm-5.2"
    monkeypatch.setenv("MODEL_PATH", model_path)
    monkeypatch.setenv("MODEL_LABEL", "glm-5.2-fp8")
    monkeypatch.setenv("HIDDEN_SIZE", "6144")
    monkeypatch.setenv("AR_NORM_PROFILE_ID", "unqualified-manual")
    serve_log = tmp_path / "serve.log"
    serve_log.write_text(
        "RUN_ENV MODEL=glm-5.2-fp8 HIDDEN=6144 "
        "PROFILE_ID=unqualified-manual HVD=0,1,2,3,4,5,6,7 "
        "WS=8 CAP=2048 FORWARD_MARKERS=1\n"
        f"ServerArgs(model='{model_path}', enable_allreduce_fusion=False)\n",
        encoding="utf-8",
    )
    proof = _model_profile_proof(
        serve_log,
        devices="0,1,2,3,4,5,6,7",
        world_size=8,
        cap=2048,
    )
    assert proof["status"] == "passed"
    assert proof["model_path"] == model_path


def test_wait_for_health_fails_fast_on_startup_error(tmp_path, monkeypatch):
    serve_log = tmp_path / "serve.log"
    serve_log.write_text(
        "startup failed: metrics server bind failed: address in use\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "benchmark.run_ar_rmsnorm_repeatability._run",
        lambda *args, **kwargs: type("Result", (), {"returncode": 1})(),
    )
    with pytest.raises(RuntimeError, match="metrics server bind failed"):
        _wait_for_health(
            "container",
            port=8100,
            timeout_seconds=1,
            log=tmp_path / "orchestration.log",
            dry_run=False,
            serve_log=serve_log,
        )


def test_single_arm_stability_summary(tmp_path):
    arm_dir = tmp_path / "block-00" / "p0-unfused"
    result_dir = arm_dir / "results"
    result_dir.mkdir(parents=True)
    (arm_dir / "arm-summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "block": 0,
                "arm": {"name": "unfused"},
            }
        ),
        encoding="utf-8",
    )
    for seed in (0, 1, 2):
        (result_dir / f"decode-seed{seed}.json").write_text(
            json.dumps({"completed": 128, "failed": 0}),
            encoding="utf-8",
        )
    summary = analyze_stability_campaign(
        tmp_path,
        arm_name="unfused",
        expected_blocks=1,
        expected_seeds=[0, 1, 2],
    )
    assert summary["passed_results"] == 3
    assert summary["stability"]["eligible"]
    assert (tmp_path / "stability-summary.json").exists()


def test_parse_amd_smi_ignores_zero_usage_ghosts():
    payload = json.dumps(
        [
            {
                "gpu": 0,
                "process_list": [
                    {
                        "process_info": {
                            "pid": 1,
                            "memory_usage": {"vram_mem": {"value": 0}},
                            "cu_occupancy": "N/A",
                        }
                    }
                ],
            },
            {
                "gpu": 3,
                "process_list": [
                    {
                        "process_info": {
                            "pid": 2,
                            "memory_usage": {"vram_mem": {"value": 1_500_000_000}},
                            "cu_occupancy": 0,
                        }
                    }
                ],
            },
        ]
    )
    parsed = parse_amd_smi_processes(payload)
    assert 0 not in parsed
    assert parsed[3][0]["pid"] == 2


def test_partition_live_gpu_processes_preserves_transient_queries():
    active = {0: [{"pid": 10}], 1: [{"pid": 20}, {"pid": 30}]}
    live, transient = partition_live_gpu_processes(
        active,
        pid_exists=lambda pid: pid in {10, 30},
    )
    assert live == {0: [{"pid": 10}], 1: [{"pid": 30}]}
    assert transient == {1: [{"pid": 20}]}


def test_read_kfd_gpu_processes_is_nonintrusive(tmp_path):
    fake_pid = 999_999_999
    process_dir = tmp_path / str(fake_pid)
    (process_dir / "stats_123").mkdir(parents=True)
    (process_dir / "vram_123").write_text("4096\n", encoding="utf-8")
    (process_dir / "stats_123" / "cu_occupancy").write_text(
        "7\n",
        encoding="utf-8",
    )
    assert read_kfd_gpu_processes({2: 123}, root=tmp_path) == {
        2: [
            {
                "pid": fake_pid,
                "name": "",
                "vram_bytes": 4096,
                "cu_occupancy": 7,
                "source": "kfd_sysfs",
            }
        ]
    }


def test_hip_mapping_uses_kfd_node_order(monkeypatch):
    payload = json.dumps(
        [
            {"gpu": 0, "node_id": 3},
            {"gpu": 3, "node_id": 2},
            {"gpu": 2, "node_id": 4},
        ]
    )
    monkeypatch.setattr(
        "subprocess.check_output",
        lambda command, **kwargs: payload,
    )
    assert hip_to_physical_gpu_map() == {0: 3, 1: 0, 2: 2}


def test_physical_to_kfd_id_map(monkeypatch):
    payload = json.dumps(
        [
            {"gpu": 0, "kfd_id": 27295},
            {"gpu": 3, "kfd_id": 36538},
        ]
    )
    monkeypatch.setattr(
        "subprocess.check_output",
        lambda command, **kwargs: payload,
    )
    assert physical_to_kfd_id_map() == {0: 27295, 3: 36538}


def test_gpu_isolation_rejects_foreign_process(tmp_path, monkeypatch):
    process_payload = json.dumps(
        [
            {
                "gpu": 0,
                "process_list": [
                    {
                        "process_info": {
                            "pid": 100,
                            "memory_usage": {"vram_mem": {"value": 10}},
                            "cu_occupancy": 0,
                        }
                    }
                ],
            },
            {
                "gpu": 1,
                "process_list": [
                    {
                        "process_info": {
                            "pid": 200,
                            "memory_usage": {"vram_mem": {"value": 10}},
                            "cu_occupancy": 0,
                        }
                    }
                ],
            },
        ]
    )

    def fake_check_output(command, **kwargs):
        if command[:3] == ["amd-smi", "process", "--json"]:
            return process_payload
        if command[:2] == ["docker", "top"]:
            return "PID\n100\n"
        raise AssertionError(command)

    monkeypatch.setattr("subprocess.check_output", fake_check_output)
    monkeypatch.setattr(
        "benchmark.run_ar_rmsnorm_repeatability.partition_live_gpu_processes",
        lambda active: (active, {}),
    )
    with pytest.raises(RuntimeError, match="GPU\\(s\\): 1"):
        _assert_gpu_isolation(
            tmp_path,
            container="test",
            selected_gpus={0, 1},
            ignored_busy_gpus=set(),
        )
    _assert_gpu_isolation(
        tmp_path,
        container="test",
        selected_gpus={0, 1},
        ignored_busy_gpus=set(),
        known_container_pids={100, 200},
    )


def test_resume_requires_clean_gpu_guard_history(tmp_path):
    assert not _gpu_guard_history_is_clean(tmp_path)
    guard = tmp_path / "gpu-guard.jsonl"
    guard.write_text(
        json.dumps({"unexpected_active_processes": {}}) + "\n",
        encoding="utf-8",
    )
    assert _gpu_guard_history_is_clean(tmp_path)
    with guard.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"unexpected_active_processes": {"1": [{"pid": 123}]}}) + "\n"
        )
    assert not _gpu_guard_history_is_clean(tmp_path)
    (tmp_path / "preflight.json").write_text(
        json.dumps({"time": "2026-07-30T01:00:00+00:00"}),
        encoding="utf-8",
    )
    with guard.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "time": "2026-07-30T01:00:01+00:00",
                    "unexpected_active_processes": {},
                }
            )
            + "\n"
        )
    assert _gpu_guard_history_is_clean(tmp_path)


def test_hierarchical_bootstrap_constant_sample():
    result = hierarchical_bootstrap(
        {0: [-2.0, -2.0], 1: [-2.0, -2.0], 2: [-2.0, -2.0]},
        samples=500,
        seed=7,
    )
    assert result == {
        "mean": -2.0,
        "ci95_low": -2.0,
        "ci95_high": -2.0,
        "n_blocks": 3,
        "n_pairs": 6,
    }


def test_command_timeout_terminates_process_group():
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        _run(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout_seconds=1,
        )
    assert time.monotonic() - started < 5


def test_measurement_command_skips_redundant_ready_probe():
    workload = Workload("decode", 128, 512, 128, 32, "tpot_capacity")
    measurement = _bench_command(
        "test",
        workload,
        0,
        ready_check=False,
    )
    warmup = _bench_command("warmup", workload, 0)
    assert measurement[-2:] == ["--ready-check-timeout-sec", "0"]
    assert "--ready-check-timeout-sec" not in warmup


def _write_trace(path: Path, names: list[str]) -> None:
    events = [
        {
            "cat": "kernel",
            "name": name,
            "ts": index,
            "dur": 1,
            "args": {"correlation": index},
        }
        for index, name in enumerate(names)
    ]
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump({"traceEvents": events}, handle)


def test_trace_signature_validation_distinguishes_gate(tmp_path):
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    for rank in range(2):
        _write_trace(
            baseline_dir / f"baseline-TP{rank}.trace.json.gz",
            ["fused_ar_rmsnorm_oneshot_blocked_kernel"]
            + ["iris_stage_one_shot_allreduce_kernel"]
            + ["_rmsnorm_kernel"] * 73,
        )
    result = validate_trace_signatures(
        baseline_dir,
        world_size=2,
        arm=ARMS[0],
    )
    assert result["status"] == "passed"

    gate_dir = tmp_path / "gate"
    gate_dir.mkdir()
    gate_names = [
        "fused_ar_rmsnorm_oneshot_blocked_kernel",
        "iris_stage_one_shot_allreduce_kernel",
    ] + ["_rmsnorm_kernel"] * 73
    for rank in range(2):
        _write_trace(
            gate_dir / f"gate-TP{rank}.trace.json.gz",
            gate_names,
        )
    result = validate_trace_signatures(
        gate_dir,
        world_size=2,
        arm=ARMS[1],
    )
    assert result["status"] == "passed"


def test_trace_signature_validation_distinguishes_post_rebase_backends(tmp_path):
    unfused, iris = comparison_arms("iris")
    unfused_dir = tmp_path / "unfused"
    unfused_dir.mkdir()
    iris_dir = tmp_path / "iris"
    iris_dir.mkdir()
    for rank in range(2):
        _write_trace(
            unfused_dir / f"unfused-TP{rank}.trace.json.gz",
            ["iris_stage_one_shot_allreduce_kernel"] + ["_rmsnorm_kernel"] * 73,
        )
        _write_trace(
            iris_dir / f"iris-TP{rank}.trace.json.gz",
            ["iris_allreduce_residual_rmsnorm_kernel"],
        )
    assert (
        validate_trace_signatures(unfused_dir, world_size=2, arm=unfused)["status"]
        == "passed"
    )
    assert (
        validate_trace_signatures(iris_dir, world_size=2, arm=iris)["status"]
        == "passed"
    )


def _write_arm(
    root: Path,
    *,
    block: int,
    position: int,
    arm: str,
    multiplier: float,
    stage: str | None = None,
) -> None:
    stage_root = root / stage if stage else root
    arm_dir = stage_root / f"block-{block:02d}" / f"p{position}-{arm}"
    results = arm_dir / "results"
    results.mkdir(parents=True)
    (arm_dir / "arm-summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "block": block,
                "arm": {"name": arm},
            }
        ),
        encoding="utf-8",
    )
    for workload in (
        "prefill-m512",
        "prefill-m1024",
        "prefill-m2048",
        "decode",
    ):
        for seed in (0, 1):
            baseline = {metric: 100.0 for metric in METRICS}
            baseline["output_throughput"] = 1000.0
            payload = {key: value * multiplier for key, value in baseline.items()}
            (results / f"{workload}-seed{seed}.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )


def test_campaign_analysis_uses_paired_blocks(tmp_path):
    for block in range(3):
        _write_arm(
            tmp_path,
            block=block,
            position=0,
            arm=ARMS[0].name,
            multiplier=1.0,
        )
        _write_arm(
            tmp_path,
            block=block,
            position=1,
            arm=ARMS[1].name,
            multiplier=0.98,
        )
    summary = analyze_campaign(
        tmp_path,
        bootstrap_samples=500,
        bootstrap_seed=9,
    )
    result = summary["workloads"]["prefill-m512"]["median_ttft_ms"]
    assert round(result["mean"], 6) == -2.0
    assert result["n_blocks"] == 3
    assert result["n_pairs"] == 6
    assert not summary["promotion"]["eligible"]
    assert any(
        "insufficient paired evidence" in reason
        for reason in summary["promotion"]["reasons"]
    )
    assert not summary["promotion"]["objectives"]["latency"]["eligible"]


def test_three_arm_analysis_reuses_one_unfused_control(tmp_path, monkeypatch):
    arms = comparison_arms("three_backend")
    monkeypatch.setattr(repeatability, "ARMS", arms)
    for block in range(3):
        for position, (arm, multiplier) in enumerate(zip(arms, (1.0, 1.02, 0.98))):
            _write_arm(
                tmp_path,
                block=block,
                position=position,
                arm=arm.name,
                multiplier=multiplier,
            )

    summary = analyze_campaign(
        tmp_path,
        bootstrap_samples=500,
        bootstrap_seed=9,
    )

    assert set(summary["comparisons"]) == {"iris_fused", "triton_profile"}
    assert (
        summary["comparisons"]["triton_profile"]["workloads"]["decode"][
            "median_tpot_ms"
        ]["n_pairs"]
        == 6
    )
    assert summary["comparison"]["candidate"] == "triton_profile"


def test_three_arm_analysis_combines_additive_stage_directories(
    tmp_path,
    monkeypatch,
):
    arms = comparison_arms("three_backend")
    monkeypatch.setattr(repeatability, "ARMS", arms)
    for stage, block in (("core", 0), ("extension", 3)):
        for position, (arm, multiplier) in enumerate(zip(arms, (1.0, 1.02, 0.98))):
            _write_arm(
                tmp_path,
                stage=stage,
                block=block,
                position=position,
                arm=arm.name,
                multiplier=multiplier,
            )

    summary = analyze_campaign(
        tmp_path,
        bootstrap_samples=500,
        bootstrap_seed=9,
    )

    result = summary["comparisons"]["triton_profile"]["workloads"]["decode"][
        "median_tpot_ms"
    ]
    assert result["n_blocks"] == 2
    assert result["n_pairs"] == 4
    assert not summary["promotion"]["objectives"]["capacity"]["eligible"]
