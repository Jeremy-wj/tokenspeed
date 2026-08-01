"""Bounded single-server AR+RMSNorm serving stability reproducer."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from datetime import datetime
from pathlib import Path

from benchmark.run_ar_rmsnorm_repeatability import (
    REPO_ROOT,
    SERVE_SCRIPT,
    Arm,
    Workload,
    _acquire_gpu_campaign_lock,
    _assert_gpu_isolation,
    _bench_command,
    _capture_gpu_diagnostics,
    _docker_image,
    _git_metadata,
    _preflight,
    _run,
    _run_guarded_benchmark,
    _serve_proof,
    _teardown,
    _wait_for_health,
    hip_to_physical_gpu_map,
    physical_to_kfd_id_map,
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--devices", default="1,2,5,6")
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument(
        "--container",
        default=os.environ.get("TOKENSPEED_CONTAINER", "jeremwan-tokenspeed-profiler"),
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "triton_shmem", "symm_mem", "iris"),
        default="triton_shmem",
    )
    parser.add_argument("--fusion", type=int, choices=(0, 1), default=1)
    parser.add_argument("--fusion-max-m", type=int, default=0)
    parser.add_argument("--double-buffer-input", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--inkernel-barrier",
        type=int,
        choices=(0, 1),
        default=int(os.environ.get("TS_TRITON_SHMEM_INKERNEL_BARRIER", "0")),
    )
    parser.add_argument(
        "--barrier-grid",
        type=int,
        default=int(os.environ.get("TS_TRITON_SHMEM_BARRIER_GRID", "0")),
    )
    parser.add_argument(
        "--deep-health-mode",
        choices=("generate", "passive", "passive_when_busy"),
        default="generate",
    )
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--prompts", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--suite", choices=("single", "full"), default="single")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--profile-steps",
        type=int,
        default=0,
        help="Capture a bounded no-stack torch trace during the workload.",
    )
    parser.add_argument("--serve-extra-args", default="")
    parser.add_argument("--ignored-busy-gpus", default="3")
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args()
    if os.environ.get("AR_NORM_PROFILE_ID") != "gpt-oss-120b-mi350x-triton-core-v3":
        parser.error(
            "source benchmark/profiles/ar_rmsnorm/"
            "gpt_oss_120b_mi350x.env before running this GPT-OSS reproducer"
        )
    return args


def main() -> int:
    args = _args()
    gpu_lock = _acquire_gpu_campaign_lock()
    if len(args.devices.split(",")) != args.world_size:
        raise ValueError("device count must equal world size")
    hip_map = hip_to_physical_gpu_map()
    selected = {hip_map[int(index)] for index in args.devices.split(",")}
    ignored = {
        int(index) for index in args.ignored_busy_gpus.split(",") if index.strip()
    }
    args.selected_physical_gpus = selected
    args.ignored_busy_gpus = ignored
    args.physical_to_kfd_id = physical_to_kfd_id_map()
    args.benchmark_timeout = args.timeout
    args.dry_run = False
    run_root = args.run_root or (
        REPO_ROOT
        / "benchmark/results/ar_rmsnorm/raw/current/gpt-oss-120b/mi350x"
        / datetime.now().strftime("%Y-%m-%d")
        / "serving-repros"
        / args.label
    )
    run_root = run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    log = run_root / "orchestration.log"
    if not args.fusion:
        signature_family = "production_unfused"
    elif args.backend in ("auto", "iris"):
        signature_family = "iris_fused"
    elif args.backend == "triton_shmem":
        signature_family = "triton_shmem_fused"
    else:
        signature_family = "symm_mem_fused"
    arm = Arm(
        "repro",
        args.fusion_max_m,
        fusion_enabled=args.fusion,
        backend=args.backend,
        signature_family=signature_family,
    )
    engine_module = (
        "tokenspeed.runtime.entrypoints.safe_smg_server"
        if args.deep_health_mode != "generate"
        else "smg_grpc_servicer.tokenspeed"
    )
    env = os.environ.copy()
    env.update(
        {
            "CONTAINER": args.container,
            "RUN_ROOT": str(run_root),
            "LOG_DIR": str(run_root),
            "RUN_LABEL": args.label,
            "ENABLE_ALLREDUCE_FUSION": str(args.fusion),
            "TS_TRITON_SHMEM_FUSION_MAX_M": str(args.fusion_max_m),
            "TS_TRITON_SHMEM_DOUBLE_BUFFER_INPUT": str(args.double_buffer_input),
            "TS_TRITON_SHMEM_INKERNEL_BARRIER": str(args.inkernel_barrier),
            "TS_TRITON_SHMEM_BARRIER_GRID": str(args.barrier_grid),
            "TS_SERVE_ENGINE_MODULE": engine_module,
            "TOKENSPEED_DEEP_HEALTH_MODE": args.deep_health_mode,
            "TOKENSPEED_PROFILE_FORWARD_MARKERS": "1",
        }
    )
    configuration = vars(args).copy()
    configuration.update(
        {
            "run_root": str(run_root),
            "selected_physical_gpus": sorted(selected),
            "ignored_busy_gpus": sorted(ignored),
            "hip_to_physical": hip_map,
            "engine_module": engine_module,
        }
    )
    summary = {
        "label": args.label,
        "configuration": configuration,
        "code_identity": _git_metadata(),
        "container_identity": _docker_image(args.container),
        "status": "running",
        "results": [],
    }
    _write(run_root / "summary.json", summary)
    phase = "initial_teardown"
    _teardown(env=env, log=log, dry_run=False)
    try:
        phase = "container_restart"
        _run(
            ["docker", "restart", args.container],
            log=log,
            timeout_seconds=60,
        )
        phase = "preflight"
        _preflight(
            run_root,
            container=args.container,
            selected_gpus=selected,
            ignored_busy_gpus=ignored,
            dry_run=False,
        )
        command = [
            "bash",
            str(SERVE_SCRIPT),
            str(args.world_size),
            args.devices,
            "2048",
            args.backend,
            *shlex.split(args.serve_extra_args),
        ]
        phase = "server_launch"
        _run(command, env=env, log=log, timeout_seconds=60)
        phase = "server_health"
        _wait_for_health(
            args.container,
            port=8100,
            timeout_seconds=900,
            log=log,
            dry_run=False,
            serve_log=run_root / f"serve-{args.label}.log",
        )
        _capture_gpu_diagnostics(
            run_root,
            label="server-ready",
            container=args.container,
            selected_gpus=selected,
            physical_to_kfd_id=args.physical_to_kfd_id,
        )
        phase = "workload_setup"
        if args.suite == "full":
            workloads = [
                Workload(f"prefill-m{m}", m // 32, 8, 128, 32, "ttft")
                for m in (128, 256, 512, 1024, 2048, 4096)
            ]
            workloads.append(Workload("decode", 128, 128, 128, 32, "decode"))
        else:
            workloads = [
                Workload(
                    "repro",
                    args.input_len,
                    args.output_len,
                    args.prompts,
                    args.concurrency,
                    "stability",
                )
            ]
        iterations = (
            list(enumerate(workloads))
            if args.suite == "full"
            else [(repeat, workloads[0]) for repeat in range(args.repeats)]
        )
        for repeat, workload in iterations:
            phase = f"{workload.name}_repeat_{repeat}_isolation"
            _assert_gpu_isolation(
                run_root,
                container=args.container,
                selected_gpus=selected,
                ignored_busy_gpus=ignored,
                physical_to_kfd_id=args.physical_to_kfd_id,
            )
            output = run_root / f"result-{workload.name}-repeat{repeat}.json"
            profile_extra: tuple[str, ...] = ()
            if args.profile_steps > 0:
                trace_dir = run_root / "traces" / f"{workload.name}-repeat{repeat}"
                profile_extra = (
                    "--profile",
                    "--profile-num-steps",
                    str(args.profile_steps),
                    "--profile-base-url",
                    "http://127.0.0.1:8101",
                    "--profile-output-dir",
                    str(trace_dir),
                    "--profile-id",
                    f"{args.label}-{workload.name}-repeat{repeat}",
                    "--no-profile-with-stack",
                    "--profile-record-shapes",
                    "--profile-activities",
                    "CPU",
                    "GPU",
                )
            phase = f"{workload.name}_repeat_{repeat}_benchmark"
            _run_guarded_benchmark(
                _bench_command(
                    f"{args.label}-repeat{repeat}",
                    workload,
                    args.seed + repeat,
                    output_file=output,
                    ready_check=False,
                    extra=profile_extra,
                ),
                args=args,
                phase_dir=run_root,
                env=env,
                log=log,
            )
            phase = f"{workload.name}_repeat_{repeat}_result_parse"
            result = json.loads(output.read_text(encoding="utf-8"))
            summary["results"].append(
                {
                    "repeat": repeat,
                    "workload": workload.name,
                    "completed": result.get("completed"),
                    "failed": result.get("failed"),
                    "median_tpot_ms": result.get("median_tpot_ms"),
                    "output_throughput": result.get("output_throughput"),
                }
            )
        phase = "serve_proof"
        summary["serve_proof"] = _serve_proof(
            run_root / f"serve-{args.label}.log",
            arm,
            2048,
            args.barrier_grid,
            args.inkernel_barrier,
            engine_module,
            args.deep_health_mode,
            args.double_buffer_input,
            "--disable-overlap-schedule" in args.serve_extra_args,
        )
        summary["status"] = "passed"
        _write(run_root / "summary.json", summary)
        return 0
    except BaseException as exc:
        summary["status"] = "failed"
        summary["failure_phase"] = phase
        summary["error"] = f"{type(exc).__name__}: {exc}"
        _capture_gpu_diagnostics(
            run_root,
            label="repro-failure",
            container=args.container,
            selected_gpus=selected,
            physical_to_kfd_id=args.physical_to_kfd_id,
            include_amd_smi_metrics=True,
        )
        _write(run_root / "summary.json", summary)
        raise
    finally:
        _teardown(env=env, log=log, dry_run=False)
        gpu_lock.close()


if __name__ == "__main__":
    sys.exit(main())
