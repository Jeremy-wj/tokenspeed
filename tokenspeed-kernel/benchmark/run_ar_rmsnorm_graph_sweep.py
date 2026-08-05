"""Run a predeclared, resumable AR+RMSNorm graph campaign.

Use ``--dry-run`` to validate and print the full schedule without creating
artifacts or launching a benchmark process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import signal
import subprocess
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ORDINARY_AR_MAX_BYTES = 512 * 1024
CONTROL_ARMS = {"upstream_unfused", "iris_fused"}
CANDIDATE_ARMS = {"triton_forced", "triton_profile"}


def _runtime_pythonpath() -> str:
    paths = [
        str(REPO_ROOT.parent / "tokenspeed-kernel-amd" / "python"),
        str(REPO_ROOT / "python"),
        str(REPO_ROOT),
    ]
    inherited = os.environ.get("PYTHONPATH")
    if inherited:
        paths.append(inherited)
    return os.pathsep.join(paths)


@dataclass(frozen=True)
class Run:
    block: str
    calls_per_graph: int
    world_size: int
    hidden_size: int
    m: int
    arm: str
    bench_impl: str
    max_token_num: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_spec(path: Path) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("schema_version") != 1:
        raise ValueError("campaign spec schema_version must be 1")
    if spec.get("status") != "planned":
        raise ValueError("campaign spec must remain status=planned before collection")
    world_sizes = [int(value) for value in spec["world_sizes"]]
    if world_sizes != [2, 4, 8]:
        raise ValueError("definitive campaign world_sizes must be [2, 4, 8]")
    arms = set(spec["arms"])
    if not CONTROL_ARMS.issubset(arms) or len(arms & CANDIDATE_ARMS) != 1:
        raise ValueError(
            "campaign requires upstream_unfused, iris_fused, and one Triton arm"
        )
    graph = spec.get("microbenchmark", {}).get("graph", spec)
    names = [block["name"] for block in graph["blocks"]]
    if len(names) != len(set(names)):
        raise ValueError("campaign block names must be unique")
    return spec


def _parse_devices(values: Sequence[str]) -> dict[int, str]:
    result: dict[int, str] = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError(
                f"invalid --devices {value!r}; expected WS=id,id"
            )
        raw_ws, raw_devices = value.split("=", 1)
        ws = int(raw_ws)
        devices = [item.strip() for item in raw_devices.split(",") if item.strip()]
        if len(devices) != ws or len(devices) != len(set(devices)):
            raise argparse.ArgumentTypeError(
                f"WS={ws} requires {ws} unique device IDs, got {devices}"
            )
        result[ws] = ",".join(devices)
    return result


def build_schedule(spec: dict[str, Any]) -> list[Run]:
    schedule = []
    hidden_size = int(spec["hidden_size"])
    profile_cap = int(spec.get("workspace_cap", spec.get("profile_cap", 0)))
    graph = spec.get("microbenchmark", {}).get("graph", spec)
    graph_calls = graph.get("calls_per_graph")
    for block in graph["blocks"]:
        arm_order = block["arm_order"]
        if set(arm_order) != set(spec["arms"]):
            raise ValueError(
                f"{block['name']} arm_order must contain every campaign arm"
            )
        for ws in spec["world_sizes"]:
            for m in block["m_values"]:
                for arm in arm_order:
                    schedule.append(
                        Run(
                            block=block["name"],
                            calls_per_graph=int(
                                block.get("calls_per_graph", graph_calls)
                            ),
                            world_size=int(ws),
                            hidden_size=hidden_size,
                            m=int(m),
                            arm=arm,
                            bench_impl=spec["arms"][arm]["bench_impl"],
                            max_token_num=(
                                profile_cap
                                if "workspace_cap" in spec
                                else max(profile_cap, int(m))
                            ),
                        )
                    )
    return schedule


def _result_path(root: Path, run: Run) -> Path:
    return (
        root
        / run.block
        / f"calls-{run.calls_per_graph}"
        / f"ws-{run.world_size}"
        / f"n-{run.hidden_size}"
        / run.arm
        / f"m{run.m}.json"
    )


def _ordinary_backend(run: Run) -> str:
    payload_bytes = 2 * run.m * run.hidden_size
    return "iris" if payload_bytes <= ORDINARY_AR_MAX_BYTES else "rccl"


def _expected_backend(spec: dict[str, Any], run: Run) -> str:
    if run.arm == "triton_forced":
        return "triton_shmem"
    if run.arm == "iris_fused":
        return "iris"
    if run.arm == "triton_profile":
        fusion_max_m = 0
        eligible_max = min(384, fusion_max_m) if fusion_max_m > 0 else 384
        return "triton_shmem" if run.m <= eligible_max else _ordinary_backend(run)
    return _ordinary_backend(run)


def _validate_result(
    path: Path,
    spec: dict[str, Any],
    run: Run,
    replays: int,
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "world_size": run.world_size,
        "N": run.hidden_size,
        "M": run.m,
        "calls_per_graph": run.calls_per_graph,
        "max_token_num": run.max_token_num,
        "repeat": replays,
        "resolved_impl": run.bench_impl,
        "expected_backend": _expected_backend(spec, run),
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"result identity mismatch in {path}: {mismatches}")


def _run_env(
    spec: dict[str, Any],
    run: Run,
    devices: str,
    output: Path,
) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "HIP_VISIBLE_DEVICES": devices,
            "TS_TRITON_SHMEM_VISIBLE_DEVICES": devices,
            "AR_NORM_WORLD_SIZE": str(run.world_size),
            "AR_NORM_DEVICES": devices,
            "BENCH_WS": str(run.world_size),
            "BENCH_N": str(run.hidden_size),
            "BENCH_M": str(run.m),
            "BENCH_MAX_TOKEN_NUM": str(run.max_token_num),
            "BENCH_CALLS_PER_GRAPH": str(run.calls_per_graph),
            "BENCH_N_WARMUP": str(
                spec.get("microbenchmark", {})
                .get("graph", {})
                .get("warmups", spec.get("warmup"))
            ),
            "BENCH_N_REPEAT": str(
                spec.get("microbenchmark", {})
                .get("graph", {})
                .get("replays", spec.get("replays"))
            ),
            "BENCH_IMPL": run.bench_impl,
            "BENCH_JSON": str(output),
            "PYTHONPATH": _runtime_pythonpath(),
        }
    )
    if run.arm == "triton_forced":
        env.update(
            {
                key: str(value)
                for key, value in spec["arms"][run.arm]["overrides"].items()
            }
        )
        cap = str(run.max_token_num)
        env.update(
            {
                "COMM_FUSION_MAX_NUM_TOKENS": cap,
                "TS_TRITON_SHMEM_ONESHOT_MAX_M": cap,
                "TS_TRITON_SHMEM_PADDED_MAX_M": cap,
            }
        )
    elif run.arm == "triton_profile":
        env["GPT_OSS_DEFINITIVE_FUSION_MAX_M"] = "0"
    return env


def _command(spec: dict[str, Any], run: Run) -> list[str]:
    profile_path = spec.get("profile_env")
    if "profiles" in spec:
        profile_path = spec["profiles"][str(run.world_size)]
    profile = (REPO_ROOT / profile_path).resolve()
    module = shlex.quote(
        spec.get("benchmark_module", "benchmark.probe_ar_rmsnorm_graph_perf")
    )
    shell = f"source {shlex.quote(str(profile))} && exec python3 -m {module}"
    return ["bash", "-lc", shell]


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_process(
    command: list[str],
    *,
    env: dict[str, str],
    log,
    timeout: int,
) -> int:
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise


def _dry_run_payload(
    spec_path: Path,
    spec: dict[str, Any],
    schedule: list[Run],
    devices: dict[int, str],
) -> dict[str, Any]:
    by_block: dict[str, int] = {}
    for run in schedule:
        by_block[run.block] = by_block.get(run.block, 0) + 1
    return {
        "dry_run": True,
        "campaign_id": spec["campaign_id"],
        "spec": str(spec_path),
        "devices": {str(key): value for key, value in devices.items()},
        "processes": len(schedule),
        "processes_by_block": by_block,
        "first_run": asdict(schedule[0]),
        "last_run": asdict(schedule[-1]),
        "schedule": [asdict(run) for run in schedule],
    }


def run_campaign(
    spec_path: Path,
    spec: dict[str, Any],
    schedule: list[Run],
    devices: dict[int, str],
    output_root: Path,
    *,
    resume: bool,
) -> int:
    output_root.mkdir(parents=True, exist_ok=True)
    spec_bytes = spec_path.read_bytes()
    manifest = {
        "schema_version": 1,
        "campaign_id": spec["campaign_id"],
        "started_at": _utc_now(),
        "spec": str(spec_path),
        "spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
        "devices": {str(key): value for key, value in devices.items()},
        "processes": len(schedule),
        "command_template": "profile selected per world size",
    }
    _write_json(output_root / "campaign-manifest.json", manifest)

    completed = skipped = failed = 0
    started = time.monotonic()
    current_block = schedule[0].block
    for index, run in enumerate(schedule, start=1):
        if run.block != current_block:
            time.sleep(int(spec["cooldown_seconds_between_blocks"]))
            current_block = run.block

        result_path = _result_path(output_root, run)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        if resume and result_path.exists():
            try:
                graph = spec.get("microbenchmark", {}).get("graph", {})
                replays = int(graph.get("replays", spec.get("replays")))
                _validate_result(result_path, spec, run, replays)
            except (ValueError, KeyError, json.JSONDecodeError):
                invalid = result_path.with_suffix(f".invalid-{int(time.time())}.json")
                result_path.replace(invalid)
            else:
                skipped += 1
                continue

        log_path = result_path.with_suffix(f".attempt-{time.time_ns()}.log")
        run_started = time.monotonic()
        command = _command(spec, run)
        status: dict[str, Any] = {
            "index": index,
            "run": asdict(run),
            "started_at": _utc_now(),
            "result": str(result_path),
            "log": str(log_path),
        }
        try:
            with log_path.open("w", encoding="utf-8") as log:
                returncode = _run_process(
                    command,
                    env=_run_env(
                        spec,
                        run,
                        devices[run.world_size],
                        result_path,
                    ),
                    log=log,
                    timeout=int(spec["process_timeout_seconds"]),
                )
            status["returncode"] = returncode
            if returncode != 0:
                raise RuntimeError(f"benchmark exited {returncode}")
            graph = spec.get("microbenchmark", {}).get("graph", {})
            replays = int(graph.get("replays", spec.get("replays")))
            _validate_result(result_path, spec, run, replays)
        except (OSError, RuntimeError, subprocess.TimeoutExpired, ValueError) as exc:
            failed += 1
            status["status"] = "failed"
            status["error"] = str(exc)
            _append_jsonl(output_root / "failures.jsonl", status)
        else:
            completed += 1
            status["status"] = "completed"
        finally:
            status["elapsed_seconds"] = time.monotonic() - run_started
            status["finished_at"] = _utc_now()
            _append_jsonl(output_root / "runtime.jsonl", status)

    summary = {
        "campaign_id": spec["campaign_id"],
        "finished_at": _utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "scheduled": len(schedule),
        "completed": completed,
        "skipped_valid_resume": skipped,
        "failed": failed,
        "complete": failed == 0 and completed + skipped == len(schedule),
    }
    _write_json(output_root / "runtime-summary.json", summary)
    return 0 if summary["complete"] else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument(
        "--devices",
        action="append",
        default=[],
        metavar="WS=ID,ID",
        help="repeat once for each world size",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    spec_path = args.spec.resolve()
    spec = _read_spec(spec_path)
    schedule = build_schedule(spec)
    devices = _parse_devices(args.devices)
    missing = sorted(set(spec["world_sizes"]) - set(devices))
    if missing:
        parser.error(f"missing --devices entries for world sizes {missing}")
    if set(devices) - set(spec["world_sizes"]):
        parser.error("device mapping contains world sizes outside the spec")

    if args.dry_run:
        print(
            json.dumps(
                _dry_run_payload(spec_path, spec, schedule, devices),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.output_root is None:
        parser.error("--output-root is required unless --dry-run is used")
    raise SystemExit(
        run_campaign(
            spec_path,
            spec,
            schedule,
            devices,
            args.output_root.resolve(),
            resume=not args.no_resume,
        )
    )


if __name__ == "__main__":
    main()
