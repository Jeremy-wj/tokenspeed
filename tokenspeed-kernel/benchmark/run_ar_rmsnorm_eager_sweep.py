"""Run the eager portion of a predeclared AR+RMSNorm campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from benchmark.run_ar_rmsnorm_graph_sweep import (
    REPO_ROOT,
    _parse_devices,
    _run_process,
    _runtime_pythonpath,
)


@dataclass(frozen=True)
class EagerRun:
    block: str
    world_size: int
    hidden_size: int
    arm: str
    bench_impl: str
    m_values: tuple[int, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_spec(path: Path) -> dict:
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("schema_version") != 1 or spec.get("status") != "planned":
        raise ValueError("expected a planned schema-v1 campaign")
    eager = spec.get("microbenchmark", {}).get("eager")
    if not eager:
        raise ValueError("campaign has no eager microbenchmark section")
    return spec


def build_schedule(spec: dict) -> list[EagerRun]:
    eager = spec["microbenchmark"]["eager"]
    schedule = []
    for block in eager["blocks"]:
        if set(block["arm_order"]) != set(spec["arms"]):
            raise ValueError(f"{block['name']} does not contain every arm")
        for ws in spec["world_sizes"]:
            for arm in block["arm_order"]:
                schedule.append(
                    EagerRun(
                        block=block["name"],
                        world_size=int(ws),
                        hidden_size=int(spec["hidden_size"]),
                        arm=arm,
                        bench_impl=spec["arms"][arm]["bench_impl"],
                        m_values=tuple(int(value) for value in block["m_values"]),
                    )
                )
    return schedule


def _result_path(root: Path, run: EagerRun) -> Path:
    return (
        root
        / run.block
        / f"ws-{run.world_size}"
        / f"n-{run.hidden_size}"
        / run.arm
        / "sweep.json"
    )


def _profile(spec: dict, run: EagerRun) -> Path:
    return (REPO_ROOT / spec["profiles"][str(run.world_size)]).resolve()


def _command(spec: dict, run: EagerRun) -> list[str]:
    profile = shlex.quote(str(_profile(spec, run)))
    return [
        "bash",
        "-lc",
        f"source {profile} && exec python3 -m benchmark.probe_ar_rmsnorm_eager_perf",
    ]


def _env(spec: dict, run: EagerRun, devices: str, output: Path) -> dict[str, str]:
    eager = spec["microbenchmark"]["eager"]
    env = dict(os.environ)
    env.update(
        {
            "HIP_VISIBLE_DEVICES": devices,
            "TS_TRITON_SHMEM_VISIBLE_DEVICES": devices,
            "AR_NORM_WORLD_SIZE": str(run.world_size),
            "AR_NORM_DEVICES": devices,
            "GPT_OSS_DEFINITIVE_FUSION_MAX_M": "0",
            "BENCH_WS": str(run.world_size),
            "BENCH_N": str(run.hidden_size),
            "BENCH_M_VALUES": ",".join(str(value) for value in run.m_values),
            "BENCH_MAX_TOKEN_NUM": str(spec["workspace_cap"]),
            "BENCH_N_WARMUP": str(eager["warmups"]),
            "BENCH_N_REPEAT": str(eager["iterations"]),
            "BENCH_IMPL": run.bench_impl,
            "BENCH_JSON": str(output),
            "PYTHONPATH": _runtime_pythonpath(),
        }
    )
    return env


def _validate(path: Path, spec: dict, run: EagerRun) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "mode": "eager",
        "resolved_impl": run.bench_impl,
        "world_size": run.world_size,
        "N": run.hidden_size,
        "max_token_num": int(spec["workspace_cap"]),
        "warmup": int(spec["microbenchmark"]["eager"]["warmups"]),
        "repeat": int(spec["microbenchmark"]["eager"]["iterations"]),
        "M_values": list(run.m_values),
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches or len(payload.get("rows", [])) != len(run.m_values):
        raise ValueError(f"eager result identity mismatch: {mismatches}")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--devices", action="append", default=[])
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

    dry_payload = {
        "dry_run": True,
        "campaign_id": spec["campaign_id"],
        "processes": len(schedule),
        "devices": {str(key): value for key, value in devices.items()},
        "schedule": [asdict(run) for run in schedule],
    }
    if args.dry_run:
        print(json.dumps(dry_payload, indent=2, sort_keys=True))
        return
    if args.output_root is None:
        parser.error("--output-root is required unless --dry-run is used")

    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    _write_json(
        root / "campaign-manifest.json",
        {
            "campaign_id": spec["campaign_id"],
            "started_at": _utc_now(),
            "spec": str(spec_path),
            "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
            "devices": {str(key): value for key, value in devices.items()},
            "processes": len(schedule),
        },
    )

    completed = skipped = failed = 0
    started = time.monotonic()
    current_block = schedule[0].block
    for index, run in enumerate(schedule, start=1):
        if run.block != current_block:
            time.sleep(int(spec["cooldown_seconds_between_blocks"]))
            current_block = run.block
        result = _result_path(root, run)
        result.parent.mkdir(parents=True, exist_ok=True)
        if not args.no_resume and result.exists():
            try:
                _validate(result, spec, run)
            except (ValueError, KeyError, json.JSONDecodeError):
                result.replace(result.with_suffix(f".invalid-{time.time_ns()}.json"))
            else:
                skipped += 1
                continue

        log = result.with_suffix(f".attempt-{time.time_ns()}.log")
        record = {
            "index": index,
            "run": asdict(run),
            "started_at": _utc_now(),
            "result": str(result),
            "log": str(log),
        }
        run_started = time.monotonic()
        try:
            with log.open("w", encoding="utf-8") as handle:
                returncode = _run_process(
                    _command(spec, run),
                    env=_env(spec, run, devices[run.world_size], result),
                    log=handle,
                    timeout=int(spec["process_timeout_seconds"]),
                )
            if returncode != 0:
                raise RuntimeError(f"benchmark exited {returncode}")
            _validate(result, spec, run)
        except (OSError, RuntimeError, subprocess.TimeoutExpired, ValueError) as exc:
            failed += 1
            record["status"] = "failed"
            record["error"] = str(exc)
        else:
            completed += 1
            record["status"] = "completed"
        record["elapsed_seconds"] = time.monotonic() - run_started
        record["finished_at"] = _utc_now()
        with (root / "runtime.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

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
    _write_json(root / "runtime-summary.json", summary)
    raise SystemExit(0 if summary["complete"] else 1)


if __name__ == "__main__":
    main()
