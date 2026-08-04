"""Orchestrate the staged GPT-OSS WS2/4/8 three-arm serving campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from benchmark.run_ar_rmsnorm_graph_sweep import (
    REPO_ROOT,
    _parse_devices,
    _run_process,
)

STAGES = {
    "core": {"blocks": 3, "offset": 0},
    "extension": {"blocks": 2, "offset": 3},
    "promotion": {"blocks": 10, "offset": 5},
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _selected_caps(
    spec: dict,
    policy: dict,
    *,
    dry_run: bool,
) -> dict[int, int]:
    if not dry_run and policy.get("status") != "frozen":
        raise ValueError("selected policy must be frozen before serving")
    allowed = set(spec["cap_policy"]["allowed_fusion_max_m"])
    result = {}
    for ws in spec["world_sizes"]:
        value = policy.get("world_sizes", {}).get(str(ws), {}).get("fusion_max_m")
        if value is None and dry_run:
            value = 0
        if value not in allowed:
            raise ValueError(f"WS={ws} selected invalid fusion_max_m={value}")
        result[int(ws)] = int(value)
    return result


def _command(
    spec: dict,
    *,
    ws: int,
    devices: str,
    cap: int,
    blocks: int,
    block_offset: int,
    stage: str,
    output_root: Path,
) -> list[str]:
    profile = (REPO_ROOT / spec["profiles"][str(ws)]).resolve()
    args = [
        "python3",
        "-m",
        "benchmark.run_ar_rmsnorm_repeatability",
        "--comparison",
        "three_backend",
        "--blocks",
        str(blocks),
        "--block-offset",
        str(block_offset),
        "--seeds",
        "0",
        "--world-size",
        str(ws),
        "--devices",
        devices,
        "--cap",
        str(spec["workspace_cap"]),
        "--decode-only",
        "--disable-overlap-schedule",
        "--definitive-diagnostics",
        "--order-seed",
        str(int(spec["end_to_end"]["arm_order_seed"]) + ws),
        "--run-root",
        str(output_root / f"ws-{ws}" / stage),
    ]
    shell = (
        f"export AR_NORM_DEVICES={shlex.quote(devices)}; "
        f"export GPT_OSS_DEFINITIVE_FUSION_MAX_M={cap}; "
        f"source {shlex.quote(str(profile))} && "
        f"exec {shlex.join(args)}"
    )
    return ["bash", "-lc", shell]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--selected-policy", type=Path)
    parser.add_argument("--devices", action="append", default=[])
    parser.add_argument("--stage", choices=tuple(STAGES), default="core")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    spec_path = args.spec.resolve()
    spec = _load_json(spec_path)
    if spec.get("status") != "planned":
        parser.error("campaign specification must remain planned")
    policy_path = (
        args.selected_policy.resolve()
        if args.selected_policy
        else spec_path.parent / spec["cap_policy"]["selected_policy_file"]
    )
    policy = _load_json(policy_path)
    devices = _parse_devices(args.devices)
    missing = sorted(set(spec["world_sizes"]) - set(devices))
    if missing:
        parser.error(f"missing --devices entries for world sizes {missing}")
    caps = _selected_caps(spec, policy, dry_run=args.dry_run)
    stage_config = STAGES[args.stage]
    blocks = stage_config["blocks"]
    block_offset = stage_config["offset"]
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else Path("/tmp/gpt-oss-definitive-e2e-dry-run")
    )
    commands = {
        ws: _command(
            spec,
            ws=ws,
            devices=devices[ws],
            cap=caps[ws],
            blocks=blocks,
            block_offset=block_offset,
            stage=args.stage,
            output_root=output_root,
        )
        for ws in spec["world_sizes"]
    }
    payload = {
        "dry_run": args.dry_run,
        "campaign_id": spec["campaign_id"],
        "stage": args.stage,
        "triplets_this_stage_per_world_size": blocks,
        "block_offset": block_offset,
        "server_lifecycles": blocks * 3 * len(spec["world_sizes"]),
        "devices": {str(key): value for key, value in devices.items()},
        "fusion_max_m": {str(key): value for key, value in caps.items()},
        "commands": {str(key): value for key, value in commands.items()},
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if args.output_root is None:
        parser.error("--output-root is required unless --dry-run is used")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        **payload,
        "dry_run": False,
        "started_at": _utc_now(),
        "spec": str(spec_path),
        "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        "selected_policy": str(policy_path),
        "selected_policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
    }
    (output_root / f"campaign-manifest-{args.stage}.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    failures = []
    started = time.monotonic()
    for ws in spec["world_sizes"]:
        log_path = output_root / f"ws-{ws}-{args.stage}.log"
        try:
            with log_path.open("w", encoding="utf-8") as log:
                returncode = _run_process(
                    commands[ws],
                    env={
                        **os.environ,
                        "PYTHONPATH": f"{REPO_ROOT / 'python'}:{REPO_ROOT}",
                    },
                    log=log,
                    timeout=int(spec.get("e2e_world_timeout_seconds", 21600)),
                )
            if returncode != 0:
                raise RuntimeError(f"WS={ws} campaign exited {returncode}")
        except (
            OSError,
            RuntimeError,
            subprocess.TimeoutExpired,
        ) as exc:
            failures.append({"world_size": ws, "error": str(exc), "log": str(log_path)})

    failed_world_sizes = {failure["world_size"] for failure in failures}
    if len(failed_world_sizes) < len(spec["world_sizes"]):
        import benchmark.run_ar_rmsnorm_repeatability as repeatability

        repeatability.ARMS = repeatability.comparison_arms("three_backend")
        for ws in spec["world_sizes"]:
            if ws in failed_world_sizes:
                continue
            repeatability.analyze_campaign(
                output_root / f"ws-{ws}",
                bootstrap_samples=10000,
                bootstrap_seed=int(spec["end_to_end"]["arm_order_seed"]) + ws,
            )

    summary = {
        "campaign_id": spec["campaign_id"],
        "stage": args.stage,
        "finished_at": _utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "world_sizes_completed": len(spec["world_sizes"]) - len(failures),
        "failures": failures,
        "complete": not failures,
    }
    (output_root / f"runtime-summary-{args.stage}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    raise SystemExit(0 if summary["complete"] else 1)


if __name__ == "__main__":
    main()
