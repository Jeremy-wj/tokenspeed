"""Run and analyze a restart-randomized AR+RMSNorm serving campaign.

The runner is deliberately model-profiled and conservative:

* every arm gets a fresh server;
* arm order is randomized independently in each restart block;
* prompt seeds and workloads are paired within a block;
* detailed request timelines and raw console logs are retained;
* every arm ends with a bounded no-stack trace and per-rank signature proof;
* all generated artifacts receive SHA256 checksums;
* paired hierarchical bootstrap intervals preserve restart-block structure.

The default post-rebase campaign compares Iris-first ``auto`` against the
upstream-unfused TP=4/N=2880 control. ``--comparison triton_shmem`` evaluates
the explicit local candidate against the same control. ``--stability-only``
runs only the control and is the canonical residual-runtime gate. Use
``--dry-run`` to inspect the schedule without touching a server or GPU.
"""
from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import os
import random
import re
import shlex
import signal
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = REPO_ROOT / "benchmark"
DEFAULT_RAW_ROOT = (
    REPO_ROOT
    / "benchmark/results/ar_rmsnorm/raw/current/gpt-oss-120b/mi350x"
)
GPU_LOCK_PATH = Path("/tmp/tokenspeed-ar-rmsnorm-gpu.lock")
SERVE_SCRIPT = BENCHMARK_DIR / "e2e_gptoss_serve.sh"
BENCH_SCRIPT = BENCHMARK_DIR / "e2e_gptoss_bench.sh"
TEARDOWN_SCRIPT = BENCHMARK_DIR / "e2e_gptoss_teardown.sh"


@dataclass(frozen=True)
class Arm:
    name: str
    fusion_max_m: int
    fusion_enabled: int = 1
    backend: str = "triton_shmem"
    signature_family: str = "triton_shmem_fused"
    double_buffer_input: int | None = None


@dataclass(frozen=True)
class Workload:
    name: str
    input_len: int
    output_len: int
    prompts: int
    concurrency: int
    objective: str


ARMS = (
    Arm("cap2048", 0),
    Arm("cap2048-gate256", 256),
)


def _upstream_unfused_arm() -> Arm:
    return Arm(
        "upstream_unfused",
        0,
        fusion_enabled=0,
        backend="auto",
        signature_family="production_unfused",
        double_buffer_input=0,
    )


def comparison_arms(name: str) -> tuple[Arm, Arm]:
    if name == "gate256":
        return ARMS
    if name == "iris":
        return (
            _upstream_unfused_arm(),
            Arm(
                "iris_auto",
                0,
                fusion_enabled=1,
                backend="auto",
                signature_family="iris_fused",
            ),
        )
    if name == "triton_shmem":
        return (
            _upstream_unfused_arm(),
            Arm(
                "triton_shmem",
                0,
                fusion_enabled=1,
                backend="triton_shmem",
                signature_family="triton_shmem_fused",
            ),
        )
    if name == "unfused":
        return (
            Arm(
                "unfused",
                0,
                fusion_enabled=0,
                backend="triton_shmem",
                signature_family="production_unfused",
                double_buffer_input=0,
            ),
            Arm("fused_stable", 0, fusion_enabled=1),
        )
    if name == "input_ring":
        return (
            Arm("single_input", 0, fusion_enabled=1, double_buffer_input=0),
            Arm("double_input", 0, fusion_enabled=1, double_buffer_input=1),
        )
    raise ValueError(f"unknown comparison {name!r}")


def campaign_arms(name: str, *, stability_only: bool = False) -> tuple[Arm, ...]:
    """Resolve paired promotion arms or the single unfused stability arm."""
    arms = comparison_arms(name)
    if not stability_only:
        return arms
    if name not in ("unfused", "iris", "triton_shmem"):
        raise ValueError(
            "--stability-only requires an upstream-unfused comparison"
        )
    return (arms[0],)


WORKLOADS = (
    Workload("prefill-m128", 4, 8, 128, 32, "ttft"),
    Workload("prefill-m256", 8, 8, 128, 32, "ttft"),
    Workload("prefill-m512", 16, 8, 128, 32, "ttft"),
    Workload("prefill-m1024", 32, 8, 128, 32, "ttft"),
    Workload("prefill-m2048", 64, 8, 128, 32, "ttft"),
    Workload("prefill-m4096", 128, 8, 128, 32, "ttft"),
    Workload("decode", 128, 512, 128, 32, "tpot_capacity"),
)

METRICS = (
    "median_tpot_ms",
    "mean_tpot_ms",
    "median_ttft_ms",
    "mean_ttft_ms",
    "output_throughput",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_csv_ints(raw: str) -> list[int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def build_schedule(
    blocks: int,
    seeds: Sequence[int],
    order_seed: int,
) -> list[dict[str, Any]]:
    """Return a deterministic block schedule with randomized arm/seed order."""
    rng = random.Random(order_seed)
    schedule = []
    for block in range(blocks):
        arm_order = [arm.name for arm in ARMS]
        rng.shuffle(arm_order)
        seed_order = list(seeds)
        rng.shuffle(seed_order)
        schedule.append(
            {
                "block": block,
                "arm_order": arm_order,
                "seed_order": seed_order,
            }
        )
    return schedule


def parse_amd_smi_processes(payload: str) -> dict[int, list[dict[str, Any]]]:
    """Parse ``amd-smi process --json`` into non-empty process records."""
    def numeric(value: Any) -> float:
        try:
            return float(value or 0)
        except (TypeError, ValueError):
            return 0.0

    parsed = json.loads(payload)
    result: dict[int, list[dict[str, Any]]] = {}
    for gpu_record in parsed:
        gpu = int(gpu_record["gpu"])
        active = []
        for wrapper in gpu_record.get("process_list", []):
            info = wrapper.get("process_info", {})
            vram = (
                info.get("memory_usage", {})
                .get("vram_mem", {})
                .get("value", 0)
            )
            occupancy = info.get("cu_occupancy", 0)
            if numeric(vram) > 0 or numeric(occupancy) > 0:
                active.append(info)
        if active:
            result[gpu] = active
    return result


def partition_live_gpu_processes(
    active: dict[int, list[dict[str, Any]]],
    *,
    pid_exists=None,
) -> tuple[
    dict[int, list[dict[str, Any]]],
    dict[int, list[dict[str, Any]]],
]:
    """Separate live jobs from transient AMD-SMI query contexts.

    Some AMD-SMI builds briefly report the querying process itself with real
    VRAM/CU usage. That PID has exited by the time ``process --json`` returns
    and must not be treated as foreign contention. Preserve it as diagnostic
    evidence rather than silently discarding it.
    """
    if pid_exists is None:
        pid_exists = lambda pid: Path(f"/proc/{pid}").exists()
    live: dict[int, list[dict[str, Any]]] = {}
    transient: dict[int, list[dict[str, Any]]] = {}
    for gpu, records in active.items():
        for record in records:
            pid = int(record.get("pid", -1))
            target = live if pid > 0 and pid_exists(pid) else transient
            target.setdefault(gpu, []).append(record)
    return live, transient


def hip_to_physical_gpu_map() -> dict[int, int]:
    """Map HIP visibility indices to AMD-SMI physical indices without contexts."""
    records = json.loads(
        subprocess.check_output(["amd-smi", "list", "--json"], text=True)
    )
    ordered = sorted(records, key=lambda record: int(record["node_id"]))
    return {
        hip_index: int(record["gpu"])
        for hip_index, record in enumerate(ordered)
    }


def physical_to_kfd_id_map() -> dict[int, int]:
    """Map physical AMD-SMI GPU indices to nonintrusive KFD sysfs IDs."""
    records = json.loads(
        subprocess.check_output(["amd-smi", "list", "--json"], text=True)
    )
    return {
        int(record["gpu"]): int(record["kfd_id"])
        for record in records
    }


def read_kfd_gpu_processes(
    physical_to_kfd_id: dict[int, int],
    *,
    root: Path = Path("/sys/class/kfd/kfd/proc"),
) -> dict[int, list[dict[str, Any]]]:
    """Read active GPU PIDs from KFD sysfs without creating a GPU context."""
    result: dict[int, list[dict[str, Any]]] = {}
    try:
        process_dirs = list(root.iterdir())
    except OSError:
        return result
    for process_dir in process_dirs:
        if not process_dir.name.isdigit():
            continue
        pid = int(process_dir.name)
        try:
            command = Path(f"/proc/{pid}/comm").read_text(
                encoding="utf-8"
            ).strip()
        except OSError:
            command = ""
        for gpu, kfd_id in physical_to_kfd_id.items():
            try:
                vram = int(
                    (process_dir / f"vram_{kfd_id}").read_text(
                        encoding="utf-8"
                    ).strip()
                )
            except (OSError, ValueError):
                vram = 0
            try:
                occupancy = int(
                    (
                        process_dir
                        / f"stats_{kfd_id}"
                        / "cu_occupancy"
                    ).read_text(encoding="utf-8").strip()
                )
            except (OSError, ValueError):
                occupancy = 0
            if vram > 0 or occupancy > 0:
                result.setdefault(gpu, []).append(
                    {
                        "pid": pid,
                        "name": command,
                        "vram_bytes": vram,
                        "cu_occupancy": occupancy,
                        "source": "kfd_sysfs",
                    }
                )
    return result


def hierarchical_bootstrap(
    values_by_block: dict[int, list[float]],
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    """Bootstrap a paired mean while preserving restart-block clustering."""
    if not values_by_block:
        raise ValueError("cannot bootstrap an empty sample")
    block_ids = sorted(values_by_block)
    if any(not values_by_block[block] for block in block_ids):
        raise ValueError("every bootstrap block must have observations")

    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        sampled_values: list[float] = []
        for block in rng.choices(block_ids, k=len(block_ids)):
            block_values = values_by_block[block]
            sampled_values.extend(
                rng.choices(block_values, k=len(block_values))
            )
        draws.append(statistics.fmean(sampled_values))
    draws.sort()

    def percentile(fraction: float) -> float:
        index = round((len(draws) - 1) * fraction)
        return draws[index]

    flattened = [
        value
        for block in block_ids
        for value in values_by_block[block]
    ]
    return {
        "mean": statistics.fmean(flattened),
        "ci95_low": percentile(0.025),
        "ci95_high": percentile(0.975),
        "n_blocks": len(block_ids),
        "n_pairs": len(flattened),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _acquire_gpu_campaign_lock(path: Path = GPU_LOCK_PATH):
    """Fail fast if another local AR+RMSNorm GPU campaign is active."""
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.seek(0)
        owner = handle.read().strip() or "unknown owner"
        handle.close()
        raise RuntimeError(f"GPU campaign lock {path} is held by {owner}")
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()} argv={shlex.join(sys.argv)}")
    handle.flush()
    return handle


def _run(
    command: Sequence[str],
    *,
    env: dict[str, str] | None = None,
    cwd: Path = REPO_ROOT,
    log: Path | None = None,
    dry_run: bool = False,
    check: bool = True,
    timeout_seconds: int | None = None,
) -> subprocess.CompletedProcess[str] | None:
    rendered = " ".join(command)
    print(f"+ {rendered}", flush=True)
    if dry_run:
        return None
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    timed_out = False
    try:
        output, _ = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            output, _ = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
    result = subprocess.CompletedProcess(
        command,
        124 if timed_out else process.returncode,
        output,
    )
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"$ {rendered}\n")
            handle.write(result.stdout)
            if not result.stdout.endswith("\n"):
                handle.write("\n")
    else:
        print(result.stdout, end="")
    if check and result.returncode != 0:
        if timed_out:
            raise TimeoutError(
                f"command exceeded {timeout_seconds}s: {rendered}"
            )
        raise subprocess.CalledProcessError(
            result.returncode,
            command,
            output=result.stdout,
        )
    return result


def _git_metadata() -> dict[str, Any]:
    def output(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
        ).strip()

    diff = subprocess.check_output(
        ["git", "diff", "--binary"],
        cwd=REPO_ROOT,
    )
    return {
        "commit": output("rev-parse", "HEAD"),
        "branch": output("branch", "--show-current"),
        "status_short": output("status", "--short"),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "scripts": {
            str(path.relative_to(REPO_ROOT)): _sha256(path)
            for path in (
                Path(__file__).resolve(),
                SERVE_SCRIPT,
                BENCH_SCRIPT,
                TEARDOWN_SCRIPT,
                BENCHMARK_DIR / "e2e_arnorm_serve.sh",
                BENCHMARK_DIR / "e2e_arnorm_bench.sh",
                BENCHMARK_DIR
                / "profiles/ar_rmsnorm/gpt_oss_120b_mi350x.env",
                REPO_ROOT
                / "python/tokenspeed_kernel/ops/communication/triton.py",
                REPO_ROOT
                / "python/tokenspeed_kernel/ops/communication/triton_shmem.py",
                REPO_ROOT
                / "python/tokenspeed_kernel/ops/communication/_triton_shmem_kernels.py",
            )
        },
    }


def _docker_image(container: str) -> dict[str, str]:
    result = subprocess.check_output(
        [
            "docker",
            "inspect",
            "--format",
            "{{.Config.Image}} {{.Image}} {{.State.Status}}",
            container,
        ],
        text=True,
    ).strip()
    image, image_id, status = result.split(maxsplit=2)
    return {"image": image, "image_id": image_id, "status": status}


def _preflight(
    root: Path,
    *,
    container: str,
    selected_gpus: set[int],
    ignored_busy_gpus: set[int],
    dry_run: bool,
) -> dict[str, Any]:
    if dry_run:
        return {
            "time": _utc_now(),
            "dry_run": True,
            "selected_gpus": sorted(selected_gpus),
            "ignored_busy_gpus": sorted(ignored_busy_gpus),
        }
    process_json = subprocess.check_output(
        ["amd-smi", "process", "--json"],
        text=True,
    )
    observed = parse_amd_smi_processes(process_json)
    # AMD-SMI 26.2 can report a short-lived query helper with real VRAM usage.
    # It exits about 0.5s after the command returns; retain it as transient
    # evidence instead of rejecting an otherwise idle rank set.
    time.sleep(0.6)
    active, transient = partition_live_gpu_processes(observed)
    unexpected = {
        gpu: records
        for gpu, records in active.items()
        if gpu in selected_gpus and gpu not in ignored_busy_gpus
    }
    snapshot = {
        "time": _utc_now(),
        "ignored_busy_gpus": sorted(ignored_busy_gpus),
        "selected_gpus": sorted(selected_gpus),
        "active_processes": active,
        "transient_processes": transient,
        "unexpected_active_processes": unexpected,
        "container": _docker_image(container),
    }
    _write_json(root / "preflight.json", snapshot)
    (root / "amd-smi-process.json").write_text(
        process_json,
        encoding="utf-8",
    )
    if snapshot["container"]["status"] != "running":
        raise RuntimeError(f"container {container!r} is not running")
    if unexpected:
        raise RuntimeError(
            "GPU preflight found unexpected active processes: "
            + ", ".join(str(gpu) for gpu in sorted(unexpected))
        )
    return snapshot


def _container_host_pids(container: str) -> set[int]:
    output = subprocess.check_output(
        ["docker", "top", container, "-eo", "pid"],
        text=True,
    )
    return {
        int(line.strip())
        for line in output.splitlines()[1:]
        if line.strip().isdigit()
    }


def _assert_gpu_isolation(
    phase_dir: Path,
    *,
    container: str,
    selected_gpus: set[int],
    ignored_busy_gpus: set[int],
    physical_to_kfd_id: dict[int, int] | None = None,
    known_container_pids: set[int] | None = None,
) -> None:
    """Reject foreign jobs using nonintrusive KFD process metadata."""
    if physical_to_kfd_id is None:
        process_json = subprocess.check_output(
            ["amd-smi", "process", "--json"],
            text=True,
        )
        observed = parse_amd_smi_processes(process_json)
        time.sleep(0.6)
        active, transient = partition_live_gpu_processes(observed)
        source = "amd-smi-process"
    else:
        active = read_kfd_gpu_processes(physical_to_kfd_id)
        transient = {}
        source = "kfd-sysfs"
    container_pids = _container_host_pids(container)
    if known_container_pids is not None:
        known_container_pids.update(container_pids)
        allowed_pids = known_container_pids
    else:
        allowed_pids = container_pids
    unexpected = {}
    for gpu, records in active.items():
        if gpu not in selected_gpus:
            continue
        if gpu in ignored_busy_gpus:
            continue
        foreign = [
            record
            for record in records
            if int(record.get("pid", -1)) not in allowed_pids
        ]
        if foreign:
            unexpected[gpu] = foreign
    guard_record = {
        "time": _utc_now(),
        "container_pids": sorted(container_pids),
        "selected_gpus": sorted(selected_gpus),
        "active_processes": active,
        "transient_processes": transient,
        "unexpected_active_processes": unexpected,
        "source": source,
    }
    guard_path = phase_dir / "gpu-guard.jsonl"
    with guard_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(guard_record, sort_keys=True) + "\n")
    if unexpected:
        raise RuntimeError(
            "GPU isolation guard found a foreign process on GPU(s): "
            + ", ".join(str(gpu) for gpu in sorted(unexpected))
        )


def _capture_gpu_diagnostics(
    phase_dir: Path,
    *,
    label: str,
    container: str,
    selected_gpus: set[int],
    physical_to_kfd_id: dict[int, int],
    include_amd_smi_metrics: bool = False,
) -> dict[str, Any]:
    """Capture bounded process/utilization evidence at phase boundaries.

    Continuous two-second monitoring remains process-only to avoid perturbing
    benchmark timing. These richer snapshots run before/after a phase and on
    failure, when the additional ``amd-smi metric`` cost is acceptable.
    """
    phase_dir.mkdir(parents=True, exist_ok=True)
    commands = {
        "container_top": [
            "docker",
            "top",
            container,
            "-eo",
            "pid,ppid,stat,etime,args",
        ],
    }
    captured: dict[str, Any] = {
        "time": _utc_now(),
        "label": label,
        "selected_gpus": sorted(selected_gpus),
        "kfd_processes": read_kfd_gpu_processes(physical_to_kfd_id),
        "commands": {},
    }
    if include_amd_smi_metrics:
        gpu_args = [str(gpu) for gpu in sorted(selected_gpus)]
        commands["metric"] = [
            "amd-smi",
            "metric",
            "--usage",
            "--mem-usage",
            "--power",
            "--clock",
            "--temperature",
            "--xgmi-err",
            "--gpu",
            *gpu_args,
            "--json",
        ]
    for name, command in commands.items():
        try:
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            record: dict[str, Any] = {
                "command": command,
                "returncode": completed.returncode,
                "stderr": completed.stderr,
            }
            if name == "metric" and completed.returncode == 0:
                try:
                    record["payload"] = json.loads(completed.stdout)
                except json.JSONDecodeError:
                    record["stdout"] = completed.stdout
            else:
                record["stdout"] = completed.stdout
            captured["commands"][name] = record
        except BaseException as exc:  # diagnostics must not mask the root fault
            captured["commands"][name] = {
                "command": command,
                "error": f"{type(exc).__name__}: {exc}",
            }
    path = phase_dir / "gpu-diagnostics.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(captured, sort_keys=True) + "\n")
    return captured


def _teardown(
    *,
    env: dict[str, str],
    log: Path,
    dry_run: bool,
) -> None:
    _run(
        ["bash", str(TEARDOWN_SCRIPT)],
        env=env,
        log=log,
        dry_run=dry_run,
        check=False,
        timeout_seconds=60,
    )


def _wait_for_health(
    container: str,
    *,
    port: int,
    timeout_seconds: int,
    log: Path,
    dry_run: bool,
    serve_log: Path | None = None,
) -> None:
    if dry_run:
        print(
            f"+ wait up to {timeout_seconds}s for "
            f"http://127.0.0.1:{port}/health"
        )
        return
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        result = _run(
            [
                "docker",
                "exec",
                container,
                "curl",
                "-fsS",
                f"http://127.0.0.1:{port}/health",
            ],
            log=log,
            check=False,
            timeout_seconds=10,
        )
        if result is not None and result.returncode == 0:
            return
        if serve_log is not None and serve_log.exists():
            text = serve_log.read_text(encoding="utf-8", errors="replace")
            fatal_signatures = (
                "startup failed:",
                "metrics server bind failed:",
                "Memory access fault by GPU node",
                "HSA_STATUS_ERROR",
                "Fatal Python error:",
                "ModuleNotFoundError:",
                "ImportError:",
            )
            fatal_line = next(
                (
                    line
                    for line in reversed(text.splitlines())
                    if any(signature in line for signature in fatal_signatures)
                ),
                None,
            )
            if fatal_line is not None:
                raise RuntimeError(
                    f"server failed before health: {fatal_line.strip()}"
                )
        time.sleep(5)
    raise TimeoutError(f"server did not become healthy in {timeout_seconds}s")


def _bench_command(
    label: str,
    workload: Workload,
    seed: int,
    *,
    output_file: Path | None = None,
    extra: Iterable[str] = (),
    ready_check: bool = True,
) -> list[str]:
    command = [
        "bash",
        str(BENCH_SCRIPT),
        label,
        str(workload.input_len),
        str(workload.output_len),
        str(workload.prompts),
        str(workload.concurrency),
        str(seed),
        "--disable-tqdm",
        "--request-id-prefix",
        f"{label}-seed{seed}-",
    ]
    if not ready_check:
        command.extend(["--ready-check-timeout-sec", "0"])
    if output_file is not None:
        command.extend(
            [
                "--save-result",
                "--save-detailed",
                "--output-file",
                str(output_file),
            ]
        )
    command.extend(extra)
    return command


def _trace_counts(path: Path) -> dict[str, int]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        events = json.load(handle).get("traceEvents", [])
    names = [
        str(event.get("name", ""))
        for event in events
        if event.get("cat") == "kernel"
    ]
    lowered = [name.lower() for name in names]
    return {
        "oneshot": sum("fused_ar_rmsnorm_oneshot" in name for name in names),
        "twoshot": sum("fused_ar_rmsnorm_twoshot" in name for name in names),
        "iris_fused": sum(
            "iris_allreduce_residual_rmsnorm_kernel" in name for name in names
        ),
        "symm_mem_fused": sum(
            "amd_allreduce_residual_rmsnorm_kernel" in name for name in names
        ),
        "iris_all_reduce": sum(
            "iris_stage_one_shot_allreduce_kernel" in name for name in names
        ),
        "legacy_all_reduce": sum(
            "amd_all_reduce_kernel" in name for name in names
        ),
        "rccl": sum(
            (
                "nccl" in name
                or "rccl" in name
                or "reduce_kernel" in name
            )
            and "rmsnorm" not in name
            for name in lowered
        ),
        "rmsnorm": sum("_rmsnorm_kernel" in name for name in names),
    }


def validate_trace_signatures(
    trace_dir: Path,
    *,
    world_size: int,
    arm: Arm,
    require_prefill_twoshot: bool = True,
) -> dict[str, Any]:
    traces = sorted(trace_dir.glob("*.trace.json*"))
    if len(traces) != world_size:
        raise RuntimeError(
            f"expected {world_size} traces in {trace_dir}, found {len(traces)}"
        )
    ranks = []
    for trace in traces:
        if trace.stat().st_size == 0:
            raise RuntimeError(f"empty trace: {trace}")
        counts = _trace_counts(trace)
        if not arm.fusion_enabled:
            fused_count = (
                counts["oneshot"]
                + counts["twoshot"]
                + counts["iris_fused"]
                + counts["symm_mem_fused"]
            )
            ordinary_count = (
                counts["iris_all_reduce"]
                + counts["legacy_all_reduce"]
                + counts["rccl"]
            )
            if fused_count or counts["rmsnorm"] < 73 or ordinary_count == 0:
                raise RuntimeError(
                    f"unfused arm signature mismatch in {trace}: {counts}"
                )
            ranks.append({"trace": trace.name, **counts})
            continue
        if arm.signature_family == "iris_fused":
            if counts["iris_fused"] == 0:
                raise RuntimeError(
                    f"missing Iris fused signature in {trace}: {counts}"
                )
            if counts["oneshot"] or counts["twoshot"]:
                raise RuntimeError(
                    f"Iris arm entered triton_shmem in {trace}: {counts}"
                )
        elif arm.signature_family == "triton_shmem_fused":
            if counts["oneshot"] == 0:
                raise RuntimeError(
                    f"missing triton_shmem fused decode signature in {trace}"
                )
            if counts["iris_fused"]:
                raise RuntimeError(
                    f"triton_shmem arm entered Iris in {trace}: {counts}"
                )
        else:
            raise RuntimeError(
                f"unsupported fused signature family {arm.signature_family!r}"
            )
        if arm.signature_family == "triton_shmem_fused" and arm.fusion_max_m:
            if counts["twoshot"] != 0 or counts["rmsnorm"] < 73:
                raise RuntimeError(
                    f"gate arm did not use unfused prefill in {trace}: {counts}"
                )
        elif (
            arm.signature_family == "triton_shmem_fused"
            and require_prefill_twoshot
            and counts["twoshot"] == 0
        ):
            raise RuntimeError(
                f"baseline arm missing fused two-shot prefill in {trace}"
            )
        ranks.append({"trace": trace.name, **counts})
    return {"status": "passed", "ranks": ranks}


def _serve_proof(
    serve_log: Path,
    arm: Arm,
    cap: int,
    barrier_grid: int,
    inkernel_barrier: int,
    engine_module: str,
    deep_health_mode: str,
    triton_ar_disable: int = 0,
    double_buffer_input: int = 0,
    disable_overlap_schedule: bool = False,
) -> dict[str, Any]:
    text = serve_log.read_text(encoding="utf-8", errors="replace")
    run_env = next(
        (line for line in text.splitlines() if line.startswith("RUN_ENV ")),
        None,
    )
    if run_env is None:
        raise RuntimeError(f"RUN_ENV missing from {serve_log}")
    required = (
        f"CAP={cap}",
        f"BACKEND={arm.backend}",
        f"ENABLE_FUSION={arm.fusion_enabled}",
        f"FUSION_MAX_M={arm.fusion_max_m}",
        f"DOUBLE_BUFFER_INPUT={double_buffer_input}",
        f"BARRIER_GRID={barrier_grid}",
        f"INKERNEL={inkernel_barrier}",
        f"ENGINE_MODULE={engine_module}",
        f"DEEP_HEALTH_MODE={deep_health_mode}",
        f"TRITON_AR_DISABLE={triton_ar_disable}",
    )
    missing = [token for token in required if token not in run_env]
    if missing:
        raise RuntimeError(f"serve proof missing {missing}: {run_env}")
    resolved_flag = bool(arm.fusion_enabled)
    if f"enable_allreduce_fusion={resolved_flag}" not in text:
        raise RuntimeError(
            f"resolved fusion flag {resolved_flag} missing from {serve_log}"
        )
    if disable_overlap_schedule and "disable_overlap_schedule=True" not in text:
        raise RuntimeError(f"overlap schedule remained enabled in {serve_log}")
    if arm.signature_family == "triton_shmem_fused":
        state_lines = [
            line
            for line in text.splitlines()
            if "triton_shmem AR+RMSNorm state:" in line
        ]
        if not state_lines or not all(
            f"max_tokens={cap}" in line for line in state_lines
        ):
            raise RuntimeError(f"workspace state proof missing from {serve_log}")
    elif arm.signature_family == "iris_fused":
        state_lines = [
            line
            for line in text.splitlines()
            if "Iris AR+RMSNorm symmetric-heap buffer allocated" in line
        ]
        if not state_lines:
            raise RuntimeError(f"Iris state proof missing from {serve_log}")
    else:
        state_lines = []
    resolved_backend_lines = [
        line
        for line in text.splitlines()
        if "AR+RMSNorm backend resolved:" in line
    ]
    if arm.signature_family == "triton_shmem_fused" and not any(
        "selected=triton_shmem" in line for line in resolved_backend_lines
    ):
        raise RuntimeError(f"triton_shmem resolution proof missing from {serve_log}")
    if arm.signature_family == "iris_fused" and not any(
        "selected=iris" in line for line in resolved_backend_lines
    ):
        raise RuntimeError(f"Iris resolution proof missing from {serve_log}")
    return {
        "run_env": run_env,
        "state_lines": state_lines,
        "backend": arm.backend,
        "signature_family": arm.signature_family,
        "resolved_enable_allreduce_fusion": resolved_flag,
        "resolved_backend_lines": resolved_backend_lines,
    }


def _qualified_profile_proof(serve_log: Path) -> dict[str, Any]:
    """Fail closed if a promotion phase drifted from the qualified profile."""
    text = serve_log.read_text(encoding="utf-8", errors="replace")
    run_env = next(
        (line for line in text.splitlines() if line.startswith("RUN_ENV ")),
        None,
    )
    if run_env is None:
        raise RuntimeError(f"RUN_ENV missing from {serve_log}")
    required_run_env = (
        "PROFILE_ID=gpt-oss-120b-mi350x-post-rebase-v1",
        "DEEP_HEALTH_MODE=passive",
        "FOLD_COPYIN=0",
        "SHMEM_OUTPUT_RING=72",
        "DOUBLE_BUFFER_INPUT=0",
        "BARRIER_GRID=0",
        "FORWARD_MARKERS=1",
    )
    required_server_args = {
        "gpu_memory_utilization=0.9": (
            r"\bgpu_memory_utilization=0\.9(?:,|\))"
        ),
        "cudagraph_capture_sizes=[32]": (
            r"\bcudagraph_capture_sizes=\[32\](?:,|\))"
        ),
        "disable_prefill_graph=True": (
            r"\bdisable_prefill_graph=True(?:,|\))"
        ),
        "disable_overlap_schedule=True": (
            r"\bdisable_overlap_schedule=True(?:,|\))"
        ),
    }
    missing = [token for token in required_run_env if token not in run_env]
    missing.extend(
        label
        for label, pattern in required_server_args.items()
        if re.search(pattern, text) is None
    )
    if missing:
        raise RuntimeError(
            f"qualified profile proof missing {missing} from {serve_log}"
        )
    return {
        "status": "passed",
        "run_env": run_env,
        "required_server_args": list(required_server_args),
    }


def _write_checksums(root: Path) -> Path:
    checksum_path = root / "checksums.sha256"
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != checksum_path:
            rows.append(f"{_sha256(path)}  {path.relative_to(root)}")
    checksum_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return checksum_path


def _arm_environment(
    args: argparse.Namespace,
    *,
    arm: Arm,
    arm_dir: Path,
    label: str,
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CONTAINER": args.container,
            "RUN_DATE": args.run_date,
            "RUN_ROOT": str(args.run_root),
            "LOG_DIR": str(arm_dir),
            "RUN_LABEL": label,
            "PORT": str(args.port),
            "ENABLE_ALLREDUCE_FUSION": str(arm.fusion_enabled),
            "TS_TRITON_SHMEM_FUSION_MAX_M": str(arm.fusion_max_m),
            "TS_TRITON_SHMEM_DOUBLE_BUFFER_INPUT": str(
                (
                    arm.double_buffer_input
                    if arm.double_buffer_input is not None
                    else args.double_buffer_input
                )
            ),
            "TS_TRITON_SHMEM_BARRIER_GRID": str(args.barrier_grid),
            "TS_TRITON_SHMEM_INKERNEL_BARRIER": str(args.inkernel_barrier),
            "TS_TRITON_AR_DISABLE": str(args.triton_ar_disable),
            "TS_SERVE_ENGINE_MODULE": args.engine_module,
            "TOKENSPEED_DEEP_HEALTH_MODE": args.deep_health_mode,
            "TOKENSPEED_PROFILE_WITH_STACK": "0",
            "TOKENSPEED_PROFILE_FORWARD_MARKERS": "1",
        }
    )
    return env


def _start_phase(
    args: argparse.Namespace,
    *,
    arm: Arm,
    phase_dir: Path,
    label: str,
) -> tuple[dict[str, str], Path]:
    phase_dir.mkdir(parents=True, exist_ok=True)
    log = phase_dir / "orchestration.log"
    env = _arm_environment(
        args,
        arm=arm,
        arm_dir=phase_dir,
        label=label,
    )
    _teardown(env=env, log=log, dry_run=args.dry_run)
    if args.restart_container_per_server and not args.dry_run:
        _run(
            ["docker", "restart", args.container],
            log=log,
            timeout_seconds=60,
        )
    _preflight(
        phase_dir,
        container=args.container,
        selected_gpus=args.selected_physical_gpus,
        ignored_busy_gpus=args.ignored_busy_gpus,
        dry_run=args.dry_run,
    )
    serve_command = [
        "bash",
        str(SERVE_SCRIPT),
        str(args.world_size),
        args.devices,
        str(args.cap),
        arm.backend,
    ]
    if args.disable_overlap_schedule:
        serve_command.append("--disable-overlap-schedule")
    serve_command.extend(shlex.split(args.serve_extra_args))
    _run(
        serve_command,
        env=env,
        log=log,
        dry_run=args.dry_run,
        timeout_seconds=60,
    )
    _wait_for_health(
        args.container,
        port=args.port,
        timeout_seconds=args.health_timeout,
        log=log,
        dry_run=args.dry_run,
        serve_log=phase_dir / f"serve-{label}.log",
    )
    if not args.dry_run:
        if args.comparison in ("unfused", "iris", "triton_shmem"):
            profile_proof = _qualified_profile_proof(
                phase_dir / f"serve-{label}.log"
            )
            _write_json(
                phase_dir / "qualified-profile-proof.json", profile_proof
            )
        _assert_gpu_isolation(
            phase_dir,
            container=args.container,
            selected_gpus=args.selected_physical_gpus,
            ignored_busy_gpus=args.ignored_busy_gpus,
            physical_to_kfd_id=args.physical_to_kfd_id,
        )
    return env, log


def _guard_phase(args: argparse.Namespace, phase_dir: Path) -> None:
    if not args.dry_run:
        _assert_gpu_isolation(
            phase_dir,
            container=args.container,
            selected_gpus=args.selected_physical_gpus,
            ignored_busy_gpus=args.ignored_busy_gpus,
            physical_to_kfd_id=args.physical_to_kfd_id,
        )


def _gpu_guard_history_is_clean(phase_dir: Path) -> bool:
    """Return whether the latest phase attempt has records and no contamination."""
    path = phase_dir / "gpu-guard.jsonl"
    if not path.exists():
        return False
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    preflight_path = phase_dir / "preflight.json"
    if preflight_path.exists():
        attempt_start = json.loads(
            preflight_path.read_text(encoding="utf-8")
        ).get("time")
        if attempt_start:
            records = [
                record
                for record in records
                if str(record.get("time", "")) >= str(attempt_start)
            ]
    return bool(records) and not any(
        record.get("unexpected_active_processes") for record in records
    )


def _run_guarded_benchmark(
    command: Sequence[str],
    *,
    args: argparse.Namespace,
    phase_dir: Path,
    env: dict[str, str],
    log: Path,
) -> subprocess.CompletedProcess[str] | None:
    """Run a benchmark while sampling foreign GPU PIDs every two seconds."""
    stop = threading.Event()
    monitor_error: list[BaseException] = []
    known_container_pids = (
        set() if args.dry_run else _container_host_pids(args.container)
    )

    def monitor() -> None:
        while not stop.wait(2.0):
            try:
                _assert_gpu_isolation(
                    phase_dir,
                    container=args.container,
                    selected_gpus=args.selected_physical_gpus,
                    ignored_busy_gpus=args.ignored_busy_gpus,
                    physical_to_kfd_id=args.physical_to_kfd_id,
                    known_container_pids=known_container_pids,
                )
            except BaseException as exc:
                monitor_error.append(exc)
                return

    thread = None
    if not args.dry_run:
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
    command_error = None
    result = None
    try:
        result = _run(
            command,
            env=env,
            log=log,
            dry_run=args.dry_run,
            timeout_seconds=args.benchmark_timeout,
        )
    except BaseException as exc:
        command_error = exc
    finally:
        stop.set()
        if thread is not None:
            thread.join(timeout=5)
    if monitor_error:
        raise RuntimeError(
            f"GPU isolation changed during benchmark: {monitor_error[0]}"
        ) from monitor_error[0]
    if command_error is not None:
        raise command_error
    return result


def _run_arm(
    args: argparse.Namespace,
    *,
    block: int,
    position: int,
    arm: Arm,
    seed_order: Sequence[int],
) -> dict[str, Any]:
    label = f"b{block:02d}-p{position}-{arm.name}"
    arm_double_buffer = (
        arm.double_buffer_input
        if arm.double_buffer_input is not None
        else args.double_buffer_input
    )
    arm_dir = args.run_root / f"block-{block:02d}" / f"p{position}-{arm.name}"
    result_dir = arm_dir / "results"
    completion = arm_dir / "arm-summary.json"
    if args.resume and completion.exists():
        prior = json.loads(completion.read_text(encoding="utf-8"))
        if prior.get("status") == "complete":
            print(f"Skipping completed {label}")
            return prior

    arm_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    started = _utc_now()
    summary: dict[str, Any] = {
        "status": "running",
        "arm": asdict(arm),
        "block": block,
        "position": position,
        "label": label,
        "seed_order": list(seed_order),
        "started_at": started,
    }
    _write_json(completion, summary)

    active_env: dict[str, str] | None = None
    active_log: Path | None = None
    try:
        produced_results = []
        serve_proofs: dict[str, Any] = {}

        # Every decode seed gets a fresh server. Reusing one server across long
        # benchmark clients has independently reproduced scheduler/GPU stalls,
        # even with M-independent separate barriers.
        decode_warmup = Workload("decode-warmup", 1, 16, 32, 32, "warmup")
        decode_workload = next(
            workload for workload in WORKLOADS if workload.name == "decode"
        )
        for seed in seed_order:
            decode_dir = arm_dir / "decode-phase" / f"seed-{seed}"
            decode_label = label + f"-decode-seed{seed}"
            output_file = result_dir / f"decode-seed{seed}.json"
            serve_log = decode_dir / f"serve-{decode_label}.log"
            if args.resume and output_file.exists() and serve_log.exists():
                result = json.loads(output_file.read_text(encoding="utf-8"))
                if (
                    result.get("failed") == 0
                    and result.get("completed") == 128
                    and _gpu_guard_history_is_clean(decode_dir)
                ):
                    serve_proofs[f"decode-seed{seed}"] = _serve_proof(
                        serve_log,
                        arm,
                        args.cap,
                        args.barrier_grid,
                        args.inkernel_barrier,
                        args.engine_module,
                        args.deep_health_mode,
                        args.triton_ar_disable,
                        arm_double_buffer,
                        args.disable_overlap_schedule,
                    )
                    produced_results.append(
                        str(output_file.relative_to(arm_dir))
                    )
                    print(f"Reusing validated {decode_label}")
                    continue
            active_env, active_log = _start_phase(
                args,
                arm=arm,
                phase_dir=decode_dir,
                label=decode_label,
            )
            if not args.dry_run:
                _capture_gpu_diagnostics(
                    decode_dir,
                    label="server-ready",
                    container=args.container,
                    selected_gpus=args.selected_physical_gpus,
                    physical_to_kfd_id=args.physical_to_kfd_id,
                )
            _guard_phase(args, decode_dir)
            _run_guarded_benchmark(
                _bench_command(
                    decode_label + "-warmup",
                    decode_warmup,
                    10000 + block,
                ),
                args=args,
                phase_dir=decode_dir,
                env=active_env,
                log=active_log,
            )
            _guard_phase(args, decode_dir)
            _run_guarded_benchmark(
                _bench_command(
                    decode_label,
                    decode_workload,
                    seed,
                    output_file=output_file,
                    ready_check=False,
                ),
                args=args,
                phase_dir=decode_dir,
                env=active_env,
                log=active_log,
            )
            produced_results.append(str(output_file.relative_to(arm_dir)))
            if not args.dry_run:
                serve_proofs[f"decode-seed{seed}"] = _serve_proof(
                    serve_log,
                    arm,
                    args.cap,
                    args.barrier_grid,
                    args.inkernel_barrier,
                    args.engine_module,
                    args.deep_health_mode,
                    args.triton_ar_disable,
                    arm_double_buffer,
                    args.disable_overlap_schedule,
                )
            if not args.dry_run:
                _capture_gpu_diagnostics(
                    decode_dir,
                    label="decode-complete",
                    container=args.container,
                    selected_gpus=args.selected_physical_gpus,
                    physical_to_kfd_id=args.physical_to_kfd_id,
                )
            _teardown(env=active_env, log=active_log, dry_run=args.dry_run)
            active_env = None
            active_log = None

        if args.decode_only:
            if not args.dry_run:
                for result_path in result_dir.glob("decode-seed*.json"):
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    if result.get("failed") != 0 or result.get("completed") != 128:
                        raise RuntimeError(f"incomplete result {result_path}")
            summary.update(
                {
                    "status": "complete",
                    "completed_at": _utc_now(),
                    "results": produced_results,
                    "trace_proof": None,
                    "serve_proofs": serve_proofs,
                }
            )
            _write_json(completion, summary)
            if not args.dry_run:
                _write_checksums(arm_dir)
            return summary

        # Prefill gets a second clean server. Buckets remain monotonic and no
        # decode measurement follows the large-M transition sequence.
        prefill_dir = arm_dir / "prefill-phase"
        prefill_label = label + "-prefill"
        active_env, active_log = _start_phase(
            args,
            arm=arm,
            phase_dir=prefill_dir,
            label=prefill_label,
        )
        if not args.dry_run:
            _capture_gpu_diagnostics(
                prefill_dir,
                label="server-ready",
                container=args.container,
                selected_gpus=args.selected_physical_gpus,
                physical_to_kfd_id=args.physical_to_kfd_id,
            )
        prefill_warmup = Workload("prefill-warmup", 4, 8, 32, 32, "warmup")
        _guard_phase(args, prefill_dir)
        _run_guarded_benchmark(
            _bench_command(
                prefill_label + "-warmup",
                prefill_warmup,
                11000 + block,
            ),
            args=args,
            phase_dir=prefill_dir,
            env=active_env,
            log=active_log,
        )
        for workload in WORKLOADS:
            if workload.name == "decode":
                continue
            for seed in seed_order:
                _guard_phase(args, prefill_dir)
                output_file = result_dir / f"{workload.name}-seed{seed}.json"
                _run_guarded_benchmark(
                    _bench_command(
                        f"{prefill_label}-{workload.name}",
                        workload,
                        seed,
                        output_file=output_file,
                        ready_check=False,
                    ),
                    args=args,
                    phase_dir=prefill_dir,
                    env=active_env,
                    log=active_log,
                )
                produced_results.append(str(output_file.relative_to(arm_dir)))

        trace_proof: dict[str, Any] | None = None
        if not args.skip_profiles:
            trace_dir = prefill_dir / "traces" / "m512-proof"
            profile_workload = Workload(
                "profile-m512",
                16,
                8,
                32,
                32,
                "signature",
            )
            profile_id = f"{prefill_label}-m512"
            _guard_phase(args, prefill_dir)
            _run_guarded_benchmark(
                _bench_command(
                    profile_id,
                    profile_workload,
                    20000 + block,
                    ready_check=False,
                    extra=(
                        "--profile",
                        "--profile-num-steps",
                        "8",
                        "--profile-base-url",
                        f"http://127.0.0.1:{args.control_port}",
                        "--profile-output-dir",
                        str(trace_dir),
                        "--profile-id",
                        profile_id,
                        "--no-profile-with-stack",
                        "--profile-record-shapes",
                        "--profile-activities",
                        "CPU",
                        "GPU",
                    ),
                ),
                args=args,
                phase_dir=prefill_dir,
                env=active_env,
                log=active_log,
            )
            if not args.dry_run:
                trace_proof = validate_trace_signatures(
                    trace_dir,
                    world_size=args.world_size,
                    arm=arm,
                )
                trace_proof["forward_analysis"] = None
                trace_proof["forward_analysis_reason"] = (
                    "m512 trace is signature-only; authoritative marker-aligned "
                    "decode traces are captured by the Level-5 profile workflow"
                )

        if not args.dry_run:
            serve_proofs["prefill"] = _serve_proof(
                prefill_dir / f"serve-{prefill_label}.log",
                arm,
                args.cap,
                args.barrier_grid,
                args.inkernel_barrier,
                args.engine_module,
                args.deep_health_mode,
                args.triton_ar_disable,
                arm_double_buffer,
                args.disable_overlap_schedule,
            )
            for result_path in result_dir.glob("*.json"):
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if result.get("failed") != 0 or result.get("completed") != 128:
                    raise RuntimeError(
                        f"incomplete benchmark result {result_path}: "
                        f"completed={result.get('completed')} "
                        f"failed={result.get('failed')}"
                    )
        if not args.dry_run:
            _capture_gpu_diagnostics(
                prefill_dir,
                label="prefill-complete",
                container=args.container,
                selected_gpus=args.selected_physical_gpus,
                physical_to_kfd_id=args.physical_to_kfd_id,
            )
        _teardown(env=active_env, log=active_log, dry_run=args.dry_run)
        active_env = None
        active_log = None

        summary.update(
            {
                "status": "complete",
                "completed_at": _utc_now(),
                "results": produced_results,
                "trace_proof": trace_proof,
                "serve_proofs": serve_proofs,
            }
        )
        _write_json(completion, summary)
        if not args.dry_run:
            _write_checksums(arm_dir)
        return summary
    except BaseException as exc:
        if not args.dry_run:
            failure_dir = active_log.parent if active_log is not None else arm_dir
            _capture_gpu_diagnostics(
                failure_dir,
                label="arm-failure",
                container=args.container,
                selected_gpus=args.selected_physical_gpus,
                physical_to_kfd_id=args.physical_to_kfd_id,
                include_amd_smi_metrics=True,
            )
        summary.update(
            {
                "status": (
                    "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
                ),
                "failed_at": _utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _write_json(completion, summary)
        raise
    finally:
        if active_env is not None and active_log is not None:
            _teardown(
                env=active_env,
                log=active_log,
                dry_run=args.dry_run,
            )


def _collect_pairs(run_root: Path) -> dict[str, dict[str, dict[int, list[float]]]]:
    records: dict[tuple[int, str, int, str], dict[str, Any]] = {}
    for summary_path in sorted(run_root.glob("block-*/*/arm-summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "complete":
            continue
        block = int(summary["block"])
        arm = str(summary["arm"]["name"])
        arm_dir = summary_path.parent
        for result_path in (arm_dir / "results").glob("*.json"):
            match = re.fullmatch(r"(.+)-seed(\d+)\.json", result_path.name)
            if match is None:
                continue
            workload, seed_text = match.groups()
            records[(block, workload, int(seed_text), arm)] = json.loads(
                result_path.read_text(encoding="utf-8")
            )

    paired: dict[str, dict[str, dict[int, list[float]]]] = {}
    for block, workload, seed, arm in sorted(records):
        if arm != ARMS[0].name:
            continue
        baseline = records[(block, workload, seed, ARMS[0].name)]
        candidate_key = (block, workload, seed, ARMS[1].name)
        if candidate_key not in records:
            continue
        candidate = records[candidate_key]
        for metric in METRICS:
            if metric not in baseline or metric not in candidate:
                continue
            baseline_value = float(baseline[metric])
            candidate_value = float(candidate[metric])
            change = (candidate_value / baseline_value - 1.0) * 100.0
            paired.setdefault(workload, {}).setdefault(metric, {}).setdefault(
                block, []
            ).append(change)
    return paired


def analyze_campaign(
    run_root: Path,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    paired = _collect_pairs(run_root)
    analyses: dict[str, Any] = {}
    for workload, metrics in paired.items():
        analyses[workload] = {}
        for metric, values_by_block in metrics.items():
            analyses[workload][metric] = hierarchical_bootstrap(
                values_by_block,
                samples=bootstrap_samples,
                seed=bootstrap_seed
                + sum(ord(character) for character in workload + metric),
            )

    reasons = []
    decode_sample = analyses.get("decode", {}).get("median_tpot_ms")
    sample_sufficient = not (
        decode_sample is None
        or decode_sample["n_blocks"] < 3
        or decode_sample["n_pairs"] < 15
    )
    if not sample_sufficient:
        reasons.append(
            "insufficient paired evidence: require >=3 blocks and >=15 pairs"
        )
    decode_tpot = analyses.get("decode", {}).get("median_tpot_ms")
    decode_throughput = analyses.get("decode", {}).get("output_throughput")
    latency_gate = bool(
        decode_tpot is not None
        and decode_throughput is not None
        and decode_tpot["mean"] <= -1.5
        and decode_tpot["ci95_high"] < 0
        and decode_throughput["ci95_low"] >= -0.5
    )
    capacity_gate = bool(
        decode_tpot is not None
        and decode_throughput is not None
        and decode_throughput["mean"] >= 1.0
        and decode_throughput["ci95_low"] > 0
        and decode_tpot["ci95_high"] <= 1.0
    )
    if sample_sufficient and not (latency_gate or capacity_gate):
        reasons.append(
            "neither latency nor capacity promotion threshold was cleared"
        )

    summary = {
        "generated_at": _utc_now(),
        "comparison": {
            "baseline": ARMS[0].name,
            "candidate": ARMS[1].name,
            "change_definition": "(candidate / baseline - 1) * 100",
        },
        "bootstrap": {
            "method": "paired hierarchical restart-block bootstrap",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
        },
        "workloads": analyses,
        "promotion": {
            "eligible": sample_sufficient and (latency_gate or capacity_gate),
            "reasons": reasons,
            "objectives": {
                "latency": {"eligible": latency_gate},
                "capacity": {"eligible": capacity_gate},
            },
            "criteria": {
                "latency": (
                    "paired TPOT mean <= -1.5%, CI95 high < 0, and "
                    "throughput CI95 low >= -0.5%"
                ),
                "capacity": (
                    "paired throughput mean >= +1.0%, CI95 low > 0, and "
                    "TPOT CI95 high <= +1.0%"
                ),
                "sample_size": ">=3 restart blocks and >=15 paired observations",
            },
        },
    }
    _write_json(run_root / "campaign-summary.json", summary)
    _write_summary_markdown(run_root / "campaign-summary.md", summary)
    return summary


def analyze_stability_campaign(
    run_root: Path,
    *,
    arm_name: str,
    expected_blocks: int,
    expected_seeds: Sequence[int],
) -> dict[str, Any]:
    """Summarize a single-arm fresh-container stability gate."""
    passed: list[dict[str, Any]] = []
    failed_arms: list[dict[str, Any]] = []
    for summary_path in sorted(run_root.glob("block-*/*/arm-summary.json")):
        arm_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if arm_summary.get("arm", {}).get("name") != arm_name:
            continue
        if arm_summary.get("status") != "complete":
            failed_arms.append(
                {
                    "path": str(summary_path.relative_to(run_root)),
                    "status": arm_summary.get("status"),
                    "error": arm_summary.get("error"),
                }
            )
            continue
        block = int(arm_summary["block"])
        result_dir = summary_path.parent / "results"
        for seed in expected_seeds:
            result_path = result_dir / f"decode-seed{seed}.json"
            if not result_path.exists():
                continue
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("completed") == 128 and result.get("failed") == 0:
                passed.append(
                    {
                        "block": block,
                        "seed": int(seed),
                        "path": str(result_path.relative_to(run_root)),
                        "metrics": {
                            metric: result.get(metric)
                            for metric in METRICS
                            if metric in result
                        },
                    }
                )
    required = expected_blocks * len(expected_seeds)
    summary = {
        "generated_at": _utc_now(),
        "kind": "single-arm fresh-container stability gate",
        "arm": arm_name,
        "required_results": required,
        "passed_results": len(passed),
        "expected_blocks": expected_blocks,
        "expected_seeds": list(expected_seeds),
        "passed": passed,
        "failed_arms": failed_arms,
        "stability": {
            "eligible": len(passed) == required and not failed_arms,
            "criterion": (
                "every expected decode seed completes 128 requests with zero "
                "failures on a fresh server"
            ),
        },
    }
    _write_json(run_root / "stability-summary.json", summary)
    return summary


def _write_summary_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# AR+RMSNorm repeatability campaign",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "Paired change is `(candidate / baseline - 1) * 100`; negative is "
        "better for latency and positive is better for throughput.",
        "",
    ]
    for workload, metrics in sorted(summary["workloads"].items()):
        lines.append(f"## {workload}")
        lines.append("")
        for metric, result in sorted(metrics.items()):
            lines.append(
                f"- `{metric}`: {result['mean']:+.3f}% "
                f"(95% CI {result['ci95_low']:+.3f}% to "
                f"{result['ci95_high']:+.3f}%; "
                f"{result['n_pairs']} pairs / {result['n_blocks']} blocks)"
            )
        lines.append("")
    promotion = summary["promotion"]
    lines.extend(
        [
            "## Promotion decision",
            "",
            f"Eligible: **{str(promotion['eligible']).lower()}**",
            "",
        ]
    )
    if promotion["reasons"]:
        lines.extend(f"- {reason}" for reason in promotion["reasons"])
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _resolve_run_root(args: argparse.Namespace) -> Path:
    if args.run_root is not None:
        return args.run_root.resolve()
    campaign_id = args.campaign_id or datetime.now().strftime(
        "repeatability-%Y%m%d-%H%M%S"
    )
    return (DEFAULT_RAW_ROOT / args.run_date / campaign_id).resolve()


def _manifest(
    args: argparse.Namespace,
    schedule: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": _utc_now(),
        "run_root": str(args.run_root),
        "configuration": {
            "world_size": args.world_size,
            "comparison": args.comparison,
            "devices": args.devices,
            "cap": args.cap,
            "barrier_grid": args.barrier_grid,
            "inkernel_barrier": args.inkernel_barrier,
            "triton_ar_disable": args.triton_ar_disable,
            "engine_module": args.engine_module,
            "deep_health_mode": args.deep_health_mode,
            "double_buffer_input": args.double_buffer_input,
            "disable_overlap_schedule": args.disable_overlap_schedule,
            "restart_container_per_server": args.restart_container_per_server,
            "serve_extra_args": args.serve_extra_args,
            "container": args.container,
            "port": args.port,
            "control_port": args.control_port,
            "benchmark_timeout": args.benchmark_timeout,
            "blocks": args.blocks,
            "seeds": args.seeds,
            "order_seed": args.order_seed,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
            "ignored_busy_gpus": sorted(args.ignored_busy_gpus),
            "hip_to_physical_gpu": args.hip_to_physical_gpu,
            "physical_to_kfd_id": args.physical_to_kfd_id,
            "selected_physical_gpus": sorted(args.selected_physical_gpus),
            "skip_profiles": args.skip_profiles,
            "decode_only": args.decode_only,
            "stability_only": args.stability_only,
            "dry_run": args.dry_run,
        },
        "arms": [asdict(arm) for arm in ARMS],
        "workloads": [asdict(workload) for workload in WORKLOADS],
        "schedule": schedule,
        "git": None if args.dry_run else _git_metadata(),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument(
        "--comparison",
        choices=(
            "iris",
            "triton_shmem",
            "gate256",
            "unfused",
            "input_ring",
        ),
        default="iris",
    )
    parser.add_argument("--seeds", type=_parse_csv_ints, default=[0, 1, 2, 3, 4])
    parser.add_argument("--order-seed", type=int, default=20260727)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260727)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--devices", default="1,2,3,5")
    parser.add_argument("--cap", type=int, default=2048)
    parser.add_argument(
        "--barrier-grid",
        type=int,
        default=0,
        help="Fixed in-kernel barrier participant count; 0 keeps M-dependent.",
    )
    parser.add_argument(
        "--inkernel-barrier",
        type=int,
        choices=(0, 1),
        default=1,
        help="Use fused in-kernel barriers (1) or separate barrier kernels (0).",
    )
    parser.add_argument(
        "--triton-ar-disable",
        type=int,
        choices=(0, 1),
        default=int(os.environ.get("TS_TRITON_AR_DISABLE", "0")),
        help="Disable standalone small-message Triton all-reduce and use RCCL.",
    )
    parser.add_argument(
        "--deep-health-mode",
        choices=("generate", "passive", "passive_when_busy"),
        default="passive",
    )
    parser.add_argument(
        "--double-buffer-input",
        type=int,
        choices=(0, 1),
        default=0,
    )
    parser.add_argument("--engine-module")
    parser.add_argument(
        "--disable-overlap-schedule",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--restart-container-per-server",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restart the dedicated container before each server lifecycle.",
    )
    parser.add_argument(
        "--serve-extra-args",
        default="",
        help="Additional server CLI arguments applied to every phase.",
    )
    parser.add_argument(
        "--container",
        default="jeremwan-tokenspeed-profiler",
    )
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--control-port", type=int, default=8101)
    parser.add_argument("--health-timeout", type=int, default=900)
    parser.add_argument(
        "--benchmark-timeout",
        type=int,
        default=300,
        help="Hard timeout for each warmup, benchmark, or profile command.",
    )
    parser.add_argument("--run-date", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--campaign-id")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument(
        "--ignored-busy-gpus",
        type=_parse_csv_ints,
        default=[3],
        help="Physical AMD-SMI indices allowed to remain occupied.",
    )
    parser.add_argument("--skip-profiles", action="store_true")
    parser.add_argument("--decode-only", action="store_true")
    parser.add_argument(
        "--stability-only",
        action="store_true",
        help=(
            "Run only the unfused arm as a fresh-container stability gate; "
            "requires --comparison unfused --decode-only."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.blocks < 1:
        parser.error("--blocks must be positive")
    if args.bootstrap_samples < 100:
        parser.error("--bootstrap-samples must be at least 100")
    if args.benchmark_timeout < 1:
        parser.error("--benchmark-timeout must be positive")
    if args.barrier_grid < 0:
        parser.error("--barrier-grid must be non-negative")
    if len(args.devices.split(",")) != args.world_size:
        parser.error("--devices count must equal --world-size")
    if args.stability_only and (
        args.comparison not in ("unfused", "iris", "triton_shmem")
        or not args.decode_only
    ):
        parser.error(
            "--stability-only requires an upstream-unfused comparison "
            "and --decode-only"
        )
    args.ignored_busy_gpus = set(args.ignored_busy_gpus)
    if args.engine_module is None:
        args.engine_module = (
            "tokenspeed.runtime.entrypoints.safe_smg_server"
            if args.deep_health_mode != "generate"
            else "smg_grpc_servicer.tokenspeed"
        )
    args.hip_to_physical_gpu = hip_to_physical_gpu_map()
    args.physical_to_kfd_id = physical_to_kfd_id_map()
    try:
        args.selected_physical_gpus = {
            args.hip_to_physical_gpu[int(device)]
            for device in args.devices.split(",")
        }
    except (KeyError, ValueError):
        parser.error("--devices contains an unknown HIP visibility index")
    args.run_root = _resolve_run_root(args)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    global ARMS
    args = _parse_args(argv)
    ARMS = campaign_arms(
        args.comparison,
        stability_only=args.stability_only,
    )
    schedule = build_schedule(args.blocks, args.seeds, args.order_seed)
    args.run_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.run_root / "campaign-manifest.json"
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"{manifest_path} already exists; pass --resume or choose "
                "another run root"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_arm_names = [arm["name"] for arm in manifest["arms"]]
        if manifest_arm_names != [arm.name for arm in ARMS]:
            raise ValueError(
                f"resume arm mismatch: {manifest_arm_names} != "
                f"{[arm.name for arm in ARMS]}"
            )
        schedule = manifest["schedule"]
        if not args.dry_run:
            manifest.setdefault("resume_events", []).append(
                {
                    "time": _utc_now(),
                    "git": _git_metadata(),
                    "reason": "resume with result and serve-proof revalidation",
                }
            )
            _write_json(manifest_path, manifest)
    else:
        manifest = _manifest(args, schedule)
        _write_json(manifest_path, manifest)

    print(json.dumps(manifest, indent=2, sort_keys=True))
    _preflight(
        args.run_root,
        container=args.container,
        selected_gpus=args.selected_physical_gpus,
        ignored_busy_gpus=args.ignored_busy_gpus,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print("Dry run complete; no server or GPU operation was launched.")
        return 0

    gpu_lock = _acquire_gpu_campaign_lock()
    campaign_env = os.environ.copy()
    campaign_env["CONTAINER"] = args.container
    try:
        for block_spec in schedule:
            block = int(block_spec["block"])
            for position, arm_name in enumerate(block_spec["arm_order"]):
                arm = next(arm for arm in ARMS if arm.name == arm_name)
                _run_arm(
                    args,
                    block=block,
                    position=position,
                    arm=arm,
                    seed_order=block_spec["seed_order"],
                )
        if args.stability_only:
            summary = analyze_stability_campaign(
                args.run_root,
                arm_name=ARMS[0].name,
                expected_blocks=args.blocks,
                expected_seeds=args.seeds,
            )
        else:
            summary = analyze_campaign(
                args.run_root,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
            )
        decision = (
            summary["stability"]
            if args.stability_only
            else summary["promotion"]
        )
        print(json.dumps(decision, indent=2, sort_keys=True))
        return 0
    finally:
        _teardown(
            env=campaign_env,
            log=args.run_root / "final-teardown.log",
            dry_run=False,
        )
        _write_checksums(args.run_root)
        gpu_lock.close()


if __name__ == "__main__":
    sys.exit(main())
