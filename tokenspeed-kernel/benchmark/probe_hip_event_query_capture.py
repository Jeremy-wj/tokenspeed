"""Reproduce the ROCm 7.2.0 cross-thread event-query graph-capture bug.

This is the runtime race hit by ProcessGroupNCCL's watchdog while TokenSpeed
captures CUDA/HIP graphs.  It does not need RCCL: one thread holds a GLOBAL
capture open while a watchdog-like thread, placed in THREAD_LOCAL mode, queries
an incomplete event.

Run on one otherwise-idle GPU:

    HIP_VISIBLE_DEVICES=1 python probe_hip_event_query_capture.py

The probe exits successfully on a fixed runtime (ROCm >= 7.2.1).  Pass
``--expect-broken`` to turn the expected ROCm 7.2.0 failure into a successful
diagnostic result.
"""

from __future__ import annotations

import argparse
import ctypes
import threading

import torch


def _hip_runtime() -> tuple[ctypes.CDLL, int]:
    hip = ctypes.CDLL("libamdhip64.so")
    version = ctypes.c_int()
    rc = hip.hipRuntimeGetVersion(ctypes.byref(version))
    if rc != 0:
        raise RuntimeError(f"hipRuntimeGetVersion failed with HIP error {rc}")
    return hip, version.value


def run_probe(sleep_cycles: int) -> tuple[int, dict[str, object]]:
    hip, runtime_version = _hip_runtime()
    torch.cuda.set_device(0)

    value = torch.ones(1, device="cuda")
    work_stream = torch.cuda.Stream()
    capture_stream = torch.cuda.Stream()
    torch.cuda.synchronize()

    incomplete = torch.cuda.Event()
    with torch.cuda.stream(work_stream):
        torch.cuda._sleep(sleep_cycles)
        incomplete.record()
    if incomplete.query():
        raise RuntimeError("delay event completed before graph capture started")

    capture_started = threading.Event()
    query_finished = threading.Event()
    result: dict[str, object] = {}

    def capture() -> None:
        try:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
                graph, stream=capture_stream, capture_error_mode="global"
            ):
                capture_started.set()
                if not query_finished.wait(10):
                    raise TimeoutError("event-query thread did not run")
                value.add_(1)
            result["capture"] = "ok"
        except BaseException as exc:
            result["capture"] = f"{type(exc).__name__}: {exc}"

    capture_thread = threading.Thread(target=capture, name="graph-capture")
    capture_thread.start()
    if not capture_started.wait(10):
        raise TimeoutError("graph-capture thread did not start")

    # Match ProcessGroupNCCL's watchdog-side CUDAStreamCaptureModeGuard.
    mode = ctypes.c_int(1)  # hipStreamCaptureModeThreadLocal
    result["mode_exchange_rc"] = hip.hipThreadExchangeStreamCaptureMode(
        ctypes.byref(mode)
    )
    previous_mode = mode.value
    try:
        result["event_query"] = incomplete.query()
    except BaseException as exc:
        result["event_query"] = f"{type(exc).__name__}: {exc}"
    finally:
        query_finished.set()

    capture_thread.join(10)
    result["capture_thread_alive"] = capture_thread.is_alive()
    work_stream.synchronize()

    mode = ctypes.c_int(previous_mode)
    result["mode_restore_rc"] = hip.hipThreadExchangeStreamCaptureMode(
        ctypes.byref(mode)
    )
    return runtime_version, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect-broken", action="store_true")
    parser.add_argument("--sleep-cycles", type=int, default=2_000_000_000)
    args = parser.parse_args()

    runtime_version, result = run_probe(args.sleep_cycles)
    passed = (
        result.get("capture") == "ok"
        and not result.get("capture_thread_alive")
        and not isinstance(result.get("event_query"), str)
    )
    print(
        f"HIP runtime={runtime_version} torch={torch.__version__} "
        f"result={result}",
        flush=True,
    )

    if passed == args.expect_broken:
        expectation = "broken" if args.expect_broken else "fixed"
        raise SystemExit(f"probe result did not match expected {expectation} runtime")


if __name__ == "__main__":
    main()
