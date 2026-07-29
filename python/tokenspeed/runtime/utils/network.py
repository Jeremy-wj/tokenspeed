# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Port, socket, and host networking helpers."""

from __future__ import annotations

import fcntl
import logging
import os
import socket
import warnings
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


def is_port_available(port):
    """Return whether a port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("", port))
            s.listen(1)
            return True
        except OSError:
            return False
        except OverflowError:
            return False


def get_free_port():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
    except OSError:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]


def get_free_port_cluster(
    offsets: Iterable[int],
    *,
    start: int = 20_000,
    end: int = 30_000,
    stride: int = 10,
    state_path: str | os.PathLike[str] = "/tmp/tokenspeed-port-cluster",
) -> int:
    """Allocate a rotating non-ephemeral local port cluster.

    Probing a port and closing the socket is inherently unreserved. A persistent
    counter under ``flock`` prevents concurrent/restarted TokenSpeed launchers
    from choosing the same still-unbound cluster, while the availability scan
    skips live and TIME_WAIT sockets. The caller receives the cluster base and
    adds the requested offsets.
    """
    normalized = tuple(sorted(set(int(offset) for offset in offsets)))
    if not normalized or normalized[0] < 0:
        raise ValueError("offsets must contain non-negative integers")
    if stride <= normalized[-1]:
        raise ValueError("stride must exceed the largest cluster offset")
    if start <= 0 or end > 65_536 or start + normalized[-1] >= end:
        raise ValueError("invalid port-cluster range")

    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as state:
        fcntl.flock(state.fileno(), fcntl.LOCK_EX)
        state.seek(0)
        raw = state.read().strip()
        candidate = int(raw) if raw else start
        if candidate < start or candidate + normalized[-1] >= end:
            candidate = start

        attempts = max(1, (end - start) // stride)
        for _ in range(attempts):
            if all(is_port_available(candidate + offset) for offset in normalized):
                following = candidate + stride
                if following + normalized[-1] >= end:
                    following = start
                state.seek(0)
                state.truncate()
                state.write(str(following))
                state.flush()
                return candidate
            candidate += stride
            if candidate + normalized[-1] >= end:
                candidate = start
    raise RuntimeError(f"no free port cluster in [{start}, {end})")


def set_uvicorn_logging_configs():
    from uvicorn.config import LOGGING_CONFIG

    LOGGING_CONFIG["formatters"]["default"][
        "fmt"
    ] = "[%(asctime)s] %(levelprefix)s %(message)s"
    LOGGING_CONFIG["formatters"]["default"]["datefmt"] = "%Y-%m-%d %H:%M:%S"
    LOGGING_CONFIG["formatters"]["access"][
        "fmt"
    ] = '[%(asctime)s] %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'
    LOGGING_CONFIG["formatters"]["access"]["datefmt"] = "%Y-%m-%d %H:%M:%S"


def get_local_ip_by_remote() -> str | None:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        pass

    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        if ip and ip != "127.0.0.1" and ip != "0.0.0.0":
            return ip
    except Exception:
        pass

    try:
        s = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        s.connect(("2001:4860:4860::8888", 80))
        return s.getsockname()[0]
    except Exception:
        raise ValueError("Can not get local ip")


def get_ip() -> str:
    from tokenspeed.runtime.utils.env import envs

    host_ip = envs.TOKENSPEED_HOST_IP.get() or os.getenv("HOST_IP", "")
    if host_ip:
        return host_ip

    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        if ip and ip != "127.0.0.1" and ip != "0.0.0.0":
            return ip
    except Exception:
        pass

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        pass

    try:
        s = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        s.connect(("2001:4860:4860::8888", 80))
        return s.getsockname()[0]
    except Exception:
        pass

    warnings.warn(
        "Failed to get the IP address, using 0.0.0.0 by default."
        "The value can be set by the environment variable"
        " TOKENSPEED_HOST_IP or HOST_IP.",
        stacklevel=2,
    )
    return "0.0.0.0"
