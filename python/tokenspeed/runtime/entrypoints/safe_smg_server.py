"""TokenSpeed SMG engine with non-invasive deep health checks.

The upstream custom HealthCheck RPC enqueues a one-token generation even while
real requests are pending, then may cancel/abort it as soon as *any* scheduler
output arrives. ``passive_when_busy`` avoids injecting that M=1 request into an
active graph stream while retaining the generation probe when the engine is
idle.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time

from smg_grpc_proto.generated import tokenspeed_scheduler_pb2
from smg_grpc_servicer.tokenspeed.servicer import TokenSpeedSchedulerServicer
from smg_grpc_servicer.tokenspeed.server import serve_grpc
from tokenspeed.runtime.utils.server_args import prepare_server_args

logger = logging.getLogger(__name__)

_ORIGINAL_HEALTH_CHECK = TokenSpeedSchedulerServicer.HealthCheck


async def _safe_health_check(self, request, context):
    mode = os.environ.get(
        "TOKENSPEED_DEEP_HEALTH_MODE", "generate"
    ).strip().lower()
    if mode == "generate":
        return await _ORIGINAL_HEALTH_CHECK(self, request, context)
    if mode not in {"passive", "passive_when_busy"}:
        raise ValueError(
            "TOKENSPEED_DEEP_HEALTH_MODE must be generate, passive, or "
            f"passive_when_busy; got {mode!r}"
        )

    if self.async_llm.gracefully_exit:
        return tokenspeed_scheduler_pb2.HealthCheckResponse(
            healthy=False,
            message="Server is shutting down",
        )

    pending = len(self.async_llm.rid_to_state)
    if mode == "passive_when_busy" and pending == 0:
        return await _ORIGINAL_HEALTH_CHECK(self, request, context)

    silence = time.time() - self.async_llm.last_receive_tstamp
    threshold = float(
        os.environ.get("TOKENSPEED_PASSIVE_HEALTH_STUCK_SEC", "30")
    )
    healthy = pending == 0 or silence <= threshold
    return tokenspeed_scheduler_pb2.HealthCheckResponse(
        healthy=healthy,
        message=(
            f"Passive health: pending={pending}, scheduler_silence={silence:.1f}s"
        ),
    )


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    TokenSpeedSchedulerServicer.HealthCheck = _safe_health_check
    server_args = prepare_server_args(argv)
    try:
        import uvloop
    except ImportError:
        uvloop = None
    if uvloop is not None:
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    logger.info(
        "Deep health mode: %s",
        os.environ.get("TOKENSPEED_DEEP_HEALTH_MODE", "generate"),
    )
    asyncio.run(serve_grpc(server_args))


if __name__ == "__main__":
    main()
