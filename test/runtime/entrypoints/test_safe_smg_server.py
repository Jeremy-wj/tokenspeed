from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

from tokenspeed.runtime.entrypoints import safe_smg_server


def _servicer(*, pending: int, silence: float = 0.0):
    return SimpleNamespace(
        async_llm=SimpleNamespace(
            gracefully_exit=False,
            rid_to_state={str(index): object() for index in range(pending)},
            last_receive_tstamp=time.time() - silence,
        )
    )


def test_passive_when_busy_does_not_generate(monkeypatch):
    called = False

    async def original(*args):
        nonlocal called
        called = True
        raise AssertionError("generation health should not run while busy")

    monkeypatch.setenv("TOKENSPEED_DEEP_HEALTH_MODE", "passive_when_busy")
    monkeypatch.setattr(safe_smg_server, "_ORIGINAL_HEALTH_CHECK", original)
    response = asyncio.run(
        safe_smg_server._safe_health_check(
            _servicer(pending=4),
            None,
            None,
        )
    )
    assert response.healthy
    assert not called


def test_passive_when_busy_keeps_idle_generation_probe(monkeypatch):
    async def original(*args):
        return "generated"

    monkeypatch.setenv("TOKENSPEED_DEEP_HEALTH_MODE", "passive_when_busy")
    monkeypatch.setattr(safe_smg_server, "_ORIGINAL_HEALTH_CHECK", original)
    response = asyncio.run(
        safe_smg_server._safe_health_check(
            _servicer(pending=0),
            None,
            None,
        )
    )
    assert response == "generated"


def test_passive_reports_stuck_busy_scheduler(monkeypatch):
    monkeypatch.setenv("TOKENSPEED_DEEP_HEALTH_MODE", "passive")
    monkeypatch.setenv("TOKENSPEED_PASSIVE_HEALTH_STUCK_SEC", "5")
    response = asyncio.run(
        safe_smg_server._safe_health_check(
            _servicer(pending=1, silence=10),
            None,
            None,
        )
    )
    assert not response.healthy
