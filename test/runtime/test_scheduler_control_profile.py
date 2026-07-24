"""Regression tests for all-rank profiler control results."""

import asyncio

import pytest

from tokenspeed.runtime.engine.io_struct import (
    ProfileReq,
    ProfileReqOutput,
    ProfileReqType,
)
from tokenspeed.runtime.engine.scheduler_control_client import SchedulerControlClient


class _StubClient:
    def __init__(self, results):
        self.results = results

    async def profile_communicator(self, req):
        _ = req
        return self.results


def _run(results):
    client = _StubClient(results)
    req = ProfileReq(type=ProfileReqType.STOP_PROFILE)
    return asyncio.run(SchedulerControlClient._execute_profile(client, req))


def test_profile_control_returns_when_all_ranks_succeed():
    result = _run(
        [
            ProfileReqOutput(success=True, message="rank zero"),
            ProfileReqOutput(success=True, message="rank one"),
        ]
    )

    assert result.message == "rank zero"


def test_profile_control_reports_nonzero_rank_failure():
    with pytest.raises(RuntimeError, match=r"rank 1: failed to finalize"):
        _run(
            [
                ProfileReqOutput(success=True, message="rank zero"),
                ProfileReqOutput(success=False, message="failed to finalize"),
            ]
        )
