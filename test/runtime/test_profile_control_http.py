"""CPU-only tests for the in-engine profiling HTTP control path."""

from types import SimpleNamespace

from fastapi.testclient import TestClient

from tokenspeed.runtime.entrypoints.vllm_compat_http import build_vllm_compat_app


class _StubAsyncLLM:
    def __init__(self):
        self.start_kwargs = None
        self.stop_calls = 0

    async def start_profile(self, **kwargs):
        self.start_kwargs = kwargs
        return SimpleNamespace(success=True, message="started")

    async def stop_profile(self):
        self.stop_calls += 1
        return SimpleNamespace(success=True, message="stopped")


def _client():
    async_llm = _StubAsyncLLM()
    app = build_vllm_compat_app(object())
    app.state.async_llm = async_llm
    return TestClient(app), async_llm


def test_start_profile_forwards_full_configuration():
    client, async_llm = _client()
    response = client.post(
        "/start_profile",
        json={
            "output_dir": "/tmp/traces",
            "start_step": 4,
            "num_steps": 8,
            "activities": ["CPU", "GPU"],
            "with_stack": False,
            "record_shapes": True,
            "profile_by_stage": False,
            "profile_id": "matched-fused",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"success": True, "message": "started"}
    assert async_llm.start_kwargs == {
        "output_dir": "/tmp/traces",
        "start_step": 4,
        "num_steps": 8,
        "activities": ["CPU", "GPU"],
        "with_stack": False,
        "record_shapes": True,
        "profile_by_stage": False,
        "profile_id": "matched-fused",
    }


def test_stop_profile_reaches_async_llm():
    client, async_llm = _client()
    response = client.post("/stop_profile")

    assert response.status_code == 200
    assert response.json() == {"success": True, "message": "stopped"}
    assert async_llm.stop_calls == 1
