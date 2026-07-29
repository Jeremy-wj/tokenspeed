from __future__ import annotations

from tokenspeed.runtime.utils import network


def test_free_port_cluster_rotates_under_persistent_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(network, "is_port_available", lambda port: True)
    state = tmp_path / "cluster-state"

    first = network.get_free_port_cluster(
        (0, 1, 3),
        start=20_000,
        end=20_040,
        stride=10,
        state_path=state,
    )
    second = network.get_free_port_cluster(
        (0, 1, 3),
        start=20_000,
        end=20_040,
        stride=10,
        state_path=state,
    )

    assert first == 20_000
    assert second == 20_010
    assert state.read_text(encoding="utf-8") == "20020"


def test_free_port_cluster_skips_unavailable_cluster(tmp_path, monkeypatch):
    monkeypatch.setattr(
        network,
        "is_port_available",
        lambda port: not (20_000 <= port < 20_010),
    )

    base = network.get_free_port_cluster(
        (0, 1, 3),
        start=20_000,
        end=20_040,
        stride=10,
        state_path=tmp_path / "cluster-state",
    )

    assert base == 20_010
