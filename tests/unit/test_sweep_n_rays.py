"""Unit test for scripts/sweep_n_rays.py (V3 §9.1 lidar-resolution sweep).

Mocks RoboVacuumSDK so NO real training runs: a fake SDK records the n_rays it
was built with and returns a tiny synthetic per-episode history. Asserts the
sweep (a) writes one entry per ray-config keyed by str(n_rays), (b) computes the
final-window mean reward/coverage correctly, and (c) stamps the run metadata.

Single-file run: uv run pytest tests/unit/test_sweep_n_rays.py --no-cov -q
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import yaml

from scripts import sweep_n_rays as sw


def test_final_window_means_computes_tail_mean() -> None:
    history = [{"reward": float(i), "coverage": i / 100.0} for i in range(10)]
    reward, coverage = sw.final_window_means(history, window=3)
    # tail = episodes 7,8,9 -> rewards mean 8.0; coverages mean 0.08
    assert abs(reward - 8.0) < 1e-9
    assert abs(coverage - 0.08) < 1e-9


def test_final_window_clamps_to_history_length() -> None:
    history = [{"reward": 2.0, "coverage": 0.5}, {"reward": 4.0, "coverage": 0.7}]
    reward, coverage = sw.final_window_means(history, window=20)
    assert abs(reward - 3.0) < 1e-9
    assert abs(coverage - 0.6) < 1e-9


class _FakeSDK:
    """Records the n_rays from the temp config; returns a deterministic history."""

    seen_rays: ClassVar[list[int]] = []

    def __init__(self, config_path: str) -> None:
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        self._n_rays = int(cfg["env"]["n_rays"])
        # also assert the reduced budget was written
        assert cfg["training"]["episodes"] == 5
        assert cfg["training"]["seeds"] == [42]
        _FakeSDK.seen_rays.append(self._n_rays)

    def train(self, seed: int, map_name: str) -> list[dict]:
        assert seed == 42
        assert map_name == "room_single"
        base = float(self._n_rays)
        return [{"reward": base + e, "coverage": (base + e) / 1000.0} for e in range(5)]


def test_run_sweep_aggregates_three_ray_configs(tmp_path, monkeypatch) -> None:
    _FakeSDK.seen_rays = []
    monkeypatch.setattr(sw, "RoboVacuumSDK", _FakeSDK)
    out = tmp_path / "sweep_n_rays.json"
    result = sw.run_sweep(
        base_config="config/config.yaml",
        ray_values=(8, 16, 24),
        episodes=5,
        seed=42,
        map_name="room_single",
        window=3,
        out_path=str(out),
        tmp_dir=str(tmp_path),
    )

    assert _FakeSDK.seen_rays == [8, 16, 24]
    assert set(result) == {"8", "16", "24", "episodes", "seed", "map", "window"}
    assert result["episodes"] == 5
    assert result["seed"] == 42
    assert result["map"] == "room_single"
    # n_rays=8 -> tail rewards {10,11,12} mean 11.0; coverage mean 0.011
    assert abs(result["8"]["final_reward"] - 11.0) < 1e-9
    assert abs(result["8"]["final_coverage"] - 0.011) < 1e-9
    assert abs(result["24"]["final_reward"] - 27.0) < 1e-9

    on_disk = json.loads(out.read_text(encoding="utf-8"))
    assert on_disk == result
