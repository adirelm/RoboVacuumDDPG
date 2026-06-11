"""Unit test for scripts/render_sensitivity.py (V3 §9.1 figure).

Writes a synthetic sweep JSON and asserts render() produces a >5 KB PNG. No real
training or sweep is run here.

Single-file run: uv run pytest tests/unit/test_render_sensitivity.py --no-cov -q
"""

from __future__ import annotations

import json

from scripts import render_sensitivity as rs


def _write_synthetic(path) -> None:
    data = {
        "episodes": 140,
        "seed": 42,
        "map": "room_single",
        "window": 20,
        "8": {"n_rays": 8, "state_dim": 12, "final_reward": 600.0, "final_coverage": 0.32},
        "16": {"n_rays": 16, "state_dim": 20, "final_reward": 850.0, "final_coverage": 0.39},
        "24": {"n_rays": 24, "state_dim": 28, "final_reward": 820.0, "final_coverage": 0.40},
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_load_series_returns_sorted_ray_axis(tmp_path) -> None:
    sweep = tmp_path / "sweep_n_rays.json"
    _write_synthetic(sweep)
    rays, rewards, coverages, meta = rs.load_series(str(sweep))
    assert rays == [8, 16, 24]
    assert rewards == [600.0, 850.0, 820.0]
    assert coverages == [0.32, 0.39, 0.40]
    assert meta["episodes"] == 140


def test_render_writes_png_over_5kb(tmp_path) -> None:
    sweep = tmp_path / "sweep_n_rays.json"
    _write_synthetic(sweep)
    out = tmp_path / "figures" / "sensitivity_n_rays.png"
    rs.render(str(sweep), str(out))
    assert out.exists()
    assert out.stat().st_size > 5000
