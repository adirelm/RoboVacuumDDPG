"""Unit tests for scripts/render_coverage_heatmap.py: heat-array orientation
(cleaned/free/void encoding at the right [ix, iy] positions) and grid->PNG
rendering.
"""

import math

from scripts import render_coverage_heatmap as rch


def test_heat_array_encodes_cleaned_free_void_at_right_positions():
    # Asymmetric 3x2 grid (nx=3, ny=2) so a transpose/flip would be caught:
    # column ix=0 cleaned-free, ix=1 free-uncleaned, ix=2 wall/void.
    grid = {
        "cleaned": [[True, True], [False, False], [False, False]],
        "free": [[True, True], [True, True], [False, False]],
        "extent": [0.0, 3.0, 0.0, 2.0],
        "walls": [],
    }
    heat = rch.heat_array(grid)
    assert heat.shape == (3, 2)  # [ix, iy] — x along axis 0, like CoverageGrid
    assert heat[0, 0] == 1.0 and heat[0, 1] == 1.0  # cleaned-free
    assert heat[1, 0] == 0.0 and heat[1, 1] == 0.0  # free-uncleaned
    assert math.isnan(heat[2, 0]) and math.isnan(heat[2, 1])  # wall/void


def test_render_writes_png(tmp_path):
    grid = {
        # 2x2 grid: one cleaned-free cell, one free-uncleaned, two wall/void.
        "cleaned": [[True, False], [False, False]],
        "free": [[True, True], [False, False]],
        "extent": [0.0, 2.0, 0.0, 2.0],
        "walls": [(0.0, 0.0, 2.0, 0.0)],
        "coverage": 0.5,
    }
    out = tmp_path / "coverage_heatmap.png"
    rch.render(grid, str(out))
    assert out.exists()
    assert out.stat().st_size > 1000


def test_render_without_coverage_key(tmp_path):
    grid = {
        "cleaned": [[False]],
        "free": [[True]],
        "extent": [0.0, 1.0, 0.0, 1.0],
        "walls": [],
    }
    out = tmp_path / "h.png"
    rch.render(grid, str(out))
    assert out.exists()
