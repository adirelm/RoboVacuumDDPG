from scripts import render_coverage_heatmap as rch


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
