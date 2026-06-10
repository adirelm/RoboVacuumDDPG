from scripts import render_trajectory as rt


def test_render_trajectory_writes_png(tmp_path, tiny_map):
    path = [(1.0, 1.0), (2.0, 2.0), (3.0, 1.5)]
    out = tmp_path / "trajectory.png"
    rt.render(tiny_map.walls, path, str(out), clean_radius=0.17)
    assert out.exists()
    assert out.stat().st_size > 5000


def test_render_trajectory_handles_single_point(tmp_path, tiny_map):
    out = tmp_path / "trajectory_single.png"
    rt.render(tiny_map.walls, [(2.0, 2.0)], str(out), clean_radius=0.17)
    assert out.exists()
