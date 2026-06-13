import json
from pathlib import Path
from typing import get_args, get_origin

import src.env.house_map as hm_mod
from src.env.house_map import HouseMap, Segment, load_house_map

# Real-schema + vendored-real-map tests live in test_house_map_real.py.


def _write_square(tmp_path: Path) -> str:
    plan = {
        "verts": [[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]],
        "bbox": {"min": [0, 0], "max": [4, 4]},
    }
    p = tmp_path / "room_single.json"
    p.write_text(json.dumps(plan))
    return str(p)


def test_load_square_room_walls_and_bounds(tmp_path):
    hm = load_house_map(_write_square(tmp_path))
    assert isinstance(hm, HouseMap)
    assert hm.bounds == (0.0, 0.0, 4.0, 4.0)
    assert len(hm.walls) == 4
    for seg in hm.walls:
        assert isinstance(seg, tuple)
        assert len(seg) == 4
    # the four square edges (order-independent set membership)
    edges = {
        (0.0, 0.0, 4.0, 0.0),
        (4.0, 0.0, 4.0, 4.0),
        (0.0, 4.0, 4.0, 4.0),
        (0.0, 0.0, 0.0, 4.0),
    }
    got = {tuple(sorted((s[0], s[2])) + sorted((s[1], s[3]))) for s in hm.walls}
    want = {tuple(sorted((e[0], e[2])) + sorted((e[1], e[3]))) for e in edges}
    assert got == want


def test_bounds_from_verts_when_bbox_absent(tmp_path):
    plan = {"verts": [[1, 2], [5, 2], [5, 8], [1, 8], [1, 2]]}
    p = tmp_path / "apt.json"
    p.write_text(json.dumps(plan))
    hm = load_house_map(str(p))
    assert hm.bounds == (1.0, 2.0, 5.0, 8.0)


def test_is_inside(tmp_path):
    hm = load_house_map(_write_square(tmp_path))
    assert hm.is_inside(2.0, 2.0) is True
    assert hm.is_inside(5.0, 5.0) is False
    assert hm.is_inside(-0.1, 2.0) is False


def test_parse_is_deterministic(tmp_path):
    path = _write_square(tmp_path)
    a = load_house_map(path)
    b = load_house_map(path)
    assert a.walls == b.walls
    assert a.bounds == b.bounds


def test_segment_alias_is_four_float_tuple():
    # Contract LAW: Segment = tuple[float, float, float, float] (single source
    # of truth in house_map.py; raycast/collision import it).
    assert Segment == tuple[float, float, float, float]
    assert get_origin(Segment) is tuple
    assert get_args(Segment) == (float, float, float, float)


def test_coord_scale_is_config_driven(tmp_path, monkeypatch):
    # A non-identity maps.coord_scale rescales verts + bounds to metres.
    monkeypatch.setattr(hm_mod, "_coord_scale", lambda: 2.0)
    hm = load_house_map(_write_square(tmp_path))
    assert hm.bounds == (0.0, 0.0, 8.0, 8.0)  # 4x4 scaled by 2
    assert hm.is_inside(6.0, 6.0) is True


def test_coord_scale_falls_back_to_default_without_config(monkeypatch):
    # If the config read raises, the loader uses the documented default (1.0).
    def _boom(*_a, **_k):
        raise FileNotFoundError

    monkeypatch.setattr(hm_mod, "load_config", _boom)
    assert hm_mod._coord_scale() == hm_mod._DEFAULT_SCALE


def test_degenerate_plan_falls_back_to_bbox_is_inside(tmp_path):
    # A <3-vertex plan can't form a polygon -> is_inside falls back to bbox.
    p = tmp_path / "degenerate.json"
    p.write_text(json.dumps({"verts": [[0, 0], [4, 4]], "bbox": {"min": [0, 0], "max": [4, 4]}}))
    hm = load_house_map(str(p))
    assert hm.is_inside(2.0, 2.0) is True
    assert hm.is_inside(9.0, 9.0) is False
