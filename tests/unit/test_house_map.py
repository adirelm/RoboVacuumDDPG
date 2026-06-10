import json
from pathlib import Path

from src.env.house_map import HouseMap, Segment, load_house_map


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


def test_segment_alias_is_tuple_type():
    assert Segment is tuple
