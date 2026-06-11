import json
import math
from pathlib import Path
from typing import get_args, get_origin

import src.env.house_map as hm_mod
from src.env.house_map import HouseMap, Segment, load_house_map

_REPO = Path(__file__).resolve().parents[2]
# A real HouseExpo plan, vendored byte-for-byte from the pinned dataset
# (data/maps/room_single.json, id 011bef0381..., room_num=1). Used so the loader
# is exercised against the REAL schema (verts/bbox/room_category/room_num/id),
# not a synthetic stand-in.
_REAL_ROOM = _REPO / "data" / "maps" / "room_single.json"


def _write_square(tmp_path: Path) -> str:
    plan = {
        "verts": [[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]],
        "bbox": {"min": [0, 0], "max": [4, 4]},
    }
    p = tmp_path / "room_single.json"
    p.write_text(json.dumps(plan))
    return str(p)


def _write_real_schema(tmp_path: Path) -> str:
    """A tiny TRIMMED real-schema plan: open (unclosed) verts + room_category + bbox.

    Mirrors the upstream HouseExpo shape (Li et al. 2019): an L-shaped, non-convex
    boundary whose verts list is NOT closed (first != last), plus per-room boxes
    that are NOT used as walls. Hand-built so expectations are exact.
    """
    plan = {
        "id": "trimmed_real_fixture",
        "room_num": 2,
        "verts": [[0, 0], [6, 0], [6, 2], [2, 2], [2, 6], [0, 6]],  # open L-shape
        "room_category": {"Bedroom": [[0, 0, 6, 2]], "Hall": [[0, 0, 2, 6]]},
        "bbox": {"min": [0, 0], "max": [6, 6]},
    }
    p = tmp_path / "trimmed_real.json"
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


# --- REAL HouseExpo schema (trimmed fixture + vendored real map) ---


def test_real_schema_open_verts_are_closed_into_walls(tmp_path):
    # 6 verts of an OPEN L-shape -> closing the ring yields 6 wall edges
    # (5 between consecutive verts + 1 closing edge back to the start).
    hm = load_house_map(_write_real_schema(tmp_path))
    assert len(hm.walls) == 6
    closing = (0.0, 0.0, 0.0, 6.0)  # last vert (0,6) -> first (0,0), after sort canonicalised
    edges = {tuple(sorted((s[0], s[2])) + sorted((s[1], s[3]))) for s in hm.walls}
    assert tuple(sorted((closing[0], closing[2])) + sorted((closing[1], closing[3]))) in edges


def test_real_schema_room_category_is_not_turned_into_walls(tmp_path):
    # The two room_category boxes must NOT add interior walls — only the 6
    # boundary edges exist (PRD-HOUSEEXPO §5.2: boundary-only, no fabrication).
    hm = load_house_map(_write_real_schema(tmp_path))
    assert len(hm.walls) == 6


def test_real_schema_point_in_polygon_beats_bbox(tmp_path):
    # (5,5) is inside the bbox (0..6) but OUTSIDE the L-shaped polygon's notch.
    hm = load_house_map(_write_real_schema(tmp_path))
    assert hm.is_inside(1.0, 1.0) is True  # inside the L
    assert hm.is_inside(5.0, 5.0) is False  # in bbox, outside polygon -> proves PIP
    assert hm.is_inside(8.0, 8.0) is False  # far outside


def test_vendored_real_map_walls_bounds_and_known_edge():
    # Hand-checked against the committed real plan room_single.json (id 011bef..).
    hm = load_house_map(str(_REAL_ROOM))
    assert hm.bounds == (0.1, 0.1, 6.9, 4.41)
    assert len(hm.walls) == 17
    # Deterministic sorted order -> first wall is the left boundary segment.
    assert hm.walls[0] == (0.1, 0.11, 0.1, 2.62)
    # Known wall: the bottom edge spans x=0.11..5.48 at y=0.10 -> length 5.37 m.
    longest = max(hm.walls, key=lambda w: math.hypot(w[2] - w[0], w[3] - w[1]))
    assert math.isclose(math.hypot(longest[2] - longest[0], longest[3] - longest[1]), 5.37, abs_tol=1e-2)


def test_vendored_real_map_is_inside_uses_polygon_not_bbox():
    hm = load_house_map(str(_REAL_ROOM))
    assert hm.is_inside(3.0, 1.0) is True  # in the main room
    # (0.5, 4.0) lies inside bbox (0.1..6.9, 0.1..4.41) but in the upper-left
    # notch the boundary cuts off -> point-in-polygon must reject it.
    assert hm.is_inside(0.5, 4.0) is False
    assert hm.is_inside(-1.0, -1.0) is False


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
