"""Real-HouseExpo-schema tests for the house_map loader: open-verts closing,
room_category is not turned into walls, point-in-polygon vs bbox, and the
vendored real map (split out of test_house_map.py to keep each file small).
"""

import json
import math
from pathlib import Path

from src.env.house_map import load_house_map

_REPO = Path(__file__).resolve().parents[2]
# Vendored real HouseExpo plan (room_single.json, id 011bef..., room_num=1) —
# exercises the loader against the REAL schema, not a synthetic stand-in.
_REAL_ROOM = _REPO / "data" / "maps" / "room_single.json"


def _write_real_schema(tmp_path: Path) -> str:
    """Tiny TRIMMED real-schema plan: open (unclosed) L-shape verts + room_category
    + bbox (mirrors HouseExpo, Li et al. 2019; room boxes are NOT walls)."""
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


def test_real_schema_open_verts_are_closed_into_walls(tmp_path):
    # 6 verts of an OPEN L-shape -> closing the ring yields 6 wall edges
    # (5 between consecutive verts + 1 closing edge back to the start).
    hm = load_house_map(_write_real_schema(tmp_path))
    assert len(hm.walls) == 6
    closing = (0.0, 0.0, 0.0, 6.0)  # last vert (0,6) -> first (0,0), canonicalised
    edges = {tuple(sorted((s[0], s[2])) + sorted((s[1], s[3]))) for s in hm.walls}
    assert tuple(sorted((closing[0], closing[2])) + sorted((closing[1], closing[3]))) in edges


def test_real_schema_room_category_is_not_turned_into_walls(tmp_path):
    # The room_category boxes must NOT add interior walls — only the 6 boundary
    # edges exist (PRD-HOUSEEXPO §5.2: boundary-only, no fabrication).
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
    # (0.5, 4.0) is inside the bbox but in the upper-left notch the boundary cuts
    # off -> point-in-polygon must reject it.
    assert hm.is_inside(0.5, 4.0) is False
    assert hm.is_inside(-1.0, -1.0) is False
