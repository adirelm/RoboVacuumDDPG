"""Real HouseExpo JSON floor-plan -> wall segments + free-space (PRD-HOUSEEXPO §5).

Parses the *real* HouseExpo schema (``verts`` exterior boundary polygon, ``bbox``
{min,max}, ``room_category`` per-room boxes, ``room_num``, ``id``). The exterior
``verts`` contour is real, non-convex geometry; it is closed into a ring and used
both as the wall set and as the polygon for a proper point-in-polygon ``is_inside``
(not just the bounding box). HouseExpo's ``room_category`` boxes are overlapping
semantic region annotations, NOT clean interior walls/doors, so they are NOT
fabricated into wall segments — see PRD-HOUSEEXPO §5.2 / ADR-005.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from src.env._house_map_geom import (
    Segment,
    Vert,
    bounds_of,
    closed_ring,
    point_in_polygon,
    verts_metric,
    walls_from_ring,
)
from src.utils.config_loader import load_config

__all__ = ["HouseMap", "Segment", "load_house_map"]

_DEFAULT_SCALE = 1.0  # real HouseExpo verts are already metric; see PRD §5 unit note.
_MIN_RING_LEN = 4  # >=3 distinct verts + the closing repeat -> a usable polygon.


@dataclass
class HouseMap:
    """Value object: wall segments + axis-aligned free-space bounds (metres)."""

    walls: list[Segment]
    bounds: tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax)
    boundary: list[Vert] = field(default_factory=list)  # closed exterior ring (for is_inside)

    def is_inside(self, x: float, y: float) -> bool:
        """True iff (x, y) lies within the exterior boundary polygon (point-in-polygon).

        Falls back to the axis-aligned bounds only when no boundary ring is
        available (e.g. a degenerate plan with <3 vertices).
        """
        if len(self.boundary) >= _MIN_RING_LEN:
            return point_in_polygon(x, y, self.boundary)
        xmin, ymin, xmax, ymax = self.bounds
        return (xmin <= x <= xmax) and (ymin <= y <= ymax)


def _coord_scale() -> float:
    """Metric scale for HouseExpo coords from config (maps.coord_scale), default 1.0."""
    try:
        return float(load_config()["maps"].get("coord_scale", _DEFAULT_SCALE))
    except (KeyError, FileNotFoundError, ValueError):
        return _DEFAULT_SCALE


def load_house_map(path: str) -> HouseMap:
    """Parse one real HouseExpo JSON plan into a deterministic HouseMap (PRD §5.1)."""
    with open(path, encoding="utf-8") as fh:
        plan = json.load(fh)
    scale = _coord_scale()
    verts = verts_metric(plan, scale)
    ring = closed_ring(verts)
    walls = walls_from_ring(ring)
    bounds = bounds_of(plan, verts, scale)
    return HouseMap(walls=walls, bounds=bounds, boundary=ring)
