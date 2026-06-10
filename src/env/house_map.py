"""HouseExpo JSON floor-plan -> wall segments + free-space bounds (PRD-HOUSEEXPO §5)."""

from __future__ import annotations

import json
from dataclasses import dataclass

Segment = tuple  # (x1, y1, x2, y2)


@dataclass
class HouseMap:
    """Value object: wall segments + axis-aligned free-space bounds (metres)."""

    walls: list[Segment]
    bounds: tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax)

    def is_inside(self, x: float, y: float) -> bool:
        """True iff (x, y) lies within the axis-aligned free-space bounds."""
        xmin, ymin, xmax, ymax = self.bounds
        return (xmin <= x <= xmax) and (ymin <= y <= ymax)


def _verts(plan: dict) -> list[tuple[float, float]]:
    raw = plan.get("verts") or plan.get("vertices") or []
    return [(float(p[0]), float(p[1])) for p in raw]


def _bounds(plan: dict, verts: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    bbox = plan.get("bbox")
    if bbox and "min" in bbox and "max" in bbox:
        lo, hi = bbox["min"], bbox["max"]
        return (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    return (min(xs), min(ys), max(xs), max(ys))


def load_house_map(path: str) -> HouseMap:
    """Parse one HouseExpo JSON plan into a deterministic HouseMap (PRD-HOUSEEXPO §5.1)."""
    with open(path, encoding="utf-8") as fh:
        plan = json.load(fh)
    verts = _verts(plan)
    walls: list[Segment] = []
    for i in range(len(verts) - 1):
        x1, y1 = verts[i]
        x2, y2 = verts[i + 1]
        if (x1, y1) != (x2, y2):
            walls.append((x1, y1, x2, y2))
    walls.sort()  # stable order -> deterministic parse (§5.1)
    return HouseMap(walls=walls, bounds=_bounds(plan, verts))
