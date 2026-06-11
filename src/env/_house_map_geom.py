"""Geometry helpers for the HouseExpo adapter (PRD-HOUSEEXPO §5).

Pure functions on the exterior boundary polygon: vertex extraction (with a
documented metric scale), polygon closure, wall-segment derivation, bounds, and
a ray-casting point-in-polygon test. Kept separate so ``house_map.py`` stays a
thin loader under the 150-LOC cap (CLAUDE.md §1; design-spec A4 split convention).
"""

from __future__ import annotations

Segment = tuple[float, float, float, float]  # (x1, y1, x2, y2)
Vert = tuple[float, float]

_MIN_VERTS_TO_CLOSE = 2  # need at least an edge before appending a closing vertex.


def verts_metric(plan: dict, scale: float) -> list[Vert]:
    """Exterior boundary vertices in metres.

    HouseExpo stores the boundary contour under ``verts`` (a few maps use
    ``vertices``). Coordinates are multiplied by ``scale`` so callers receive
    metres; real HouseExpo plans are already metric (``scale = 1.0``), but the
    factor is config-driven so a pixel-resolution source could be rescaled
    without touching this module (CLAUDE.md §4).
    """
    raw = plan.get("verts") or plan.get("vertices") or []
    return [(float(p[0]) * scale, float(p[1]) * scale) for p in raw]


def closed_ring(verts: list[Vert]) -> list[Vert]:
    """Return the polygon ring with the first vertex repeated at the end.

    Real HouseExpo ``verts`` are an *open* contour (first != last); closing the
    ring makes the last edge (back to the start) an explicit wall and lets the
    point-in-polygon test treat the boundary as a closed loop.
    """
    if len(verts) >= _MIN_VERTS_TO_CLOSE and verts[0] != verts[-1]:
        return [*verts, verts[0]]
    return list(verts)


def walls_from_ring(ring: list[Vert]) -> list[Segment]:
    """Consecutive vertices of a closed ring -> deterministic wall segments.

    Zero-length edges (duplicate consecutive vertices) are dropped; the result
    is sorted so parsing the same plan twice is byte-identical (PRD §5.1).
    """
    walls: list[Segment] = []
    for i in range(len(ring) - 1):
        x1, y1 = ring[i]
        x2, y2 = ring[i + 1]
        if (x1, y1) != (x2, y2):
            walls.append((x1, y1, x2, y2))
    walls.sort()
    return walls


def bounds_of(plan: dict, verts: list[Vert], scale: float) -> tuple[float, float, float, float]:
    """Axis-aligned free-space bounds (xmin, ymin, xmax, ymax) in metres."""
    bbox = plan.get("bbox")
    if bbox and "min" in bbox and "max" in bbox:
        lo, hi = bbox["min"], bbox["max"]
        return (float(lo[0]) * scale, float(lo[1]) * scale, float(hi[0]) * scale, float(hi[1]) * scale)
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    return (min(xs), min(ys), max(xs), max(ys))


def point_in_polygon(x: float, y: float, ring: list[Vert]) -> bool:
    """Ray-casting (even-odd) point-in-polygon test against the boundary ring.

    A proper interior test (not just the bounding box) so non-convex, real
    HouseExpo plans expose only their navigable interior as free space — the
    invariant ``coverage.py`` relies on via ``HouseMap.is_inside`` (PRD §5).
    """
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside
