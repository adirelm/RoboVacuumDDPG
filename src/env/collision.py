"""Robot-disc vs wall-segment collision test (PRD-SIM §3.4, FR-5)."""

from __future__ import annotations

import math

from src.env.house_map import Segment  # single source of truth (4-float tuple)


def _point_segment_distance(px: float, py: float, seg: Segment) -> float:
    """Shortest Euclidean distance from point (px, py) to segment seg."""
    ax, ay, bx, by = seg
    abx, aby = bx - ax, by - ay
    denom = abx * abx + aby * aby
    if denom == 0.0:  # degenerate segment = point
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * abx + (py - ay) * aby) / denom
    t = max(0.0, min(1.0, t))  # clamp projection to the segment
    cx, cy = ax + t * abx, ay + t * aby
    return math.hypot(px - cx, py - cy)


def collides(x: float, y: float, robot_radius: float, walls: list[Segment]) -> bool:
    """True iff the robot disc (centre (x, y), radius robot_radius) intersects any wall."""
    return any(_point_segment_distance(x, y, seg) < robot_radius for seg in walls)
