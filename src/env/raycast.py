"""Lidar raycasting via ray-segment intersection (PRD-SIM §3.2)."""

from __future__ import annotations

import math

import numpy as np

Segment = tuple  # (x1, y1, x2, y2)


def _ray_segment_t(ox: float, oy: float, dx: float, dy: float, seg: Segment, max_range: float):  # noqa: PLR0913
    """Return hit distance t in [0, max_range] for one segment, or None (cross-product form)."""
    ax, ay, bx, by = seg
    sx, sy = bx - ax, by - ay  # wall direction s = B - A
    rxs = dx * sy - dy * sx  # r x s
    if rxs == 0.0:  # parallel / collinear -> no hit
        return None
    qpx, qpy = ax - ox, ay - oy  # A - origin
    t = (qpx * sy - qpy * sx) / rxs
    u = (qpx * dy - qpy * dx) / rxs
    if 0.0 <= t <= max_range and 0.0 <= u <= 1.0:
        return t
    return None


def cast_ray(x: float, y: float, angle: float, walls: list[Segment], max_range: float) -> float:
    """Distance to the nearest wall along `angle`, capped at `max_range`."""
    dx, dy = math.cos(angle), math.sin(angle)
    best = max_range
    for seg in walls:
        t = _ray_segment_t(x, y, dx, dy, seg, max_range)
        if t is not None and t < best:
            best = t
    return best


def cast_lidar(  # noqa: PLR0913
    x: float, y: float, theta: float, n_rays: int, walls: list[Segment], max_range: float
) -> np.ndarray:
    """`n_rays` raw distances, evenly spaced over 2π relative to `theta`; shape (n_rays,)."""
    out = np.empty(n_rays, dtype=np.float32)
    for i in range(n_rays):
        phi = theta + 2.0 * math.pi * i / n_rays
        out[i] = cast_ray(x, y, phi, walls, max_range)
    return out
