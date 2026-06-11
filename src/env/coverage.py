"""Cleaned-cell grid + coverage fraction + nearest-uncleaned bearing (PRD-SIM §3.3)."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np


class CoverageGrid:
    """Boolean grid of cells over the map's axis-aligned bounding box.

    The denominator for `fraction()` is the count of REACHABLE free cells —
    those whose centre lies inside the map free space per the optional `is_free`
    callable (e.g. ``HouseMap.is_inside``). Cells inside walls / outside the
    floor plan are excluded, so on a non-convex plan ``coverage_target`` stays
    reachable. When `is_free` is omitted, every bounding-box cell counts as free,
    preserving the legacy behaviour on convex box maps.
    """

    def __init__(
        self,
        bounds,
        cell_size: float,
        clean_radius: float,
        is_free: Callable[[float, float], bool] | None = None,
    ):
        self.xmin, self.ymin, xmax, ymax = bounds
        self.cell_size = cell_size
        self.clean_radius = clean_radius
        self.nx = max(1, round((xmax - self.xmin) / cell_size))
        self.ny = max(1, round((ymax - self.ymin) / cell_size))
        # cell centres
        self.cx = self.xmin + (np.arange(self.nx) + 0.5) * cell_size
        self.cy = self.ymin + (np.arange(self.ny) + 0.5) * cell_size
        self.cleaned = np.zeros((self.nx, self.ny), dtype=bool)
        # precompute the reachable-free mask once (centre inside the floor plan)
        if is_free is None:
            self.free = np.ones((self.nx, self.ny), dtype=bool)
        else:
            self.free = np.array(
                [[bool(is_free(float(x), float(y))) for y in self.cy] for x in self.cx],
                dtype=bool,
            )
        self.n_free = max(1, int(self.free.sum()))

    def mark(self, x: float, y: float) -> int:
        """Mark cells whose centre is within clean_radius; return NEWLY cleaned count."""
        dx = self.cx[:, None] - x
        dy = self.cy[None, :] - y
        within = (dx * dx + dy * dy) <= (self.clean_radius * self.clean_radius)
        newly = within & ~self.cleaned
        count = int(newly.sum())
        self.cleaned |= newly
        return count

    def fraction(self) -> float:
        """Cleaned-and-free cells / total reachable-free cells, in [0, 1]."""
        return float((self.cleaned & self.free).sum()) / float(self.n_free)

    def nearest_uncleaned_bearing(self, x: float, y: float, theta: float) -> tuple[float, float]:
        """(cos, sin) of the bearing to the nearest uncleaned cell, in the robot frame.

        Only REACHABLE free cells count as targets (``self.free & ~self.cleaned``),
        matching `fraction()` and the done-condition; otherwise the cue would point
        the policy at wall/exterior cells it can never clean.
        """
        unclean = self.free & ~self.cleaned
        if not unclean.any():
            return (0.0, 0.0)
        ix, iy = np.where(unclean)
        dx = self.cx[ix] - x
        dy = self.cy[iy] - y
        j = int(np.argmin(dx * dx + dy * dy))
        ang = math.atan2(dy[j], dx[j]) - theta  # rotate into robot frame
        return (math.cos(ang), math.sin(ang))

    def reset(self) -> None:
        """Clear all cleaned flags (per-episode grid)."""
        self.cleaned[:] = False
