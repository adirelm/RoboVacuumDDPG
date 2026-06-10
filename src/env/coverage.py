"""Cleaned-cell grid + coverage fraction + nearest-uncleaned bearing (PRD-SIM §3.3)."""

from __future__ import annotations

import math

import numpy as np


class CoverageGrid:
    """Boolean grid of cells over the map's axis-aligned bounding box.

    The denominator for `fraction()` is every cell in the bounding box. For the
    convex box maps used here this equals the free-space; on a non-convex plan
    it would also count cells inside walls (those stay uncleanable forever).
    """

    def __init__(self, bounds, cell_size: float, clean_radius: float):
        self.xmin, self.ymin, xmax, ymax = bounds
        self.cell_size = cell_size
        self.clean_radius = clean_radius
        self.nx = max(1, round((xmax - self.xmin) / cell_size))
        self.ny = max(1, round((ymax - self.ymin) / cell_size))
        # cell centres
        self.cx = self.xmin + (np.arange(self.nx) + 0.5) * cell_size
        self.cy = self.ymin + (np.arange(self.ny) + 0.5) * cell_size
        self.cleaned = np.zeros((self.nx, self.ny), dtype=bool)

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
        """Cleaned cells / total cells in the bounding-box grid, in [0, 1]."""
        return float(self.cleaned.sum()) / float(self.cleaned.size)

    def nearest_uncleaned_bearing(self, x: float, y: float, theta: float) -> tuple[float, float]:
        """(cos, sin) of the bearing to the nearest uncleaned cell, in the robot frame."""
        unclean = ~self.cleaned
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
