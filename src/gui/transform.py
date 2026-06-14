"""World(metres) <-> screen(pixels) mapping with an aspect-preserving letterbox."""

from __future__ import annotations


class View:
    """Maps map-space metres to window pixels, preserving aspect (centred/letterboxed).

    A single uniform `scale` is used for both axes so square metres stay square on
    screen; screen-y is flipped (world-up = screen-up).
    """

    def __init__(self, bounds: tuple[float, float, float, float], win_w: int, win_h: int, pad: int = 20):
        xmin, ymin, xmax, ymax = bounds
        self.xmin = xmin
        self.ymin = ymin
        span_x = max(xmax - xmin, 1e-9)
        self.span_y = max(ymax - ymin, 1e-9)
        avail_w = max(win_w - 2 * pad, 1)
        avail_h = max(win_h - 2 * pad, 1)
        self.scale = min(avail_w / span_x, avail_h / self.span_y)
        self.off_x = (win_w - self.scale * span_x) / 2.0
        self.off_y = (win_h - self.scale * self.span_y) / 2.0

    def to_screen(self, x: float, y: float) -> tuple[int, int]:
        """Map world (x, y) in metres to integer screen pixels (y flipped)."""
        px = self.off_x + (x - self.xmin) * self.scale
        py = self.off_y + (self.span_y - (y - self.ymin)) * self.scale
        return (round(px), round(py))

    def scale_len(self, metres: float) -> int:
        """Scale a length in metres to a pixel length."""
        return round(metres * self.scale)
