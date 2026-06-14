"""Render the live per-episode reward curve onto a pygame Surface."""

from __future__ import annotations

import pygame

from src.gui import palette

_MIN_PTS = 2  # need >=2 points for a polyline
_EPS = 1e-9  # floor for a flat (zero-range) reward series


def _rolling(values: list[float], w: int) -> list[float]:
    """Trailing rolling mean (expanding for the first w-1 points), same length."""
    out = []
    for i in range(len(values)):
        window = values[max(0, i - w + 1) : i + 1]
        out.append(sum(window) / len(window))
    return out


def draw_curve(surface: pygame.Surface, rect: pygame.Rect, rewards: list[float], rolling: int = 10) -> None:
    """Plot per-episode `rewards` inside `rect` (autoscaled) with a zero line + rolling mean."""
    if not rewards:
        return
    lo, hi = min(rewards), max(rewards)
    if hi - lo < _EPS:
        hi = lo + 1.0
    n = len(rewards)

    def to_pt(i: int, v: float) -> tuple[int, int]:
        x = rect.left + rect.width * (i / max(n - 1, 1))
        y = rect.bottom - rect.height * (v - lo) / (hi - lo)
        return (int(x), int(y))

    if lo < 0.0 < hi:
        zy = int(rect.bottom - rect.height * (0.0 - lo) / (hi - lo))
        pygame.draw.line(surface, palette.ZERO_LINE, (rect.left, zy), (rect.right, zy), 1)
    if n >= _MIN_PTS:
        raw = [to_pt(i, v) for i, v in enumerate(rewards)]
        pygame.draw.lines(surface, palette.CURVE, False, raw, palette.CURVE_W)
        roll = [to_pt(i, v) for i, v in enumerate(_rolling(rewards, rolling))]
        pygame.draw.lines(surface, palette.ROLL, False, roll, palette.CURVE_W)
