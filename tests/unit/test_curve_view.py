import pygame

from src.gui import palette
from src.gui.curve_view import _rolling, draw_curve


def test_empty_rewards_is_noop(gui_surface):
    draw_curve(gui_surface, pygame.Rect(10, 10, 200, 100), [])  # must not raise


def test_rolling_mean_hand_computed():
    assert _rolling([1.0, 2.0, 3.0, 4.0], 2) == [1.0, 1.5, 2.5, 3.5]


def test_curve_draws_the_reward_polylines(gui_surface):
    # Strictly-positive series => no zero line, so any CURVE/ROLL pixel must come
    # from the reward polylines themselves (guards against a vacuous "any non-BG").
    rect = pygame.Rect(10, 10, 300, 150)
    gui_surface.fill(palette.BG)
    draw_curve(gui_surface, rect, [10.0, 50.0, 30.0, 80.0, 60.0])
    pixels = {
        gui_surface.get_at((x, y))[:3]
        for x in range(rect.left, rect.right, 2)
        for y in range(rect.top, rect.bottom, 2)
    }
    assert palette.CURVE in pixels, "raw reward polyline not drawn"
    assert palette.ROLL in pixels, "rolling-mean polyline not drawn"


def test_sign_crossing_series_draws_zero_line(gui_surface):
    # A series that straddles 0 (lo<0<hi) must draw the ZERO_LINE axis at the
    # computed zero-y, spanning the rect width (curve_view.py:37-38). Strictly
    # positive/empty series skip this branch, so it needs its own case.
    rect = pygame.Rect(10, 10, 300, 150)
    gui_surface.fill(palette.BG)
    rewards = [-20.0, 30.0, -5.0, 40.0]
    draw_curve(gui_surface, rect, rewards)
    lo, hi = min(rewards), max(rewards)
    zy = int(rect.bottom - rect.height * (0.0 - lo) / (hi - lo))
    row = {gui_surface.get_at((x, zy))[:3] for x in range(rect.left, rect.right)}
    assert palette.ZERO_LINE in row, "zero-reference axis not drawn at the computed y"
