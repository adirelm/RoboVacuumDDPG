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
