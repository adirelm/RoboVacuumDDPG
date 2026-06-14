import pygame

from src.gui import palette
from src.gui.curve_view import _rolling, draw_curve


def test_empty_rewards_is_noop(gui_surface):
    draw_curve(gui_surface, pygame.Rect(10, 10, 200, 100), [])  # must not raise


def test_rolling_mean_hand_computed():
    assert _rolling([1.0, 2.0, 3.0, 4.0], 2) == [1.0, 1.5, 2.5, 3.5]


def test_curve_draws_non_background_pixels(gui_surface):
    rect = pygame.Rect(10, 10, 300, 150)
    gui_surface.fill(palette.BG)
    draw_curve(gui_surface, rect, [-100.0, 0.0, 100.0, 50.0, 200.0])
    found = any(
        gui_surface.get_at((x, y))[:3] != palette.BG
        for x in range(rect.left, rect.right, 4)
        for y in range(rect.top, rect.bottom, 4)
    )
    assert found
