"""Pure mappings from pygame key events to app commands / a manual drive action."""

from __future__ import annotations

import pygame

_KEY_COMMAND = {
    pygame.K_ESCAPE: "quit",
    pygame.K_SPACE: "pause",
    pygame.K_r: "reset",
    pygame.K_t: "mode_train",
    pygame.K_p: "mode_play",
    pygame.K_d: "mode_drive",
    pygame.K_EQUALS: "speed_up",
    pygame.K_PLUS: "speed_up",
    pygame.K_MINUS: "speed_down",
    pygame.K_l: "toggle_lidar",
    pygame.K_c: "toggle_coverage",
    pygame.K_TAB: "cycle_map",
    pygame.K_s: "cycle_seed",
}


def command_for(key: int) -> str | None:
    """Map a pygame key constant to an app command, or None if the key is unbound."""
    return _KEY_COMMAND.get(key)


def drive_action(pressed) -> list[float]:
    """Build a [throttle, steer] action in [-1, 1]^2 from the arrow keys held now.

    `pressed` is a pygame.key.get_pressed()-style sequence indexable by key code.
    """
    throttle = float(bool(pressed[pygame.K_UP])) - float(bool(pressed[pygame.K_DOWN]))
    steer = float(bool(pressed[pygame.K_RIGHT])) - float(bool(pressed[pygame.K_LEFT]))
    return [throttle, steer]
