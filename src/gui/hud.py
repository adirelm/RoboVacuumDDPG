"""Render the text HUD (mode, coverage%, step, reward, episode, sigma) onto a Surface."""

from __future__ import annotations

import pygame

from src.gui import palette

_CONTROLS = "T/P/D mode | space pause | r reset | +/- speed | L lidar | C cover | tab map | esc quit"


class Hud:
    """Lazily-initialised pygame font that blits the per-frame stats panel."""

    def __init__(self, size: int = 20):
        pygame.font.init()
        self.font = pygame.font.Font(None, size)

    def draw(
        self, surface: pygame.Surface, frame, mode: str, speed: int, *, no_checkpoint: bool = False
    ) -> None:
        """Blit the HUD stat lines top-left + a controls hint, plus a no-checkpoint badge."""
        lines = [
            f"mode: {mode}    speed x{speed}",
            f"episode {frame.episode}    step {frame.step}",
            f"coverage {frame.coverage * 100:.1f}%    reward {frame.reward:+.2f}",
            f"sigma {frame.sigma:.3f}    buffer {frame.buffer_size}",
        ]
        if no_checkpoint:
            lines.append("no checkpoint - untrained policy")
        y = 8
        last = len(lines) - 1
        for i, text in enumerate(lines):
            colour = palette.BADGE if (no_checkpoint and i == last) else palette.TEXT
            surface.blit(self.font.render(text, True, colour), (8, y))
            y += self.font.get_height() + 2
        hint = self.font.render(_CONTROLS, True, palette.TEXT)
        surface.blit(hint, (8, surface.get_height() - self.font.get_height() - 6))
