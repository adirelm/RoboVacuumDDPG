"""Pygame live viewer — watch DDPG train, play a trained policy, or drive manually.

uv run python scripts/play.py [--map room_single] [--seed 42] [--checkpoint assets/demo_policy.pt]

Modes: [T] train (default), [P] play trained, [D] drive (arrow keys). The window
loop only runs under `__main__`, so importing this module (tests) opens nothing.
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pygame  # noqa: E402

from src.gui import input_map  # noqa: E402
from src.gui.app import App  # noqa: E402


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RoboVacuumDDPG live viewer")
    p.add_argument("--map", default="room_single")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint", default=None, help="trained policy for PLAY mode")
    return p.parse_args(argv)


def _loop(app: App, screen: pygame.Surface, clock: pygame.time.Clock, fps: int) -> None:
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                cmd = input_map.command_for(event.key)
                if cmd == "quit":
                    running = False
                elif cmd:
                    app.handle_command(cmd)
        app.update()
        app.draw(screen)
        pygame.display.flip()
        clock.tick(fps)


def main(argv=None) -> int:
    args = _parse_args(argv)
    app = App([args.map], args.seed, checkpoint_path=args.checkpoint)
    if args.checkpoint is None:  # default PLAY-mode policy = the committed demo
        app.checkpoint_path = app.sdk.cfg["gui"]["demo_checkpoint"]
    g = app.sdk.cfg["gui"]
    pygame.init()
    pygame.display.set_caption("RoboVacuumDDPG - live viewer")
    screen = pygame.display.set_mode((g["window_width"], g["window_height"]))
    _loop(app, screen, pygame.time.Clock(), int(g["fps"]))
    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
