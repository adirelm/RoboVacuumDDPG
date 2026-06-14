"""Dev tool: render headless screenshots of each GUI mode into assets/screenshots/.

Uses the SDL dummy driver so it needs no display (runs on CI too).

uv run python scripts/capture_screenshots.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pygame  # noqa: E402

from src.gui.app import App  # noqa: E402

_OUT = _REPO_ROOT / "assets" / "screenshots"
_DEMO = str(_REPO_ROOT / "assets" / "demo_policy.pt")


def _shot(app: App, steps: int, name: str) -> None:
    for _ in range(steps):
        app.update()
    surface = pygame.Surface((app.win[0], app.win[1]))
    app.draw(surface)
    _OUT.mkdir(parents=True, exist_ok=True)
    pygame.image.save(surface, str(_OUT / name))
    print("wrote", _OUT / name)


def main() -> int:
    pygame.init()
    _shot(App(["room_single"], 42, checkpoint_path=None), 80, "train.png")

    play = App(["room_single"], 42, checkpoint_path=_DEMO)
    play.handle_command("mode_play")
    _shot(play, 180, "play.png")  # mid-episode (< max_steps/speed) so coverage is visible

    # Simulate held arrow keys (forward + right) so the drive shot shows a real path
    # (headless capture has no real keyboard state).
    held = {pygame.K_UP: True, pygame.K_DOWN: False, pygame.K_LEFT: False, pygame.K_RIGHT: True}
    pygame.key.get_pressed = lambda: held
    drive = App(["room_single"], 42)
    drive.handle_command("mode_drive")
    _shot(drive, 120, "drive.png")

    nockpt = App(["room_single"], 42, checkpoint_path="assets/_absent_.pt")
    nockpt.handle_command("mode_play")
    _shot(nockpt, 80, "play_no_checkpoint.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
