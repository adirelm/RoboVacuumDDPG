"""App: orchestrates the live viewer — owns the SDK + LiveSession, advances + draws frames.

Imports ONLY src.sdk (+ pygame / intra-gui), so all business logic stays behind the
SDK. Holds no env/agent internals — it consumes Frames from the session.
"""

from __future__ import annotations

import pygame

from src.gui import curve_view, env_view, hud, input_map
from src.gui.transform import View
from src.sdk.sdk import RoboVacuumSDK

_MODE_CMDS = {"mode_train": "train", "mode_play": "play", "mode_drive": "drive"}
_MAX_SPEED = 50


class App:
    """Holds the SDK + a LiveSession and renders one frame per tick."""

    def __init__(
        self, maps: list[str], seed: int, checkpoint_path: str | None = None, config_path: str | None = None
    ):
        self.sdk = RoboVacuumSDK(config_path)
        g = self.sdk.cfg["gui"]
        self.win = (g["window_width"], g["window_height"])
        self.speed = int(g["train_steps_per_frame"])
        self.trail_len = int(g["trail_length"])
        self.maps = maps
        self.map_idx = 0
        self.seed = seed
        self.checkpoint_path = checkpoint_path
        self.mode = "train"
        self.paused = False
        self.show_lidar = True
        self.show_coverage = True
        self._hud = hud.Hud()
        self._new_session()

    def _new_session(self) -> None:
        ckpt = self.checkpoint_path if self.mode == "play" else None
        self.session = self.sdk.live_session(self.maps[self.map_idx], self.seed, self.mode, ckpt)
        self.meta = self.session.meta
        self.view = View(self.meta["bounds"], self.win[0], self.win[1])
        self.covered: list[tuple[float, float]] = []
        self.trail: list[tuple[float, float]] = []
        self.episode_rewards: list[float] = []
        self._ep_reward = 0.0
        self.last_frame = self.session.step()
        self._absorb(self.last_frame)

    def _absorb(self, frame) -> None:
        self.covered.extend(frame.new_cells)
        self.trail.append((frame.pose[0], frame.pose[1]))
        if len(self.trail) > self.trail_len:
            self.trail = self.trail[-self.trail_len :]
        self._ep_reward += frame.reward
        if frame.done:
            self.episode_rewards.append(self._ep_reward)
            self._ep_reward = 0.0
            self.covered = []
            self.trail = []

    def update(self) -> None:
        """Advance `speed` steps (unless paused), accumulating trail/coverage/rewards."""
        if self.paused:
            return
        action = input_map.drive_action(pygame.key.get_pressed()) if self.mode == "drive" else None
        for _ in range(self.speed):
            self.last_frame = self.session.step(action)
            self._absorb(self.last_frame)

    def draw(self, surface: pygame.Surface) -> None:
        """Compose env_view + the reward curve + the HUD onto `surface`."""
        scene = env_view.EnvScene(
            walls=self.meta["walls"],
            pose=self.last_frame.pose,
            robot_radius=self.meta["robot_radius"],
            cell_size=self.meta["cell_size"],
            covered=self.covered,
            trail=self.trail,
            lidar_endpoints=self.last_frame.lidar_endpoints,
            show_lidar=self.show_lidar,
            show_coverage=self.show_coverage,
        )
        env_view.draw_env(surface, self.view, scene)
        w, h = surface.get_size()
        rect = pygame.Rect(int(w * 0.62), int(h * 0.70), int(w * 0.36), int(h * 0.27))
        curve_view.draw_curve(surface, rect, self.episode_rewards)
        no_ckpt = self.mode == "play" and not self.session.has_checkpoint
        self._hud.draw(surface, self.last_frame, self.mode, self.speed, no_checkpoint=no_ckpt)

    def handle_command(self, cmd: str) -> None:
        """Apply a command (mode switch / pause / toggles / cycling), rebuilding when needed."""
        if cmd in _MODE_CMDS:
            self.mode = _MODE_CMDS[cmd]
            self._new_session()
        elif cmd == "pause":
            self.paused = not self.paused
        elif cmd == "reset":
            self._new_session()
        elif cmd == "speed_up":
            self.speed = min(self.speed + 1, _MAX_SPEED)
        elif cmd == "speed_down":
            self.speed = max(self.speed - 1, 1)
        elif cmd == "toggle_lidar":
            self.show_lidar = not self.show_lidar
        elif cmd == "toggle_coverage":
            self.show_coverage = not self.show_coverage
        elif cmd == "cycle_map":
            self.map_idx = (self.map_idx + 1) % len(self.maps)
            self._new_session()
        elif cmd == "cycle_seed":
            self.seed += 1
            self._new_session()
