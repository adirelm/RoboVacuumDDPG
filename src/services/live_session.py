"""LiveSession: a per-step driver that streams Frames for the GUI (train/play/drive).

The GUI never touches src.env/src.ddpg directly — it consumes Frames from this
session through the SDK. LiveSession lives in src.services (business logic), so it
MAY use the env internals (cast_lidar, the coverage grid) the GUI is not allowed
to import.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.ddpg.agent import DDPGAgent
from src.env.house_map import HouseMap
from src.env.raycast import cast_lidar
from src.env.vacuum_env import VacuumEnv
from src.services.trainer import Trainer

_MODES = ("train", "play", "drive")


@dataclass(frozen=True)
class Frame:
    """One rendered step: pose, sensors, per-step deltas, and training stats."""

    mode: str
    pose: tuple[float, float, float]
    lidar_endpoints: list[tuple[float, float]]
    new_cells: list[tuple[float, float]]
    coverage: float
    reward: float
    collision: bool
    episode: int
    step: int
    sigma: float
    buffer_size: int
    done: bool


class LiveSession:
    """Advance one step at a time in a chosen mode, returning a Frame each call."""

    def __init__(self, cfg: dict, house_map: HouseMap, mode: str, seed=None, checkpoint_path=None):
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        self.cfg = cfg
        self.mode = mode
        self.house_map = house_map
        self.env = VacuumEnv(house_map, cfg, seed=seed)
        self.agent = DDPGAgent(self.env.state_dim, self.env.action_dim, cfg, seed=seed)
        self.has_checkpoint = False
        if mode == "play" and checkpoint_path is not None:
            try:
                self.agent.load(checkpoint_path)
                self.has_checkpoint = True
            except FileNotFoundError:
                self.has_checkpoint = False
        self.trainer = Trainer(self.env, self.agent, cfg) if mode == "train" else None
        self._state = self.env.reset()
        self._episode = 0
        self._step = 0
        self._fresh = True  # first frame (and post-reset) emits the spawn footprint as new_cells

    @property
    def meta(self) -> dict:
        """Static draw metadata the GUI needs (bounds, walls, cell/robot sizes, lidar)."""
        e = self.cfg["env"]
        return {
            "bounds": self.house_map.bounds,
            "walls": self.house_map.walls,
            "cell_size": e["coverage_cell"],
            "clean_radius": e["clean_radius"],
            "robot_radius": e["robot_radius"],
            "n_rays": e["n_rays"],
            "ray_max": e["ray_max"],
        }

    def _advance(self, action) -> tuple[np.ndarray, float, bool, bool]:
        before = self.env.coverage.cleaned.copy()
        if self.mode == "train":
            self._state, info, done = self.trainer.step(self._state)
            reward, collision = info["reward"], info["collision"]
        else:
            greedy = self.agent.act(self._state, explore=False)
            act = action if (self.mode == "drive" and action is not None) else greedy
            self._state, reward, done, env_info = self.env.step(act)
            collision = env_info["collision"]
        return before, float(reward), bool(collision), bool(done)

    def step(self, action=None) -> Frame:
        """Advance one step (mode-specific) and return the resulting Frame."""
        before, reward, collision, done = self._advance(action)
        if self._fresh:  # surface the spawn footprint (already cleaned by reset) once
            before = np.zeros_like(before)
            self._fresh = False
        self._step += 1
        x, y, theta = self.env.pose
        frame = Frame(
            mode=self.mode,
            pose=(float(x), float(y), float(theta)),
            lidar_endpoints=self._lidar(x, y, theta),
            new_cells=self._new_cells(before),
            coverage=float(self.env.coverage.fraction()),
            reward=reward,
            collision=collision,
            episode=self._episode,
            step=self._step,
            sigma=float(self.agent.noise.sigma),
            buffer_size=len(self.agent.buffer),
            done=done,
        )
        if done:
            self._state = self.env.reset()
            self._episode += 1
            self._fresh = True  # next frame re-emits the new spawn footprint
        return frame

    def reset(self) -> None:
        """Restart the current episode (fresh spawn + cleared coverage grid)."""
        self._state = self.env.reset()

    def _lidar(self, x: float, y: float, theta: float) -> list[tuple[float, float]]:
        e = self.cfg["env"]
        dists = cast_lidar(x, y, theta, e["n_rays"], self.house_map.walls, e["ray_max"])
        out = []
        for i, d in enumerate(dists):
            phi = theta + 2.0 * math.pi * i / e["n_rays"]
            out.append((float(x + d * math.cos(phi)), float(y + d * math.sin(phi))))
        return out

    def _new_cells(self, before: np.ndarray) -> list[tuple[float, float]]:
        cov = self.env.coverage
        ixs, iys = np.where(cov.cleaned & ~before)
        return [(float(cov.cx[ix]), float(cov.cy[iy])) for ix, iy in zip(ixs, iys, strict=True)]
