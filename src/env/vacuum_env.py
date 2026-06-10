"""VacuumEnv: from-scratch 2D vacuum MDP, 4-tuple step, NO Gymnasium (PRD-SIM §3.6)."""

from __future__ import annotations

import math

import numpy as np

from src.env.collision import collides
from src.env.coverage import CoverageGrid
from src.env.house_map import HouseMap
from src.env.kinematics import step_unicycle
from src.env.raycast import cast_lidar
from src.env.reward import compute_reward
from src.env.state import assemble_state


class VacuumEnv:
    """Custom (non-Gym) robotic-vacuum environment over a HouseMap."""

    def __init__(self, house_map: HouseMap, cfg: dict, seed: int | None = None):
        self.house_map = house_map
        self.cfg = cfg
        self.e = cfg["env"]
        self.r = cfg["reward"]
        self.rng = np.random.default_rng(seed)
        self.coverage = CoverageGrid(
            house_map.bounds,
            self.e["coverage_cell"],
            self.e["clean_radius"],
            is_free=house_map.is_inside,
        )
        self.action_dim = 2
        self.state_dim = self.e["n_rays"] + 4
        self.pose = (0.0, 0.0, 0.0)
        self.v = 0.0
        self.omega = 0.0
        self.step_count = 0

    def _spawn(self) -> tuple[float, float, float]:
        xmin, ymin, xmax, ymax = self.house_map.bounds
        pad = self.e["robot_radius"]
        for _ in range(100):
            x = float(self.rng.uniform(xmin + pad, xmax - pad))
            y = float(self.rng.uniform(ymin + pad, ymax - pad))
            if not collides(x, y, self.e["robot_radius"], self.house_map.walls):
                return (x, y, float(self.rng.uniform(-math.pi, math.pi)))
        return ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0, 0.0)

    def _state(self) -> np.ndarray:
        x, y, theta = self.pose
        lidar = cast_lidar(x, y, theta, self.e["n_rays"], self.house_map.walls, self.e["ray_max"])
        cos_b, sin_b = self.coverage.nearest_uncleaned_bearing(x, y, theta)
        return assemble_state(
            lidar,
            self.v,
            self.omega,
            cos_b,
            sin_b,
            self.e["ray_max"],
            self.e["v_max"],
            self.e["omega_max"],
        )

    def reset(self) -> np.ndarray:
        self.coverage.reset()
        self.pose = self._spawn()
        self.v = 0.0
        self.omega = 0.0
        self.step_count = 0
        self.coverage.mark(self.pose[0], self.pose[1])
        return self._state()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        throttle, steer = float(a[0]), float(a[1])
        cand = step_unicycle(
            self.pose,
            throttle,
            steer,
            self.e["v_max"],
            self.e["omega_max"],
            self.e["dt"],
        )
        collision = collides(cand[0], cand[1], self.e["robot_radius"], self.house_map.walls)
        if collision:
            self.v = 0.0
            self.omega = 0.0
        else:
            self.pose = cand
            self.v = throttle * self.e["v_max"]
            self.omega = steer * self.e["omega_max"]
        new_cells = self.coverage.mark(self.pose[0], self.pose[1])
        reward = compute_reward(
            new_cells,
            collision,
            self.r["k_coverage"],
            self.r["k_collision"],
            self.r["k_step"],
        )
        self.step_count += 1
        target = self.e.get("coverage_target", 1.0)
        done = (self.step_count >= self.e["max_steps"]) or (self.coverage.fraction() >= target)
        info = {"coverage": self.coverage.fraction(), "collision": collision, "pose": self.pose}
        return self._state(), float(reward), bool(done), info
