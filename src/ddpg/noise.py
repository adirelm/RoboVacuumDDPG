"""Gaussian exploration noise (PRD-DDPG checkpoint 4, F5.10-F5.13; ADR-003).

sigma decays LINEARLY from sigma_start toward sigma_end over decay_steps, then
clamps at sigma_end. Brief mandates Gaussian (not Ornstein-Uhlenbeck).
"""

from __future__ import annotations

import numpy as np


class GaussianNoise:
    def __init__(
        self,
        action_dim: int,
        sigma_start: float,
        sigma_end: float,
        decay_steps: int,
        seed: int | None = None,
    ):
        self.action_dim = action_dim
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.decay_steps = decay_steps
        self.sigma = sigma_start
        self._step = 0
        self._rng = np.random.default_rng(seed)

    def sample(self) -> np.ndarray:
        return self._rng.normal(0.0, self.sigma, size=self.action_dim).astype(np.float32)

    def decay(self) -> None:
        self._step += 1
        frac = min(self._step / self.decay_steps, 1.0)
        self.sigma = self.sigma_start + (self.sigma_end - self.sigma_start) * frac
