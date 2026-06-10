"""Observation assembly: normalized lidar + (v, w) + heading cue (spec §3, PRD-SIM §2.2)."""

from __future__ import annotations

import numpy as np


def assemble_state(  # noqa: PLR0913
    lidar: np.ndarray,
    v: float,
    omega: float,
    heading_cos: float,
    heading_sin: float,
    ray_max: float,
    v_max: float,
    omega_max: float,
) -> np.ndarray:
    """Normalized float32 state, shape (n_rays + 4,): lidar/ray_max + v/v_max + w/w_max + cos + sin."""
    rays = np.asarray(lidar, dtype=np.float32) / ray_max
    tail = np.array(
        [v / v_max, omega / omega_max, heading_cos, heading_sin],
        dtype=np.float32,
    )
    return np.concatenate([rays, tail]).astype(np.float32)
