"""Explicit-Euler unicycle pose integrator (PRD-SIM §3.1, ADR-002). Pure function."""

from __future__ import annotations

import math


def _wrap(theta: float) -> float:
    """Wrap angle into (-pi, pi]."""
    wrapped = (theta + math.pi) % (2.0 * math.pi) - math.pi
    if wrapped == -math.pi:  # keep the half-open (-pi, pi] convention
        wrapped = math.pi
    return wrapped


def step_unicycle(  # noqa: PLR0913
    pose: tuple[float, float, float],
    throttle: float,
    steer: float,
    v_max: float,
    omega_max: float,
    dt: float,
) -> tuple[float, float, float]:
    """Integrate one timestep: returns new (x, y, theta); theta wrapped to (-pi, pi]."""
    x, y, theta = pose
    v = throttle * v_max
    omega = steer * omega_max
    x_new = x + v * math.cos(theta) * dt
    y_new = y + v * math.sin(theta) * dt
    theta_new = _wrap(theta + omega * dt)
    return (x_new, y_new, theta_new)
