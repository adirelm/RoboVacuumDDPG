"""Reward shaping: r = k_coverage*new_cells - k_collision*collision - k_step (PRD-SIM §3.5)."""

from __future__ import annotations


def compute_reward(
    new_cells: int,
    collision: bool,
    k_coverage: float,
    k_collision: float,
    k_step: float,
) -> float:
    """r = k_coverage*new_cells - k_collision*collision - k_step (signs per FR-6)."""
    return k_coverage * new_cells - k_collision * float(collision) - k_step
