"""Shared pytest fixtures for RoboVacuumDDPG.

Provides:
  cfg        — the real config dict loaded from config/config.yaml.
  house_map  — a tiny synthetic 4-wall square room mirroring the contract
               HouseMap shape (walls / bounds / is_inside), with NO dependency
               on src.env.house_map (built in a later phase). Later phases may
               override `house_map` locally to use the real loader.

This file is incremental (contract F12): later phases APPEND fixtures
(e.g. `tiny_map`) — they never replace or drop the ones defined here.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.utils.config_loader import load_config

Segment = tuple[float, float, float, float]


@dataclass
class FakeHouseMap:
    """Synthetic HouseMap stand-in (same public surface as the contract dataclass)."""

    walls: list[Segment]
    bounds: tuple[float, float, float, float]

    def is_inside(self, x: float, y: float) -> bool:
        xmin, ymin, xmax, ymax = self.bounds
        return xmin < x < xmax and ymin < y < ymax


@pytest.fixture
def cfg() -> dict:
    """Return the project config dict (config/config.yaml)."""
    return load_config()


@pytest.fixture
def house_map() -> FakeHouseMap:
    """A 4x4 m square room: four wall segments, bounds (0,0,4,4)."""
    walls: list[Segment] = [
        (0.0, 0.0, 4.0, 0.0),
        (4.0, 0.0, 4.0, 4.0),
        (4.0, 4.0, 0.0, 4.0),
        (0.0, 4.0, 0.0, 0.0),
    ]
    return FakeHouseMap(walls=walls, bounds=(0.0, 0.0, 4.0, 4.0))
