from pathlib import Path

import numpy as np
import pytest

from src.env.house_map import HouseMap, load_house_map
from src.env.vacuum_env import VacuumEnv
from src.utils.config_loader import load_config

_REPO = Path(__file__).resolve().parents[2]

# Curated maps derived from config (train + holdout) — adding a map to config
# auto-extends the spawn-inside regression below; no hardcoded list to drift.
_CFG_MAPS = load_config()["maps"]
_ALL_MAPS = _CFG_MAPS["train"] + _CFG_MAPS["holdout"]

# 4x4 square room; walls on the four edges
WALLS = [
    (0.0, 0.0, 4.0, 0.0),
    (4.0, 0.0, 4.0, 4.0),
    (0.0, 4.0, 4.0, 4.0),
    (0.0, 0.0, 0.0, 4.0),
]
HMAP = HouseMap(walls=WALLS, bounds=(0.0, 0.0, 4.0, 4.0))

CFG = {
    "env": {
        "n_rays": 16,
        "ray_max": 5.0,
        "dt": 0.1,
        "v_max": 0.5,
        "omega_max": 1.5,
        "robot_radius": 0.17,
        "clean_radius": 0.17,
        "coverage_cell": 0.10,
        "max_steps": 1000,
        "coverage_target": 0.9,
    },
    "reward": {"k_coverage": 1.0, "k_collision": 10.0, "k_step": 0.01},
}


def test_dims_set_from_config():
    env = VacuumEnv(HMAP, CFG, seed=42)
    assert env.action_dim == 2
    assert env.state_dim == 20  # n_rays + 4


def test_reset_returns_state_vector():
    env = VacuumEnv(HMAP, CFG, seed=42)
    s = env.reset()
    assert isinstance(s, np.ndarray)
    assert s.shape == (20,)


def test_step_returns_four_tuple():
    env = VacuumEnv(HMAP, CFG, seed=42)
    env.reset()
    out = env.step(np.array([1.0, 0.0], dtype=np.float32))
    assert isinstance(out, tuple) and len(out) == 4
    s, r, done, info = out
    assert isinstance(s, np.ndarray) and s.shape == (20,)
    assert isinstance(r, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert set(info.keys()) == {"coverage", "collision", "pose"}


def test_out_of_range_action_is_clipped():
    env = VacuumEnv(HMAP, CFG, seed=42)
    env.reset()
    # action well outside [-1, 1] must not crash and must integrate as clipped
    s, r, _done, _info = env.step(np.array([5.0, -9.0], dtype=np.float32))
    assert np.isfinite(r)
    assert np.all(np.isfinite(s))


def test_collision_reverts_pose():
    env = VacuumEnv(HMAP, CFG, seed=1)
    env.reset()
    # force the robot hard against the left wall, heading west, then drive into it
    env.pose = (0.18, 2.0, np.pi)  # just inside the left wall (x=0), radius 0.17
    before = env.pose
    _s, r, _done, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
    assert info["collision"] is True
    assert info["pose"] == before  # move rejected -> no tunneling
    assert r < 0.0  # collision penalty dominates


def test_done_at_max_steps():
    cfg = {**CFG, "env": {**CFG["env"], "max_steps": 2}}
    env = VacuumEnv(HMAP, cfg, seed=42)
    env.reset()
    _, _, d1, _ = env.step(np.array([0.0, 1.0], dtype=np.float32))
    _, _, d2, _ = env.step(np.array([0.0, 1.0], dtype=np.float32))
    assert d1 is False
    assert d2 is True  # step_count == max_steps


def test_reset_is_deterministic_for_same_seed():
    a = VacuumEnv(HMAP, CFG, seed=7).reset()
    b = VacuumEnv(HMAP, CFG, seed=7).reset()
    assert np.array_equal(a, b)


def test_env_runs_on_real_curated_houseexpo_map():
    # Smoke test: VacuumEnv reset/step on a REAL, non-convex HouseExpo plan
    # (apt_multi, room_num=5) with the project config — no error, finite state,
    # and the robot spawns inside the polygon (point-in-polygon free space).
    hm = load_house_map(str(_REPO / "data" / "maps" / "apt_multi.json"))
    assert len(hm.walls) > 4  # real interior contour, not a 4-edge box
    env = VacuumEnv(hm, load_config(), seed=42)
    s = env.reset()
    assert s.shape == (env.state_dim,)
    assert hm.is_inside(env.pose[0], env.pose[1])  # spawned in real free space
    for _ in range(20):
        s, r, done, info = env.step(np.array([1.0, 0.2], dtype=np.float32))
        assert np.all(np.isfinite(s))
        assert np.isfinite(r)
        if done:
            break
    assert 0.0 <= info["coverage"] <= 1.0


@pytest.mark.parametrize("map_name", _ALL_MAPS)
def test_spawn_is_inside_polygon_across_maps_and_seeds(map_name):
    # The robot must spawn INSIDE the floor polygon on every curated map and seed
    # — not merely collision-free, which on a non-convex plan can land in an
    # exterior pocket. Regression for the _spawn is_inside fix (would fail RED on
    # office/room_single/apt_small/apt_large before it).
    hm = load_house_map(str(_REPO / "data" / "maps" / f"{map_name}.json"))
    cfg = load_config()
    for seed in range(25):
        env = VacuumEnv(hm, cfg, seed=seed)
        env.reset()
        x, y, _theta = env.pose
        assert hm.is_inside(x, y), f"{map_name} seed={seed} spawned outside polygon at ({x:.2f}, {y:.2f})"


def test_spawn_falls_back_to_free_cell_when_random_search_exhausts(monkeypatch):
    # Force every random sample to be rejected so _spawn falls through to the
    # guaranteed-interior free-cell scan (vacuum_env.py fallback branch).
    env = VacuumEnv(HMAP, CFG, seed=3)  # free mask built here (bbox -> all free)
    monkeypatch.setattr(env.house_map, "is_inside", lambda _x, _y: False)
    env.reset()
    x, y, theta = env.pose
    assert theta == 0.0  # the fallback returns heading 0.0 (random path returns a random angle)
    # the chosen point is a collision-free interior cell centre, not the bbox centre
    assert 0.17 <= x <= 3.83 and 0.17 <= y <= 3.83
