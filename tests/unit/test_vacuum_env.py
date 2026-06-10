import numpy as np

from src.env.house_map import HouseMap
from src.env.vacuum_env import VacuumEnv

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
