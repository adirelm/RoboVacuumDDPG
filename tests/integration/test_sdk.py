from src.ddpg.agent import DDPGAgent
from src.env.vacuum_env import VacuumEnv
from src.sdk.sdk import RoboVacuumSDK


def test_build_env_returns_vacuum_env_for_train_map():
    sdk = RoboVacuumSDK()
    env = sdk.build_env(map_name="room_single", seed=42)
    assert isinstance(env, VacuumEnv)
    assert env.state_dim == 20  # n_rays=16 + 4
    assert env.action_dim == 2


def test_rollout_returns_list_of_xy_pairs():
    sdk = RoboVacuumSDK()
    env = sdk.build_env(map_name="room_single", seed=42)
    agent = DDPGAgent(env.state_dim, env.action_dim, sdk.cfg, seed=42)
    path = sdk.rollout(agent, env, max_steps=5)
    assert isinstance(path, list)
    assert len(path) == 5
    assert all(isinstance(p, tuple) and len(p) == 2 for p in path)
    assert all(isinstance(p[0], float) and isinstance(p[1], float) for p in path)


def test_coverage_report_keys():
    sdk = RoboVacuumSDK()
    env = sdk.build_env(map_name="room_single", seed=42)
    agent = DDPGAgent(env.state_dim, env.action_dim, sdk.cfg, seed=42)
    report = sdk.coverage_report(agent, env)
    assert set(report) == {"coverage", "steps", "collisions"}
    assert 0.0 <= report["coverage"] <= 1.0
    assert isinstance(report["collisions"], int)


def test_map_walls_returns_segments():
    sdk = RoboVacuumSDK()
    walls = sdk.map_walls("room_single")
    assert isinstance(walls, list) and len(walls) >= 1
    assert all(len(seg) == 4 for seg in walls)


def test_evaluate_loads_checkpoint_and_reports(tmp_path):
    sdk = RoboVacuumSDK()
    env = sdk.build_env(map_name="room_single", seed=1)
    agent = DDPGAgent(env.state_dim, env.action_dim, sdk.cfg, seed=1)
    ckpt = tmp_path / "agent.pt"
    agent.save(str(ckpt))
    report = sdk.evaluate(str(ckpt), map_name="room_single", seed=1)
    assert set(report) == {"coverage", "steps", "collisions"}
    assert 0.0 <= report["coverage"] <= 1.0


def test_trajectory_loads_checkpoint_and_returns_path(tmp_path):
    sdk = RoboVacuumSDK()
    env = sdk.build_env(map_name="room_single", seed=2)
    agent = DDPGAgent(env.state_dim, env.action_dim, sdk.cfg, seed=2)
    ckpt = tmp_path / "agent.pt"
    agent.save(str(ckpt))
    path = sdk.trajectory("room_single", checkpoint_path=str(ckpt), seed=2)
    assert isinstance(path, list) and len(path) >= 1
    assert all(len(p) == 2 for p in path)


def test_coverage_grid_returns_renderable_dict():
    sdk = RoboVacuumSDK()
    grid = sdk.coverage_grid("room_single", seed=42, max_steps=5)
    assert set(grid) >= {"cleaned", "free", "extent", "walls", "coverage"}
    assert len(grid["extent"]) == 4
    # cleaned/free are nx-by-ny boolean grids of identical shape.
    assert len(grid["cleaned"]) == len(grid["free"]) >= 1
    assert len(grid["cleaned"][0]) == len(grid["free"][0]) >= 1
    assert 0.0 <= grid["coverage"] <= 1.0
    assert all(len(seg) == 4 for seg in grid["walls"])
