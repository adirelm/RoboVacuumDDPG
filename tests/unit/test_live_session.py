import numpy as np
import pytest

from src.services.live_session import Frame, LiveSession
from src.utils.config_loader import load_config


def _session(mode, tiny_map, checkpoint_path=None):
    return LiveSession(load_config(), tiny_map, mode, seed=0, checkpoint_path=checkpoint_path)


def test_invalid_mode_raises(tiny_map):
    with pytest.raises(ValueError, match="mode must be"):
        LiveSession(load_config(), tiny_map, "fly", seed=0)


def test_meta_has_render_keys(tiny_map):
    meta = _session("play", tiny_map).meta
    assert set(meta) >= {"bounds", "walls", "cell_size", "clean_radius", "robot_radius", "n_rays", "ray_max"}
    assert meta["bounds"] == tiny_map.bounds


def test_play_step_returns_wellformed_frame(tiny_map):
    f = _session("play", tiny_map).step()
    assert isinstance(f, Frame)
    assert f.mode == "play"
    assert len(f.lidar_endpoints) == load_config()["env"]["n_rays"]
    assert all(len(p) == 2 for p in f.lidar_endpoints)
    assert isinstance(f.new_cells, list)
    assert 0.0 <= f.coverage <= 1.0
    assert len(f.pose) == 3
    assert isinstance(f.done, bool)


def test_train_step_advances_buffer_and_step_counter(tiny_map):
    s = _session("train", tiny_map)
    f1 = s.step()
    f2 = s.step()
    assert f2.step == 2 and f1.step == 1
    assert f2.buffer_size >= f1.buffer_size >= 1


def test_drive_uses_supplied_action(tiny_map):
    s = _session("drive", tiny_map)
    start = s.env.pose
    f = s.step(np.array([1.0, 0.0], dtype=np.float32))  # full throttle
    assert s.env.pose != start or f.collision  # it moved (unless it hit a wall)


def test_play_without_checkpoint_falls_back(tiny_map):
    s = _session("play", tiny_map, checkpoint_path="results/checkpoints/_absent_.pt")
    assert s.has_checkpoint is False
    assert s.step().mode == "play"


def test_reset_restarts_episode(tiny_map):
    s = _session("play", tiny_map)
    s.step()
    s.reset()  # no error; fresh episode state
    assert s.step().coverage >= 0.0
