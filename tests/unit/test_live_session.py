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


def test_reset_re_emits_spawn_footprint(tiny_map):
    # A manual reset must behave like the internal done-driven reset: the next
    # frame re-surfaces the spawn footprint as new_cells (regression guard for
    # _fresh being set in reset()).
    s = _session("play", tiny_map)
    s.step()  # consume the initial spawn-prime frame
    s.reset()
    assert s.step().new_cells, "reset() should re-prime new_cells with the spawn footprint"


def test_done_drives_episode_boundary_and_respawn(tiny_map):
    # Shrink max_steps to 1 so step() returns done=True: the live session must
    # then reset, bump the episode counter, and re-prime _fresh so the NEXT frame
    # re-emits the spawn footprint as new_cells (live_session.py:115-118).
    cfg = load_config()
    cfg["env"]["max_steps"] = 1
    cfg["env"]["coverage_target"] = 1.0  # isolate the max_steps trigger
    s = LiveSession(cfg, tiny_map, "play", seed=0)
    first = s.step()  # step_count -> 1 >= max_steps -> done, then internal reset
    assert first.done is True and first.episode == 0
    second = s.step()  # post-reset frame in the new episode
    assert second.episode == 1, "episode counter must increment after a done step"
    assert second.new_cells, "the new episode's spawn footprint must be re-emitted"
