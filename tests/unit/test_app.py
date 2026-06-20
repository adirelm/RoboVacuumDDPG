import yaml

from src.gui.app import App
from src.utils.config_loader import load_config


def _app():
    return App(["room_single"], seed=42, checkpoint_path=None)


def _short_episode_config(tmp_path):
    """Write a config copy that forces an episode to end after one step."""
    cfg = load_config()
    cfg["env"]["max_steps"] = 1
    cfg["env"]["coverage_target"] = 1.0  # isolate the max_steps episode trigger
    path = tmp_path / "short_config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(path)


def test_app_update_then_draw_runs(gui_surface):
    app = _app()
    app.update()
    app.draw(gui_surface)  # composes env_view + curve + hud without error


def test_mode_switch_rebuilds_session(gui_surface):
    app = _app()
    assert app.mode == "train"
    app.handle_command("mode_play")
    assert app.mode == "play"
    app.update()
    app.draw(gui_surface)


def test_toggle_flags(gui_surface):
    app = _app()
    assert app.show_lidar is True and app.show_coverage is True
    app.handle_command("toggle_lidar")
    app.handle_command("toggle_coverage")
    assert app.show_lidar is False and app.show_coverage is False


def test_pause_stops_advancement(gui_surface):
    app = _app()
    app.handle_command("pause")
    assert app.paused is True
    before = app.last_frame.step
    app.update()  # paused -> no advancement
    assert app.last_frame.step == before


def test_speed_clamps_within_bounds(gui_surface):
    app = _app()
    for _ in range(60):
        app.handle_command("speed_up")
    assert app.speed == 50
    for _ in range(60):
        app.handle_command("speed_down")
    assert app.speed == 1


def test_cycle_seed_increments(gui_surface):
    app = _app()
    s0 = app.seed
    app.handle_command("cycle_seed")
    assert app.seed == s0 + 1


def test_reset_rebuilds_session(gui_surface):
    app = _app()
    app.update()
    app.handle_command("reset")  # fresh session: trail/covered cleared, one step taken
    assert app.last_frame.step == 1
    app.draw(gui_surface)


def test_cycle_map_wraps(gui_surface):
    app = App(["room_single", "apt_small"], seed=1, checkpoint_path=None)
    assert app.map_idx == 0
    app.handle_command("cycle_map")
    assert app.map_idx == 1
    app.handle_command("cycle_map")
    assert app.map_idx == 0  # wraps


def test_completed_episode_records_reward_and_resets(gui_surface, tmp_path):
    # With max_steps=1 every step ends the episode, so _absorb hits the frame.done
    # branch (app.py:59-63): the per-episode reward is banked into episode_rewards
    # and the live trail/coverage are cleared for the next episode.
    app = App(["room_single"], seed=42, checkpoint_path=None, config_path=_short_episode_config(tmp_path))
    # _new_session already took one (done) step during construction.
    assert app.episode_rewards, "a completed episode must append its reward"
    assert app.covered == [] and app.trail == [], "trail/coverage must reset after a done step"
    n = len(app.episode_rewards)
    app.update()  # advances `speed` more one-step episodes
    assert len(app.episode_rewards) > n, "each completed episode appends another reward"
