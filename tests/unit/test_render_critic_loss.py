import json
import math

from scripts import render_critic_loss as rcl


def test_episode_series_truncates_to_shortest_seed_with_hand_computed_mean(tmp_path):
    # Seed A: 4 episodes of critic_loss [10, 20, 30, 40]; seed B: 2 episodes
    # [30, 40]. episode_series must truncate to the shortest run (2 episodes)
    # and average per episode: [(10+30)/2, (20+40)/2] = [20, 30].
    history_dir = tmp_path / "history"
    history_dir.mkdir(parents=True)
    for seed, losses in ((1, [10.0, 20.0, 30.0, 40.0]), (2, [30.0, 40.0])):
        records = [
            {"episode": e, "reward": 1.0, "critic_loss": v, "coverage": 0.5, "steps": 10}
            for e, v in enumerate(losses)
        ]
        (history_dir / f"seed_{seed}.json").write_text(json.dumps(records), encoding="utf-8")
    episodes, mean, ci = rcl.episode_series(str(history_dir))
    assert list(episodes) == [0, 1]
    assert list(mean) == [20.0, 30.0]
    # 95% CI with ddof=1 over {10,30}: std=14.142..., sem=10, ci=19.6
    assert math.isclose(float(ci[0]), 19.6, rel_tol=1e-9)


def _write_synthetic(history_dir, seeds):
    history_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        records = [
            {
                "episode": e,
                "reward": 1.0,
                "critic_loss": float(seed) / (e + 1),
                # F29: step-level critic loss per gradient update (3 updates/episode)
                "critic_losses": [float(seed) / (e + 1)] * 3,
                "coverage": 0.5,
                "steps": 10,
            }
            for e in range(4)
        ]
        (history_dir / f"seed_{seed}.json").write_text(json.dumps(records), encoding="utf-8")


def test_step_series_flattens_per_seed_critic_losses(tmp_path):
    _write_synthetic(tmp_path / "history", [42, 7])
    steps, mean, ci = rcl.step_series(str(tmp_path / "history"))
    # 4 episodes * 3 updates each => 12 gradient-update steps on the x-axis.
    assert list(steps) == list(range(12))
    assert len(mean) == 12 and len(ci) == 12


def test_step_series_falls_back_to_per_episode_mean_when_absent(tmp_path):
    history_dir = tmp_path / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    for seed in (42, 7):
        records = [
            {"episode": e, "reward": 1.0, "critic_loss": 0.1, "coverage": 0.5, "steps": 10} for e in range(4)
        ]
        (history_dir / f"seed_{seed}.json").write_text(json.dumps(records), encoding="utf-8")
    steps, mean, _ci = rcl.step_series(str(history_dir))
    # No critic_losses lists -> fall back to one point per episode (4).
    assert list(steps) == [0, 1, 2, 3]
    assert len(mean) == 4


def test_render_writes_png_over_5kb(tmp_path):
    _write_synthetic(tmp_path / "history", [42, 7])
    out = tmp_path / "figures" / "critic_loss.png"
    rcl.render(str(tmp_path / "history"), str(out))
    assert out.exists()
    assert out.stat().st_size > 5000
