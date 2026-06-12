import json

import numpy as np

from scripts import render_learning_curve as rlc


def test_rolling_mean_hand_computed():
    # trailing window w=2 over [1,2,3,4]: expanding head [1.0, 1.5], then
    # (2+3)/2=2.5, (3+4)/2=3.5 — locks the cumsum offsets (off-by-one guard).
    out = rlc.rolling(np.array([1.0, 2.0, 3.0, 4.0]), 2)
    assert list(out) == [1.0, 1.5, 2.5, 3.5]


def test_rolling_shorter_than_window_is_expanding_mean():
    out = rlc.rolling(np.array([5.0, 7.0]), 10)
    assert list(out) == [5.0, 6.0]


def _write_synthetic(history_dir, seeds):
    history_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        records = [
            {
                "episode": e,
                "reward": float(e + seed),
                "critic_loss": 0.1,
                "coverage": 0.5,
                "steps": 10,
            }
            for e in range(3)
        ]
        (history_dir / f"seed_{seed}.json").write_text(json.dumps(records), encoding="utf-8")


def test_mean_ci_shapes_match_episode_count(tmp_path):
    _write_synthetic(tmp_path / "history", [42, 7])
    episodes, mean, ci = rlc.mean_ci(str(tmp_path / "history"), "reward")
    assert list(episodes) == [0, 1, 2]
    assert len(mean) == 3 and len(ci) == 3
    # episode 0: rewards {42, 7} -> mean 24.5
    assert abs(float(mean[0]) - 24.5) < 1e-9


def test_render_writes_png_over_5kb(tmp_path):
    _write_synthetic(tmp_path / "history", [42, 7, 123])
    out = tmp_path / "figures" / "learning_curve.png"
    rlc.render(str(tmp_path / "history"), str(out))
    assert out.exists()
    assert out.stat().st_size > 5000
