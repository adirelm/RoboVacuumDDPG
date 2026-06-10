import json

from scripts import render_learning_curve as rlc


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
