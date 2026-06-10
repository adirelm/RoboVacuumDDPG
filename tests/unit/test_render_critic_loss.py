import json

from scripts import render_critic_loss as rcl


def _write_synthetic(history_dir, seeds):
    history_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        records = [
            {
                "episode": e,
                "reward": 1.0,
                "critic_loss": float(seed) / (e + 1),
                "coverage": 0.5,
                "steps": 10,
            }
            for e in range(4)
        ]
        (history_dir / f"seed_{seed}.json").write_text(json.dumps(records), encoding="utf-8")


def test_render_writes_png_over_5kb(tmp_path):
    _write_synthetic(tmp_path / "history", [42, 7])
    out = tmp_path / "figures" / "critic_loss.png"
    rcl.render(str(tmp_path / "history"), str(out))
    assert out.exists()
    assert out.stat().st_size > 5000
