"""Multi-seed DDPG training driver.

uv run python scripts/train.py
Saves results/history/seed_<seed>.json + results/checkpoints/seed_<seed>.pt.
The checkpoint holds the trained actor/critic state_dicts (F6/F30) so the
trajectory figure + held-out eval reload the trained policy.
"""

import json
import sys
from pathlib import Path

from src.sdk.sdk import RoboVacuumSDK


def _save_history(results_dir: str, seed: int, history: list[dict]) -> None:
    out = Path(results_dir) / "history"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"seed_{seed}.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


def _checkpoint_path(results_dir: str, seed: int) -> str:
    out = Path(results_dir) / "checkpoints"
    out.mkdir(parents=True, exist_ok=True)
    return str(out / f"seed_{seed}.pt")


def run_seeds(results_dir: str = "results") -> list[int]:
    sdk = RoboVacuumSDK()
    seeds = list(sdk.cfg["training"]["seeds"])
    for seed in seeds:
        ckpt = _checkpoint_path(results_dir, seed)
        history = sdk.train(seed=seed, checkpoint_path=ckpt)
        _save_history(results_dir, seed, history)
        print(f"seed={seed} episodes={len(history)} saved -> {results_dir}")
    return seeds


def main() -> int:
    run_seeds()
    return 0


if __name__ == "__main__":
    sys.exit(main())
