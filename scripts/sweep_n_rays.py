"""Reduced lidar-resolution (n_rays) sensitivity sweep (V3 §9.1).

For each n_rays in {8, 16, 24}: write a temp config (copy of config/config.yaml
with env.n_rays overridden and a REDUCED budget — fewer episodes, single seed),
train via the SDK on room_single, and record the final-window mean reward +
coverage. Writes results/sweep_n_rays.json and prints a summary table.

Each n_rays changes state_dim (= n_rays + 4), so every config builds its own
actor/critic from scratch — that is exactly the axis under study. The reduced
budget (REDUCED_EPISODES, single seed 42) keeps the whole sweep to ~30-40 min;
it is a sensitivity probe, NOT the headline 5-seed / 500-episode result.

uv run python scripts/sweep_n_rays.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.sdk.sdk import RoboVacuumSDK  # noqa: E402

RAY_VALUES = (8, 16, 24)
REDUCED_EPISODES = 140  # reduced from the headline 500 to fit the ~30-40 min probe budget
SEED = 42
MAP_NAME = "room_single"
WINDOW = 20  # final-N-episode tail averaged for the reported figure
BASE_CONFIG = "config/config.yaml"
OUT_PATH = "results/sweep_n_rays.json"


def final_window_means(history: list[dict], window: int = WINDOW) -> tuple[float, float]:
    """Mean reward + coverage over the final `window` episodes (clamped to length)."""
    tail = history[-window:] if window < len(history) else history
    n = len(tail)
    reward = sum(float(r["reward"]) for r in tail) / n
    coverage = sum(float(r["coverage"]) for r in tail) / n
    return reward, coverage


def _write_temp_config(base_config: str, n_rays: int, episodes: int, seed: int, tmp_dir: str) -> str:
    """Copy the base config, override env.n_rays + the reduced budget, write to tmp_dir."""
    cfg = yaml.safe_load(Path(base_config).read_text(encoding="utf-8"))
    cfg["env"]["n_rays"] = int(n_rays)
    cfg["training"]["episodes"] = int(episodes)
    cfg["training"]["seeds"] = [int(seed)]
    out = Path(tmp_dir) / f"config_n_rays_{n_rays}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return str(out)


def run_sweep(  # noqa: PLR0913
    base_config: str = BASE_CONFIG,
    ray_values: tuple[int, ...] = RAY_VALUES,
    episodes: int = REDUCED_EPISODES,
    seed: int = SEED,
    map_name: str = MAP_NAME,
    window: int = WINDOW,
    out_path: str = OUT_PATH,
    tmp_dir: str = "results/_sweep_tmp",
) -> dict:
    """Train one DDPG agent per n_rays and aggregate final-window reward/coverage."""
    result: dict = {"episodes": episodes, "seed": seed, "map": map_name, "window": window}
    for n_rays in ray_values:
        tmp_cfg = _write_temp_config(base_config, n_rays, episodes, seed, tmp_dir)
        history = RoboVacuumSDK(tmp_cfg).train(seed=seed, map_name=map_name)
        reward, coverage = final_window_means(history, window)
        result[str(n_rays)] = {
            "n_rays": n_rays,
            "state_dim": n_rays + 4,
            "final_reward": reward,
            "final_coverage": coverage,
        }
        print(f"n_rays={n_rays:>2d} (dim={n_rays + 4}) -> reward={reward:8.1f}  coverage={coverage:.4f}")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _print_table(result: dict, ray_values: tuple[int, ...] = RAY_VALUES) -> None:
    print(
        "\n=== n_rays sensitivity (final-{} mean, {} episodes, seed {}, map {}) ===".format(
            result["window"], result["episodes"], result["seed"], result["map"]
        )
    )
    print(f"{'n_rays':>7} {'state_dim':>10} {'reward':>10} {'coverage':>10}")
    for n_rays in ray_values:
        row = result[str(n_rays)]
        rwd, cov = row["final_reward"], row["final_coverage"]
        print(f"{n_rays:>7} {row['state_dim']:>10} {rwd:>10.1f} {cov:>10.4f}")


def main() -> int:
    result = run_sweep()
    _print_table(result)
    print(f"\nsaved -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
