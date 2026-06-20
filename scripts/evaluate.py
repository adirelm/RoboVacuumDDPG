"""Held-out generalization eval: greedy coverage on config.maps.holdout maps,
aggregated over the trained seeds as mean ± 95% CI (ADR-008).

Loads each trained checkpoint (F6/F30) and evaluates the **greedy trained
policy**. It fails loudly if a checkpoint is missing — it NEVER silently falls
back to a fresh untrained agent (that silent fallback previously published
untrained random-policy numbers as the held-out result). Imports only RoboVacuumSDK.

uv run python scripts/evaluate.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np

_LOG = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.sdk.sdk import RoboVacuumSDK  # noqa: E402

OUT_JSON = "results/holdout_eval.json"
CKPT_DIR = Path("results/checkpoints")
_MIN_CI_SAMPLES = 2  # need ≥2 samples for a sample-standard-deviation CI


def _mean_ci(values: list[float]) -> tuple[float, float]:
    """Mean and 95% CI half-width (1.96·SEM) over the seed samples."""
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    if arr.size < _MIN_CI_SAMPLES:
        return mean, 0.0
    sem = float(arr.std(ddof=1) / np.sqrt(arr.size))
    return mean, 1.96 * sem


def evaluate_holdout(seeds: list[int] | None = None) -> dict:
    """Greedy held-out coverage per holdout map, aggregated over the trained seeds.

    Raises FileNotFoundError if a requested trained checkpoint is missing — the
    eval never silently rolls out a fresh untrained agent.
    """
    sdk = RoboVacuumSDK()
    seeds = seeds if seeds is not None else sdk.cfg["training"]["seeds"]
    results: dict[str, dict] = {}
    for name in sdk.cfg["maps"]["holdout"]:
        per_seed = []
        for seed in seeds:
            ckpt = CKPT_DIR / f"seed_{seed}.pt"
            if not ckpt.exists():
                raise FileNotFoundError(
                    f"Missing trained checkpoint {ckpt}: run scripts/train.py first. "
                    "Held-out eval must use the trained policy (F6/F30), never a fresh agent."
                )
            per_seed.append(sdk.evaluate(str(ckpt), name, seed=seed))
        cov = [r["coverage"] for r in per_seed]
        col = [float(r["collisions"]) for r in per_seed]
        mean, ci = _mean_ci(cov)
        results[name] = {
            "coverage_mean": mean,
            "coverage_ci": ci,
            "coverage_per_seed": cov,
            "collisions_mean": float(np.mean(col)),
            "steps": per_seed[0]["steps"],
            "seeds": list(seeds),
        }
    return results


def main() -> int:
    results = evaluate_holdout()
    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT_JSON).write_text(json.dumps(results, indent=2), encoding="utf-8")
    for name, rep in results.items():
        _LOG.info("held-out %s: coverage_mean=%.4f ± %.4f", name, rep["coverage_mean"], rep["coverage_ci"])
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
