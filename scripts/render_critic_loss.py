"""Render results/figures/critic_loss.png: critic MSE loss vs gradient-update
step (mean +/- 95% CI over the training seeds).

Plots the STEP-level critic_losses list (contract F29, spec section 7) that
Trainer.train records per gradient update, flattened across episodes. Falls
back to the per-episode mean critic_loss when older histories lack the list.

uv run python scripts/render_critic_loss.py
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HISTORY_DIR = "results/history"
OUT_PNG = "results/figures/critic_loss.png"


def _seed_step_losses(records: list[dict]) -> list[float]:
    """Flatten one seed's per-episode critic_losses into a step-level series.

    Falls back to the per-episode mean critic_loss (one point per episode) when
    the step-level list is absent or empty.
    """
    has_steps = any(r.get("critic_losses") for r in records)
    if has_steps:
        return [float(v) for r in records for v in r.get("critic_losses", [])]
    return [float(r["critic_loss"]) for r in records]


def step_series(history_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (steps, mean, ci) of critic loss vs gradient-update step.

    Seeds are aligned by step index and truncated to the shortest run so the
    mean/CI are well-defined at every plotted step.
    """
    files = sorted(Path(history_dir).glob("seed_*.json"))
    rows = [_seed_step_losses(json.loads(fp.read_text(encoding="utf-8"))) for fp in files]
    n_steps = min(len(r) for r in rows)
    matrix = np.asarray([r[:n_steps] for r in rows], dtype=np.float64)
    steps = np.arange(n_steps)
    mean = matrix.mean(axis=0)
    n = matrix.shape[0]
    sem = matrix.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    ci = 1.96 * sem
    return steps, mean, ci


def render(history_dir: str = HISTORY_DIR, out_png: str = OUT_PNG) -> str:
    steps, mean, ci = step_series(history_dir)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    n_seeds = len(sorted(Path(history_dir).glob("seed_*.json")))
    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    ax.plot(steps, mean, color="#d62728", label="mean critic loss")
    ax.fill_between(
        steps,
        np.maximum(mean - ci, 0.0),
        mean + ci,
        color="#d62728",
        alpha=0.25,
        label="95% CI",
    )
    ax.set_xlabel("gradient-update step")
    ax.set_ylabel("critic MSE loss")
    ax.set_title(f"Critic loss (mean +/- 95% CI over {n_seeds} seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


def main() -> int:
    print(render())
    return 0


if __name__ == "__main__":
    sys.exit(main())
