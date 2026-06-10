"""Render results/figures/critic_loss.png: critic loss vs training step
(per-episode mean +/- 95% CI over the training seeds).

uv run python scripts/render_critic_loss.py
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np

from scripts.render_learning_curve import mean_ci

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HISTORY_DIR = "results/history"
OUT_PNG = "results/figures/critic_loss.png"


def render(history_dir: str = HISTORY_DIR, out_png: str = OUT_PNG) -> str:
    episodes, mean, ci = mean_ci(history_dir, "critic_loss")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    n_seeds = len(sorted(Path(history_dir).glob("seed_*.json")))
    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    ax.plot(episodes, mean, color="#d62728", label="mean critic loss")
    ax.fill_between(
        episodes,
        np.maximum(mean - ci, 0.0),
        mean + ci,
        color="#d62728",
        alpha=0.25,
        label="95% CI",
    )
    ax.set_xlabel("episode (training step proxy)")
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
