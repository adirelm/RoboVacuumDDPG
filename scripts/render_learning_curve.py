"""Render results/figures/learning_curve.png: cumulative reward vs episode,
mean +/- 95% CI over the training seeds.

uv run python scripts/render_learning_curve.py
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HISTORY_DIR = "results/history"
OUT_PNG = "results/figures/learning_curve.png"


def _load_matrix(history_dir: str, key: str) -> np.ndarray:
    files = sorted(Path(history_dir).glob("seed_*.json"))
    rows = []
    for fp in files:
        records = json.loads(fp.read_text(encoding="utf-8"))
        rows.append([float(r[key]) for r in records])
    return np.asarray(rows, dtype=np.float64)


def mean_ci(history_dir: str, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = _load_matrix(history_dir, key)
    episodes = np.arange(matrix.shape[1])
    mean = matrix.mean(axis=0)
    n = matrix.shape[0]
    sem = matrix.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    ci = 1.96 * sem
    return episodes, mean, ci


def render(history_dir: str = HISTORY_DIR, out_png: str = OUT_PNG) -> str:
    episodes, mean, ci = mean_ci(history_dir, "reward")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    n_seeds = len(sorted(Path(history_dir).glob("seed_*.json")))
    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    ax.plot(episodes, mean, color="#1f77b4", label="mean reward")
    ax.fill_between(episodes, mean - ci, mean + ci, color="#1f77b4", alpha=0.25, label="95% CI")
    ax.set_xlabel("episode")
    ax.set_ylabel("cumulative reward")
    ax.set_title(f"Learning curve (mean +/- 95% CI over {n_seeds} seeds)")
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
