"""Render results/figures/critic_loss.png: PER-EPISODE-mean critic MSE vs
episode (mean +/- 95% CI over the training seeds) — the bounded "critic loss
vs convergence time" view (final-20 tail ~12-26; spec section 7).

step_series() is retained as the auxiliary per-step view of the same signal
(the raw critic_losses list Trainer.train records per gradient update,
contract F29) — noisier, reaching ~10^3-10^4 on individual updates.

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


def episode_series(history_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (episodes, mean, ci) of the PER-EPISODE-mean critic loss.

    This is the bounded "critic loss vs convergence time" view (the 12-26 tail
    in metrics_summary.json). Plotting the per-episode mean — rather than the
    raw per-step MSE, which spikes to ~10e3-10e4 on individual updates — keeps
    the figure consistent with the Q3 "bounded, no divergence" claim.
    """
    files = sorted(Path(history_dir).glob("seed_*.json"))
    rows = [[float(r["critic_loss"]) for r in json.loads(fp.read_text(encoding="utf-8"))] for fp in files]
    n_ep = min(len(r) for r in rows)
    matrix = np.asarray([r[:n_ep] for r in rows], dtype=np.float64)
    episodes = np.arange(n_ep)
    mean = matrix.mean(axis=0)
    n = matrix.shape[0]
    sem = matrix.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    return episodes, mean, 1.96 * sem


def render(history_dir: str = HISTORY_DIR, out_png: str = OUT_PNG) -> str:
    episodes, mean, ci = episode_series(history_dir)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    n_seeds = len(sorted(Path(history_dir).glob("seed_*.json")))
    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    ax.fill_between(
        episodes, np.maximum(mean - ci, 0.0), mean + ci, color="#d62728", alpha=0.2, label="95% CI"
    )
    ax.plot(episodes, mean, color="#d62728", linewidth=1.8, label="per-episode-mean critic loss")
    ax.set_xlabel("episode (convergence time)")
    ax.set_ylabel("critic MSE loss (per-episode mean)")
    ax.set_title(f"Critic loss — per-episode mean +/- 95% CI ({n_seeds} seeds)")
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
