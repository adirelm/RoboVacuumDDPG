"""Render results/figures/sensitivity_n_rays.png from results/sweep_n_rays.json.

Final-window reward (bars, left axis) and coverage (line, right axis) vs lidar
resolution n_rays in {8, 16, 24} — the V3 §9.1 sensitivity figure. Reads the
committed sweep JSON only; the heavy per-episode histories are never touched.

uv run python scripts/render_sensitivity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SWEEP_JSON = "results/sweep_n_rays.json"
OUT_PNG = "results/figures/sensitivity_n_rays.png"
_META_KEYS = {"episodes", "seed", "map", "window"}


def load_series(sweep_json: str = SWEEP_JSON) -> tuple[list[int], list[float], list[float], dict]:
    """Return (rays, rewards, coverages, meta) with the ray axis ascending."""
    data = json.loads(Path(sweep_json).read_text(encoding="utf-8"))
    meta = {k: data[k] for k in _META_KEYS if k in data}
    rays = sorted(int(k) for k in data if k not in _META_KEYS)
    rewards = [float(data[str(r)]["final_reward"]) for r in rays]
    coverages = [float(data[str(r)]["final_coverage"]) for r in rays]
    return rays, rewards, coverages, meta


def render(sweep_json: str = SWEEP_JSON, out_png: str = OUT_PNG) -> str:
    rays, rewards, coverages, meta = load_series(sweep_json)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    x = list(range(len(rays)))
    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    bars = ax.bar(x, rewards, width=0.5, color="#1f77b4", alpha=0.8, label="final-window reward")
    ax.set_xlabel("lidar resolution n_rays (state_dim = n_rays + 4)")
    ax.set_ylabel("final-window mean reward", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in rays])
    for rect, rwd in zip(bars, rewards, strict=True):
        cx = rect.get_x() + rect.get_width() / 2
        ax.text(cx, rect.get_height(), f"{rwd:.0f}", ha="center", va="bottom")

    ax2 = ax.twinx()
    (line,) = ax2.plot(x, coverages, color="#d62728", marker="o", label="final-window coverage")
    ax2.set_ylabel("final-window mean coverage", color="#d62728")
    ax2.set_ylim(0, max(coverages) * 1.4 if coverages else 1.0)

    episodes = meta.get("episodes", "?")
    seed = meta.get("seed", "?")
    map_name = meta.get("map", "?")
    ax.set_title(f"n_rays sensitivity (reduced budget: {episodes} eps, seed {seed}, {map_name})")
    ax.legend(handles=[bars, line], loc="upper left")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


def main() -> int:
    print(render())
    return 0


if __name__ == "__main__":
    sys.exit(main())
