"""Render a rollout trajectory over the 2D map: walls (black), path (colour),
and a shaded covered-area footprint, proving wall-avoidance + coverage.

uv run python scripts/render_trajectory.py
"""

import sys
from pathlib import Path

import matplotlib

from src.sdk.sdk import RoboVacuumSDK

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_PNG = "results/figures/trajectory.png"


def render(
    walls: list[tuple[float, float, float, float]],
    path: list[tuple[float, float]],
    out_png: str = OUT_PNG,
    clean_radius: float = 0.17,
) -> str:
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    for x1, y1, x2, y2 in walls:
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=2.0)
    for x, y in path:
        ax.add_patch(plt.Circle((x, y), clean_radius, color="#2ca02c", alpha=0.18))
    if path:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, color="#1f77b4", linewidth=1.5, label="robot path")
        ax.scatter([xs[0]], [ys[0]], color="green", zorder=5, label="start")
        ax.scatter([xs[-1]], [ys[-1]], color="red", zorder=5, label="end")
        ax.legend(loc="upper right")
    ax.set_aspect("equal")
    ax.set_title("Rollout trajectory + covered area")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


def main() -> int:
    sdk = RoboVacuumSDK()
    name = sdk.cfg["maps"]["train"][0]
    seed = sdk.cfg["training"]["seeds"][0]
    ckpt = Path("results/checkpoints") / f"seed_{seed}.pt"
    checkpoint = str(ckpt) if ckpt.exists() else None
    path = sdk.trajectory(name, checkpoint_path=checkpoint, seed=seed)
    walls = sdk.map_walls(name)
    print(render(walls, path, clean_radius=sdk.cfg["env"]["clean_radius"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
