"""Render results/figures/coverage_heatmap.png: the cumulative cleaned-cell grid
(a 2D coverage map) over the JSON walls for the trained policy — directly
answers the brief's Q2 ("the cleaning-coverage map in the maze").

uv run python scripts/render_coverage_heatmap.py
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.sdk.sdk import RoboVacuumSDK  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_PNG = "results/figures/coverage_heatmap.png"


def heat_array(grid: dict) -> np.ndarray:
    """[ix, iy] heat values: cleaned-free = 1, free-uncleaned = 0, wall/void = NaN.

    Indexed like CoverageGrid (x along axis 0, y along axis 1); render() shows it
    as imshow(heat.T, origin="lower", extent=...) so x runs right and y runs up.
    """
    cleaned = np.asarray(grid["cleaned"], dtype=float)
    free = np.asarray(grid["free"], dtype=float)
    return np.where(free > 0, np.where(cleaned > 0, 1.0, 0.0), np.nan)


def render(grid: dict, out_png: str = OUT_PNG) -> str:
    heat = heat_array(grid)
    cmap = plt.cm.YlGn.copy()
    cmap.set_bad("white")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    # grid is indexed [ix, iy]; transpose so y is vertical with origin at lower-left.
    ax.imshow(heat.T, origin="lower", extent=grid["extent"], cmap=cmap, vmin=0.0, vmax=1.0)
    for x1, y1, x2, y2 in grid["walls"]:
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=2.0)
    ax.set_aspect("equal")
    cov = grid.get("coverage")
    title = "Coverage map — cleaned cells"
    if cov is not None:
        title += f" ({cov * 100:.0f}% of free space)"
    ax.set_title(title, fontsize=11)
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
    grid = sdk.coverage_grid(name, checkpoint_path=checkpoint, seed=seed)
    print(render(grid))
    return 0


if __name__ == "__main__":
    sys.exit(main())
