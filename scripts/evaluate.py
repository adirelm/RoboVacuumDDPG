"""Held-out generalization eval: greedy coverage on config.maps.holdout maps.

Loads the trained checkpoint (F6/F30) when present so eval uses the trained
policy; falls back to a fresh agent otherwise. Imports only RoboVacuumSDK.

uv run python scripts/evaluate.py
"""

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.sdk.sdk import RoboVacuumSDK  # noqa: E402

OUT_JSON = "results/holdout_eval.json"


def evaluate_holdout(seed: int = 0) -> dict:
    sdk = RoboVacuumSDK()
    ckpt = Path("results/checkpoints") / f"seed_{seed}.pt"
    checkpoint = str(ckpt) if ckpt.exists() else None
    results: dict[str, dict] = {}
    for name in sdk.cfg["maps"]["holdout"]:
        results[name] = sdk.evaluate(checkpoint, name, seed=seed)
    return results


def main() -> int:
    results = evaluate_holdout()
    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT_JSON).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
