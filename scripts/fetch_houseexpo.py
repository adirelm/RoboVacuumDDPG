"""Clone HouseExpo at a pinned SHA into git-ignored data/houseexpo_full/,
copy the curated subset into data/maps/, and stamp the SHA into config.

uv run python scripts/fetch_houseexpo.py
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

from src.sdk.sdk import RoboVacuumSDK

FULL_DIR = "data/houseexpo_full"
SENTINEL = "PINNED_AT_FETCH"


def curated_names(cfg: dict) -> list[str]:
    return list(cfg["maps"]["train"]) + list(cfg["maps"]["holdout"])


def clone_repo(repo_url: str, sha: str, dest: str) -> str:
    dest_path = Path(dest)
    if not dest_path.exists():
        subprocess.run(["git", "clone", repo_url, dest], check=True)
    if sha != SENTINEL:
        subprocess.run(["git", "-C", dest, "checkout", sha], check=True)
    resolved = subprocess.run(
        ["git", "-C", dest, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return resolved.stdout.strip()


def copy_curated(json_dir: str, maps_dir: str, names: list[str]) -> int:
    Path(maps_dir).mkdir(parents=True, exist_ok=True)
    copied = 0
    for name in names:
        src_file = Path(json_dir) / f"{name}.json"
        if src_file.exists():
            shutil.copy2(src_file, Path(maps_dir) / f"{name}.json")
            copied += 1
    return copied


def stamp_sha(config_path: str, sha: str) -> None:
    text = Path(config_path).read_text(encoding="utf-8")
    new = re.sub(r'(dataset_sha:\s*")[^"]*(")', rf"\g<1>{sha}\g<2>", text, count=1)
    Path(config_path).write_text(new, encoding="utf-8")


def main() -> int:
    sdk = RoboVacuumSDK()
    cfg = sdk.cfg
    sha = clone_repo(cfg["maps"]["dataset_repo"], cfg["maps"]["dataset_sha"], FULL_DIR)
    copied = copy_curated(f"{FULL_DIR}/json", cfg["paths"]["maps_dir"], curated_names(cfg))
    stamp_sha(sdk.config_path or "config/config.yaml", sha)
    print(f"fetched SHA={sha} curated={copied} into {cfg['paths']['maps_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
