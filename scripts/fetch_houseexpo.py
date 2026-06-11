"""Fetch the REAL HouseExpo dataset at a pinned SHA into git-ignored data/, copy
the curated subset into data/maps/, and stamp the SHA into config.

Downloads the upstream ``HouseExpo/json.tar.gz`` (~25 MB, all 35 126 real maps)
and extracts it with Python's stdlib ``tarfile`` — NO ``7z`` needed. If you
prefer the repo's own instructions: ``git clone github.com/TeaganLi/HouseExpo &&
cd HouseExpo && tar -xvzf json.tar.gz`` yields the same ``json/<id>.json`` files.

    uv run python scripts/fetch_houseexpo.py
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

# Run as a file (`uv run python scripts/fetch_houseexpo.py`) puts scripts/ on
# sys.path[0]; add the repo root so both `scripts.*` and `src.*` resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._fetch_archive import archive_url, download, extract_json
from src.sdk.sdk import RoboVacuumSDK

FULL_DIR = "data/houseexpo_full"
ARCHIVE = f"{FULL_DIR}/json.tar.gz"
JSON_DIR = f"{FULL_DIR}/json"
SENTINEL = "PINNED_AT_FETCH"


def curated_names(cfg: dict) -> list[str]:
    return list(cfg["maps"]["train"]) + list(cfg["maps"]["holdout"])


def copy_curated(json_dir: str, maps_dir: str, names: list[str]) -> int:
    """Copy ``<name>.json`` files (already logical-named) into maps_dir; idempotent."""
    Path(maps_dir).mkdir(parents=True, exist_ok=True)
    copied = 0
    for name in names:
        src_file = Path(json_dir) / f"{name}.json"
        if src_file.exists():
            shutil.copy2(src_file, Path(maps_dir) / f"{name}.json")
            copied += 1
    return copied


def copy_curated_by_id(json_dir: str, maps_dir: str, id_map: dict[str, str]) -> int:
    """Copy real ``<id>.json`` files into ``maps_dir/<logical-name>.json``."""
    Path(maps_dir).mkdir(parents=True, exist_ok=True)
    copied = 0
    for name, map_id in id_map.items():
        src_file = Path(json_dir) / f"{map_id}.json"
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
    maps_cfg = cfg["maps"]
    sha = maps_cfg["dataset_sha"]
    if sha == SENTINEL:
        print(f"ERROR: pin a real SHA in config.maps.dataset_sha (not {SENTINEL!r}).")
        return 1
    url = archive_url(maps_cfg["dataset_repo"], sha, maps_cfg["archive_path"])
    size = download(url, ARCHIVE)
    n_extracted = extract_json(ARCHIVE, JSON_DIR)
    maps_dir = cfg["paths"]["maps_dir"]
    id_map = maps_cfg.get("curated_ids") or {}
    copied = copy_curated_by_id(JSON_DIR, maps_dir, id_map) if id_map else 0
    stamp_sha(sdk.config_path or "config/config.yaml", sha)
    print(f"fetched real HouseExpo @ {sha}")
    print(f"  archive: {ARCHIVE} ({size / 1e6:.1f} MB)")
    print(f"  extracted: {n_extracted} real JSON maps -> {JSON_DIR}/")
    print(f"  curated: {copied}/{len(id_map)} maps -> {maps_dir}/ ({', '.join(curated_names(cfg))})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
