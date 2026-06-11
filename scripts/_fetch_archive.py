"""Download + extract the real HouseExpo JSON archive (helper for fetch_houseexpo.py).

The upstream repo ships every map as ``HouseExpo/json.tar.gz`` (a standard gzip
tar — extractable with Python's stdlib ``tarfile``, NO ``7z`` required). We fetch
that single ~25 MB blob from raw.githubusercontent.com at the pinned SHA (config-
driven URL) instead of cloning all 35 k files. Targets are git-ignored.
"""

from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

_RAW = "https://raw.githubusercontent.com"


def archive_url(repo_url: str, sha: str, archive_path: str) -> str:
    """Build the raw.githubusercontent.com URL for the pinned archive blob."""
    owner_repo = repo_url.rstrip("/").removeprefix("https://github.com/")
    return f"{_RAW}/{owner_repo}/{sha}/{archive_path}"


def download(url: str, dest: str) -> int:
    """Download ``url`` to ``dest`` (skips if already present); return byte size."""
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if not dest_path.exists():
        with urllib.request.urlopen(url, timeout=180) as resp:  # pinned https blob only
            dest_path.write_bytes(resp.read())
    return dest_path.stat().st_size


def extract_json(archive: str, out_dir: str) -> int:
    """Extract ``*.json`` members of the tar.gz into ``out_dir``; return file count.

    Members are flattened (the ``json/`` prefix is stripped) so files land as
    ``out_dir/<id>.json``. Path traversal is rejected (only basenames are kept).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if not (member.isfile() and member.name.endswith(".json")):
                continue
            fh = tar.extractfile(member)
            if fh is None:
                continue
            (out / Path(member.name).name).write_bytes(fh.read())
            count += 1
    return count
