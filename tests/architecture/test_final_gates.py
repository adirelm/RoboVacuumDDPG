"""Final-gate contract (CLAUDE.md §1/§6, spec §2, TODO T04-08/09): the
≤150-LOC guard runs green and the official cover sheet adrl-001-ex05.pdf exists.
The coverage/ruff/format gates are asserted by the Step-4 command sweep.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_file_size_guard_passes() -> None:
    script = _REPO_ROOT / "scripts" / "check_file_sizes.py"
    assert script.exists(), "scripts/check_file_sizes.py missing"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"file-size guard failed:\n{result.stdout}\n{result.stderr}"


def test_submission_cover_sheet_exists() -> None:
    pdf = _REPO_ROOT / "adrl-001-ex05.pdf"
    assert pdf.exists(), "adrl-001-ex05.pdf (Moodle cover sheet) missing"
    assert pdf.stat().st_size > 0, "cover sheet is empty"
