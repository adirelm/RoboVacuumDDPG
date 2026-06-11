"""Final-gate contract (CLAUDE.md §1/§6, spec §2, TODO T04-08/09): the
≤150-LOC guard runs green and, when present locally, the official cover sheet
adrl-001-ex05.pdf is a non-empty PDF. The coverage/ruff/format gates are
asserted by the Step-4 command sweep.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

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


def test_submission_cover_sheet_nonempty_when_present() -> None:
    """The Moodle cover sheet (adrl-001-ex05.pdf) carries PII (ID + name +
    self-grade) and is git-ignored — never committed, so it is absent on a
    fresh CI clone. We SKIP (not fail) when absent: shipping it is a
    submission-step concern, not a code gate. When it *is* present locally,
    validate it is a non-empty PDF before upload.
    """
    pdf = _REPO_ROOT / "adrl-001-ex05.pdf"
    if not pdf.exists():
        pytest.skip("cover sheet is git-ignored (Moodle-only PII); absent on CI / fresh clone")
    assert pdf.stat().st_size > 0, "cover sheet is empty"
