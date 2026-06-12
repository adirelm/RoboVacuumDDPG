"""Doc contract (CLAUDE.md §1.4, TODO T04-06): PROMPTS.md is the verbatim
architect -> implementer prompt log, each mapped to a commit hash with a
human-judgment annotation — and every cited hash must RESOLVE in the current
history (a history rewrite once left all of them dangling; never again).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
_DOC = _REPO / "docs" / "shared" / "PROMPTS.md"


def test_prompts_doc_skeleton() -> None:
    text = _DOC.read_text(encoding="utf-8")
    assert "Human ↔ AI Responsibility Contract" in text
    assert "| Prompt | Commit | Human-judgment annotation |" in text
    for phase in ("Phase 0", "Phase 1", "Phase 2", "Phase 3", "Phase 4"):
        assert phase in text, f"missing phase row group: {phase}"
    # Finalized: every phase's Commit cell carries real SHAs, no leftover stubs.
    assert "PENDING" not in text, "PROMPTS.md still has a PENDING commit stub"


def test_prompts_doc_shas_resolve_in_history() -> None:
    """Every backticked short SHA in PROMPTS.md must exist in this repo's history."""
    if not (_REPO / ".git").exists():
        pytest.skip("not a git checkout (e.g. source archive)")
    shas = re.findall(r"`([0-9a-f]{7,40})`", _DOC.read_text(encoding="utf-8"))
    assert len(shas) >= 20, "PROMPTS.md should cite the per-phase commit SHAs"
    for sha in shas:
        ok = subprocess.run(
            ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
            cwd=_REPO,
            capture_output=True,
            check=False,
        )
        assert ok.returncode == 0, f"PROMPTS.md cites dangling commit SHA {sha}"
