"""Doc contract (spec §2 §10, submission guidelines §10): UX.md records the
N/A verdict — the surface is CLI + static figures, there is no GUI to evaluate.
"""

from __future__ import annotations

from pathlib import Path

_DOC = Path(__file__).resolve().parent.parent.parent / "docs" / "UX.md"


def test_ux_doc_records_na_verdict() -> None:
    text = _DOC.read_text(encoding="utf-8")
    assert "## 1. Verdict — §10 Not Applicable (no GUI)" in text
    assert "## 2. The interaction surface (CLI + static figures)" in text
    assert "## 3. What a GUI would have added (and why it is out of scope)" in text
    assert "RoboVacuumSDK" in text
    assert "results/figures/" in text
