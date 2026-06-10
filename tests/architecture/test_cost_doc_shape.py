"""Doc contract (spec §2/§11, TODO T04-03): COST_ANALYSIS.md has the tiktoken
headline, chars/bytes appendix, training-runtime + compute envelope, and a
named architect-decided spend cap.
"""

from __future__ import annotations

from pathlib import Path

_DOC = Path(__file__).resolve().parent.parent.parent / "docs" / "COST_ANALYSIS.md"


def test_cost_doc_sections_present() -> None:
    text = _DOC.read_text(encoding="utf-8")
    for header in (
        "## 1. Headline — tiktoken (cl100k_base)",
        "## 2. Appendix — chars & bytes",
        "## 3. AI-tooling cost",
        "## 4. Training runtime & compute envelope",
        "## 5. Cost envelope — architect spend cap vs running total",
    ):
        assert header in text, f"missing header: {header}"
    assert "src/cost/meter.py" in text
    assert "PENDING" in text
