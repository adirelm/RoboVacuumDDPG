"""Doc contract (spec §2/§11, TODO T04-03): COST_ANALYSIS.md has the
artifact-corpus-size headline, chars/bytes appendix, training-runtime + compute
envelope, and a named architect-decided spend cap. Training is complete, so the
runtime envelope is filled in (no PENDING placeholders left).
"""

from __future__ import annotations

from pathlib import Path

_DOC = Path(__file__).resolve().parent.parent.parent / "docs" / "COST_ANALYSIS.md"


def test_cost_doc_sections_present() -> None:
    text = _DOC.read_text(encoding="utf-8")
    for header in (
        "## 1. Headline — artifact corpus size (concrete)",
        "## 2. Appendix — chars & bytes",
        "## 3. AI-tooling cost",
        "## 4. Training runtime & compute envelope",
        "## 5. Cost envelope — architect spend cap vs running total",
    ):
        assert header in text, f"missing header: {header}"
    assert "src/cost/meter.py" in text
    # Training complete: the runtime envelope is filled, no PENDING left.
    assert "PENDING" not in text


def test_cost_doc_reports_training_runtime() -> None:
    text = _DOC.read_text(encoding="utf-8")
    # The real per-seed CPU wall-clock envelope (~40-80 min/seed, no paid API).
    assert "40" in text
    assert "80 min" in text
    assert "CPU" in text
