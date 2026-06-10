"""Doc contract (spec §2, TODO T04-04): QUALITY.md covers all EIGHT ISO/IEC
25010 product-quality characteristics with evidence + honest limitations.
"""

from __future__ import annotations

from pathlib import Path

_DOC = Path(__file__).resolve().parent.parent.parent / "docs" / "QUALITY.md"

_CHARACTERISTICS = (
    "## 1. Functional Suitability",
    "## 2. Performance Efficiency",
    "## 3. Compatibility",
    "## 4. Usability",
    "## 5. Reliability",
    "## 6. Security",
    "## 7. Maintainability",
    "## 8. Portability",
)


def test_quality_doc_covers_eight_characteristics() -> None:
    text = _DOC.read_text(encoding="utf-8")
    for header in _CHARACTERISTICS:
        assert header in text, f"missing ISO 25010 characteristic: {header}"
    assert "ISO/IEC 25010" in text
    assert "## 9. Honest limitations" in text


def test_quality_doc_carries_evidence_pointers() -> None:
    text = _DOC.read_text(encoding="utf-8")
    assert "fail_under=85" in text
    assert "150" in text
    assert "ruff" in text.lower()
