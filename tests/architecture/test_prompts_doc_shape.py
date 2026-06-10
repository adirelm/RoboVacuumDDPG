"""Doc contract (CLAUDE.md §1.4, TODO T04-06): PROMPTS.md is the verbatim
architect -> implementer prompt log, each mapped to a commit hash with a
human-judgment annotation.
"""

from __future__ import annotations

from pathlib import Path

_DOC = Path(__file__).resolve().parent.parent.parent / "docs" / "shared" / "PROMPTS.md"


def test_prompts_doc_skeleton() -> None:
    text = _DOC.read_text(encoding="utf-8")
    assert "Human ↔ AI Responsibility Contract" in text
    assert "| Prompt | Commit | Human-judgment annotation |" in text
    for phase in ("Phase 0", "Phase 1", "Phase 2", "Phase 3", "Phase 4"):
        assert phase in text, f"missing phase row group: {phase}"
    assert "PENDING" in text
