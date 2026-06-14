"""Doc contract (spec §2 §10, submission guidelines §10): UX.md documents the
Pygame GUI — the three modes with screenshots, a controls reference, Nielsen's
10 heuristics mapped to the UI, and accessibility notes.
"""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
_DOC = _ROOT / "docs" / "UX.md"


def test_ux_doc_covers_the_gui() -> None:
    text = _DOC.read_text(encoding="utf-8")
    assert "Pygame" in text
    assert "scripts/play.py" in text
    assert "RoboVacuumSDK" in text
    # Nielsen's 10 heuristics are enumerated 1..10.
    assert "Nielsen" in text
    for n in range(1, 11):
        assert f"{n}." in text, f"missing heuristic {n}"
    # No leftover "Not Applicable" verdict now that a GUI exists.
    assert "Not Applicable" not in text


def test_ux_doc_embeds_mode_screenshots() -> None:
    text = _DOC.read_text(encoding="utf-8")
    for shot in ("train.png", "play.png", "drive.png", "play_no_checkpoint.png"):
        assert f"assets/screenshots/{shot}" in text, f"missing screenshot {shot}"
        assert (_ROOT / "assets" / "screenshots" / shot).exists(), f"{shot} not on disk"
