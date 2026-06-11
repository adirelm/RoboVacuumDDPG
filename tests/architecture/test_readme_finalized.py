"""Doc contract (spec §7/§10, TODO T04-05): README is the submission-report
shell — embeds the three figures, the uv install/run + SDK usage, links the
analysis docs, and (training complete) carries the honest real-results summary
with no PENDING placeholders left.
"""

from __future__ import annotations

from pathlib import Path

_DOC = Path(__file__).resolve().parent.parent.parent / "README.md"


def test_readme_embeds_three_figures() -> None:
    text = _DOC.read_text(encoding="utf-8")
    assert "results/figures/learning_curve.png" in text
    assert "results/figures/critic_loss.png" in text
    assert "results/figures/trajectory.png" in text


def test_readme_has_install_run_sdk_and_links() -> None:
    text = _DOC.read_text(encoding="utf-8")
    assert "uv sync --dev" in text
    assert "RoboVacuumSDK" in text
    assert "docs/ANALYSIS.md" in text
    assert "docs/THEORY.md" in text
    # Training complete: no PENDING placeholders left in the report shell.
    assert "PENDING" not in text


def test_readme_carries_honest_results_summary() -> None:
    text = _DOC.read_text(encoding="utf-8")
    # The one-line honest summary: learns on room_single, does not yet generalize.
    assert "691.3" in text
    assert "room_single" in text
