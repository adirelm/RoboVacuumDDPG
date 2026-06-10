"""Doc contract (spec §7, PRD-DDPG §4/§5.6): ANALYSIS.md answers the brief's
THREE analysis questions, embeds the three figures, and marks result numbers
PENDING until training completes.
"""

from __future__ import annotations

from pathlib import Path

_DOC = Path(__file__).resolve().parent.parent.parent / "docs" / "ANALYSIS.md"


def test_analysis_doc_has_three_question_headers() -> None:
    text = _DOC.read_text(encoding="utf-8")
    assert "## Q1 — Why DDPG (not DQN, not PPO)" in text
    assert "## Q2 — Removing Gaussian exploration noise early" in text
    assert "## Q3 — Target networks + soft updates prevent critic collapse" in text


def test_analysis_doc_embeds_figures_and_pending_numbers() -> None:
    text = _DOC.read_text(encoding="utf-8")
    assert "results/figures/learning_curve.png" in text
    assert "results/figures/critic_loss.png" in text
    assert "results/figures/trajectory.png" in text
    assert "ΔReward" in text
    assert "PENDING" in text
