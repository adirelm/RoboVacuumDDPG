"""Doc contract (spec §7, PRD-DDPG §4/§5.6): ANALYSIS.md answers the brief's
THREE analysis questions, embeds the three figures, and now (training complete)
reports the real seeded results with NO remaining PENDING placeholders.
"""

from __future__ import annotations

from pathlib import Path

_DOC = Path(__file__).resolve().parent.parent.parent / "docs" / "ANALYSIS.md"


def test_analysis_doc_has_three_question_headers() -> None:
    text = _DOC.read_text(encoding="utf-8")
    assert "## Q1 — Why DDPG (not DQN, not PPO)" in text
    assert "## Q2 — Removing Gaussian exploration noise early" in text
    assert "## Q3 — Target networks + soft updates prevent critic collapse" in text


def test_analysis_doc_embeds_figures_and_reports_results() -> None:
    text = _DOC.read_text(encoding="utf-8")
    assert "results/figures/learning_curve.png" in text
    assert "results/figures/critic_loss.png" in text
    assert "results/figures/trajectory.png" in text
    assert "ΔReward" in text
    # Training is complete: the real across-seed result is reported and no
    # PENDING placeholder is left behind (spec §10 honesty stance).
    assert "691.3" in text
    assert "PENDING" not in text


def test_analysis_doc_reports_holdout_overfitting() -> None:
    text = _DOC.read_text(encoding="utf-8")
    # The honest held-out generalization limitation must be stated plainly.
    assert "apt_large" in text
    assert "office" in text
    assert "overfit" in text.lower()
