"""Unit tests for the runtime cost meter (contract: src/cost/meter.py).

RuntimeMeter is a small wall-clock + step/episode counter (contract F23).
"""

from __future__ import annotations

from src.cost.meter import RuntimeMeter


def test_fresh_meter_has_zero_counters() -> None:
    m = RuntimeMeter()
    assert m.steps == 0
    assert m.episodes == 0


def test_tick_increments_steps() -> None:
    m = RuntimeMeter()
    m.tick()
    m.tick(3)
    assert m.steps == 4  # noqa: PLR2004


def test_episode_increments_episodes() -> None:
    m = RuntimeMeter()
    m.episode()
    m.episode()
    assert m.episodes == 2  # noqa: PLR2004


def test_elapsed_is_nonnegative_and_monotonic() -> None:
    m = RuntimeMeter()
    first = m.elapsed()
    second = m.elapsed()
    assert first >= 0.0
    assert second >= first


def test_summary_reports_counters_and_elapsed() -> None:
    m = RuntimeMeter()
    m.tick(5)
    m.episode()
    summary = m.summary()
    assert summary["steps"] == 5  # noqa: PLR2004
    assert summary["episodes"] == 1
    assert summary["elapsed_s"] >= 0.0


def test_reset_zeroes_counters_and_restarts_clock() -> None:
    m = RuntimeMeter()
    m.tick(2)
    m.episode()
    m.reset()
    assert m.steps == 0
    assert m.episodes == 0
    assert m.elapsed() >= 0.0
