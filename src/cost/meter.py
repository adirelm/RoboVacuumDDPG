"""Runtime cost meter — wall-clock timer + step/episode counters (contract F23).

A tiny, dependency-free instrument the training loop can poll to report how
much wall-clock time and how many environment steps / episodes a run consumed,
keeping the "cost-budget envelope" honest (CLAUDE.md responsibility contract).
"""

from __future__ import annotations

import time


class RuntimeMeter:
    """Track elapsed wall-clock time and cumulative step / episode counts."""

    def __init__(self) -> None:
        self._start = time.perf_counter()
        self.steps = 0
        self.episodes = 0

    def tick(self, n: int = 1) -> None:
        """Record `n` environment steps (default 1)."""
        self.steps += n

    def episode(self) -> None:
        """Record one completed episode."""
        self.episodes += 1

    def elapsed(self) -> float:
        """Seconds of wall-clock time since construction (or last reset)."""
        return time.perf_counter() - self._start

    def summary(self) -> dict:
        """Snapshot of counters and elapsed wall-clock time."""
        return {
            "steps": self.steps,
            "episodes": self.episodes,
            "elapsed_s": self.elapsed(),
        }

    def reset(self) -> None:
        """Zero the counters and restart the wall-clock timer."""
        self._start = time.perf_counter()
        self.steps = 0
        self.episodes = 0
