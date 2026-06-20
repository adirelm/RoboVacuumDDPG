import copy

import pytest

from scripts import evaluate as evaluate_script
from src.sdk.sdk import RoboVacuumSDK


def _smoke_init(monkeypatch):
    orig_init = RoboVacuumSDK.__init__

    def patched_init(self, config_path=None):
        orig_init(self, config_path)
        # Deep-copy so the shrunk smoke config never leaks into the shared cache.
        self.cfg = copy.deepcopy(self.cfg)
        self.cfg["env"]["max_steps"] = 3  # smoke speed

    monkeypatch.setattr(RoboVacuumSDK, "__init__", patched_init)


def test_evaluate_holdout_aggregates_trained_seeds(monkeypatch):
    _smoke_init(monkeypatch)
    results = evaluate_script.evaluate_holdout(seeds=[42, 7])
    holdout = RoboVacuumSDK().cfg["maps"]["holdout"]
    assert set(results) == set(holdout)
    for name in holdout:
        report = results[name]
        assert set(report) == {
            "coverage_mean",
            "coverage_ci",
            "coverage_per_seed",
            "collisions_mean",
            "steps",
            "seeds",
        }
        assert 0.0 <= report["coverage_mean"] <= 1.0
        assert report["coverage_ci"] >= 0.0
        assert len(report["coverage_per_seed"]) == 2
        assert report["seeds"] == [42, 7]


def test_evaluate_holdout_raises_on_missing_checkpoint(monkeypatch):
    # The silent fresh-agent fallback is gone: a missing checkpoint must raise,
    # never publish untrained numbers as the held-out result.
    _smoke_init(monkeypatch)
    with pytest.raises(FileNotFoundError):
        evaluate_script.evaluate_holdout(seeds=[999999])
