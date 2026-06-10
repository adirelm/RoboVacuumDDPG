import copy

from scripts import evaluate as evaluate_script
from src.sdk.sdk import RoboVacuumSDK


def test_evaluate_holdout_returns_report_per_map(monkeypatch):
    orig_init = RoboVacuumSDK.__init__

    def patched_init(self, config_path=None):
        orig_init(self, config_path)
        # Deep-copy so the shrunk smoke config never leaks into the shared cache.
        self.cfg = copy.deepcopy(self.cfg)
        self.cfg["env"]["max_steps"] = 3  # smoke speed

    monkeypatch.setattr(RoboVacuumSDK, "__init__", patched_init)

    results = evaluate_script.evaluate_holdout()
    holdout = RoboVacuumSDK().cfg["maps"]["holdout"]
    assert set(results) == set(holdout)
    for name in holdout:
        report = results[name]
        assert set(report) == {"coverage", "steps", "collisions"}
        assert 0.0 <= report["coverage"] <= 1.0
