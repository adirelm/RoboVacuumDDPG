import copy
import json

import torch

from scripts import train as train_script
from src.sdk.sdk import RoboVacuumSDK


def test_run_seeds_writes_history_and_checkpoint(tmp_path, monkeypatch):
    # Shrink the workload to a 1-episode, 3-step, single-seed smoke run.
    orig_init = RoboVacuumSDK.__init__

    def patched_init(self, config_path=None):
        orig_init(self, config_path)
        # Deep-copy so the shrunken smoke config never leaks into the shared
        # cached config dict (load_config caches and returns the same object).
        self.cfg = copy.deepcopy(self.cfg)
        self.cfg["training"]["episodes"] = 1
        self.cfg["training"]["seeds"] = [42]
        self.cfg["env"]["max_steps"] = 3
        self.cfg["ddpg"]["warmup_steps"] = 1
        self.cfg["ddpg"]["batch_size"] = 2

    monkeypatch.setattr(RoboVacuumSDK, "__init__", patched_init)

    out = train_script.run_seeds(results_dir=str(tmp_path))
    assert out == [42]
    hist_file = tmp_path / "history" / "seed_42.json"
    ckpt_file = tmp_path / "checkpoints" / "seed_42.pt"
    assert hist_file.exists() and ckpt_file.exists()
    history = json.loads(hist_file.read_text(encoding="utf-8"))
    assert len(history) == 1
    assert set(history[0]) >= {"episode", "reward", "critic_loss", "coverage", "steps"}
    # Checkpoint is a real reloadable agent state_dict bundle (F6/F30).
    ckpt = torch.load(str(ckpt_file), weights_only=True)
    assert set(ckpt) == {"actor", "critic", "actor_target", "critic_target"}
