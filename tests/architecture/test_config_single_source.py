"""Architecture contract (CLAUDE.md §4, PRD-DDPG checkpoint 3, spec §5.3):
hyperparameters live ONLY in config/config.yaml, reached via config_loader.
No module-level hyperparameter literal may be baked into src/.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.utils.config_loader import get, load_config

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCAN_DIRS = ("src/ddpg", "src/model", "src/env", "src/services")
_FORBIDDEN_NAMES = {
    "gamma",
    "tau",
    "lr_actor",
    "lr_critic",
    "batch_size",
    "buffer_size",
    "grad_clip",
    "warmup_steps",
    "sigma_start",
    "sigma_end",
    "sigma_decay_steps",
    "k_coverage",
    "k_collision",
    "k_step",
    "v_max",
    "omega_max",
    "ray_max",
}


def test_every_hyperparameter_resolves_through_config() -> None:
    load_config()
    ddpg = get("ddpg")
    for key in (
        "gamma",
        "tau",
        "lr_actor",
        "lr_critic",
        "batch_size",
        "buffer_size",
        "hidden_sizes",
        "grad_clip",
        "warmup_steps",
    ):
        assert key in ddpg, f"ddpg.{key} missing from config"
    noise = get("noise")
    for key in ("type", "sigma_start", "sigma_end", "sigma_decay_steps"):
        assert key in noise, f"noise.{key} missing from config"
    reward = get("reward")
    for key in ("k_coverage", "k_collision", "k_step"):
        assert key in reward, f"reward.{key} missing from config"
    gui = get("gui")
    gui_keys = (
        "window_width",
        "window_height",
        "fps",
        "train_steps_per_frame",
        "trail_length",
        "demo_checkpoint",
    )
    for key in gui_keys:
        assert key in gui, f"gui.{key} missing from config"
    assert abs(float(ddpg["gamma"]) - 0.99) < 1e-9
    assert abs(float(ddpg["tau"]) - 0.005) < 1e-9


def _scan_py() -> list[Path]:
    files: list[Path] = []
    for rel in _SCAN_DIRS:
        d = _REPO_ROOT / rel
        if d.exists():
            files.extend(p for p in d.rglob("*.py") if p.is_file())
    return sorted(files)


_PY = _scan_py()
_IDS = [str(p.relative_to(_REPO_ROOT)) for p in _PY]


@pytest.mark.skipif(not _PY, reason="No src/ddpg|model|env|services .py yet")
@pytest.mark.parametrize("py_file", _PY, ids=_IDS)
def test_no_module_level_hyperparameter_literal(py_file: Path) -> None:
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    bad: list[str] = []
    for node in tree.body:  # module level only
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if (
                    isinstance(tgt, ast.Name)
                    and tgt.id.lower() in _FORBIDDEN_NAMES
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, (int, float))
                ):
                    bad.append(f"{py_file}:{node.lineno}: {tgt.id} = {node.value.value}")
    assert not bad, "hardcoded hyperparameter literal in src/ (CLAUDE.md §4):\n" + "\n".join(bad)
