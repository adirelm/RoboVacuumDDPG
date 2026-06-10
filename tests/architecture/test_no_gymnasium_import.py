"""Architectural contract (spec §1, §3, §8): NO gymnasium/gym and NO
stable_baselines3 / rllib / ray.rllib import in src/ (contract amendment F22).

Brief "demand": no ready simulation platforms or RL libraries — the simulator
and DDPG are from scratch. AST-level (not grep) so string occurrences in
comments/docstrings do not false-positive. Parametrized over every .py under
src/ so a regression fails its own case.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

FORBIDDEN_ROOTS = ("gymnasium", "gym", "stable_baselines3", "rllib", "ray")
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC = _REPO_ROOT / "src"


def _is_forbidden(name: str) -> bool:
    if name == "ray" or name.startswith("ray."):
        # Only ray.rllib is banned (plain ray would be fine, but we vendor none).
        return name == "ray" or name.startswith("ray.rllib")
    return any(name == root or name.startswith(root + ".") for root in FORBIDDEN_ROOTS)


def _iter_src_py() -> list[Path]:
    return sorted(p for p in _SRC.rglob("*.py") if p.is_file())


def _offenders(tree: ast.AST, path: Path) -> list[str]:
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_forbidden(alias.name):
                    bad.append(f"{path}:{node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if _is_forbidden(mod):
                bad.append(f"{path}:{node.lineno}: from {mod} import ...")
    return bad


_PY = _iter_src_py()
_IDS = [str(p.relative_to(_REPO_ROOT)) for p in _PY]


@pytest.mark.skipif(not _PY, reason="No .py files under src/ yet")
@pytest.mark.parametrize("py_file", _PY, ids=_IDS)
def test_no_rl_library_import(py_file: Path) -> None:
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    bad = _offenders(tree, py_file)
    assert not bad, "RL-library import contract violation (spec §1/§3/§8, F22):\n" + "\n".join(bad)


def test_scan_actually_covered_src_files() -> None:
    """Guard: the parametrization must have found the real env/ddpg modules."""
    names = {p.name for p in _PY}
    assert "vacuum_env.py" in names
    assert "agent.py" in names
