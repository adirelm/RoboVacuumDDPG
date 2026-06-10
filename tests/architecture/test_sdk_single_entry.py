"""Architecture contract (CLAUDE.md §3, PRD-SIM FR-10, spec §8): the UI layer
(scripts/, notebooks/) imports ONLY src.sdk — no direct src.env / src.ddpg /
src.model / src.services / src.utils imports leak into a UI module.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.sdk.sdk import RoboVacuumSDK

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_UI_DIRS = ("scripts", "notebooks")
_ALLOWED = {"src.sdk", "src.sdk.sdk"}


def _ui_py_files() -> list[Path]:
    files: list[Path] = []
    for rel in _UI_DIRS:
        d = _REPO_ROOT / rel
        if d.exists():
            files.extend(p for p in d.rglob("*.py") if p.is_file())
    return sorted(files)


def _src_imports(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names if a.name.split(".")[0] == "src"]
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod.split(".")[0] == "src":
                names.append(mod)
    return names


def _is_allowed(mod: str) -> bool:
    return mod in _ALLOWED or mod.startswith("src.sdk.")


_UI = _ui_py_files()
_IDS = [str(p.relative_to(_REPO_ROOT)) for p in _UI]


@pytest.mark.skipif(not _UI, reason="No scripts/ or notebooks/ .py files yet")
@pytest.mark.parametrize("py_file", _UI, ids=_IDS)
def test_ui_imports_only_sdk(py_file: Path) -> None:
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    leaks = [m for m in _src_imports(tree) if not _is_allowed(m)]
    assert not leaks, f"{py_file} imports src.* outside the SDK: {leaks}"


def test_at_least_one_script_uses_the_sdk() -> None:
    uses = [
        p.name
        for p in (_REPO_ROOT / "scripts").rglob("*.py")
        if "RoboVacuumSDK" in p.read_text(encoding="utf-8")
    ]
    assert uses, "expected at least one script under scripts/ to import RoboVacuumSDK"


def test_sdk_exposes_contract_surface() -> None:
    for method in ("build_env", "train", "rollout", "coverage_report"):
        assert callable(getattr(RoboVacuumSDK, method)), f"SDK missing {method}"
