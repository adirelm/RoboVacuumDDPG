"""Architecture contract (CLAUDE.md §3, PRD-SIM FR-10, spec §8): business logic is
reached ONLY through the SDK.

- The UI layer (scripts/, notebooks/) imports only `src.sdk` or `src.gui` (the
  presentation App) — never `src.env` / `src.ddpg` / `src.model` / `src.services`.
- The presentation layer `src/gui/*` itself imports only `src.sdk` (+ intra-`src.gui`)
  — it consumes Frames via the SDK and never touches env/ddpg internals directly.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.sdk.sdk import RoboVacuumSDK

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_UI_DIRS = ("scripts", "notebooks")
# Both the UI entry layer and the GUI presentation layer may reach the SDK; the GUI
# may also import sibling gui modules. Neither may import env/ddpg/model/services.
_PRESENTATION = ("src.sdk", "src.gui")


def _py_files(*rel_dirs: str) -> list[Path]:
    files: list[Path] = []
    for rel in rel_dirs:
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


def _violations(tree: ast.AST, allowed: tuple[str, ...]) -> list[str]:
    return [m for m in _src_imports(tree) if not any(m == p or m.startswith(p + ".") for p in allowed)]


_UI = _py_files(*_UI_DIRS)
_GUI = _py_files("src/gui")
_UI_IDS = [str(p.relative_to(_REPO_ROOT)) for p in _UI]
_GUI_IDS = [str(p.relative_to(_REPO_ROOT)) for p in _GUI]


@pytest.mark.skipif(not _UI, reason="No scripts/ or notebooks/ .py files yet")
@pytest.mark.parametrize("py_file", _UI, ids=_UI_IDS)
def test_ui_imports_only_sdk_or_gui(py_file: Path) -> None:
    leaks = _violations(ast.parse(py_file.read_text(encoding="utf-8")), _PRESENTATION)
    assert not leaks, f"{py_file} imports src.* outside the SDK/GUI: {leaks}"


@pytest.mark.skipif(not _GUI, reason="No src/gui/ .py files yet")
@pytest.mark.parametrize("py_file", _GUI, ids=_GUI_IDS)
def test_gui_imports_only_sdk(py_file: Path) -> None:
    # The GUI must reach business logic only via the SDK (no env/ddpg/services).
    leaks = _violations(ast.parse(py_file.read_text(encoding="utf-8")), _PRESENTATION)
    assert not leaks, f"{py_file} (gui) imports business logic directly: {leaks}"


def test_at_least_one_script_uses_the_sdk() -> None:
    uses = [
        p.name
        for p in (_REPO_ROOT / "scripts").rglob("*.py")
        if "RoboVacuumSDK" in p.read_text(encoding="utf-8")
    ]
    assert uses, "expected at least one script under scripts/ to import RoboVacuumSDK"


def test_sdk_exposes_contract_surface() -> None:
    for method in ("build_env", "train", "rollout", "coverage_report", "live_session"):
        assert callable(getattr(RoboVacuumSDK, method)), f"SDK missing {method}"
