"""Doc/architecture contract (spec §9.2, CLAUDE.md §3): notebooks/analysis.ipynb
is a valid nbformat-4 notebook that consumes ONLY the SDK (no direct
src.env/model/ddpg/services imports), recomputes the results tables, and carries
the DDPG LaTeX + citations. nbformat/nbclient are not project deps, so we
validate structurally with stdlib json (it must at least parse and be shaped).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

_NB = Path(__file__).resolve().parent.parent.parent / "notebooks" / "analysis.ipynb"
_ALLOWED_SRC = {"src.sdk", "src.sdk.sdk"}


def _load() -> dict:
    return json.loads(_NB.read_text(encoding="utf-8"))


def _code_source() -> str:
    nb = _load()
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    return "\n".join(("".join(c["source"]) if isinstance(c["source"], list) else c["source"]) for c in cells)


def _markdown_source() -> str:
    nb = _load()
    cells = [c for c in nb["cells"] if c["cell_type"] == "markdown"]
    return "\n".join(("".join(c["source"]) if isinstance(c["source"], list) else c["source"]) for c in cells)


def test_notebook_is_valid_nbformat4() -> None:
    nb = _load()
    assert nb["nbformat"] == 4, "notebook must be nbformat 4"
    assert nb["cells"], "notebook has no cells"
    for cell in nb["cells"]:
        assert cell["cell_type"] in {"code", "markdown"}
        assert "source" in cell


def test_notebook_imports_only_the_sdk() -> None:
    tree = ast.parse(_code_source())
    src_mods: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            src_mods += [a.name for a in node.names if a.name.split(".")[0] == "src"]
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod.split(".")[0] == "src":
                src_mods.append(mod)
    leaks = [m for m in src_mods if m not in _ALLOWED_SRC and not m.startswith("src.sdk.")]
    assert not leaks, f"notebook imports src.* outside the SDK: {leaks}"
    assert "src.sdk.sdk" in src_mods, "notebook must consume RoboVacuumSDK"


def test_notebook_loads_results_and_has_latex_citations() -> None:
    code = _code_source()
    assert "metrics_summary.json" in code
    assert "holdout_eval.json" in code
    md = _markdown_source()
    assert "\\nabla_\\theta J" in md, "missing DDPG objective LaTeX"
    assert "\\tau" in md, "missing Polyak soft-update LaTeX"
    assert "1509.02971" in md, "missing Lillicrap DDPG citation"
    assert "1903.09845" in md, "missing HouseExpo citation"
