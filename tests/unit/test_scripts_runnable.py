"""Regression: src-importing scripts must be runnable directly (`python scripts/X.py`).

Before the sys.path bootstrap, `python scripts/train.py` raised
ModuleNotFoundError: No module named 'src' (sys.path[0] is scripts/, not the repo
root). runpy.run_path with a non-'__main__' run_name re-runs module-level imports
(reproducing the bug) but skips the `if __name__ == '__main__'` block (no training).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "script", ["train", "evaluate", "render_trajectory", "sweep_n_rays", "render_sensitivity"]
)
def test_script_imports_resolve_when_run_directly(script: str) -> None:
    code = f"import runpy; runpy.run_path('scripts/{script}.py', run_name='smoke')"
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=_REPO, capture_output=True, text=True, check=False
    )
    assert "ModuleNotFoundError" not in proc.stderr, proc.stderr
    assert proc.returncode == 0, proc.stderr
