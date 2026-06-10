"""Unit tests for the ≤150-LOC file-size guard (scripts/check_file_sizes.py)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent.parent / "scripts" / "check_file_sizes.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_file_sizes", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_count_loc_excludes_blanks_and_comments(tmp_path) -> None:
    mod = _load_module()
    f = tmp_path / "sample.py"
    f.write_text("# comment\n\nx = 1\n   \ny = 2  # trailing\n", encoding="utf-8")
    assert mod.count_loc(f) == 2  # noqa: PLR2004


def test_count_loc_counts_code_lines(tmp_path) -> None:
    mod = _load_module()
    f = tmp_path / "code.py"
    f.write_text("\n".join(f"a{i} = {i}" for i in range(10)) + "\n", encoding="utf-8")
    assert mod.count_loc(f) == 10  # noqa: PLR2004


def test_scan_dirs_flags_oversized_file(tmp_path) -> None:
    mod = _load_module()
    src = tmp_path / "src"
    src.mkdir()
    big = src / "big.py"
    big.write_text("\n".join(f"v{i} = {i}" for i in range(160)) + "\n", encoding="utf-8")
    over = mod.scan_dirs(tmp_path, ("src",))
    names = [p.name for p, _ in over]
    assert "big.py" in names


def test_scan_dirs_passes_small_file(tmp_path) -> None:
    mod = _load_module()
    src = tmp_path / "src"
    src.mkdir()
    (src / "small.py").write_text("x = 1\n", encoding="utf-8")
    assert mod.scan_dirs(tmp_path, ("src",)) == []


def test_repo_source_tree_under_limit() -> None:
    mod = _load_module()
    root = _SCRIPT.resolve().parent.parent
    over = mod.scan_dirs(root, ("src", "tests"))
    assert over == [], f"Files over 150 LOC: {over}"
