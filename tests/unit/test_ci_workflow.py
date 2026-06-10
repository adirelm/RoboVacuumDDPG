"""Structural checks on the CI workflow (.github/workflows/ci.yml)."""

from __future__ import annotations

from pathlib import Path

import yaml

_CI = Path(__file__).resolve().parent.parent.parent / ".github" / "workflows" / "ci.yml"


def _run_commands() -> str:
    data = yaml.safe_load(_CI.read_text(encoding="utf-8"))
    jobs = data["jobs"]
    cmds = []
    for job in jobs.values():
        for step in job["steps"]:
            if "run" in step:
                cmds.append(step["run"])
    return "\n".join(cmds)


def test_ci_file_exists() -> None:
    assert _CI.exists()


def test_ci_has_required_steps() -> None:
    runs = _run_commands()
    assert "uv sync --dev" in runs
    assert "ruff check" in runs
    assert "ruff format --check" in runs
    assert "scripts/check_file_sizes.py" in runs
    assert "pytest" in runs and "--cov" in runs


def test_ci_uses_uv_setup_action() -> None:
    text = _CI.read_text(encoding="utf-8")
    assert "astral-sh/setup-uv" in text


def test_ci_triggers_on_push_and_pr() -> None:
    data = yaml.safe_load(_CI.read_text(encoding="utf-8"))
    triggers = data[True] if True in data else data["on"]
    assert "push" in triggers
    assert "pull_request" in triggers
