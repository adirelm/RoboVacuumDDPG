# RoboVacuumDDPG Implementation Plan

> **FROZEN pre-implementation plan (historical).** This is the task-by-task plan
> as authored *before* the build. It is kept verbatim for provenance and is
> **superseded by the shipped docs** — any embedded first-draft snippets (e.g. an
> early `COST_ANALYSIS` sketch with a "tiktoken" headline, or `PENDING` result
> cells) do **not** reflect the final repo. For live state see `docs/COST_ANALYSIS.md`,
> `docs/ANALYSIS.md`, `README.md`, and `docs/shared/PROMPTS.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build — from scratch — a 2D robotic-vacuum simulator over real HouseExpo maps and a DDPG agent that learns continuous coverage navigation.

**Architecture:** A custom non-Gymnasium `VacuumEnv` (lidar raycasting + unicycle kinematics + coverage grid + collision) wrapped by a single `RoboVacuumSDK`; a from-scratch DDPG agent (actor/critic + target nets, uniform replay, Gaussian exploration, Polyak soft-updates) driven by a custom training loop. Results: learning-curve + critic-loss graphs + a trajectory visualization over the map.

**Tech Stack:** Python 3.11, PyTorch, NumPy, Matplotlib, `uv`, pytest, ruff. NO Gymnasium / Gazebo / Stable-Baselines3 (from scratch — brief mandate).

---

## Interface contract & corrections (READ FIRST)

Signatures are pinned in [`_contract.md`](_contract.md) — **LAW**. Its *Contract amendments* section OVERRIDES any task text below where they differ. During execution, apply these reconcile-pass corrections (F-ids from the plan audit):

- **[F6/F30] Persist & reload trained weights.** Implement `DDPGAgent.save/load` (state_dicts). `scripts/train.py` calls `agent.save(...)`; `scripts/render_trajectory.py` and `scripts/evaluate.py` **load the checkpoint** — never roll out a fresh untrained agent. The trajectory figure + eval MUST use the trained policy.
- **[F12] `tests/conftest.py` is incremental.** Each phase that touches it APPENDS fixtures (`cfg`, `house_map`, `tiny_map`) — never replaces/drops earlier ones. Canonical real-map fixture is `house_map`; `tiny_map` is an alias of the same 4-wall room.
- **[F28] `scripts/check_file_sizes.py` authored ONCE (Phase 0)** — scans `src/ tests/ scripts/`, exposes `count_loc()`+`scan_dirs()`. Phase 4 only *runs* it (do not recreate with a different body).
- **[F21] Hermetic test map.** Commit a synthetic `data/maps/room_single.json` (valid HouseExpo shape) so integration/SDK tests do not depend on a real dataset fetch.
- **[F22] No-RL-library gate.** The architecture test bans `gymnasium`/`gym` **and** `stable_baselines3`/`rllib`/`ray.rllib` imports under `src/`.
- **[F29] Step-level critic loss.** `Trainer.train` history carries a `critic_losses` list (one per gradient update); `render_critic_loss.py` plots that (step-granular, spec §7).
- **[F5] `RoboVacuumSDK.evaluate(checkpoint_path, map_name)`** added (resolves spec §4).
- **[F23] `src/cost/meter.py`** = a small `RuntimeMeter` (wall-clock + step/episode counters), authored in Phase 0.
- **[F14/F24] Already exist (do NOT recreate):** `docs/THEORY.md`, `.env-example` (from the scaffold/planning phase).
- **[F26] Phase-0 conftest:** do not import the unused `field` symbol (keep Ruff-clean as written).
- **Spec coverage confirmed:** 2 graphs ✅, trajectory viz ✅, 3 analysis questions ✅, no-gym ✅, Gaussian noise ✅, Polyak ✅, HouseExpo ✅, from-scratch ✅.

---

## Phase 0 — Bootstrap Completion (config loader · file-size guard · test fixtures · CI)

> Scope: the repo/config/docs/scaffold are already in place (see `pyproject.toml`,
> `config/config.yaml` version `1.0.0`, `src/**/__init__.py`, `tests/{unit,integration,architecture}/__init__.py`,
> `.gitignore`). This phase wires the four bootstrap primitives every later phase
> depends on: the cached config loader (`load_config`/`get` per `_contract.md`),
> the ≤150-LOC file-size guard, shared pytest fixtures (`cfg`, a 4-wall synthetic
> `HouseMap`), and the CI workflow. TDD throughout (RED → GREEN → REFACTOR). `uv` only.
> All absolute paths below are under the repo root
> `"<REPO_ROOT>"`.

### Task 0.1: Cached config loader (`load_config` + `get`)

**Files:**
- Create: `src/utils/config_loader.py`
- Test: `tests/unit/test_config_loader.py`

- [ ] **Step 1: Write the failing test** — `tests/unit/test_config_loader.py`:
```python
"""Unit tests for the cached YAML config loader (contract: src/utils/config_loader.py)."""

from __future__ import annotations

import pytest

from src.utils.config_loader import get, load_config


def test_load_config_returns_dict_with_version() -> None:
    cfg = load_config()
    assert isinstance(cfg, dict)
    assert cfg["version"] == "1.0.0"


def test_load_config_is_cached_same_object() -> None:
    assert load_config() is load_config()


def test_get_ddpg_block_has_tau() -> None:
    ddpg = get("ddpg")
    assert isinstance(ddpg, dict)
    assert ddpg["tau"] == 0.005
    assert ddpg["gamma"] == 0.99


def test_get_env_block_has_n_rays() -> None:
    env = get("env")
    assert env["n_rays"] == 16
    assert env["ray_max"] == 5.0


def test_get_missing_section_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        get("missing")


def test_load_config_explicit_path(tmp_path) -> None:
    custom = tmp_path / "c.yaml"
    custom.write_text('version: "9.9.9"\nddpg: {tau: 1.0}\n', encoding="utf-8")
    data = load_config(str(custom))
    assert data["version"] == "9.9.9"
    assert data["ddpg"]["tau"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails** —
  `uv run pytest tests/unit/test_config_loader.py -v`
  Expected: FAIL (`ModuleNotFoundError: No module named 'src.utils.config_loader'` — module does not exist yet).

- [ ] **Step 3: Write minimal implementation** — `src/utils/config_loader.py`:
```python
"""Cached YAML config loader for RoboVacuumDDPG — single source of truth.

Contract (docs/superpowers/plans/_contract.md):
  load_config(path: str | None = None) -> dict   # parse config/config.yaml; caches
  get(section: str) -> dict                       # top-level block; KeyError if missing
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "config.yaml"


@lru_cache(maxsize=None)
def _load_cached(cfg_path: str) -> dict:
    path = Path(cfg_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at {path}. Expected config/config.yaml at repo root."
        )
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} did not parse to a dict.")
    return data


def load_config(path: str | None = None) -> dict:
    """Load and cache the YAML config; repeated calls return the same dict object."""
    cfg_path = path if path is not None else str(_DEFAULT_CONFIG_PATH)
    return _load_cached(cfg_path)


def get(section: str) -> dict:
    """Return a top-level config block; raise KeyError if the section is missing."""
    cfg = load_config()
    if section not in cfg:
        raise KeyError(f"Config section '{section}' not found.")
    return cfg[section]
```

- [ ] **Step 4: Run test to verify it passes** —
  `uv run pytest tests/unit/test_config_loader.py -v`
  Expected: PASS (6 tests; `load_config()` returns version `1.0.0`, `get('ddpg')` has `tau=0.005`, `get('missing')` raises `KeyError`).

- [ ] **Step 5: Commit** —
  `git add src/utils/config_loader.py tests/unit/test_config_loader.py && git commit -m "Phase0 T0.1: cached config loader (load_config/get) + tests"`

---

### Task 0.2: File-size guard script (`check_file_sizes.py`, ≤150 LOC)

**Files:**
- Create: `scripts/check_file_sizes.py`
- Test: `tests/unit/test_check_file_sizes.py`

- [ ] **Step 1: Write the failing test** — `tests/unit/test_check_file_sizes.py`:
```python
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
    assert mod.count_loc(f) == 2


def test_count_loc_counts_code_lines(tmp_path) -> None:
    mod = _load_module()
    f = tmp_path / "code.py"
    f.write_text("\n".join(f"a{i} = {i}" for i in range(10)) + "\n", encoding="utf-8")
    assert mod.count_loc(f) == 10


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
```

- [ ] **Step 2: Run test to verify it fails** —
  `uv run pytest tests/unit/test_check_file_sizes.py -v`
  Expected: FAIL (`FileNotFoundError`/`spec is None` — `scripts/check_file_sizes.py` does not exist yet).

- [ ] **Step 3: Write minimal implementation** — `scripts/check_file_sizes.py`:
```python
"""Fail if any .py under src/ or tests/ exceeds 150 LOC (CLAUDE.md §1).

LOC excludes blank lines and pure-comment lines, so docstrings count but
`# divider` separators do not. Exit 1 if any file is over the limit.
"""

from __future__ import annotations

import sys
from pathlib import Path

LIMIT = 150
SCAN = ("src", "tests")
EXCLUDED_DIRS = {".venv", ".git", "build", "dist", "__pycache__", ".ruff_cache", ".pytest_cache", "vendor"}


def count_loc(path: Path) -> int:
    """Count non-blank, non-comment lines in a Python file."""
    loc = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        loc += 1
    return loc


def scan_dirs(root: Path, dirs: tuple[str, ...] = SCAN) -> list[tuple[Path, int]]:
    """Return [(path, loc)] for every .py over LIMIT under the given top-level dirs."""
    over: list[tuple[Path, int]] = []
    for top in dirs:
        base = root / top
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if any(part in EXCLUDED_DIRS for part in path.parts):
                continue
            loc = count_loc(path)
            if loc > LIMIT:
                over.append((path.relative_to(root), loc))
    return over


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    over = scan_dirs(root)
    if over:
        print(f"FAIL: {len(over)} file(s) exceed {LIMIT} LOC:")
        for path, loc in over:
            print(f"  {path}: {loc} LOC")
        return 1
    print(f"OK: all .py files under {SCAN} are <= {LIMIT} LOC")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes** —
  `uv run pytest tests/unit/test_check_file_sizes.py -v && uv run python scripts/check_file_sizes.py`
  Expected: PASS (5 tests; script prints `OK: all .py files ... <= 150 LOC` and exits 0).

- [ ] **Step 5: Commit** —
  `git add scripts/check_file_sizes.py tests/unit/test_check_file_sizes.py && git commit -m "Phase0 T0.2: ≤150-LOC file-size guard + tests"`

---

### Task 0.3: Shared pytest fixtures (`cfg`, synthetic `HouseMap`)

**Files:**
- Create: `tests/conftest.py`
- Test: `tests/unit/test_conftest_fixtures.py`

> Fixture `house_map` builds a synthetic 4-wall square room as a plain object that
> mirrors the contract `HouseMap` shape (`walls: list[Segment]`, `bounds: tuple`,
> `is_inside(x, y) -> bool`) WITHOUT importing `src.env.house_map` (that module is
> built in a later phase). A 4×4 m room with corners (0,0)→(4,4): four wall
> segments `(0,0,4,0)`, `(4,0,4,4)`, `(4,4,0,4)`, `(0,4,0,0)`; bounds `(0,0,4,4)`;
> `is_inside` is a strict-interior axis-aligned box test. Later phases needing the
> real loader override this fixture locally; this synthetic stand-in keeps Phase 0
> self-contained.

- [ ] **Step 1: Write the failing test** — `tests/unit/test_conftest_fixtures.py`:
```python
"""Verify the shared conftest fixtures (cfg dict + synthetic 4-wall HouseMap)."""

from __future__ import annotations


def test_cfg_fixture_is_config_dict(cfg) -> None:
    assert isinstance(cfg, dict)
    assert cfg["version"] == "1.0.0"
    assert cfg["ddpg"]["tau"] == 0.005
    assert cfg["env"]["n_rays"] == 16


def test_house_map_has_four_walls(house_map) -> None:
    assert len(house_map.walls) == 4
    for seg in house_map.walls:
        assert len(seg) == 4
        assert all(isinstance(c, float) for c in seg)


def test_house_map_bounds(house_map) -> None:
    assert house_map.bounds == (0.0, 0.0, 4.0, 4.0)


def test_house_map_is_inside_interior(house_map) -> None:
    assert house_map.is_inside(2.0, 2.0) is True
    assert house_map.is_inside(0.0, 0.0) is False
    assert house_map.is_inside(5.0, 2.0) is False
    assert house_map.is_inside(2.0, -1.0) is False
```

- [ ] **Step 2: Run test to verify it fails** —
  `uv run pytest tests/unit/test_conftest_fixtures.py -v`
  Expected: FAIL (`fixture 'cfg' not found` / `fixture 'house_map' not found` — `tests/conftest.py` does not define them yet).

- [ ] **Step 3: Write minimal implementation** — `tests/conftest.py`:
```python
"""Shared pytest fixtures for RoboVacuumDDPG.

Provides:
  cfg        — the real config dict loaded from config/config.yaml.
  house_map  — a tiny synthetic 4-wall square room mirroring the contract
               HouseMap shape (walls / bounds / is_inside), with NO dependency
               on src.env.house_map (built in a later phase). Later phases may
               override `house_map` locally to use the real loader.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from src.utils.config_loader import load_config

Segment = tuple[float, float, float, float]


@dataclass
class FakeHouseMap:
    """Synthetic HouseMap stand-in (same public surface as the contract dataclass)."""

    walls: list[Segment]
    bounds: tuple[float, float, float, float]

    def is_inside(self, x: float, y: float) -> bool:
        xmin, ymin, xmax, ymax = self.bounds
        return xmin < x < xmax and ymin < y < ymax


@pytest.fixture
def cfg() -> dict:
    """Return the project config dict (config/config.yaml)."""
    return load_config()


@pytest.fixture
def house_map() -> FakeHouseMap:
    """A 4x4 m square room: four wall segments, bounds (0,0,4,4)."""
    walls: list[Segment] = [
        (0.0, 0.0, 4.0, 0.0),
        (4.0, 0.0, 4.0, 4.0),
        (4.0, 4.0, 0.0, 4.0),
        (0.0, 4.0, 0.0, 0.0),
    ]
    return FakeHouseMap(walls=walls, bounds=(0.0, 0.0, 4.0, 4.0))
```

> Note: `field` is imported only if a later edit adds default-factory fields; if
> Ruff F401 flags it unused, drop the `, field` from the import. The minimal code
> above does not use it — REMOVE `field` from the import line before committing to
> keep Ruff clean: `from dataclasses import dataclass`.

- [ ] **Step 4: Run test to verify it passes** —
  `uv run pytest tests/unit/test_conftest_fixtures.py -v && uv run ruff check tests/conftest.py`
  Expected: PASS (4 tests; `cfg` is the real config with `tau=0.005`; `house_map` has 4 float-tuple walls, bounds `(0,0,4,4)`, interior `is_inside`). Ruff: zero violations.

- [ ] **Step 5: Commit** —
  `git add tests/conftest.py tests/unit/test_conftest_fixtures.py && git commit -m "Phase0 T0.3: shared conftest fixtures (cfg + synthetic 4-wall HouseMap)"`

---

### Task 0.4: CI workflow (`.github/workflows/ci.yml`)

**Files:**
- Create: `.github/workflows/ci.yml`
- Test: `tests/unit/test_ci_workflow.py`

> The workflow runs the full bootstrap gate on push/PR to `main` and `assignment-5`:
> `uv sync --dev` → `ruff check` → `ruff format --check` → `check_file_sizes.py`
> → `pytest --cov`. The test parses the YAML and asserts every required step is
> present (no prose checks beyond structure).

- [ ] **Step 1: Write the failing test** — `tests/unit/test_ci_workflow.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails** —
  `uv run pytest tests/unit/test_ci_workflow.py -v`
  Expected: FAIL (`FileNotFoundError` — `.github/workflows/ci.yml` does not exist yet).

- [ ] **Step 3: Write minimal implementation** — `.github/workflows/ci.yml`:
```yaml
name: CI

on:
  push:
    branches: [main, assignment-5]
  pull_request:
    branches: [main, assignment-5]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        uses: astral-sh/setup-uv@v3
        with:
          version: "latest"
      - name: Set up Python
        run: uv python install 3.11
      - name: Install deps
        run: uv sync --dev
      - name: Ruff lint
        run: uv run ruff check src/ tests/ scripts/
      - name: Ruff format check
        run: uv run ruff format --check src/ tests/ scripts/
      - name: File-size guard (<=150 LOC)
        run: uv run python scripts/check_file_sizes.py
      - name: Pytest with coverage
        run: uv run pytest tests/ --cov=src --cov-report=term-missing --cov-report=xml -v
      - name: Upload coverage
        uses: actions/upload-artifact@v4
        with:
          name: coverage-report
          path: coverage.xml
          retention-days: 7
```

> Note on `test_ci_triggers_on_push_and_pr`: PyYAML parses the YAML key `on:` as
> the Python boolean `True` (YAML 1.1 truthy). The test handles both `data[True]`
> and `data["on"]`, so it passes regardless of the PyYAML version's normalisation.

- [ ] **Step 4: Run test to verify it passes** —
  `uv run pytest tests/unit/test_ci_workflow.py -v`
  Expected: PASS (4 tests; workflow has all 5 required steps, uses `astral-sh/setup-uv`, triggers on push + pull_request).

- [ ] **Step 5: Commit** —
  `git add .github/workflows/ci.yml tests/unit/test_ci_workflow.py && git commit -m "Phase0 T0.4: CI workflow (uv sync, ruff, file-size guard, pytest --cov) + test"`

---

**Phase 0 Definition of Done (verify before moving to Phase 1):**
- `uv run pytest tests/ -v` — all Phase-0 tests green.
- `uv run ruff check src/ tests/ scripts/` — zero violations.
- `uv run ruff format --check src/ tests/ scripts/` — clean.
- `uv run python scripts/check_file_sizes.py` — exit 0 (every new file ≤150 LOC).
- `load_config()["version"] == "1.0.0"`, `get("ddpg")["tau"] == 0.005`, `get("missing")` raises `KeyError`.

---

## Phase 1 — From-Scratch 2D Vacuum Simulator (`src/env/`)

> Ground truth: spec `docs/superpowers/specs/2026-06-10-robovacuum-ddpg-design.md`
> §3–§4, contract `docs/superpowers/plans/_contract.md` (signatures are LAW),
> PRD-SIM `docs/prd/PRD-SIM.md` §3 (component design) + §5 (hand-computed
> acceptance tests K/R/C/X/W), PRD-HOUSEEXPO `docs/prd/PRD-HOUSEEXPO.md` §5
> (parse semantics + §5.1 determinism invariant). Config defaults in
> `config/config.yaml` (`env`, `reward` blocks).

This phase builds the deterministic physics core in contract order:
`house_map → raycast → kinematics → collision → coverage → reward → state →
vacuum_env`. Every module is a focused pure unit (≤150 LOC, CLAUDE.md §1),
TDD RED→GREEN→REFACTOR (CLAUDE.md §2). Tests use **hand-computed** expectations
and self-contained config dicts / temp JSON, so Phase 1 does NOT depend on
`config_loader` or the vendored HouseExpo dataset (both out of scope here).
No `gymnasium` / `gazebo` / SB3 anywhere (spec §1, FR-1). `uv` only.

All commands are run from the repo root
`<REPO_ROOT>`.

---

### Task 1: `house_map` — load HouseExpo JSON → walls + bounds; `is_inside`

**Files:**
- Create `src/env/house_map.py`
- Test `tests/unit/test_house_map.py`

A HouseExpo plan is parsed into the `HouseMap` value object the rest of the env
raycasts and collision-tests against (PRD-HOUSEEXPO §5). We read the room
boundary polygon (`verts`: a closed contour of `[x, y]` vertices) into wall
**segments** between consecutive vertices, and the axis-aligned `bounds` from
the `bbox` (`{"min":[x,y],"max":[x,y]}`) when present, else from the vert
extent. Segment order is stabilised by sorting (PRD-HOUSEEXPO §5.1 determinism).
`is_inside` is an axis-aligned bounds check (the navigable interior is the
rectangle the bounds describe for these convex test plans).

- [ ] **Step 1: Write the failing test** — hand-computed: a 4×4 square room
  contour `(0,0)→(4,0)→(4,4)→(0,4)→(0,0)` yields 4 wall segments and bounds
  `(0,0,4,4)`; `is_inside` true at the centre `(2,2)`, false outside `(5,5)`;
  parsing twice is byte-identical (determinism).

```python
import json
from pathlib import Path

import pytest

from src.env.house_map import HouseMap, Segment, load_house_map


def _write_square(tmp_path: Path) -> str:
    plan = {
        "verts": [[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]],
        "bbox": {"min": [0, 0], "max": [4, 4]},
    }
    p = tmp_path / "room_single.json"
    p.write_text(json.dumps(plan))
    return str(p)


def test_load_square_room_walls_and_bounds(tmp_path):
    hm = load_house_map(_write_square(tmp_path))
    assert isinstance(hm, HouseMap)
    assert hm.bounds == (0.0, 0.0, 4.0, 4.0)
    assert len(hm.walls) == 4
    for seg in hm.walls:
        assert isinstance(seg, tuple)
        assert len(seg) == 4
    # the four square edges (order-independent set membership)
    edges = {
        (0.0, 0.0, 4.0, 0.0),
        (4.0, 0.0, 4.0, 4.0),
        (0.0, 4.0, 4.0, 4.0),
        (0.0, 0.0, 0.0, 4.0),
    }
    got = {tuple(sorted((s[0], s[2])) + sorted((s[1], s[3]))) for s in hm.walls}
    want = {tuple(sorted((e[0], e[2])) + sorted((e[1], e[3]))) for e in edges}
    assert got == want


def test_bounds_from_verts_when_bbox_absent(tmp_path):
    plan = {"verts": [[1, 2], [5, 2], [5, 8], [1, 8], [1, 2]]}
    p = tmp_path / "apt.json"
    p.write_text(json.dumps(plan))
    hm = load_house_map(str(p))
    assert hm.bounds == (1.0, 2.0, 5.0, 8.0)


def test_is_inside(tmp_path):
    hm = load_house_map(_write_square(tmp_path))
    assert hm.is_inside(2.0, 2.0) is True
    assert hm.is_inside(5.0, 5.0) is False
    assert hm.is_inside(-0.1, 2.0) is False


def test_parse_is_deterministic(tmp_path):
    path = _write_square(tmp_path)
    a = load_house_map(path)
    b = load_house_map(path)
    assert a.walls == b.walls
    assert a.bounds == b.bounds


def test_segment_alias_is_tuple_type():
    assert Segment is tuple
```

- [ ] **Step 2: Run test to verify it fails** —
  `uv run pytest tests/unit/test_house_map.py -v`
  Expected: FAIL (`ModuleNotFoundError: No module named 'src.env.house_map'`).

- [ ] **Step 3: Write minimal implementation** —

```python
"""HouseExpo JSON floor-plan -> wall segments + free-space bounds (PRD-HOUSEEXPO §5)."""
from __future__ import annotations

import json
from dataclasses import dataclass

Segment = tuple  # (x1, y1, x2, y2)


@dataclass
class HouseMap:
    """Value object: wall segments + axis-aligned free-space bounds (metres)."""

    walls: list[Segment]
    bounds: tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax)

    def is_inside(self, x: float, y: float) -> bool:
        """True iff (x, y) lies within the axis-aligned free-space bounds."""
        xmin, ymin, xmax, ymax = self.bounds
        return (xmin <= x <= xmax) and (ymin <= y <= ymax)


def _verts(plan: dict) -> list[tuple[float, float]]:
    raw = plan.get("verts") or plan.get("vertices") or []
    return [(float(p[0]), float(p[1])) for p in raw]


def _bounds(plan: dict, verts: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    bbox = plan.get("bbox")
    if bbox and "min" in bbox and "max" in bbox:
        lo, hi = bbox["min"], bbox["max"]
        return (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    return (min(xs), min(ys), max(xs), max(ys))


def load_house_map(path: str) -> HouseMap:
    """Parse one HouseExpo JSON plan into a deterministic HouseMap (PRD-HOUSEEXPO §5.1)."""
    with open(path, encoding="utf-8") as fh:
        plan = json.load(fh)
    verts = _verts(plan)
    walls: list[Segment] = []
    for i in range(len(verts) - 1):
        x1, y1 = verts[i]
        x2, y2 = verts[i + 1]
        if (x1, y1) != (x2, y2):
            walls.append((x1, y1, x2, y2))
    walls.sort()  # stable order -> deterministic parse (§5.1)
    return HouseMap(walls=walls, bounds=_bounds(plan, verts))
```

- [ ] **Step 4: Run test to verify it passes** —
  `uv run pytest tests/unit/test_house_map.py -v`
  Expected: PASS (5 tests green).

- [ ] **Step 5: Commit** —
  `git add src/env/house_map.py tests/unit/test_house_map.py && git commit -m "feat(env): house_map loader + is_inside (PRD-SIM §3, PRD-HOUSEEXPO §5; TDD)"`

---

### Task 2: `raycast` — `cast_ray` then `cast_lidar`

**Files:**
- Create `src/env/raycast.py`
- Test `tests/unit/test_raycast.py`

Ray–segment intersection via the 2-D cross-product form (PRD-SIM §3.2). A single
ray returns the nearest hit distance `t` along `angle`, capped at `max_range`;
parallel/collinear walls (`rxs == 0`) are no-hits. `cast_lidar` fires `n_rays`
rays evenly spaced over `2π` relative to `theta`, returning raw distances
(not normalized) as `np.ndarray` shape `(n_rays,)`.

- [ ] **Step 1: Write the failing test** — hand-computed (PRD-SIM R-1..R-4):
  east ray in a square room of half-width 2 hits at `2.0`; out-of-range wall
  clamps to `max_range`; oblique `π/4` ray hits wall `x=3` at `3√2 ≈ 4.2426`;
  parallel wall is a no-hit (clamps to `max_range`); lidar shape `(16,)` with
  ray 0 (φ=θ) matching `cast_ray` at `theta`.

```python
import math

import numpy as np

from src.env.raycast import cast_lidar, cast_ray

RAY_MAX = 5.0


def test_east_ray_hits_wall_at_two():
    # square room half-width 2 centred at origin; east wall x=2 from (2,-2)->(2,2)
    walls = [(2.0, -2.0, 2.0, 2.0)]
    d = cast_ray(0.0, 0.0, 0.0, walls, RAY_MAX)
    assert math.isclose(d, 2.0, abs_tol=1e-9)


def test_no_wall_within_range_clamps_to_max():
    walls = [(6.0, -1.0, 6.0, 1.0)]  # x=6 > ray_max=5
    d = cast_ray(0.0, 0.0, 0.0, walls, RAY_MAX)
    assert math.isclose(d, RAY_MAX, abs_tol=1e-9)


def test_oblique_hit_at_three_root_two():
    walls = [(3.0, 0.0, 3.0, 4.0)]
    d = cast_ray(0.0, 0.0, math.pi / 4, walls, RAY_MAX)
    assert math.isclose(d, 3.0 * math.sqrt(2.0), abs_tol=1e-6)  # 4.2426


def test_parallel_wall_is_no_hit():
    # ray along +x; wall also horizontal (parallel) -> rxs == 0 -> clamp
    walls = [(1.0, 1.0, 4.0, 1.0)]
    d = cast_ray(0.0, 0.0, 0.0, walls, RAY_MAX)
    assert math.isclose(d, RAY_MAX, abs_tol=1e-9)


def test_nearest_of_two_walls():
    walls = [(4.0, -2.0, 4.0, 2.0), (2.0, -2.0, 2.0, 2.0)]
    d = cast_ray(0.0, 0.0, 0.0, walls, RAY_MAX)
    assert math.isclose(d, 2.0, abs_tol=1e-9)


def test_cast_lidar_shape_and_ray_zero_matches_cast_ray():
    walls = [(2.0, -2.0, 2.0, 2.0)]
    theta = 0.0
    out = cast_lidar(0.0, 0.0, theta, 16, walls, RAY_MAX)
    assert isinstance(out, np.ndarray)
    assert out.shape == (16,)
    # ray 0 points along phi = theta -> east -> 2.0
    assert math.isclose(out[0], 2.0, abs_tol=1e-9)
    # all distances within [0, ray_max]
    assert np.all(out >= 0.0) and np.all(out <= RAY_MAX + 1e-9)
```

- [ ] **Step 2: Run test to verify it fails** —
  `uv run pytest tests/unit/test_raycast.py -v`
  Expected: FAIL (`ModuleNotFoundError: No module named 'src.env.raycast'`).

- [ ] **Step 3: Write minimal implementation** —

```python
"""Lidar raycasting via ray-segment intersection (PRD-SIM §3.2)."""
from __future__ import annotations

import math

import numpy as np

Segment = tuple  # (x1, y1, x2, y2)


def _ray_segment_t(ox: float, oy: float, dx: float, dy: float, seg: Segment, max_range: float):
    """Return hit distance t in [0, max_range] for one segment, or None (cross-product form)."""
    ax, ay, bx, by = seg
    sx, sy = bx - ax, by - ay  # wall direction s = B - A
    rxs = dx * sy - dy * sx  # r x s
    if rxs == 0.0:  # parallel / collinear -> no hit
        return None
    qpx, qpy = ax - ox, ay - oy  # A - origin
    t = (qpx * sy - qpy * sx) / rxs
    u = (qpx * dy - qpy * dx) / rxs
    if 0.0 <= t <= max_range and 0.0 <= u <= 1.0:
        return t
    return None


def cast_ray(x: float, y: float, angle: float, walls: list[Segment], max_range: float) -> float:
    """Distance to the nearest wall along `angle`, capped at `max_range`."""
    dx, dy = math.cos(angle), math.sin(angle)
    best = max_range
    for seg in walls:
        t = _ray_segment_t(x, y, dx, dy, seg, max_range)
        if t is not None and t < best:
            best = t
    return best


def cast_lidar(
    x: float, y: float, theta: float, n_rays: int, walls: list[Segment], max_range: float
) -> np.ndarray:
    """`n_rays` raw distances, evenly spaced over 2π relative to `theta`; shape (n_rays,)."""
    out = np.empty(n_rays, dtype=np.float32)
    for i in range(n_rays):
        phi = theta + 2.0 * math.pi * i / n_rays
        out[i] = cast_ray(x, y, phi, walls, max_range)
    return out
```

- [ ] **Step 4: Run test to verify it passes** —
  `uv run pytest tests/unit/test_raycast.py -v`
  Expected: PASS (6 tests green).

- [ ] **Step 5: Commit** —
  `git add src/env/raycast.py tests/unit/test_raycast.py && git commit -m "feat(env): raycast cast_ray + cast_lidar (PRD-SIM §3.2 R-1..R-4; TDD)"`

---

### Task 3: `kinematics` — `step_unicycle`

**Files:**
- Create `src/env/kinematics.py`
- Test `tests/unit/test_kinematics.py`

Pure explicit-Euler unicycle integrator (PRD-SIM §3.1, FR-2): `v = throttle·v_max`,
`ω = steer·omega_max`, then `x += v·cosθ·dt`, `y += v·sinθ·dt`, `θ += ω·dt`,
with `θ` wrapped to `(−π, π]`.

- [ ] **Step 1: Write the failing test** — hand-computed (PRD-SIM K-1..K-4):
  throttle=1, steer=0 advances `v_max·dt = 0.05` along +x; pure rotation moves θ
  only; heading π/2 moves +y; θ wraps into `(−π, π]`.

```python
import math

from src.env.kinematics import step_unicycle

V_MAX, OMEGA_MAX, DT = 0.5, 1.5, 0.1


def test_pure_translation_east():
    x, y, th = step_unicycle((0.0, 0.0, 0.0), 1.0, 0.0, V_MAX, OMEGA_MAX, DT)
    assert math.isclose(x, 0.05, abs_tol=1e-12)  # v_max * dt
    assert math.isclose(y, 0.0, abs_tol=1e-12)
    assert math.isclose(th, 0.0, abs_tol=1e-12)


def test_pure_rotation():
    x, y, th = step_unicycle((0.0, 0.0, 0.0), 0.0, 1.0, V_MAX, OMEGA_MAX, DT)
    assert math.isclose(x, 0.0, abs_tol=1e-12)
    assert math.isclose(y, 0.0, abs_tol=1e-12)
    assert math.isclose(th, 0.15, abs_tol=1e-12)  # omega_max * dt


def test_translation_at_heading_north():
    x, y, th = step_unicycle((0.0, 0.0, math.pi / 2), 1.0, 0.0, V_MAX, OMEGA_MAX, DT)
    assert math.isclose(x, 0.0, abs_tol=1e-9)
    assert math.isclose(y, 0.05, abs_tol=1e-12)
    assert math.isclose(th, math.pi / 2, abs_tol=1e-12)


def test_theta_wraps_into_minus_pi_pi():
    # start near +pi, rotate positive until it crosses; result stays in (-pi, pi]
    th = 3.10
    for _ in range(10):
        _, _, th = step_unicycle((0.0, 0.0, th), 0.0, 1.0, V_MAX, OMEGA_MAX, DT)
    assert -math.pi < th <= math.pi


def test_negative_throttle_moves_backward():
    x, _, _ = step_unicycle((0.0, 0.0, 0.0), -1.0, 0.0, V_MAX, OMEGA_MAX, DT)
    assert math.isclose(x, -0.05, abs_tol=1e-12)
```

- [ ] **Step 2: Run test to verify it fails** —
  `uv run pytest tests/unit/test_kinematics.py -v`
  Expected: FAIL (`ModuleNotFoundError: No module named 'src.env.kinematics'`).

- [ ] **Step 3: Write minimal implementation** —

```python
"""Explicit-Euler unicycle pose integrator (PRD-SIM §3.1, ADR-002). Pure function."""
from __future__ import annotations

import math


def _wrap(theta: float) -> float:
    """Wrap angle into (−π, π]."""
    wrapped = (theta + math.pi) % (2.0 * math.pi) - math.pi
    if wrapped == -math.pi:  # keep the half-open (−π, π] convention
        wrapped = math.pi
    return wrapped


def step_unicycle(
    pose: tuple[float, float, float],
    throttle: float,
    steer: float,
    v_max: float,
    omega_max: float,
    dt: float,
) -> tuple[float, float, float]:
    """Integrate one timestep: returns new (x, y, theta); theta wrapped to (−π, π]."""
    x, y, theta = pose
    v = throttle * v_max
    omega = steer * omega_max
    x_new = x + v * math.cos(theta) * dt
    y_new = y + v * math.sin(theta) * dt
    theta_new = _wrap(theta + omega * dt)
    return (x_new, y_new, theta_new)
```

- [ ] **Step 4: Run test to verify it passes** —
  `uv run pytest tests/unit/test_kinematics.py -v`
  Expected: PASS (5 tests green).

- [ ] **Step 5: Commit** —
  `git add src/env/kinematics.py tests/unit/test_kinematics.py && git commit -m "feat(env): step_unicycle integrator (PRD-SIM §3.1 K-1..K-4; TDD)"`

---

### Task 4: `collision` — `collides`

**Files:**
- Create `src/env/collision.py`
- Test `tests/unit/test_collision.py`

Robot disc vs wall-segment test (PRD-SIM §3.4, FR-5): collision iff the
point-to-segment distance from the robot centre to any wall is `< robot_radius`.
The point-to-segment distance projects the centre onto the segment, clamps the
projection parameter to `[0, 1]`, and measures Euclidean distance to that point.

- [ ] **Step 1: Write the failing test** — hand-computed (PRD-SIM X-1, X-2):
  centre `(0,0)`, vertical wall at `x=0.10` → distance `0.10 < 0.17` → True;
  wall at `x=0.20` → `0.20 > 0.17` → False; clamped-endpoint case.

```python
from src.env.collision import collides

ROBOT_R = 0.17


def test_collision_when_wall_within_radius():
    walls = [(0.10, -1.0, 0.10, 1.0)]  # distance 0.10 < 0.17
    assert collides(0.0, 0.0, ROBOT_R, walls) is True


def test_no_collision_when_wall_outside_radius():
    walls = [(0.20, -1.0, 0.20, 1.0)]  # distance 0.20 > 0.17
    assert collides(0.0, 0.0, ROBOT_R, walls) is False


def test_endpoint_clamped_distance():
    # nearest point on the segment is its endpoint (0.0, 0.30); distance 0.30 > 0.17
    walls = [(0.0, 0.30, 1.0, 0.30)]
    assert collides(0.0, 0.0, ROBOT_R, walls) is False
    # move the wall endpoint closer: nearest point endpoint (0.0, 0.10) -> 0.10 < 0.17
    walls = [(0.0, 0.10, 1.0, 0.40)]
    assert collides(0.0, 0.0, ROBOT_R, walls) is True


def test_no_walls_means_no_collision():
    assert collides(0.0, 0.0, ROBOT_R, []) is False


def test_collision_against_nearest_of_many():
    walls = [(5.0, -1.0, 5.0, 1.0), (0.05, -1.0, 0.05, 1.0)]
    assert collides(0.0, 0.0, ROBOT_R, walls) is True
```

- [ ] **Step 2: Run test to verify it fails** —
  `uv run pytest tests/unit/test_collision.py -v`
  Expected: FAIL (`ModuleNotFoundError: No module named 'src.env.collision'`).

- [ ] **Step 3: Write minimal implementation** —

```python
"""Robot-disc vs wall-segment collision test (PRD-SIM §3.4, FR-5)."""
from __future__ import annotations

import math

Segment = tuple  # (x1, y1, x2, y2)


def _point_segment_distance(px: float, py: float, seg: Segment) -> float:
    """Shortest Euclidean distance from point (px, py) to segment seg."""
    ax, ay, bx, by = seg
    abx, aby = bx - ax, by - ay
    denom = abx * abx + aby * aby
    if denom == 0.0:  # degenerate segment = point
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * abx + (py - ay) * aby) / denom
    t = max(0.0, min(1.0, t))  # clamp projection to the segment
    cx, cy = ax + t * abx, ay + t * aby
    return math.hypot(px - cx, py - cy)


def collides(x: float, y: float, robot_radius: float, walls: list[Segment]) -> bool:
    """True iff the robot disc (centre (x, y), radius robot_radius) intersects any wall."""
    return any(_point_segment_distance(x, y, seg) < robot_radius for seg in walls)
```

- [ ] **Step 4: Run test to verify it passes** —
  `uv run pytest tests/unit/test_collision.py -v`
  Expected: PASS (5 tests green).

- [ ] **Step 5: Commit** —
  `git add src/env/collision.py tests/unit/test_collision.py && git commit -m "feat(env): collides disc-vs-segment test (PRD-SIM §3.4 X-1,X-2; TDD)"`

---

### Task 5: `coverage` — `CoverageGrid.mark/fraction/nearest_uncleaned_bearing/reset`

**Files:**
- Create `src/env/coverage.py`
- Test `tests/unit/test_coverage.py`

The free space is a square grid of edge `cell_size`; a cell is cleaned when its
**centre lies within `clean_radius`** of the robot centre (PRD-SIM §3.3, FR-4).
`mark` returns the count of NEWLY cleaned cells; `fraction` = cleaned / total
free cells; `nearest_uncleaned_bearing` returns the `(cos, sin)` of the bearing
to the nearest uncleaned cell **in the robot frame** (`(0.0, 0.0)` if fully
cleaned); `reset` clears all cleaned flags.

- [ ] **Step 1: Write the failing test** — hand-computed (PRD-SIM C-1..C-3):
  first `mark` from a fresh grid cleans `> 0` cells; an immediate re-mark at the
  same pose cleans `0`; `fraction` rises after a mark and returns to `0.0` after
  `reset`; the bearing unit vector has norm ≈ 1 while uncleaned and is `(0,0)`
  once the whole grid is cleaned.

```python
import math

from src.env.coverage import CoverageGrid

BOUNDS = (0.0, 0.0, 4.0, 4.0)
CELL = 0.10
CLEAN_R = 0.17


def test_first_mark_cleans_some_then_remark_cleans_none():
    grid = CoverageGrid(BOUNDS, CELL, CLEAN_R)
    first = grid.mark(2.0, 2.0)
    assert first > 0  # disc of radius 0.17 covers >=1 cell centre
    again = grid.mark(2.0, 2.0)  # C-2: no reward for revisiting
    assert again == 0


def test_fraction_rises_then_resets():
    grid = CoverageGrid(BOUNDS, CELL, CLEAN_R)
    assert grid.fraction() == 0.0
    grid.mark(2.0, 2.0)
    f = grid.fraction()
    assert 0.0 < f < 1.0  # a single disc never covers a 4x4 room
    grid.reset()  # C-3: cleared on reset
    assert grid.fraction() == 0.0


def test_nearest_uncleaned_bearing_is_unit_vector():
    grid = CoverageGrid(BOUNDS, CELL, CLEAN_R)
    grid.mark(2.0, 2.0)  # clean a patch near the centre
    cos_b, sin_b = grid.nearest_uncleaned_bearing(2.0, 2.0, 0.0)
    assert math.isclose(math.hypot(cos_b, sin_b), 1.0, abs_tol=1e-6)


def test_bearing_zero_when_fully_cleaned():
    # tiny room: one cell, clean it, then bearing is (0,0)
    grid = CoverageGrid((0.0, 0.0, 0.05, 0.05), CELL, CLEAN_R)
    grid.mark(0.0, 0.0)
    assert grid.fraction() == 1.0
    assert grid.nearest_uncleaned_bearing(0.0, 0.0, 0.0) == (0.0, 0.0)


def test_bearing_is_in_robot_frame():
    # nearest uncleaned cell straight ahead in world (+x); robot heading +x ->
    # robot-frame bearing ~ (1, 0). Heading rotated by pi/2 -> bearing ~ (0, -1).
    grid = CoverageGrid((0.0, 0.0, 1.0, 1.0), CELL, CLEAN_R)
    grid.mark(0.0, 0.5)  # clean near the left edge, leave +x side uncleaned
    cf, _ = grid.nearest_uncleaned_bearing(0.0, 0.5, 0.0)
    cf_rot, sf_rot = grid.nearest_uncleaned_bearing(0.0, 0.5, math.pi / 2)
    assert cf > 0.0  # uncleaned mass is ahead in the robot frame
    assert sf_rot < cf  # rotating the heading rotates the robot-frame bearing
```

- [ ] **Step 2: Run test to verify it fails** —
  `uv run pytest tests/unit/test_coverage.py -v`
  Expected: FAIL (`ModuleNotFoundError: No module named 'src.env.coverage'`).

- [ ] **Step 3: Write minimal implementation** —

```python
"""Cleaned-cell grid + coverage fraction + nearest-uncleaned bearing (PRD-SIM §3.3)."""
from __future__ import annotations

import math

import numpy as np


class CoverageGrid:
    """Boolean grid of cleanable cells over the map's free-space bounds."""

    def __init__(self, bounds, cell_size: float, clean_radius: float):
        self.xmin, self.ymin, xmax, ymax = bounds
        self.cell_size = cell_size
        self.clean_radius = clean_radius
        self.nx = max(1, int(round((xmax - self.xmin) / cell_size)))
        self.ny = max(1, int(round((ymax - self.ymin) / cell_size)))
        # cell centres
        self.cx = self.xmin + (np.arange(self.nx) + 0.5) * cell_size
        self.cy = self.ymin + (np.arange(self.ny) + 0.5) * cell_size
        self.cleaned = np.zeros((self.nx, self.ny), dtype=bool)

    def mark(self, x: float, y: float) -> int:
        """Mark cells whose centre is within clean_radius; return NEWLY cleaned count."""
        dx = self.cx[:, None] - x
        dy = self.cy[None, :] - y
        within = (dx * dx + dy * dy) <= (self.clean_radius * self.clean_radius)
        newly = within & ~self.cleaned
        count = int(newly.sum())
        self.cleaned |= newly
        return count

    def fraction(self) -> float:
        """Cleaned free-cells / total free-cells, in [0, 1]."""
        return float(self.cleaned.sum()) / float(self.cleaned.size)

    def nearest_uncleaned_bearing(self, x: float, y: float, theta: float) -> tuple[float, float]:
        """(cos, sin) of the bearing to the nearest uncleaned cell, in the robot frame."""
        unclean = ~self.cleaned
        if not unclean.any():
            return (0.0, 0.0)
        ix, iy = np.where(unclean)
        dx = self.cx[ix] - x
        dy = self.cy[iy] - y
        j = int(np.argmin(dx * dx + dy * dy))
        ang = math.atan2(dy[j], dx[j]) - theta  # rotate into robot frame
        return (math.cos(ang), math.sin(ang))

    def reset(self) -> None:
        """Clear all cleaned flags (per-episode grid)."""
        self.cleaned[:] = False
```

- [ ] **Step 4: Run test to verify it passes** —
  `uv run pytest tests/unit/test_coverage.py -v`
  Expected: PASS (5 tests green).

- [ ] **Step 5: Commit** —
  `git add src/env/coverage.py tests/unit/test_coverage.py && git commit -m "feat(env): CoverageGrid mark/fraction/bearing/reset (PRD-SIM §3.3 C-1..C-3; TDD)"`

---

### Task 6: `reward` — `compute_reward`

**Files:**
- Create `src/env/reward.py`
- Test `tests/unit/test_reward.py`

Pure reward function (PRD-SIM §3.5, FR-6):
`r = k_coverage·new_cells − k_collision·collision − k_step`. The only positive
term is coverage; collision and per-step time are penalties (sign flips with a
collision).

- [ ] **Step 1: Write the failing test** — hand-computed (PRD-SIM W-1, W-2):
  `new_cells=3, collision=False` → `2.99`; `new_cells=0, collision=True` →
  `−10.01`; the reward sign flips when a collision is introduced.

```python
import math

from src.env.reward import compute_reward

K_COV, K_COL, K_STEP = 1.0, 10.0, 0.01


def test_positive_reward_on_coverage():
    r = compute_reward(3, False, K_COV, K_COL, K_STEP)
    assert math.isclose(r, 2.99, abs_tol=1e-9)  # 1.0*3 - 0 - 0.01


def test_negative_reward_on_collision():
    r = compute_reward(0, True, K_COV, K_COL, K_STEP)
    assert math.isclose(r, -10.01, abs_tol=1e-9)  # 0 - 10.0 - 0.01


def test_sign_flips_with_collision():
    clean = compute_reward(3, False, K_COV, K_COL, K_STEP)
    crash = compute_reward(3, True, K_COV, K_COL, K_STEP)
    assert clean > 0.0
    assert crash < 0.0  # 3 - 10 - 0.01 = -7.01
    assert math.isclose(crash, -7.01, abs_tol=1e-9)


def test_idle_step_cost_only():
    r = compute_reward(0, False, K_COV, K_COL, K_STEP)
    assert math.isclose(r, -0.01, abs_tol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails** —
  `uv run pytest tests/unit/test_reward.py -v`
  Expected: FAIL (`ModuleNotFoundError: No module named 'src.env.reward'`).

- [ ] **Step 3: Write minimal implementation** —

```python
"""Reward shaping: r = k_coverage*new_cells − k_collision*collision − k_step (PRD-SIM §3.5)."""
from __future__ import annotations


def compute_reward(
    new_cells: int,
    collision: bool,
    k_coverage: float,
    k_collision: float,
    k_step: float,
) -> float:
    """r = k_coverage·new_cells − k_collision·collision − k_step (signs per FR-6)."""
    return k_coverage * new_cells - k_collision * float(collision) - k_step
```

- [ ] **Step 4: Run test to verify it passes** —
  `uv run pytest tests/unit/test_reward.py -v`
  Expected: PASS (4 tests green).

- [ ] **Step 5: Commit** —
  `git add src/env/reward.py tests/unit/test_reward.py && git commit -m "feat(env): compute_reward shaping (PRD-SIM §3.5 W-1,W-2; TDD)"`

---

### Task 7: `state` — `assemble_state` (shape (20,) at n_rays=16, all normalized)

**Files:**
- Create `src/env/state.py`
- Test `tests/unit/test_state.py`

Observation assembly (spec §3, PRD-SIM §2.2): normalized float32 vector of shape
`(n_rays + 4,)` = `lidar/ray_max` ⊕ `v/v_max` ⊕ `omega/omega_max` ⊕
`heading_cos` ⊕ `heading_sin`. At `n_rays=16` → `(20,)`. Lidar components land in
`[0, 1]`; `(v, ω)` in `[−1, 1]`; heading cue is a passed-through unit vector.

- [ ] **Step 1: Write the failing test** — hand-computed (PRD-SIM W-3): 16 rays
  at `ray_max=5` → shape `(20,)`, dtype float32; a lidar of all `5.0`
  normalizes to all `1.0`; `v=v_max → 1.0`, `omega=omega_max → 1.0`; heading
  components copied verbatim; ray slice ∈ [0,1], (v,ω) ∈ [−1,1].

```python
import numpy as np

from src.env.state import assemble_state

RAY_MAX, V_MAX, OMEGA_MAX = 5.0, 0.5, 1.5


def test_state_dim_is_20_at_16_rays():
    lidar = np.full(16, 5.0, dtype=np.float32)
    s = assemble_state(lidar, 0.0, 0.0, 1.0, 0.0, RAY_MAX, V_MAX, OMEGA_MAX)
    assert isinstance(s, np.ndarray)
    assert s.shape == (20,)
    assert s.dtype == np.float32


def test_components_are_normalized():
    lidar = np.full(16, 5.0, dtype=np.float32)  # at max range -> 1.0
    s = assemble_state(lidar, V_MAX, OMEGA_MAX, 0.6, 0.8, RAY_MAX, V_MAX, OMEGA_MAX)
    assert np.allclose(s[:16], 1.0)          # lidar / ray_max
    assert np.isclose(s[16], 1.0)            # v / v_max
    assert np.isclose(s[17], 1.0)            # omega / omega_max
    assert np.isclose(s[18], 0.6)            # heading_cos passthrough
    assert np.isclose(s[19], 0.8)            # heading_sin passthrough


def test_ray_slice_in_unit_interval_and_speed_in_signed_unit():
    lidar = np.array([0.0, 2.5, 5.0] + [1.0] * 13, dtype=np.float32)
    s = assemble_state(lidar, -V_MAX, -OMEGA_MAX, 1.0, 0.0, RAY_MAX, V_MAX, OMEGA_MAX)
    assert np.all(s[:16] >= 0.0) and np.all(s[:16] <= 1.0)
    assert np.isclose(s[0], 0.0) and np.isclose(s[1], 0.5)
    assert np.isclose(s[16], -1.0) and np.isclose(s[17], -1.0)  # signed unit


def test_dim_scales_with_n_rays():
    lidar = np.full(8, 1.0, dtype=np.float32)
    s = assemble_state(lidar, 0.0, 0.0, 1.0, 0.0, RAY_MAX, V_MAX, OMEGA_MAX)
    assert s.shape == (12,)  # n_rays + 4
```

- [ ] **Step 2: Run test to verify it fails** —
  `uv run pytest tests/unit/test_state.py -v`
  Expected: FAIL (`ModuleNotFoundError: No module named 'src.env.state'`).

- [ ] **Step 3: Write minimal implementation** —

```python
"""Observation assembly: normalized lidar ⊕ (v, ω) ⊕ heading cue (spec §3, PRD-SIM §2.2)."""
from __future__ import annotations

import numpy as np


def assemble_state(
    lidar: np.ndarray,
    v: float,
    omega: float,
    heading_cos: float,
    heading_sin: float,
    ray_max: float,
    v_max: float,
    omega_max: float,
) -> np.ndarray:
    """Normalized float32 state, shape (n_rays + 4,): lidar/ray_max ⊕ v/v_max ⊕ ω/ω_max ⊕ cos ⊕ sin."""
    rays = np.asarray(lidar, dtype=np.float32) / ray_max
    tail = np.array(
        [v / v_max, omega / omega_max, heading_cos, heading_sin],
        dtype=np.float32,
    )
    return np.concatenate([rays, tail]).astype(np.float32)
```

- [ ] **Step 4: Run test to verify it passes** —
  `uv run pytest tests/unit/test_state.py -v`
  Expected: PASS (4 tests green).

- [ ] **Step 5: Commit** —
  `git add src/env/state.py tests/unit/test_state.py && git commit -m "feat(env): assemble_state normalized 20-dim observation (spec §3, PRD-SIM W-3; TDD)"`

---

### Task 8: `vacuum_env` — `VacuumEnv.reset/step` (4-tuple)

**Files:**
- Create `src/env/vacuum_env.py`
- Test `tests/unit/test_vacuum_env.py`

`VacuumEnv` orchestrates the seven modules above (PRD-SIM §3.6, FR-7..FR-9). It
takes a `HouseMap` + the config dict + seed; builds a `CoverageGrid`; sets
`action_dim = 2` and `state_dim = n_rays + 4`. `reset()` re-spawns at a random
free cell, clears coverage, zeroes `(v, ω)`, returns the initial state.
`step(action)` clips action to `[−1, 1]`, integrates kinematics, on collision
reverts the move + sets `collision=True`, marks coverage, builds reward + next
state, and returns the **4-tuple** `(np.ndarray, float, bool, dict)` with
`info = {"coverage", "collision", "pose"}`; `done` at `max_steps` or coverage
target. This test builds the config dict inline (Phase 1 does not depend on
`config_loader`).

- [ ] **Step 1: Write the failing test** — hand-computed: `reset()` returns a
  state of shape `(20,)`; `step` returns exactly `(np.ndarray, float, bool, dict)`
  with `info` keys `{"coverage","collision","pose"}`; an out-of-range action is
  clipped (does not crash); a step driving into a near wall reports
  `collision=True` and the pose reverts (no tunneling); `done` becomes `True` at
  `max_steps`; same `(map, seed)` reset is deterministic.

```python
import numpy as np

from src.env.house_map import HouseMap
from src.env.vacuum_env import VacuumEnv

# 4x4 square room; walls on the four edges
WALLS = [
    (0.0, 0.0, 4.0, 0.0),
    (4.0, 0.0, 4.0, 4.0),
    (0.0, 4.0, 4.0, 4.0),
    (0.0, 0.0, 0.0, 4.0),
]
HMAP = HouseMap(walls=WALLS, bounds=(0.0, 0.0, 4.0, 4.0))

CFG = {
    "env": {
        "n_rays": 16,
        "ray_max": 5.0,
        "dt": 0.1,
        "v_max": 0.5,
        "omega_max": 1.5,
        "robot_radius": 0.17,
        "clean_radius": 0.17,
        "coverage_cell": 0.10,
        "max_steps": 1000,
    },
    "reward": {"k_coverage": 1.0, "k_collision": 10.0, "k_step": 0.01},
}


def test_dims_set_from_config():
    env = VacuumEnv(HMAP, CFG, seed=42)
    assert env.action_dim == 2
    assert env.state_dim == 20  # n_rays + 4


def test_reset_returns_state_vector():
    env = VacuumEnv(HMAP, CFG, seed=42)
    s = env.reset()
    assert isinstance(s, np.ndarray)
    assert s.shape == (20,)


def test_step_returns_four_tuple():
    env = VacuumEnv(HMAP, CFG, seed=42)
    env.reset()
    out = env.step(np.array([1.0, 0.0], dtype=np.float32))
    assert isinstance(out, tuple) and len(out) == 4
    s, r, done, info = out
    assert isinstance(s, np.ndarray) and s.shape == (20,)
    assert isinstance(r, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert set(info.keys()) == {"coverage", "collision", "pose"}


def test_out_of_range_action_is_clipped():
    env = VacuumEnv(HMAP, CFG, seed=42)
    env.reset()
    # action well outside [-1, 1] must not crash and must integrate as clipped
    s, r, done, info = env.step(np.array([5.0, -9.0], dtype=np.float32))
    assert np.isfinite(r)
    assert np.all(np.isfinite(s))


def test_collision_reverts_pose():
    env = VacuumEnv(HMAP, CFG, seed=1)
    env.reset()
    # force the robot hard against the left wall, heading west, then drive into it
    env.pose = (0.18, 2.0, np.pi)  # just inside the left wall (x=0), radius 0.17
    before = env.pose
    s, r, done, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
    assert info["collision"] is True
    assert info["pose"] == before  # move rejected -> no tunneling
    assert r < 0.0  # collision penalty dominates


def test_done_at_max_steps():
    cfg = {**CFG, "env": {**CFG["env"], "max_steps": 2}}
    env = VacuumEnv(HMAP, cfg, seed=42)
    env.reset()
    _, _, d1, _ = env.step(np.array([0.0, 1.0], dtype=np.float32))
    _, _, d2, _ = env.step(np.array([0.0, 1.0], dtype=np.float32))
    assert d1 is False
    assert d2 is True  # step_count == max_steps


def test_reset_is_deterministic_for_same_seed():
    a = VacuumEnv(HMAP, CFG, seed=7).reset()
    b = VacuumEnv(HMAP, CFG, seed=7).reset()
    assert np.array_equal(a, b)
```

- [ ] **Step 2: Run test to verify it fails** —
  `uv run pytest tests/unit/test_vacuum_env.py -v`
  Expected: FAIL (`ModuleNotFoundError: No module named 'src.env.vacuum_env'`).

- [ ] **Step 3: Write minimal implementation** —

```python
"""VacuumEnv: from-scratch 2D vacuum MDP, 4-tuple step, NO Gymnasium (PRD-SIM §3.6)."""
from __future__ import annotations

import math

import numpy as np

from src.env.collision import collides
from src.env.coverage import CoverageGrid
from src.env.house_map import HouseMap
from src.env.kinematics import step_unicycle
from src.env.raycast import cast_lidar
from src.env.reward import compute_reward
from src.env.state import assemble_state


class VacuumEnv:
    """Custom (non-Gym) robotic-vacuum environment over a HouseMap."""

    def __init__(self, house_map: HouseMap, cfg: dict, seed: int | None = None):
        self.house_map = house_map
        self.cfg = cfg
        self.e = cfg["env"]
        self.r = cfg["reward"]
        self.rng = np.random.default_rng(seed)
        self.coverage = CoverageGrid(
            house_map.bounds, self.e["coverage_cell"], self.e["clean_radius"]
        )
        self.action_dim = 2
        self.state_dim = self.e["n_rays"] + 4
        self.pose = (0.0, 0.0, 0.0)
        self.v = 0.0
        self.omega = 0.0
        self.step_count = 0

    def _spawn(self) -> tuple[float, float, float]:
        xmin, ymin, xmax, ymax = self.house_map.bounds
        pad = self.e["robot_radius"]
        for _ in range(100):
            x = float(self.rng.uniform(xmin + pad, xmax - pad))
            y = float(self.rng.uniform(ymin + pad, ymax - pad))
            if not collides(x, y, self.e["robot_radius"], self.house_map.walls):
                return (x, y, float(self.rng.uniform(-math.pi, math.pi)))
        return ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0, 0.0)

    def _state(self) -> np.ndarray:
        x, y, theta = self.pose
        lidar = cast_lidar(x, y, theta, self.e["n_rays"], self.house_map.walls, self.e["ray_max"])
        cos_b, sin_b = self.coverage.nearest_uncleaned_bearing(x, y, theta)
        return assemble_state(
            lidar, self.v, self.omega, cos_b, sin_b,
            self.e["ray_max"], self.e["v_max"], self.e["omega_max"],
        )

    def reset(self) -> np.ndarray:
        self.coverage.reset()
        self.pose = self._spawn()
        self.v = 0.0
        self.omega = 0.0
        self.step_count = 0
        self.coverage.mark(self.pose[0], self.pose[1])
        return self._state()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        throttle, steer = float(a[0]), float(a[1])
        cand = step_unicycle(
            self.pose, throttle, steer,
            self.e["v_max"], self.e["omega_max"], self.e["dt"],
        )
        collision = collides(cand[0], cand[1], self.e["robot_radius"], self.house_map.walls)
        if collision:
            self.v = 0.0
            self.omega = 0.0
        else:
            self.pose = cand
            self.v = throttle * self.e["v_max"]
            self.omega = steer * self.e["omega_max"]
        new_cells = self.coverage.mark(self.pose[0], self.pose[1])
        reward = compute_reward(
            new_cells, collision,
            self.r["k_coverage"], self.r["k_collision"], self.r["k_step"],
        )
        self.step_count += 1
        done = (self.step_count >= self.e["max_steps"]) or (self.coverage.fraction() >= 1.0)
        info = {"coverage": self.coverage.fraction(), "collision": collision, "pose": self.pose}
        return self._state(), float(reward), bool(done), info
```

- [ ] **Step 4: Run test to verify it passes** —
  `uv run pytest tests/unit/test_vacuum_env.py -v`
  Expected: PASS (7 tests green).

- [ ] **Step 5: Commit** —
  `git add src/env/vacuum_env.py tests/unit/test_vacuum_env.py && git commit -m "feat(env): VacuumEnv reset/step 4-tuple orchestration (PRD-SIM §3.6 W-3,W-4; TDD)"`

---

### Phase 1 exit gate

After Task 8, run the full deterministic-core gate (PRD-SIM §5.1, CLAUDE.md §2/§6/§7):

```bash
uv run pytest tests/unit/ -v --cov=src/env --cov-report=term-missing
uv run ruff check src/env tests/unit
```

Expected: all Phase-1 unit tests PASS, `src/env/` coverage ≥85%, zero Ruff
violations, every `src/env/*.py` ≤150 LOC. No `gymnasium`/`gazebo`/SB3 imports
introduced (verified by the architecture test in a later phase).

---

## Phase 2 — DDPG from scratch (Actor · Critic · Replay · Noise · Agent)

This phase implements the learning core exactly per the contract
(`docs/superpowers/plans/_contract.md` §model, §ddpg) and `PRD-DDPG.md`
checkpoints 1–2 and acceptance tests §5.1–§5.4. Each `.py` ≤150 LOC, TDD
RED→GREEN→REFACTOR, `uv` only, no `gymnasium`/SB3/RLlib. All hyperparameters
flow from `config/config.yaml` (`ddpg.*`, `noise.*`) — no literals in `src/`.

Assumes Phase 1 delivered `src/utils/config_loader.py` (`load_config`, `get`).
Tasks 1–4 (actor, critic, replay_buffer, noise) are independent; Task 5 (agent)
depends on all four.

---

### Task 1: Actor network (Tanh-bounded deterministic policy)
**Files:**
- Create: `src/model/actor.py`
- Test: `tests/unit/test_actor.py`

- [ ] **Step 1: Write the failing test** — `tests/unit/test_actor.py`:

```python
import torch

from src.model.actor import Actor


def test_actor_forward_shape_and_bounds():
    torch.manual_seed(0)
    state_dim, action_dim = 20, 2
    actor = Actor(state_dim, action_dim, [256, 256])
    batch = 64
    state = torch.randn(batch, state_dim)
    action = actor.forward(state)
    assert action.shape == (batch, action_dim)
    assert torch.all(action >= -1.0)
    assert torch.all(action <= 1.0)


def test_actor_adversarial_preactivation_stays_bounded():
    # Huge inputs must still saturate within [-1, 1] via tanh (PRD-DDPG §5.2).
    actor = Actor(20, 2, [256, 256])
    state = torch.full((1000, 20), 1.0e6)
    action = actor.forward(state)
    assert torch.all(action >= -1.0)
    assert torch.all(action <= 1.0)
    assert action.shape == (1000, 2)


def test_actor_hidden_sizes_respected():
    actor = Actor(20, 2, [256, 256])
    linears = [m for m in actor.modules() if isinstance(m, torch.nn.Linear)]
    assert linears[0].in_features == 20
    assert linears[0].out_features == 256
    assert linears[-1].out_features == 2
```

- [ ] **Step 2: Run test to verify it fails** —
  `uv run pytest tests/unit/test_actor.py -v`
  Expected: FAIL (`ModuleNotFoundError: No module named 'src.model.actor'`).

- [ ] **Step 3: Write minimal implementation** — `src/model/actor.py`:

```python
"""Actor MLP: state -> Tanh-bounded deterministic action in (-1, 1).

PRD-DDPG checkpoint 1 (Actor-Critic). Tanh guarantees the action stays in
[-1, 1] (acceptance test §5.2). No hardcoded sizes: hidden_sizes is injected.
"""
from __future__ import annotations

import torch
from torch import nn


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        in_features = state_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden))
            layers.append(nn.ReLU())
            in_features = hidden
        layers.append(nn.Linear(in_features, action_dim))
        self.body = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.body(state))
```

- [ ] **Step 4: Run test to verify it passes** —
  `uv run pytest tests/unit/test_actor.py -v`
  Expected: PASS (3 tests green).

- [ ] **Step 5: Commit** —
  `git add src/model/actor.py tests/unit/test_actor.py && git commit -m "feat(model): Actor MLP with Tanh-bounded action (PRD-DDPG checkpoint 1, §5.2)"`

---

### Task 2: Critic network (state ⊕ action → scalar Q)
**Files:**
- Create: `src/model/critic.py`
- Test: `tests/unit/test_critic.py`

- [ ] **Step 1: Write the failing test** — `tests/unit/test_critic.py`:

```python
import torch

from src.model.critic import Critic


def test_critic_forward_shape():
    torch.manual_seed(0)
    state_dim, action_dim = 20, 2
    critic = Critic(state_dim, action_dim, [256, 256])
    batch = 32
    state = torch.randn(batch, state_dim)
    action = torch.randn(batch, action_dim)
    q = critic.forward(state, action)
    assert q.shape == (batch, 1)
    assert torch.all(torch.isfinite(q))


def test_critic_first_layer_concatenates_state_and_action():
    # in_features of the first linear must be state_dim + action_dim (§5.3, F5.16).
    critic = Critic(20, 2, [256, 256])
    first = next(m for m in critic.modules() if isinstance(m, torch.nn.Linear))
    assert first.in_features == 22
    last = [m for m in critic.modules() if isinstance(m, torch.nn.Linear)][-1]
    assert last.out_features == 1
```

- [ ] **Step 2: Run test to verify it fails** —
  `uv run pytest tests/unit/test_critic.py -v`
  Expected: FAIL (`ModuleNotFoundError: No module named 'src.model.critic'`).

- [ ] **Step 3: Write minimal implementation** — `src/model/critic.py`:

```python
"""Critic MLP: (state, action) -> scalar Q. PRD-DDPG checkpoint 1, §5.3.

State and action are concatenated at the first layer (F5.16), so the first
linear has in_features == state_dim + action_dim; output is shape (B, 1).
"""
from __future__ import annotations

import torch
from torch import nn


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        in_features = state_dim + action_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden))
            layers.append(nn.ReLU())
            in_features = hidden
        layers.append(nn.Linear(in_features, 1))
        self.body = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.body(torch.cat([state, action], dim=1))
```

- [ ] **Step 4: Run test to verify it passes** —
  `uv run pytest tests/unit/test_critic.py -v`
  Expected: PASS (2 tests green).

- [ ] **Step 5: Commit** —
  `git add src/model/critic.py tests/unit/test_critic.py && git commit -m "feat(model): Critic MLP (state+action -> Q), §5.3 / F5.16"`

---

### Task 3: Replay buffer (uniform experience replay)
**Files:**
- Create: `src/ddpg/replay_buffer.py`
- Test: `tests/unit/test_replay_buffer.py`

- [ ] **Step 1: Write the failing test** — `tests/unit/test_replay_buffer.py`:

```python
import numpy as np
import torch

from src.ddpg.replay_buffer import ReplayBuffer


def _dummy(state_dim, action_dim):
    s = np.ones(state_dim, dtype=np.float32)
    a = np.full(action_dim, 0.5, dtype=np.float32)
    s2 = np.zeros(state_dim, dtype=np.float32)
    return s, a, s2


def test_len_grows_then_caps_at_capacity():
    buf = ReplayBuffer(capacity=3, state_dim=20, action_dim=2, seed=0)
    assert len(buf) == 0
    s, a, s2 = _dummy(20, 2)
    for _ in range(5):
        buf.add(s, a, 1.0, s2, False)
    assert len(buf) == 3  # capped


def test_sample_returns_five_float32_tensors_with_right_shapes():
    state_dim, action_dim, batch = 20, 2, 8
    buf = ReplayBuffer(capacity=100, state_dim=state_dim, action_dim=action_dim, seed=1)
    s, a, s2 = _dummy(state_dim, action_dim)
    for i in range(50):
        buf.add(s, a, float(i), s2, bool(i % 2))
    out = buf.sample(batch)
    assert len(out) == 5
    bs, ba, br, bs2, bd = out
    for t in out:
        assert isinstance(t, torch.Tensor)
        assert t.dtype == torch.float32
    assert bs.shape == (batch, state_dim)
    assert ba.shape == (batch, action_dim)
    assert br.shape == (batch, 1)
    assert bs2.shape == (batch, state_dim)
    assert bd.shape == (batch, 1)


def test_same_seed_same_sample_indices():
    state_dim, action_dim = 20, 2
    s, a, s2 = _dummy(state_dim, action_dim)
    b1 = ReplayBuffer(capacity=100, state_dim=state_dim, action_dim=action_dim, seed=7)
    b2 = ReplayBuffer(capacity=100, state_dim=state_dim, action_dim=action_dim, seed=7)
    for i in range(50):
        b1.add(s, a, float(i), s2, False)
        b2.add(s, a, float(i), s2, False)
    r1 = b1.sample(8)[2]
    r2 = b2.sample(8)[2]
    assert torch.equal(r1, r2)
```

- [ ] **Step 2: Run test to verify it fails** —
  `uv run pytest tests/unit/test_replay_buffer.py -v`
  Expected: FAIL (`ModuleNotFoundError: No module named 'src.ddpg.replay_buffer'`).

- [ ] **Step 3: Write minimal implementation** — `src/ddpg/replay_buffer.py`:

```python
"""Uniform experience replay (PRD-DDPG F5.6). Pre-allocated float32 ring buffer.

sample() returns (s, a, r, s2, done) as float32 tensors; r and done are (B, 1).
A seeded numpy Generator makes index draws reproducible (acceptance §5.4).
"""
from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int, seed: int | None = None):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._idx = 0
        self._size = 0

    def add(self, s: np.ndarray, a: np.ndarray, r: float, s2: np.ndarray, done: bool) -> None:
        i = self._idx
        self.states[i] = s
        self.actions[i] = a
        self.rewards[i, 0] = r
        self.next_states[i] = s2
        self.dones[i, 0] = float(done)
        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        idx = self._rng.integers(0, self._size, size=batch_size)
        return (
            torch.as_tensor(self.states[idx], dtype=torch.float32),
            torch.as_tensor(self.actions[idx], dtype=torch.float32),
            torch.as_tensor(self.rewards[idx], dtype=torch.float32),
            torch.as_tensor(self.next_states[idx], dtype=torch.float32),
            torch.as_tensor(self.dones[idx], dtype=torch.float32),
        )

    def __len__(self) -> int:
        return self._size
```

- [ ] **Step 4: Run test to verify it passes** —
  `uv run pytest tests/unit/test_replay_buffer.py -v`
  Expected: PASS (3 tests green).

- [ ] **Step 5: Commit** —
  `git add src/ddpg/replay_buffer.py tests/unit/test_replay_buffer.py && git commit -m "feat(ddpg): uniform ReplayBuffer with seeded sampling (F5.6, §5.4)"`

---

### Task 4: Gaussian exploration noise (σ-schedule)
**Files:**
- Create: `src/ddpg/noise.py`
- Test: `tests/unit/test_noise.py`

- [ ] **Step 1: Write the failing test** — `tests/unit/test_noise.py`:

```python
import numpy as np

from src.ddpg.noise import GaussianNoise


def test_sample_shape_is_action_dim():
    noise = GaussianNoise(action_dim=2, sigma_start=0.2, sigma_end=0.05, decay_steps=50000, seed=0)
    sample = noise.sample()
    assert sample.shape == (2,)


def test_decay_moves_sigma_from_start_toward_end_linearly_and_monotonic():
    start, end, steps = 0.2, 0.05, 100
    noise = GaussianNoise(action_dim=2, sigma_start=start, sigma_end=end, decay_steps=steps, seed=0)
    assert noise.sigma == start
    prev = noise.sigma
    for k in range(1, steps + 1):
        noise.decay()
        assert noise.sigma <= prev + 1e-12  # monotonic non-increasing
        expected = start + (end - start) * min(k / steps, 1.0)
        assert abs(noise.sigma - expected) < 1e-9  # linear
        prev = noise.sigma
    assert abs(noise.sigma - end) < 1e-9
    noise.decay()  # clamp past horizon
    assert abs(noise.sigma - end) < 1e-9


def test_same_seed_same_samples():
    n1 = GaussianNoise(action_dim=2, sigma_start=0.2, sigma_end=0.05, decay_steps=50000, seed=42)
    n2 = GaussianNoise(action_dim=2, sigma_start=0.2, sigma_end=0.05, decay_steps=50000, seed=42)
    s1 = np.stack([n1.sample() for _ in range(5)])
    s2 = np.stack([n2.sample() for _ in range(5)])
    assert np.array_equal(s1, s2)
```

- [ ] **Step 2: Run test to verify it fails** —
  `uv run pytest tests/unit/test_noise.py -v`
  Expected: FAIL (`ModuleNotFoundError: No module named 'src.ddpg.noise'`).

- [ ] **Step 3: Write minimal implementation** — `src/ddpg/noise.py`:

```python
"""Gaussian exploration noise (PRD-DDPG checkpoint 4, F5.10-F5.13; ADR-003).

sigma decays LINEARLY from sigma_start toward sigma_end over decay_steps, then
clamps at sigma_end. Brief mandates Gaussian (not Ornstein-Uhlenbeck).
"""
from __future__ import annotations

import numpy as np


class GaussianNoise:
    def __init__(
        self,
        action_dim: int,
        sigma_start: float,
        sigma_end: float,
        decay_steps: int,
        seed: int | None = None,
    ):
        self.action_dim = action_dim
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.decay_steps = decay_steps
        self.sigma = sigma_start
        self._step = 0
        self._rng = np.random.default_rng(seed)

    def sample(self) -> np.ndarray:
        return self._rng.normal(0.0, self.sigma, size=self.action_dim).astype(np.float32)

    def decay(self) -> None:
        self._step += 1
        frac = min(self._step / self.decay_steps, 1.0)
        self.sigma = self.sigma_start + (self.sigma_end - self.sigma_start) * frac
```

- [ ] **Step 4: Run test to verify it passes** —
  `uv run pytest tests/unit/test_noise.py -v`
  Expected: PASS (3 tests green).

- [ ] **Step 5: Commit** —
  `git add src/ddpg/noise.py tests/unit/test_noise.py && git commit -m "feat(ddpg): Gaussian noise with linear sigma decay (F5.10-F5.13, ADR-003)"`

---

### Task 5: DDPGAgent (act · update · Polyak soft_update)
**Files:**
- Create: `src/ddpg/agent.py`
- Test: `tests/unit/test_agent.py`

Depends on Tasks 1–4 (Actor, Critic, ReplayBuffer, GaussianNoise) and Phase 1's
`config_loader`. The `cfg` dict is the full `load_config()` output; the agent
reads `cfg["ddpg"]` (gamma, tau, lr_actor, lr_critic, batch_size, buffer_size,
hidden_sizes, grad_clip) and `cfg["noise"]` (sigma_start, sigma_end,
sigma_decay_steps). Verifies acceptance §5.1 (Polyak math element-wise) and
§5.4 (finite, seeded update).

- [ ] **Step 1: Write the failing test** — `tests/unit/test_agent.py`:

```python
import numpy as np
import torch

from src.ddpg.agent import DDPGAgent

CFG = {
    "ddpg": {
        "gamma": 0.99,
        "tau": 0.005,
        "lr_actor": 1.0e-4,
        "lr_critic": 1.0e-3,
        "batch_size": 16,
        "buffer_size": 1000,
        "hidden_sizes": [32, 32],
        "grad_clip": 1.0,
    },
    "noise": {"sigma_start": 0.2, "sigma_end": 0.05, "sigma_decay_steps": 1000},
}


def test_act_returns_action_dim_clipped_to_bounds():
    agent = DDPGAgent(state_dim=20, action_dim=2, cfg=CFG, seed=0)
    state = np.ones(20, dtype=np.float32)
    a = agent.act(state, explore=True)
    assert a.shape == (2,)
    assert np.all(a >= -1.0) and np.all(a <= 1.0)
    a_greedy = agent.act(state, explore=False)
    assert a_greedy.shape == (2,)
    assert np.all(a_greedy >= -1.0) and np.all(a_greedy <= 1.0)


def test_update_returns_finite_losses_after_prefilled_buffer():
    agent = DDPGAgent(state_dim=20, action_dim=2, cfg=CFG, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(CFG["ddpg"]["batch_size"] + 5):
        s = rng.standard_normal(20).astype(np.float32)
        a = rng.uniform(-1, 1, 2).astype(np.float32)
        s2 = rng.standard_normal(20).astype(np.float32)
        agent.store(s, a, float(rng.standard_normal()), s2, bool(rng.integers(2)))
    out = agent.update()
    assert set(out) == {"critic_loss", "actor_loss"}
    assert np.isfinite(out["critic_loss"])
    assert np.isfinite(out["actor_loss"])


def test_update_returns_empty_when_buffer_below_batch():
    agent = DDPGAgent(state_dim=20, action_dim=2, cfg=CFG, seed=0)
    s = np.zeros(20, dtype=np.float32)
    agent.store(s, np.zeros(2, dtype=np.float32), 0.0, s, False)
    assert agent.update() == {}


def test_soft_update_is_exact_polyak_with_hand_set_weights():
    # PRD-DDPG §5.1: online=1.0, target=0.0, tau=0.005 -> target == 0.005 exactly;
    # a second call -> 0.005*1 + 0.995*0.005 = 0.009975.
    agent = DDPGAgent(state_dim=20, action_dim=2, cfg=CFG, seed=0)
    with torch.no_grad():
        for p in agent.actor.parameters():
            p.fill_(1.0)
        for p in agent.actor_target.parameters():
            p.fill_(0.0)
        for p in agent.critic.parameters():
            p.fill_(1.0)
        for p in agent.critic_target.parameters():
            p.fill_(0.0)
    agent.soft_update()
    for p in agent.actor_target.parameters():
        assert torch.allclose(p, torch.full_like(p, 0.005), atol=1e-9)
    for p in agent.critic_target.parameters():
        assert torch.allclose(p, torch.full_like(p, 0.005), atol=1e-9)
    agent.soft_update()
    for p in agent.actor_target.parameters():
        assert torch.allclose(p, torch.full_like(p, 0.009975), atol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails** —
  `uv run pytest tests/unit/test_agent.py -v`
  Expected: FAIL (`ModuleNotFoundError: No module named 'src.ddpg.agent'`).

- [ ] **Step 3: Write minimal implementation** — `src/ddpg/agent.py`:

```python
"""DDPGAgent: act / update / Polyak soft_update (PRD-DDPG §3, checkpoints 1-2).

Online + hard-copied target nets; Adam optimizers (lr_actor, lr_critic);
ReplayBuffer; GaussianNoise. update() does the single-step TD critic loss,
the deterministic-policy-gradient actor loss, grad-norm clip, then soft_update.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from src.ddpg.noise import GaussianNoise
from src.ddpg.replay_buffer import ReplayBuffer
from src.model.actor import Actor
from src.model.critic import Critic


class DDPGAgent:
    def __init__(self, state_dim: int, action_dim: int, cfg: dict, seed: int | None = None):
        if seed is not None:
            torch.manual_seed(seed)
        d = cfg["ddpg"]
        n = cfg["noise"]
        self.gamma = d["gamma"]
        self.tau = d["tau"]
        self.batch_size = d["batch_size"]
        self.grad_clip = d["grad_clip"]
        self.action_dim = action_dim
        hidden = d["hidden_sizes"]
        self.actor = Actor(state_dim, action_dim, hidden)
        self.critic = Critic(state_dim, action_dim, hidden)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=d["lr_actor"])
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=d["lr_critic"])
        self.buffer = ReplayBuffer(d["buffer_size"], state_dim, action_dim, seed)
        self.noise = GaussianNoise(
            action_dim, n["sigma_start"], n["sigma_end"], n["sigma_decay_steps"], seed
        )

    def act(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            action = self.actor(s).squeeze(0).numpy()
        if explore:
            action = action + self.noise.sample()
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def store(self, s, a, r, s2, done) -> None:
        self.buffer.add(s, a, r, s2, done)

    def update(self) -> dict:
        if len(self.buffer) < self.batch_size:
            return {}
        s, a, r, s2, done = self.buffer.sample(self.batch_size)
        with torch.no_grad():
            target_q = self.critic_target(s2, self.actor_target(s2))
            y = r + self.gamma * (1.0 - done) * target_q
        critic_loss = nn.functional.mse_loss(self.critic(s, a), y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_opt.step()
        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_opt.step()
        self.soft_update()
        return {"critic_loss": float(critic_loss.item()), "actor_loss": float(actor_loss.item())}

    def soft_update(self) -> None:
        for online, target in (
            (self.actor, self.actor_target),
            (self.critic, self.critic_target),
        ):
            with torch.no_grad():
                for po, pt in zip(online.parameters(), target.parameters(), strict=True):
                    pt.mul_(1.0 - self.tau).add_(self.tau * po)
```

- [ ] **Step 4: Run test to verify it passes** —
  `uv run pytest tests/unit/test_agent.py -v`
  Expected: PASS (4 tests green; Polyak math exact to 1e-9).

- [ ] **Step 5: Commit** —
  `git add src/ddpg/agent.py tests/unit/test_agent.py && git commit -m "feat(ddpg): DDPGAgent act/update/Polyak soft_update (checkpoints 1-2, §5.1/§5.4)"`

---

### Phase 2 Definition of Done
- [ ] `uv run pytest tests/unit/test_actor.py tests/unit/test_critic.py tests/unit/test_replay_buffer.py tests/unit/test_noise.py tests/unit/test_agent.py -v` — all green.
- [ ] `uv run ruff check src/model src/ddpg tests/unit` — zero violations.
- [ ] Every new `.py` ≤150 LOC (actor 21 · critic 21 · replay_buffer 50 · noise 38 · agent 80).
- [ ] No `gymnasium`/SB3/RLlib import anywhere in the new files.
- [ ] All hyperparameters sourced from `cfg["ddpg"]` / `cfg["noise"]` — no literals in `src/`.

---

## Phase 3 — Training loop, SDK, data fetch, and results rendering

> **Scope.** Wires the deterministic simulator (Phase 1) and the from-scratch
> DDPG agent (Phase 2) into a runnable pipeline: `Trainer.train`, the
> `RoboVacuumSDK` single entry point, the pinned HouseExpo fetcher + curated
> subset, the multi-seed training driver, the two required figures
> (`learning_curve.png`, `critic_loss.png`), the trajectory visualization, and
> the held-out generalization evaluator. Deterministic/render parts are tested
> on **synthetic inputs**; the training scripts get an `episodes=1` smoke test.
>
> **Contract is LAW.** All signatures come verbatim from
> `"<REPO_ROOT>/docs/superpowers/plans/_contract.md"`.
> Config keys come from
> `"<REPO_ROOT>/config/config.yaml"`.
> `uv` only. No `gymnasium`, no SB3. Every `.py` ≤ 150 LOC.
>
> **Prerequisites (must be GREEN before Phase 3):** Phase 1 `src/env/*`
> (`HouseMap`, `VacuumEnv`, `CoverageGrid`, …) and Phase 2 `src/model/*`,
> `src/ddpg/*` (`Actor`, `Critic`, `DDPGAgent`, `ReplayBuffer`, `GaussianNoise`),
> plus the Phase 0 `RoboVacuumSDK` stub. Task 1 below provisions a shared
> `tests/conftest.py` `tiny_map` fixture this phase depends on (it is written
> idempotently — if an earlier phase already created the same fixture, replace
> its body with this canonical one).

---

### Task 1: Shared `tiny_map` conftest fixture

**Files:**
- Create: `tests/conftest.py`
- Test: `tests/unit/test_conftest_tiny_map.py`

This fixture is a 4 m × 4 m closed box (4 walls) used by the trainer smoke test
and the trajectory render test. It returns a real `HouseMap` so downstream code
exercises the production `VacuumEnv` path, not a mock.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_conftest_tiny_map.py
from src.env.house_map import HouseMap


def test_tiny_map_is_closed_box(tiny_map):
    assert isinstance(tiny_map, HouseMap)
    assert len(tiny_map.walls) == 4
    assert tiny_map.bounds == (0.0, 0.0, 4.0, 4.0)
    # centre of the box is inside; far outside corner is not
    assert tiny_map.is_inside(2.0, 2.0) is True
    assert tiny_map.is_inside(10.0, 10.0) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_conftest_tiny_map.py::test_tiny_map_is_closed_box -v`
Expected: FAIL with `fixture 'tiny_map' not found` (conftest does not exist yet).

- [ ] **Step 3: Write minimal implementation**

```python
# tests/conftest.py
import pytest

from src.env.house_map import HouseMap


@pytest.fixture
def tiny_map() -> HouseMap:
    """A 4 m x 4 m closed box: four axis-aligned wall segments."""
    walls = [
        (0.0, 0.0, 4.0, 0.0),
        (4.0, 0.0, 4.0, 4.0),
        (4.0, 4.0, 0.0, 4.0),
        (0.0, 4.0, 0.0, 0.0),
    ]
    return HouseMap(walls=walls, bounds=(0.0, 0.0, 4.0, 4.0))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_conftest_tiny_map.py::test_tiny_map_is_closed_box -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py tests/unit/test_conftest_tiny_map.py
git commit -m "Phase 3: shared tiny_map conftest fixture (4x4 closed box)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `Trainer.train` — per-episode collect→store→update loop

**Files:**
- Create: `src/services/trainer.py`
- Test: `tests/integration/test_trainer_smoke.py`

Contract: `class Trainer.__init__(self, env, agent, cfg)`;
`train(self, episodes) -> list[dict]` returning per-episode dicts with keys
`episode`, `reward`, `critic_loss`, `coverage`, `steps`; honors
`cfg["ddpg"]["warmup_steps"]` (random actions before learning).

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_trainer_smoke.py
from src.ddpg.agent import DDPGAgent
from src.services.trainer import Trainer
from src.utils.config_loader import load_config
from src.env.vacuum_env import VacuumEnv


def _tiny_cfg() -> dict:
    cfg = load_config()
    cfg["env"]["max_steps"] = 5          # tiny episode for speed
    cfg["ddpg"]["warmup_steps"] = 2      # learning starts mid-episode
    cfg["ddpg"]["batch_size"] = 2        # update can fire on a tiny buffer
    return cfg


def test_train_one_episode_returns_history_with_required_keys(tiny_map):
    cfg = _tiny_cfg()
    env = VacuumEnv(tiny_map, cfg, seed=0)
    agent = DDPGAgent(env.state_dim, env.action_dim, cfg, seed=0)
    history = Trainer(env, agent, cfg).train(episodes=1)

    assert isinstance(history, list) and len(history) == 1
    record = history[0]
    assert set(record) == {"episode", "reward", "critic_loss", "coverage", "steps"}
    assert record["episode"] == 0
    assert record["steps"] == 5
    assert isinstance(record["reward"], float)
    assert 0.0 <= record["coverage"] <= 1.0


def test_train_honors_warmup_no_update_before_threshold(tiny_map):
    cfg = _tiny_cfg()
    cfg["env"]["max_steps"] = 1          # 1 step < warmup_steps=2 ⇒ no update
    env = VacuumEnv(tiny_map, cfg, seed=0)
    agent = DDPGAgent(env.state_dim, env.action_dim, cfg, seed=0)
    history = Trainer(env, agent, cfg).train(episodes=1)

    # No learning step fired during warmup ⇒ critic_loss defaults to 0.0.
    assert history[0]["critic_loss"] == 0.0
    assert history[0]["steps"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/integration/test_trainer_smoke.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.services.trainer'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/services/trainer.py
import numpy as np

from src.ddpg.agent import DDPGAgent
from src.env.vacuum_env import VacuumEnv


class Trainer:
    """Custom DDPG training loop: collect -> store -> update -> log.

    No Gymnasium loop. Honors ddpg.warmup_steps (random actions, no learning)
    and returns a per-episode history list for the renderers/SDK.
    """

    def __init__(self, env: VacuumEnv, agent: DDPGAgent, cfg: dict) -> None:
        self.env = env
        self.agent = agent
        self.cfg = cfg
        self.warmup_steps = int(cfg["ddpg"]["warmup_steps"])
        self.action_dim = env.action_dim
        self._global_step = 0
        self._rng = np.random.default_rng(0)

    def _select_action(self, state: np.ndarray) -> np.ndarray:
        if self._global_step < self.warmup_steps:
            return self._rng.uniform(-1.0, 1.0, self.action_dim).astype(np.float32)
        return self.agent.act(state, explore=True)

    def _run_episode(self, episode: int) -> dict:
        state = self.env.reset()
        total_reward = 0.0
        critic_losses: list[float] = []
        steps = 0
        coverage = 0.0
        done = False
        while not done:
            action = self._select_action(state)
            next_state, reward, done, info = self.env.step(action)
            self.agent.store(state, action, reward, next_state, done)
            self._global_step += 1
            if self._global_step >= self.warmup_steps:
                metrics = self.agent.update()
                if metrics:
                    critic_losses.append(metrics["critic_loss"])
            state = next_state
            total_reward += reward
            steps += 1
            coverage = info["coverage"]
        mean_loss = float(np.mean(critic_losses)) if critic_losses else 0.0
        return {
            "episode": episode,
            "reward": float(total_reward),
            "critic_loss": mean_loss,
            "coverage": float(coverage),
            "steps": steps,
        }

    def train(self, episodes: int) -> list[dict]:
        return [self._run_episode(ep) for ep in range(episodes)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/integration/test_trainer_smoke.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add src/services/trainer.py tests/integration/test_trainer_smoke.py
git commit -m "Phase 3: Trainer.train collect->store->update loop (warmup honored)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `RoboVacuumSDK` — single business-logic entry point

**Files:**
- Modify: `src/sdk/sdk.py` (replace the Phase 0 `NotImplementedError` stubs)
- Test: `tests/integration/test_sdk.py`

Contract: `RoboVacuumSDK(config_path=None)` with `build_env(map_name, seed=None)`,
`train(seed, map_name=None) -> list[dict]`,
`rollout(agent, env, max_steps=None) -> list[tuple[float,float]]`,
`coverage_report(agent, env) -> dict` (`coverage`/`steps`/`collisions`).
Map names resolve to `config.paths.maps_dir/<name>.json`.

> **One additive method beyond the contract's five:** `map_walls(map_name) ->
> list[Segment]` — a read-only geometry accessor so `scripts/render_trajectory.py`
> can draw walls *through the SDK* without importing `src.env` (preserving the
> single-entry rule from Task 4). It does not alter any contracted signature.
> Flagged for human sign-off (CLAUDE.md §1.4: new public SDK method).

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_sdk.py
from src.ddpg.agent import DDPGAgent
from src.sdk.sdk import RoboVacuumSDK
from src.env.vacuum_env import VacuumEnv


def test_build_env_returns_vacuum_env_for_train_map():
    sdk = RoboVacuumSDK()
    env = sdk.build_env(map_name="room_single", seed=42)
    assert isinstance(env, VacuumEnv)
    assert env.state_dim == 20  # n_rays=16 + 4
    assert env.action_dim == 2


def test_rollout_returns_list_of_xy_pairs():
    sdk = RoboVacuumSDK()
    env = sdk.build_env(map_name="room_single", seed=42)
    agent = DDPGAgent(env.state_dim, env.action_dim, sdk.cfg, seed=42)
    path = sdk.rollout(agent, env, max_steps=5)
    assert isinstance(path, list)
    assert len(path) == 5
    assert all(isinstance(p, tuple) and len(p) == 2 for p in path)
    assert all(isinstance(p[0], float) and isinstance(p[1], float) for p in path)


def test_coverage_report_keys():
    sdk = RoboVacuumSDK()
    env = sdk.build_env(map_name="room_single", seed=42)
    agent = DDPGAgent(env.state_dim, env.action_dim, sdk.cfg, seed=42)
    report = sdk.coverage_report(agent, env)
    assert set(report) == {"coverage", "steps", "collisions"}
    assert 0.0 <= report["coverage"] <= 1.0
    assert isinstance(report["collisions"], int)


def test_map_walls_returns_segments():
    sdk = RoboVacuumSDK()
    walls = sdk.map_walls("room_single")
    assert isinstance(walls, list) and len(walls) >= 1
    assert all(len(seg) == 4 for seg in walls)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/integration/test_sdk.py -v`
Expected: FAIL — the Phase 0 stub methods raise `NotImplementedError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/sdk/sdk.py
from src.ddpg.agent import DDPGAgent
from src.env.house_map import load_house_map
from src.env.vacuum_env import VacuumEnv
from src.services.trainer import Trainer
from src.utils.config_loader import load_config


class RoboVacuumSDK:
    """The single business-logic entry point. UIs/scripts/notebooks import
    ONLY this class — never src.env / src.ddpg / src.services directly."""

    def __init__(self, config_path: str | None = None) -> None:
        self.config_path = config_path
        self.cfg = load_config(config_path)

    def _map_path(self, map_name: str) -> str:
        return f"{self.cfg['paths']['maps_dir']}/{map_name}.json"

    def build_env(self, map_name: str, seed: int | None = None) -> VacuumEnv:
        house_map = load_house_map(self._map_path(map_name))
        return VacuumEnv(house_map, self.cfg, seed=seed)

    def map_walls(self, map_name: str) -> list[tuple[float, float, float, float]]:
        # Read-only geometry accessor so the trajectory renderer can draw walls
        # through the single SDK entry point (scripts never import src.env).
        return load_house_map(self._map_path(map_name)).walls

    def train(self, seed: int, map_name: str | None = None) -> list[dict]:
        name = map_name or self.cfg["maps"]["train"][0]
        env = self.build_env(name, seed=seed)
        agent = DDPGAgent(env.state_dim, env.action_dim, self.cfg, seed=seed)
        trainer = Trainer(env, agent, self.cfg)
        return trainer.train(self.cfg["training"]["episodes"])

    def rollout(
        self, agent: DDPGAgent, env: VacuumEnv, max_steps: int | None = None
    ) -> list[tuple[float, float]]:
        limit = max_steps if max_steps is not None else self.cfg["env"]["max_steps"]
        state = env.reset()
        path: list[tuple[float, float]] = []
        for _ in range(limit):
            action = agent.act(state, explore=False)
            state, _reward, done, info = env.step(action)
            x, y, _theta = info["pose"]
            path.append((float(x), float(y)))
            if done:
                break
        return path

    def coverage_report(self, agent: DDPGAgent, env: VacuumEnv) -> dict:
        state = env.reset()
        steps = 0
        collisions = 0
        info = {"coverage": 0.0, "collision": False}
        done = False
        while not done:
            action = agent.act(state, explore=False)
            state, _reward, done, info = env.step(action)
            steps += 1
            if info["collision"]:
                collisions += 1
        return {
            "coverage": float(info["coverage"]),
            "steps": steps,
            "collisions": collisions,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/integration/test_sdk.py -v`
Expected: PASS (all three tests).

- [ ] **Step 5: Commit**

```bash
git add src/sdk/sdk.py tests/integration/test_sdk.py
git commit -m "Phase 3: implement RoboVacuumSDK (build_env/train/rollout/coverage_report)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: SDK is the only import surface (architecture test)

**Files:**
- Create: `tests/architecture/test_sdk_single_entry.py`

Asserts (AST + grep-equivalent over the source tree) that no module under
`scripts/` imports `src.env`, `src.ddpg`, `src.model`, or `src.services`
directly — only `src.sdk.sdk`. Pure-string scan; no new source needed.

- [ ] **Step 1: Write the failing test**

```python
# tests/architecture/test_sdk_single_entry.py
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
FORBIDDEN = ("src.env", "src.ddpg", "src.model", "src.services")


def _script_files() -> list[pathlib.Path]:
    return sorted((REPO / "scripts").rglob("*.py"))


def test_scripts_import_only_the_sdk():
    offenders: list[str] = []
    for path in _script_files():
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN:
            if token in text:
                offenders.append(f"{path.name}:{token}")
    assert offenders == [], f"scripts must import only RoboVacuumSDK: {offenders}"


def test_at_least_one_script_uses_the_sdk():
    uses = [p.name for p in _script_files() if "RoboVacuumSDK" in p.read_text(encoding="utf-8")]
    assert uses, "expected at least one script under scripts/ to import RoboVacuumSDK"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/architecture/test_sdk_single_entry.py -v`
Expected: FAIL — `test_at_least_one_script_uses_the_sdk` fails because no SDK-using
script exists yet (the fetch script in Task 5 does not count; the SDK-using
`scripts/train.py` lands in Task 6). Keep this test RED until Task 6, then it goes GREEN.

- [ ] **Step 3: Write minimal implementation**

No production code — this is a guardrail test. It turns GREEN once `scripts/train.py`
(Task 6) imports `RoboVacuumSDK`. Re-run after Task 6.

- [ ] **Step 4: Run test to verify it passes**

Run (after Task 6 lands): `uv run pytest tests/architecture/test_sdk_single_entry.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/architecture/test_sdk_single_entry.py
git commit -m "Phase 3: architecture test — scripts import only RoboVacuumSDK

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: `scripts/fetch_houseexpo.py` — pinned, idempotent clone + curate

**Files:**
- Create: `scripts/fetch_houseexpo.py`
- Test: `tests/unit/test_fetch_houseexpo.py`

Clones `config.maps.dataset_repo` at a SHA into git-ignored
`data/houseexpo_full/`, copies the curated `config.maps.train + holdout` names
into `config.paths.maps_dir`, and stamps the resolved SHA back into
`config.maps.dataset_sha` (replacing the `PINNED_AT_FETCH` sentinel). Idempotent:
a second run with a real SHA is a no-op re-checkout. The git clone is mocked in
the unit test (the network/disk dump is not part of the ≥85% core gate).

> **First script task — make `scripts/` an importable package** so the smoke
> tests can do `from scripts import <module>`. Create the marker file in Step 0
> below (it is committed alongside this script).

- [ ] **Step 0: Make `scripts/` a package**

```bash
touch scripts/__init__.py
```

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_fetch_houseexpo.py
import json

from scripts import fetch_houseexpo


def test_resolve_target_names_are_train_plus_holdout():
    cfg = {"maps": {"train": ["a", "b"], "holdout": ["c"]}}
    assert fetch_houseexpo.curated_names(cfg) == ["a", "b", "c"]


def test_stamp_sha_replaces_sentinel(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text('maps:\n  dataset_sha: "PINNED_AT_FETCH"\n', encoding="utf-8")
    fetch_houseexpo.stamp_sha(str(cfg_file), "a" * 40)
    text = cfg_file.read_text(encoding="utf-8")
    assert "PINNED_AT_FETCH" not in text
    assert "a" * 40 in text


def test_copy_curated_is_idempotent(tmp_path):
    src = tmp_path / "full" / "json"
    src.mkdir(parents=True)
    (src / "room_single.json").write_text(json.dumps({"verts": [[0, 0]]}), encoding="utf-8")
    dst = tmp_path / "maps"
    n1 = fetch_houseexpo.copy_curated(str(src), str(dst), ["room_single"])
    n2 = fetch_houseexpo.copy_curated(str(src), str(dst), ["room_single"])
    assert n1 == 1 and n2 == 1
    assert (dst / "room_single.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_fetch_houseexpo.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.fetch_houseexpo'`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/fetch_houseexpo.py
"""Clone HouseExpo at a pinned SHA into git-ignored data/houseexpo_full/,
copy the curated subset into data/maps/, and stamp the SHA into config.

uv run python scripts/fetch_houseexpo.py
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

from src.sdk.sdk import RoboVacuumSDK

FULL_DIR = "data/houseexpo_full"
SENTINEL = "PINNED_AT_FETCH"


def curated_names(cfg: dict) -> list[str]:
    return list(cfg["maps"]["train"]) + list(cfg["maps"]["holdout"])


def clone_repo(repo_url: str, sha: str, dest: str) -> str:
    dest_path = Path(dest)
    if not dest_path.exists():
        subprocess.run(["git", "clone", repo_url, dest], check=True)
    if sha != SENTINEL:
        subprocess.run(["git", "-C", dest, "checkout", sha], check=True)
    resolved = subprocess.run(
        ["git", "-C", dest, "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    )
    return resolved.stdout.strip()


def copy_curated(json_dir: str, maps_dir: str, names: list[str]) -> int:
    Path(maps_dir).mkdir(parents=True, exist_ok=True)
    copied = 0
    for name in names:
        src_file = Path(json_dir) / f"{name}.json"
        if src_file.exists():
            shutil.copy2(src_file, Path(maps_dir) / f"{name}.json")
            copied += 1
    return copied


def stamp_sha(config_path: str, sha: str) -> None:
    text = Path(config_path).read_text(encoding="utf-8")
    new = re.sub(
        r'(dataset_sha:\s*")[^"]*(")', rf"\g<1>{sha}\g<2>", text, count=1
    )
    Path(config_path).write_text(new, encoding="utf-8")


def main() -> int:
    sdk = RoboVacuumSDK()
    cfg = sdk.cfg
    sha = clone_repo(cfg["maps"]["dataset_repo"], cfg["maps"]["dataset_sha"], FULL_DIR)
    copied = copy_curated(f"{FULL_DIR}/json", cfg["paths"]["maps_dir"], curated_names(cfg))
    stamp_sha(sdk.config_path or "config/config.yaml", sha)
    print(f"fetched SHA={sha} curated={copied} into {cfg['paths']['maps_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_fetch_houseexpo.py -v`
Expected: PASS (all three tests; `clone_repo`/`main` are smoke-only, not unit-tested).

- [ ] **Step 5: Commit**

```bash
git add scripts/__init__.py scripts/fetch_houseexpo.py tests/unit/test_fetch_houseexpo.py
git commit -m "Phase 3: scripts/fetch_houseexpo.py (pinned clone, curate, stamp SHA)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: `scripts/train.py` — multi-seed training driver

**Files:**
- Create: `scripts/train.py`
- Test: `tests/integration/test_train_script.py`

Loops `config.training.seeds`, saves per-seed history JSON
(`results/history/seed_<seed>.json`) and a checkpoint
(`results/checkpoints/seed_<seed>.pt`). Imports only `RoboVacuumSDK` (satisfies
Task 4). Smoke-tested with `episodes` monkeypatched to 1 and one seed.

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_train_script.py
import json

from scripts import train as train_script


def test_run_seeds_writes_history_and_checkpoint(tmp_path, monkeypatch):
    from src.sdk.sdk import RoboVacuumSDK

    # Shrink the workload to a 1-episode, 3-step, single-seed smoke run.
    orig_init = RoboVacuumSDK.__init__

    def patched_init(self, config_path=None):
        orig_init(self, config_path)
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
    assert set(history[0]) == {"episode", "reward", "critic_loss", "coverage", "steps"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/integration/test_train_script.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.train'`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/train.py
"""Multi-seed DDPG training driver.

uv run python scripts/train.py
Saves results/history/seed_<seed>.json + results/checkpoints/seed_<seed>.pt.
"""
import json
import sys
from pathlib import Path

import torch

from src.sdk.sdk import RoboVacuumSDK


def _save_history(results_dir: str, seed: int, history: list[dict]) -> None:
    out = Path(results_dir) / "history"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"seed_{seed}.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


def _save_checkpoint(results_dir: str, seed: int, sdk: RoboVacuumSDK) -> None:
    out = Path(results_dir) / "checkpoints"
    out.mkdir(parents=True, exist_ok=True)
    # Minimal, reproducible artifact: seed + the run config it was trained under.
    torch.save({"seed": seed, "config": sdk.cfg}, out / f"seed_{seed}.pt")


def run_seeds(results_dir: str = "results") -> list[int]:
    sdk = RoboVacuumSDK()
    seeds = list(sdk.cfg["training"]["seeds"])
    for seed in seeds:
        history = sdk.train(seed=seed)
        _save_history(results_dir, seed, history)
        _save_checkpoint(results_dir, seed, sdk)
        print(f"seed={seed} episodes={len(history)} saved -> {results_dir}")
    return seeds


def main() -> int:
    run_seeds()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/integration/test_train_script.py tests/architecture/test_sdk_single_entry.py -v`
Expected: PASS (train smoke + the Task 4 architecture test now goes GREEN).

- [ ] **Step 5: Commit**

```bash
git add scripts/train.py tests/integration/test_train_script.py
git commit -m "Phase 3: scripts/train.py multi-seed driver (per-seed history JSON + checkpoint)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: `scripts/render_learning_curve.py` — reward vs episode, mean ± 95% CI

**Files:**
- Create: `scripts/render_learning_curve.py`
- Test: `tests/unit/test_render_learning_curve.py`

Reads per-seed `results/history/seed_<seed>.json`, computes per-episode
mean ± 95% CI across seeds, saves `results/figures/learning_curve.png` (> 5 KB).
Tested on **synthetic** history JSON written into a tmp dir.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_render_learning_curve.py
import json

from scripts import render_learning_curve as rlc


def _write_synthetic(history_dir, seeds):
    history_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        records = [
            {"episode": e, "reward": float(e + seed), "critic_loss": 0.1,
             "coverage": 0.5, "steps": 10}
            for e in range(3)
        ]
        (history_dir / f"seed_{seed}.json").write_text(json.dumps(records), encoding="utf-8")


def test_mean_ci_shapes_match_episode_count(tmp_path):
    _write_synthetic(tmp_path / "history", [42, 7])
    episodes, mean, ci = rlc.mean_ci(str(tmp_path / "history"), "reward")
    assert list(episodes) == [0, 1, 2]
    assert len(mean) == 3 and len(ci) == 3
    # episode 0: rewards {42, 7} -> mean 24.5
    assert abs(float(mean[0]) - 24.5) < 1e-9


def test_render_writes_png_over_5kb(tmp_path):
    _write_synthetic(tmp_path / "history", [42, 7, 123])
    out = tmp_path / "figures" / "learning_curve.png"
    rlc.render(str(tmp_path / "history"), str(out))
    assert out.exists()
    assert out.stat().st_size > 5000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_render_learning_curve.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.render_learning_curve'`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/render_learning_curve.py
"""Render results/figures/learning_curve.png: cumulative reward vs episode,
mean +/- 95% CI over the training seeds.

uv run python scripts/render_learning_curve.py
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HISTORY_DIR = "results/history"
OUT_PNG = "results/figures/learning_curve.png"


def _load_matrix(history_dir: str, key: str) -> np.ndarray:
    files = sorted(Path(history_dir).glob("seed_*.json"))
    rows = []
    for fp in files:
        records = json.loads(fp.read_text(encoding="utf-8"))
        rows.append([float(r[key]) for r in records])
    return np.asarray(rows, dtype=np.float64)


def mean_ci(history_dir: str, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = _load_matrix(history_dir, key)
    episodes = np.arange(matrix.shape[1])
    mean = matrix.mean(axis=0)
    n = matrix.shape[0]
    sem = matrix.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    ci = 1.96 * sem
    return episodes, mean, ci


def render(history_dir: str = HISTORY_DIR, out_png: str = OUT_PNG) -> str:
    episodes, mean, ci = mean_ci(history_dir, "reward")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    n_seeds = len(sorted(Path(history_dir).glob("seed_*.json")))
    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    ax.plot(episodes, mean, color="#1f77b4", label="mean reward")
    ax.fill_between(episodes, mean - ci, mean + ci, color="#1f77b4", alpha=0.25,
                    label="95% CI")
    ax.set_xlabel("episode")
    ax.set_ylabel("cumulative reward")
    ax.set_title(f"Learning curve (mean +/- 95% CI over {n_seeds} seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


def main() -> int:
    print(render())
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_render_learning_curve.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/render_learning_curve.py tests/unit/test_render_learning_curve.py
git commit -m "Phase 3: render_learning_curve.py (reward vs episode, mean +/- 95% CI)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: `scripts/render_critic_loss.py` — critic loss vs training step

**Files:**
- Create: `scripts/render_critic_loss.py`
- Test: `tests/unit/test_render_critic_loss.py`

Reads per-seed history JSON, plots per-episode `critic_loss` mean ± 95% CI
across seeds, saves `results/figures/critic_loss.png` (> 5 KB). The shared
mean±CI math is reused from `render_learning_curve.mean_ci` (DRY).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_render_critic_loss.py
import json

from scripts import render_critic_loss as rcl


def _write_synthetic(history_dir, seeds):
    history_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        records = [
            {"episode": e, "reward": 1.0, "critic_loss": float(seed) / (e + 1),
             "coverage": 0.5, "steps": 10}
            for e in range(4)
        ]
        (history_dir / f"seed_{seed}.json").write_text(json.dumps(records), encoding="utf-8")


def test_render_writes_png_over_5kb(tmp_path):
    _write_synthetic(tmp_path / "history", [42, 7])
    out = tmp_path / "figures" / "critic_loss.png"
    rcl.render(str(tmp_path / "history"), str(out))
    assert out.exists()
    assert out.stat().st_size > 5000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_render_critic_loss.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.render_critic_loss'`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/render_critic_loss.py
"""Render results/figures/critic_loss.png: critic loss vs training step
(per-episode mean +/- 95% CI over the training seeds).

uv run python scripts/render_critic_loss.py
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np

from scripts.render_learning_curve import mean_ci

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HISTORY_DIR = "results/history"
OUT_PNG = "results/figures/critic_loss.png"


def render(history_dir: str = HISTORY_DIR, out_png: str = OUT_PNG) -> str:
    episodes, mean, ci = mean_ci(history_dir, "critic_loss")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    n_seeds = len(sorted(Path(history_dir).glob("seed_*.json")))
    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    ax.plot(episodes, mean, color="#d62728", label="mean critic loss")
    ax.fill_between(episodes, np.maximum(mean - ci, 0.0), mean + ci,
                    color="#d62728", alpha=0.25, label="95% CI")
    ax.set_xlabel("episode (training step proxy)")
    ax.set_ylabel("critic MSE loss")
    ax.set_title(f"Critic loss (mean +/- 95% CI over {n_seeds} seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


def main() -> int:
    print(render())
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_render_critic_loss.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/render_critic_loss.py tests/unit/test_render_critic_loss.py
git commit -m "Phase 3: render_critic_loss.py (critic loss vs step, mean +/- 95% CI)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 9: `scripts/render_trajectory.py` — robot path over the map + covered area

**Files:**
- Create: `scripts/render_trajectory.py`
- Test: `tests/unit/test_render_trajectory.py`

Draws wall segments, the rollout path in colour, and a shaded covered-area
footprint, saving a PNG. Tested with a synthetic 3-point path and the
`tiny_map` fixture (no training needed — pure rendering).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_render_trajectory.py
from scripts import render_trajectory as rt


def test_render_trajectory_writes_png(tmp_path, tiny_map):
    path = [(1.0, 1.0), (2.0, 2.0), (3.0, 1.5)]
    out = tmp_path / "trajectory.png"
    rt.render(tiny_map.walls, path, str(out), clean_radius=0.17)
    assert out.exists()
    assert out.stat().st_size > 5000


def test_render_trajectory_handles_single_point(tmp_path, tiny_map):
    out = tmp_path / "trajectory_single.png"
    rt.render(tiny_map.walls, [(2.0, 2.0)], str(out), clean_radius=0.17)
    assert out.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_render_trajectory.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.render_trajectory'`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/render_trajectory.py
"""Render a rollout trajectory over the 2D map: walls (black), path (colour),
and a shaded covered-area footprint, proving wall-avoidance + coverage.

uv run python scripts/render_trajectory.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_PNG = "results/figures/trajectory.png"


def render(
    walls: list[tuple[float, float, float, float]],
    path: list[tuple[float, float]],
    out_png: str = OUT_PNG,
    clean_radius: float = 0.17,
) -> str:
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    for x1, y1, x2, y2 in walls:
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=2.0)
    for x, y in path:
        ax.add_patch(plt.Circle((x, y), clean_radius, color="#2ca02c", alpha=0.18))
    if path:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, color="#1f77b4", linewidth=1.5, label="robot path")
        ax.scatter([xs[0]], [ys[0]], color="green", zorder=5, label="start")
        ax.scatter([xs[-1]], [ys[-1]], color="red", zorder=5, label="end")
        ax.legend(loc="upper right")
    ax.set_aspect("equal")
    ax.set_title("Rollout trajectory + covered area")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


def main() -> int:
    from src.ddpg.agent import DDPGAgent
    from src.sdk.sdk import RoboVacuumSDK

    sdk = RoboVacuumSDK()
    name = sdk.cfg["maps"]["train"][0]
    env = sdk.build_env(name, seed=sdk.cfg["training"]["seeds"][0])
    agent = DDPGAgent(env.state_dim, env.action_dim, sdk.cfg, seed=0)
    path = sdk.rollout(agent, env)
    walls = sdk.map_walls(name)
    print(render(walls, path, clean_radius=sdk.cfg["env"]["clean_radius"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_render_trajectory.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/render_trajectory.py tests/unit/test_render_trajectory.py
git commit -m "Phase 3: render_trajectory.py (path over walls + shaded coverage)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 10: `scripts/evaluate.py` — held-out maps coverage (generalization)

**Files:**
- Create: `scripts/evaluate.py`
- Test: `tests/integration/test_evaluate_script.py`

Runs greedy `coverage_report` rollouts on every `config.maps.holdout` map via the
SDK and returns a `{map_name: report}` dict. Imports only `RoboVacuumSDK`.
Smoke-tested with a fresh (untrained) agent and `max_steps` shrunk.

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_evaluate_script.py
from scripts import evaluate as evaluate_script


def test_evaluate_holdout_returns_report_per_map(monkeypatch):
    from src.sdk.sdk import RoboVacuumSDK

    orig_init = RoboVacuumSDK.__init__

    def patched_init(self, config_path=None):
        orig_init(self, config_path)
        self.cfg["env"]["max_steps"] = 3  # smoke speed

    monkeypatch.setattr(RoboVacuumSDK, "__init__", patched_init)

    results = evaluate_script.evaluate_holdout()
    holdout = RoboVacuumSDK().cfg["maps"]["holdout"]
    assert set(results) == set(holdout)
    for name in holdout:
        report = results[name]
        assert set(report) == {"coverage", "steps", "collisions"}
        assert 0.0 <= report["coverage"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/integration/test_evaluate_script.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.evaluate'`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/evaluate.py
"""Held-out generalization eval: greedy coverage on config.maps.holdout maps.

uv run python scripts/evaluate.py
"""
import json
import sys
from pathlib import Path

from src.ddpg.agent import DDPGAgent
from src.sdk.sdk import RoboVacuumSDK

OUT_JSON = "results/holdout_eval.json"


def evaluate_holdout(seed: int = 0) -> dict:
    sdk = RoboVacuumSDK()
    results: dict[str, dict] = {}
    for name in sdk.cfg["maps"]["holdout"]:
        env = sdk.build_env(name, seed=seed)
        agent = DDPGAgent(env.state_dim, env.action_dim, sdk.cfg, seed=seed)
        results[name] = sdk.coverage_report(agent, env)
    return results


def main() -> int:
    results = evaluate_holdout()
    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT_JSON).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/integration/test_evaluate_script.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/evaluate.py tests/integration/test_evaluate_script.py
git commit -m "Phase 3: scripts/evaluate.py held-out coverage generalization eval

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Phase 3 exit gate

After Task 10, run the full gate and confirm GREEN before marking the phase done:

```bash
uv run pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=85
uv run ruff check src/ tests/ scripts/
```

Expected: all Phase 3 tests PASS, coverage ≥ 85% on `src/`, zero Ruff violations,
every `.py` ≤ 150 LOC (the architecture test in Task 4 is GREEN once `scripts/train.py`
exists). The two required figures plus the trajectory render are reproducible via
`uv run python scripts/render_learning_curve.py`, `render_critic_loss.py`,
`render_trajectory.py` after `scripts/train.py` has populated `results/history/`.

---

## Phase 4 — Architecture Tests, Analysis, Docs & Final Gates

Scope: the five blocking **architecture tests** (no-gymnasium AST walk, actor
Tanh bounds, Polyak soft-update math, SDK single-entry, config single-source),
the **analysis/quality/cost/UX docs**, the **PROMPTS** log skeleton, the
**README** finalization with embedded figures, and the **FINAL GATES** task
(pytest `--cov≥85`, `ruff check`, `ruff format --check`, `check_file_sizes`
green) + tag `v1.0.0` + the `adrl-001-ex05.pdf` cover sheet.

All paths are absolute under the repo root
`<REPO_ROOT>/`.
Commands are run from that root with `uv` only. Contract signatures
(`src/sdk/sdk.py::RoboVacuumSDK`, `src/model/actor.py::Actor`,
`src/ddpg/agent.py::DDPGAgent.soft_update`, `src/utils/config_loader.py::load_config/get`)
are LAW — never invented. Numeric result figures are referenced as **PENDING**
placeholders to be filled after the seeded training run completes (spec §10
honesty stance).

Prereq: Phases 1–3 have landed `src/env/*`, `src/model/{actor,critic}.py`,
`src/ddpg/{replay_buffer,noise,agent}.py`, `src/services/trainer.py`,
`src/sdk/sdk.py`, `scripts/{fetch_houseexpo,render_learning_curve,render_critic_loss,render_trajectory}.py`,
and `results/figures/{learning_curve,critic_loss}.png` + a trajectory figure.

---

### Task 1: Architecture test — no `gymnasium`/`gym` import under `src/`
**Files:**
- Create: `tests/architecture/test_no_gymnasium_import.py`
- Test: `tests/architecture/test_no_gymnasium_import.py` (this IS the test)

- [ ] **Step 1: Write the failing test** — AST-walks every `.py` under `src/`, fails on any `import gymnasium`/`import gym` or `from gymnasium … import …`. Parametrized one case per file so a regression names the offending file.

```python
"""Architectural contract (spec §1, §3, §8): NO gymnasium/gym import in src/.

Brief "דרישת חובה": no ready simulation platforms — the simulator is from
scratch. AST-level (not grep) so string occurrences in comments/docstrings do
not false-positive. Parametrized over every .py under src/ so a regression
fails its own case.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

FORBIDDEN_ROOTS = ("gymnasium", "gym")
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC = _REPO_ROOT / "src"


def _is_forbidden(name: str) -> bool:
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
def test_no_gymnasium_or_gym_import(py_file: Path) -> None:
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    bad = _offenders(tree, py_file)
    assert not bad, "gymnasium/gym import contract violation (spec §1/§3/§8):\n" + "\n".join(bad)


def test_scan_actually_covered_src_files() -> None:
    """Guard: the parametrization must have found the real env/ddpg modules."""
    names = {p.name for p in _PY}
    assert "vacuum_env.py" in names
    assert "agent.py" in names
```

- [ ] **Step 2: Run test to verify it fails** —
```bash
uv run pytest "tests/architecture/test_no_gymnasium_import.py" -v
```
Expected: FAIL — `test_no_gymnasium_import.py` does not yet exist (collection error / file-not-found), so the test cannot pass until it is authored on disk.

- [ ] **Step 3: Write minimal implementation** — no production code; the deliverable IS the test file above. The src/ tree from Phases 1–3 already contains zero gymnasium imports (spec §1), so once the file lands the parametrized cases pass.

- [ ] **Step 4: Run test to verify it passes** —
```bash
uv run pytest "tests/architecture/test_no_gymnasium_import.py" -v
```
Expected: PASS — every `src/**.py` case green; `test_scan_actually_covered_src_files` confirms `vacuum_env.py` and `agent.py` were scanned.

- [ ] **Step 5: Commit** —
```bash
git add tests/architecture/test_no_gymnasium_import.py && git commit -m "Phase 4: architecture test — no gymnasium/gym import under src/ (spec §1/§8)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Architecture test — Actor action is Tanh-bounded to [−1, 1]
**Files:**
- Create: `tests/architecture/test_actor_action_bounded.py`
- Test: `tests/architecture/test_actor_action_bounded.py`

- [ ] **Step 1: Write the failing test** — builds the contract `Actor(state_dim, action_dim, hidden_sizes)`, asserts forward output is element-wise within `[−1, 1]` for 1000 random states AND for adversarially large pre-activation states (PRD-DDPG §5.2 / F5.15).

```python
"""Architecture contract (PRD-DDPG §5.2, spec §5.1): Actor output is Tanh-
bounded to [-1, 1] for all inputs, including adversarially large states.
"""

from __future__ import annotations

import torch

from src.model.actor import Actor
from src.utils.config_loader import get, load_config


def _build_actor() -> tuple[Actor, int]:
    load_config()
    env = get("env")
    state_dim = int(env["n_rays"]) + 4
    hidden = list(get("ddpg")["hidden_sizes"])
    return Actor(state_dim=state_dim, action_dim=2, hidden_sizes=hidden), state_dim


def test_actor_output_within_unit_box_random() -> None:
    torch.manual_seed(0)
    actor, state_dim = _build_actor()
    states = torch.randn(1000, state_dim)
    actions = actor(states)
    assert actions.shape == (1000, 2)
    assert torch.all(actions >= -1.0)
    assert torch.all(actions <= 1.0)


def test_actor_output_within_unit_box_adversarial() -> None:
    actor, state_dim = _build_actor()
    states = torch.full((64, state_dim), 1.0e6)
    actions = actor(states)
    assert torch.all(actions >= -1.0)
    assert torch.all(actions <= 1.0)
    assert torch.all(torch.isfinite(actions))
```

- [ ] **Step 2: Run test to verify it fails** —
```bash
uv run pytest "tests/architecture/test_actor_action_bounded.py::test_actor_output_within_unit_box_random" -v
```
Expected: FAIL — the test file does not yet exist (collection error), so it cannot pass until authored.

- [ ] **Step 3: Write minimal implementation** — no production code; `src/model/actor.py::Actor.forward` already returns `torch.tanh(...)` (contract §model), guaranteeing the bound. The deliverable is the test file above.

- [ ] **Step 4: Run test to verify it passes** —
```bash
uv run pytest "tests/architecture/test_actor_action_bounded.py" -v
```
Expected: PASS — both random and adversarial batches stay in `[−1, 1]` and finite.

- [ ] **Step 5: Commit** —
```bash
git add tests/architecture/test_actor_action_bounded.py && git commit -m "Phase 4: architecture test — Actor action Tanh-bounded to [-1,1] (PRD-DDPG §5.2)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Architecture test — soft-update is Polyak averaging
**Files:**
- Create: `tests/architecture/test_soft_update_is_polyak.py`
- Test: `tests/architecture/test_soft_update_is_polyak.py`

- [ ] **Step 1: Write the failing test** — builds `DDPGAgent`, forces online params to 1.0 and target params to 0.0, calls `soft_update()` and asserts every target param == τ exactly, then a second call yields `τ·1 + (1−τ)·τ` (PRD-DDPG §5.1 hand-computed). τ is read from `ddpg.tau` (no hardcode).

```python
"""Architecture contract (PRD-DDPG §5.1, spec §5.2): soft_update() is Polyak
averaging  theta_target <- tau*theta + (1-tau)*theta_target  for BOTH targets.

Hand-computed: online=1.0, target=0.0, tau -> target == tau after one call;
second call -> tau*1 + (1-tau)*tau.
"""

from __future__ import annotations

import torch

from src.ddpg.agent import DDPGAgent
from src.utils.config_loader import get, load_config


def _build_agent() -> tuple[DDPGAgent, float]:
    cfg = load_config()
    tau = float(get("ddpg")["tau"])
    state_dim = int(get("env")["n_rays"]) + 4
    agent = DDPGAgent(state_dim=state_dim, action_dim=2, cfg=cfg, seed=0)
    return agent, tau


def _set_all(module: torch.nn.Module, value: float) -> None:
    with torch.no_grad():
        for p in module.parameters():
            p.fill_(value)


def _assert_all_close(module: torch.nn.Module, value: float) -> None:
    for p in module.parameters():
        assert torch.allclose(p, torch.full_like(p, value), atol=1e-7), (
            f"param not ~{value}: got {p.flatten()[0].item()}"
        )


def test_soft_update_moves_targets_toward_online_by_tau() -> None:
    agent, tau = _build_agent()
    for online, target in ((agent.actor, agent.actor_target), (agent.critic, agent.critic_target)):
        _set_all(online, 1.0)
        _set_all(target, 0.0)
    agent.soft_update()
    _assert_all_close(agent.actor_target, tau)
    _assert_all_close(agent.critic_target, tau)
    agent.soft_update()
    expected = tau * 1.0 + (1.0 - tau) * tau
    _assert_all_close(agent.actor_target, expected)
    _assert_all_close(agent.critic_target, expected)
```

- [ ] **Step 2: Run test to verify it fails** —
```bash
uv run pytest "tests/architecture/test_soft_update_is_polyak.py::test_soft_update_moves_targets_toward_online_by_tau" -v
```
Expected: FAIL — the test file does not yet exist (collection error), so it cannot pass until authored.

- [ ] **Step 3: Write minimal implementation** — no production code; `src/ddpg/agent.py::DDPGAgent.soft_update` already applies Polyak averaging with `τ = cfg["ddpg"]["tau"]` over both `actor_target` and `critic_target` (contract §ddpg). The deliverable is the test file above. (If the agent named its target attributes differently, this test pins the contract names `actor_target` / `critic_target`.)

- [ ] **Step 4: Run test to verify it passes** —
```bash
uv run pytest "tests/architecture/test_soft_update_is_polyak.py" -v
```
Expected: PASS — first call lands every target at `τ = 0.005`; second at `0.005 + 0.995·0.005 = 0.009975`.

- [ ] **Step 5: Commit** —
```bash
git add tests/architecture/test_soft_update_is_polyak.py && git commit -m "Phase 4: architecture test — soft_update is Polyak averaging for both targets (PRD-DDPG §5.1)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Architecture test — SDK is the single entry point
**Files:**
- Create: `tests/architecture/test_sdk_single_entry.py`
- Test: `tests/architecture/test_sdk_single_entry.py`

- [ ] **Step 1: Write the failing test** — AST-walks every `.py` under `scripts/` and `notebooks/` (CLI/scripts/notebook layer) and asserts none imports anything under `src.*` except `src.sdk` / `src.sdk.sdk` (CLAUDE.md §3, PRD-SIM FR-10). Also asserts the SDK exposes exactly the contract methods.

```python
"""Architecture contract (CLAUDE.md §3, PRD-SIM FR-10, spec §8): the UI layer
(scripts/, notebooks/) imports ONLY src.sdk — no direct src.env / src.ddpg /
src.model / src.services / src.utils imports leak into a UI module.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

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


def test_sdk_exposes_contract_surface() -> None:
    from src.sdk.sdk import RoboVacuumSDK

    for method in ("build_env", "train", "rollout", "coverage_report"):
        assert callable(getattr(RoboVacuumSDK, method)), f"SDK missing {method}"
```

- [ ] **Step 2: Run test to verify it fails** —
```bash
uv run pytest "tests/architecture/test_sdk_single_entry.py::test_sdk_exposes_contract_surface" -v
```
Expected: FAIL — the test file does not yet exist (collection error), so it cannot pass until authored.

- [ ] **Step 3: Write minimal implementation** — no production code; Phase 2/3 already route scripts/notebook through `RoboVacuumSDK` (contract §sdk: `build_env`/`train`/`rollout`/`coverage_report`). The deliverable is the test file above.

- [ ] **Step 4: Run test to verify it passes** —
```bash
uv run pytest "tests/architecture/test_sdk_single_entry.py" -v
```
Expected: PASS — every script/notebook imports only `src.sdk`, and the SDK exposes the four contract methods.

- [ ] **Step 5: Commit** —
```bash
git add tests/architecture/test_sdk_single_entry.py && git commit -m "Phase 4: architecture test — SDK single entry; UI imports only src.sdk (CLAUDE.md §3)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Architecture test — config is the single source of hyperparameters
**Files:**
- Create: `tests/architecture/test_config_single_source.py`
- Test: `tests/architecture/test_config_single_source.py`

- [ ] **Step 1: Write the failing test** — asserts (a) every DDPG/noise/env/reward hyperparameter named in the contract resolves through `load_config()`/`get()`, and (b) AST-walks `src/ddpg/`, `src/model/`, `src/env/`, `src/services/` and fails if any module defines a module-level numeric literal assignment to a hyperparameter-looking name (e.g. `gamma`, `tau`, `lr_actor`, `batch_size`, `sigma_start`) — those must come from config, not source (CLAUDE.md §4, PRD-DDPG checkpoint 3).

```python
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
    "gamma", "tau", "lr_actor", "lr_critic", "batch_size", "buffer_size",
    "grad_clip", "warmup_steps", "sigma_start", "sigma_end", "sigma_decay_steps",
    "k_coverage", "k_collision", "k_step", "v_max", "omega_max", "ray_max",
}


def test_every_hyperparameter_resolves_through_config() -> None:
    load_config()
    ddpg = get("ddpg")
    for key in ("gamma", "tau", "lr_actor", "lr_critic", "batch_size",
                "buffer_size", "hidden_sizes", "grad_clip", "warmup_steps"):
        assert key in ddpg, f"ddpg.{key} missing from config"
    noise = get("noise")
    for key in ("type", "sigma_start", "sigma_end", "sigma_decay_steps"):
        assert key in noise, f"noise.{key} missing from config"
    reward = get("reward")
    for key in ("k_coverage", "k_collision", "k_step"):
        assert key in reward, f"reward.{key} missing from config"
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
```

- [ ] **Step 2: Run test to verify it fails** —
```bash
uv run pytest "tests/architecture/test_config_single_source.py::test_every_hyperparameter_resolves_through_config" -v
```
Expected: FAIL — the test file does not yet exist (collection error), so it cannot pass until authored.

- [ ] **Step 3: Write minimal implementation** — no production code; `config/config.yaml` already carries every key (verified) and Phase 1/2 modules pull constants from `cfg`. The deliverable is the test file above.

- [ ] **Step 4: Run test to verify it passes** —
```bash
uv run pytest "tests/architecture/test_config_single_source.py" -v
```
Expected: PASS — all keys resolve through config and no module-level hyperparameter literal exists in `src/`.

- [ ] **Step 5: Commit** —
```bash
git add tests/architecture/test_config_single_source.py && git commit -m "Phase 4: architecture test — config single-source; no hardcoded hyperparams in src/ (CLAUDE.md §4)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: `docs/ANALYSIS.md` — the brief's 3 analysis questions (with PENDING result placeholders)
**Files:**
- Create: `docs/ANALYSIS.md`
- Test: `tests/architecture/test_analysis_doc_shape.py`

- [ ] **Step 1: Write the failing test** — asserts `docs/ANALYSIS.md` exists, has the exact section headers for all three brief questions, embeds the three figures, and carries the `ΔReward`/`coverage` PENDING placeholders (no invented numbers — spec §10).

```python
"""Doc contract (spec §7, PRD-DDPG §4/§5.6): ANALYSIS.md answers the brief's
THREE analysis questions, embeds the three figures, and marks result numbers
PENDING until training completes.
"""

from __future__ import annotations

from pathlib import Path

_DOC = Path(__file__).resolve().parent.parent.parent / "docs" / "ANALYSIS.md"


def test_analysis_doc_has_three_question_headers() -> None:
    text = _DOC.read_text(encoding="utf-8")
    assert "## Q1 — Why DDPG (not DQN, not PPO)" in text
    assert "## Q2 — Removing Gaussian exploration noise early" in text
    assert "## Q3 — Target networks + soft updates prevent critic collapse" in text


def test_analysis_doc_embeds_figures_and_pending_numbers() -> None:
    text = _DOC.read_text(encoding="utf-8")
    assert "results/figures/learning_curve.png" in text
    assert "results/figures/critic_loss.png" in text
    assert "results/figures/trajectory.png" in text
    assert "ΔReward" in text
    assert "PENDING" in text
```

- [ ] **Step 2: Run test to verify it fails** —
```bash
uv run pytest "tests/architecture/test_analysis_doc_shape.py" -v
```
Expected: FAIL — `docs/ANALYSIS.md` does not exist yet (`FileNotFoundError`).

- [ ] **Step 3: Write minimal implementation** — author `docs/ANALYSIS.md`:

```markdown
# ANALYSIS — RoboVacuumDDPG (the brief's three questions)

> Source of truth: design spec §7; PRD-DDPG §4/§5.6. Result numbers are
> **PENDING** until a seeded 500-episode run completes (spec §10 honesty
> stance — no invented metrics). Figures are regenerated by
> `scripts/render_learning_curve.py`, `scripts/render_critic_loss.py`,
> `scripts/render_trajectory.py`.

## Q1 — Why DDPG (not DQN, not PPO)

The vacuum's command is a **continuous** motor pair `a = [throttle, steer] ∈
[−1,1]²` driving a unicycle body (spec §3). The choice follows three forces:

- **DQN** is discrete-action: `argmax_a Q(s,a)` cannot be enumerated over a
  continuous box. Discretizing the `[throttle, steer]` grid throws away the
  smooth control the physical actuators need.
- **PPO** is on-policy and *stochastic*: it discards each rollout after one
  update and cannot reuse HouseExpo experience the way an off-policy replay
  buffer can. The motors are deterministic actuators, so a stochastic policy
  is a poor structural match.
- **DDPG** is off-policy with a **deterministic** actor + Q-critic: continuous
  control, sample-efficient replay reuse, and Polyak-stabilized targets.
  Matches Lillicrap et al. 2016 (arXiv:1509.02971) and Lecture 09.

Evidence pointers: actor `src/model/actor.py`, critic `src/model/critic.py`,
deterministic update `src/ddpg/agent.py`.

## Q2 — Removing Gaussian exploration noise early

Ablation (documented, not a pass condition — PRD-DDPG §5.6). With Gaussian
noise (`src/ddpg/noise.py`, σ_start=0.2 → σ_end=0.05 over 50k steps) the agent
explores laterally and fills open floor; with noise removed at the start the
deterministic actor commits to one heading and the **coverage map collapses to
a narrow path** — the robot retraces a single corridor.

| Variant | Final coverage % | ΔReward vs warmup baseline |
|---|---|---|
| Gaussian noise (default) | PENDING | PENDING (ΔReward) |
| Noise OFF from step 0 | PENDING | PENDING (ΔReward) |

![Learning curve](../results/figures/learning_curve.png)
*Cumulative reward vs episode, mean ± CI over the 5 seeds. Numbers PENDING.*

![Trajectory](../results/figures/trajectory.png)
*Greedy rollout path over the 2D HouseExpo map; covered area shaded. PENDING.*

## Q3 — Target networks + soft updates prevent critic collapse

The TD target `y = r + γ(1−d)·Q′(s′, μ′(s′))` is computed from **separate
target networks** updated by Polyak averaging `θ_t ← τθ + (1−τ)θ_t`
(τ=0.005, `src/ddpg/agent.py::soft_update`). A slowly-moving target breaks the
"chasing a moving target" feedback loop that makes a single-network critic
diverge. We surface this as a **non-increasing rolling `critic_loss` slope**
over the final 20% of training.

![Critic loss](../results/figures/critic_loss.png)
*Critic MSE vs training step; rolling slope ≤ 0 in the final window. PENDING.*

| Metric (final 20% window, 5 seeds) | Value |
|---|---|
| Mean final coverage % | PENDING |
| ΔReward (final window − warmup window) | PENDING |
| Critic-loss rolling slope | PENDING |

## References
Lillicrap et al. 2016 (arXiv:1509.02971); Silver et al. 2014 (ICML);
Fujimoto et al. 2018 (TD3, arXiv:1802.09477); Li et al. 2019 HouseExpo
(arXiv:1903.09845).
```

- [ ] **Step 4: Run test to verify it passes** —
```bash
uv run pytest "tests/architecture/test_analysis_doc_shape.py" -v
```
Expected: PASS — three question headers, three embedded figures, `ΔReward` + `PENDING` markers all present.

- [ ] **Step 5: Commit** —
```bash
git add docs/ANALYSIS.md tests/architecture/test_analysis_doc_shape.py && git commit -m "Phase 4: ANALYSIS.md — brief's 3 questions + figures + PENDING result placeholders (spec §7)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: `docs/COST_ANALYSIS.md` — training runtime + compute envelope (§11)
**Files:**
- Create: `docs/COST_ANALYSIS.md`
- Test: `tests/architecture/test_cost_doc_shape.py`

- [ ] **Step 1: Write the failing test** — asserts the cost doc exists and carries the §11 section headers (tiktoken headline, chars/bytes appendix, AI-tooling table, training-runtime/compute envelope, architect spend cap) with PENDING placeholders for not-yet-measured numbers.

```python
"""Doc contract (spec §2/§11, TODO T04-03): COST_ANALYSIS.md has the tiktoken
headline, chars/bytes appendix, training-runtime + compute envelope, and a
named architect-decided spend cap.
"""

from __future__ import annotations

from pathlib import Path

_DOC = Path(__file__).resolve().parent.parent.parent / "docs" / "COST_ANALYSIS.md"


def test_cost_doc_sections_present() -> None:
    text = _DOC.read_text(encoding="utf-8")
    for header in (
        "## 1. Headline — tiktoken (cl100k_base)",
        "## 2. Appendix — chars & bytes",
        "## 3. AI-tooling cost",
        "## 4. Training runtime & compute envelope",
        "## 5. Cost envelope — architect spend cap vs running total",
    ):
        assert header in text, f"missing header: {header}"
    assert "src/cost/meter.py" in text
    assert "PENDING" in text
```

- [ ] **Step 2: Run test to verify it fails** —
```bash
uv run pytest "tests/architecture/test_cost_doc_shape.py" -v
```
Expected: FAIL — `docs/COST_ANALYSIS.md` does not exist yet (`FileNotFoundError`).

- [ ] **Step 3: Write minimal implementation** — author `docs/COST_ANALYSIS.md`:

```markdown
# COST_ANALYSIS — RoboVacuumDDPG (spec §11)

> Token counts are produced by `src/cost/meter.py` (tiktoken `cl100k_base`
> headline + chars/bytes appendix). Training-runtime and dollar numbers are
> **PENDING** until the seeded run + the AI-tooling tally complete — no
> invented spend (spec §10 honesty stance). The architect owns the spend cap
> (CLAUDE.md §1.4 cost-budget row).

## 1. Headline — tiktoken (cl100k_base)

Total prompt/response tokens across the build, counted by
`src/cost/meter.py` with the `cl100k_base` encoder.

| Channel | Tokens (tiktoken cl100k_base) |
|---|---|
| Architect → implementer prompts | PENDING |
| Implementer responses | PENDING |
| **Total** | PENDING |

## 2. Appendix — chars & bytes

Encoder-independent fallback (same corpus as §1), reported for reproducibility.

| Measure | Value |
|---|---|
| Characters | PENDING |
| Bytes (UTF-8) | PENDING |

## 3. AI-tooling cost

| Item | Unit | Qty | Cost |
|---|---|---|---|
| Claude Code session(s) | session | PENDING | PENDING |
| Token spend (from §1 × rate) | USD | PENDING | PENDING |
| **AI-tooling subtotal** | USD | — | PENDING |

## 4. Training runtime & compute envelope

`training.episodes = 500`, `training.seeds = [42, 7, 123, 314, 271]` (5 seeds),
`env.max_steps = 1000`. Compute is **CPU/laptop-class** (no GPU assumed); the
DDPG nets are small (`hidden_sizes = [256, 256]`).

| Quantity | Value |
|---|---|
| Episodes per seed | 500 |
| Seeds | 5 |
| Max steps / episode | 1000 |
| Wall-clock per seed | PENDING |
| Total wall-clock (5 seeds) | PENDING |
| Peak RSS | PENDING |
| Device | PENDING (CPU model) |

## 5. Cost envelope — architect spend cap vs running total

The architect set a spend cap (CLAUDE.md §1.4 *cost-budget envelope* row).

| | Amount |
|---|---|
| Architect-decided cap (USD) | PENDING |
| Running total (AI-tooling + compute) | PENDING |
| Headroom remaining | PENDING |

If the running total approaches the cap, training is time-boxed and any
partial convergence is reported honestly (spec §10), not masked.
```

- [ ] **Step 4: Run test to verify it passes** —
```bash
uv run pytest "tests/architecture/test_cost_doc_shape.py" -v
```
Expected: PASS — all five headers + `src/cost/meter.py` reference + `PENDING` present.

- [ ] **Step 5: Commit** —
```bash
git add docs/COST_ANALYSIS.md tests/architecture/test_cost_doc_shape.py && git commit -m "Phase 4: COST_ANALYSIS.md — tiktoken headline + training runtime + compute envelope (spec §11)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: `docs/QUALITY.md` — ISO/IEC 25010 eight characteristics
**Files:**
- Create: `docs/QUALITY.md`
- Test: `tests/architecture/test_quality_doc_shape.py`

- [ ] **Step 1: Write the failing test** — asserts the ISO/IEC 25010 doc names all **eight** product-quality characteristics as section headers, each with an evidence pointer, plus an honest-limitations section.

```python
"""Doc contract (spec §2, TODO T04-04): QUALITY.md covers all EIGHT ISO/IEC
25010 product-quality characteristics with evidence + honest limitations.
"""

from __future__ import annotations

from pathlib import Path

_DOC = Path(__file__).resolve().parent.parent.parent / "docs" / "QUALITY.md"

_CHARACTERISTICS = (
    "## 1. Functional Suitability",
    "## 2. Performance Efficiency",
    "## 3. Compatibility",
    "## 4. Usability",
    "## 5. Reliability",
    "## 6. Security",
    "## 7. Maintainability",
    "## 8. Portability",
)


def test_quality_doc_covers_eight_characteristics() -> None:
    text = _DOC.read_text(encoding="utf-8")
    for header in _CHARACTERISTICS:
        assert header in text, f"missing ISO 25010 characteristic: {header}"
    assert "ISO/IEC 25010" in text
    assert "## 9. Honest limitations" in text


def test_quality_doc_carries_evidence_pointers() -> None:
    text = _DOC.read_text(encoding="utf-8")
    assert "fail_under=85" in text
    assert "150" in text
    assert "ruff" in text.lower()
```

- [ ] **Step 2: Run test to verify it fails** —
```bash
uv run pytest "tests/architecture/test_quality_doc_shape.py" -v
```
Expected: FAIL — `docs/QUALITY.md` does not exist yet (`FileNotFoundError`).

- [ ] **Step 3: Write minimal implementation** — author `docs/QUALITY.md`:

```markdown
# QUALITY — RoboVacuumDDPG (ISO/IEC 25010 product-quality model)

> The eight **ISO/IEC 25010** product-quality characteristics, each with a
> concrete evidence pointer into this repo and an honest bound. Gates:
> `fail_under=85` coverage, every `.py` ≤ 150 LOC, zero ruff violations.

## 1. Functional Suitability
The simulator emits a custom `(state, reward, done, info)` 4-tuple and the
DDPG agent learns continuous coverage control. Evidence: `src/env/vacuum_env.py`,
`src/ddpg/agent.py`; hand-computed unit tests in `tests/` (kinematics K-1..K-4,
Polyak math, Tanh bounds, critic shape).

## 2. Performance Efficiency
Small MLPs (`hidden_sizes = [256, 256]`), vectorized ray–segment math
(`src/env/raycast.py`), coarse coverage grid (`coverage_cell = 0.10`). Runtime
envelope tracked in `docs/COST_ANALYSIS.md` §4.

## 3. Compatibility
Pure Python + PyTorch/NumPy/Matplotlib; no Gymnasium/Gazebo/SB3 coupling
(architecture test `tests/architecture/test_no_gymnasium_import.py`). Reads the
upstream HouseExpo JSON format unchanged (`src/env/house_map.py`).

## 4. Usability
Single entry point `RoboVacuumSDK` (`src/sdk/sdk.py`); CLI + figures, no GUI
(see `docs/UX.md`). One-command install (`uv sync --dev`) and run.

## 5. Reliability
Seeded determinism across 5 seeds `[42, 7, 123, 314, 271]`; gradient-norm clip
`grad_clip = 1.0`; target networks + Polyak soft updates prevent critic
collapse (`docs/ANALYSIS.md` Q3). Collision revert keeps the robot inside walls
(`src/env/collision.py`).

## 6. Security
No secrets in source; `.env-example` documents required env without values; no
PII in committed files (final-gate grep). No network calls at train/eval time
(dataset fetched offline by `scripts/fetch_houseexpo.py` at a pinned SHA).

## 7. Maintainability
Every `.py` ≤ **150** LOC (`scripts/check_file_sizes.py`); zero **ruff**
violations; ≥85% coverage (`fail_under=85`); strict module boundaries
(env / model / ddpg / services / sdk); ADR-001..008 record every architecture
decision.

## 8. Portability
`uv`-only, `requires-python >= 3.11`, CPU-class compute (no GPU required);
config-driven (`config/config.yaml`) so the `n_rays` ablation (8/16/24) and all
hyperparameters change without editing source.

## 9. Honest limitations
- Convergence numbers are **PENDING** until the seeded run completes; partial
  convergence will be reported honestly (spec §10), not masked.
- Raycasting cost grows with map complexity; mitigated by capped ray count and
  a coarse coverage grid, but very large held-out plans may slow eval.
- Security scope is "no secrets / no PII / offline" — there is no adversarial
  threat model; this is a research artifact, not a deployed service.
```

- [ ] **Step 4: Run test to verify it passes** —
```bash
uv run pytest "tests/architecture/test_quality_doc_shape.py" -v
```
Expected: PASS — eight ISO 25010 headers + limitations section + evidence pointers present.

- [ ] **Step 5: Commit** —
```bash
git add docs/QUALITY.md tests/architecture/test_quality_doc_shape.py && git commit -m "Phase 4: QUALITY.md — ISO/IEC 25010 eight characteristics + honest limitations

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 9: `docs/UX.md` — §10 N/A justification (CLI + static figures, no GUI)
**Files:**
- Create: `docs/UX.md`
- Test: `tests/architecture/test_ux_doc_shape.py`

- [ ] **Step 1: Write the failing test** — asserts the UX doc records the §10 N/A verdict with the exact section headers and names the CLI + static-figure surface (no GUI).

```python
"""Doc contract (spec §2 §10, submission guidelines §10): UX.md records the
N/A verdict — the surface is CLI + static figures, there is no GUI to evaluate.
"""

from __future__ import annotations

from pathlib import Path

_DOC = Path(__file__).resolve().parent.parent.parent / "docs" / "UX.md"


def test_ux_doc_records_na_verdict() -> None:
    text = _DOC.read_text(encoding="utf-8")
    assert "## 1. Verdict — §10 Not Applicable (no GUI)" in text
    assert "## 2. The interaction surface (CLI + static figures)" in text
    assert "## 3. What a GUI would have added (and why it is out of scope)" in text
    assert "RoboVacuumSDK" in text
    assert "results/figures/" in text
```

- [ ] **Step 2: Run test to verify it fails** —
```bash
uv run pytest "tests/architecture/test_ux_doc_shape.py" -v
```
Expected: FAIL — `docs/UX.md` does not exist yet (`FileNotFoundError`).

- [ ] **Step 3: Write minimal implementation** — author `docs/UX.md`:

```markdown
# UX — RoboVacuumDDPG (§10)

## 1. Verdict — §10 Not Applicable (no GUI)
Submission-guideline §10 (UX/UI) is **N/A** for this assignment: RoboVacuumDDPG
ships **no interactive GUI**. The deliverable surface is a command-line workflow
plus **static analysis figures**, which is the appropriate interface for a
from-scratch RL training/evaluation artifact (the brief grades the DDPG code +
analysis, not a UI). This mirrors the A4 precedent and is recorded here so the
N/A is an explicit, evidenced decision rather than an omission.

## 2. The interaction surface (CLI + static figures)
- **Single entry point** `RoboVacuumSDK` (`src/sdk/sdk.py`): `build_env`,
  `train`, `rollout`, `coverage_report`.
- **CLI / scripts** drive training and rendering:
  `scripts/render_learning_curve.py`, `scripts/render_critic_loss.py`,
  `scripts/render_trajectory.py`, `scripts/fetch_houseexpo.py`.
- **Static figures** under `results/figures/`: `learning_curve.png`,
  `critic_loss.png`, and the trajectory visualization over the 2D HouseExpo map.
- **Notebook** `notebooks/analysis.ipynb` consumes only the SDK and renders the
  same figures from saved artifacts (read-only, not interactive control).

## 3. What a GUI would have added (and why it is out of scope)
A live Pygame-style viewer (as in A1's DroneRL) would let a user watch the
vacuum sweep in real time, but it adds zero grading value here and would risk
the ≤150-LOC and single-entry constraints. The static trajectory figure already
proves wall-avoidance + smooth continuous coverage (spec §7), so a GUI is
deliberately out of scope. Should real-time visualization ever be wanted, it
would attach to `RoboVacuumSDK.rollout()` without touching the env/agent layers.
```

- [ ] **Step 4: Run test to verify it passes** —
```bash
uv run pytest "tests/architecture/test_ux_doc_shape.py" -v
```
Expected: PASS — N/A verdict + interaction-surface + GUI-out-of-scope headers and SDK/figures references present.

- [ ] **Step 5: Commit** —
```bash
git add docs/UX.md tests/architecture/test_ux_doc_shape.py && git commit -m "Phase 4: UX.md — §10 N/A (CLI + static figures, no GUI)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 10: `docs/shared/PROMPTS.md` — prompt log skeleton (architect → implementer trail)
**Files:**
- Create: `docs/shared/PROMPTS.md`
- Test: `tests/architecture/test_prompts_doc_shape.py`

- [ ] **Step 1: Write the failing test** — asserts the PROMPTS log exists with the §1.4 architect↔implementer framing, a per-phase prompt table mapping prompt → commit hash + human-judgment annotation, and PENDING placeholders for hashes not yet known.

```python
"""Doc contract (CLAUDE.md §1.4, TODO T04-06): PROMPTS.md is the verbatim
architect -> implementer prompt log, each mapped to a commit hash with a
human-judgment annotation.
"""

from __future__ import annotations

from pathlib import Path

_DOC = Path(__file__).resolve().parent.parent.parent / "docs" / "shared" / "PROMPTS.md"


def test_prompts_doc_skeleton() -> None:
    text = _DOC.read_text(encoding="utf-8")
    assert "Human ↔ AI Responsibility Contract" in text
    assert "| Prompt | Commit | Human-judgment annotation |" in text
    for phase in ("Phase 0", "Phase 1", "Phase 2", "Phase 3", "Phase 4"):
        assert phase in text, f"missing phase row group: {phase}"
    assert "PENDING" in text
```

- [ ] **Step 2: Run test to verify it fails** —
```bash
uv run pytest "tests/architecture/test_prompts_doc_shape.py" -v
```
Expected: FAIL — `docs/shared/PROMPTS.md` does not exist yet (`FileNotFoundError`).

- [ ] **Step 3: Write minimal implementation** — author `docs/shared/PROMPTS.md`:

```markdown
# PROMPTS — RoboVacuumDDPG (architect → implementer trail)

> Evidence for the **Human ↔ AI Responsibility Contract** (CLAUDE.md §1.4): the
> developer is the architect (decides PRD/architecture/acceptance/sign-off), the
> AI is the implementer (code against an approved spec). Each row records the
> verbatim prompt, the commit it produced, and the human-judgment call attached.
> Commit hashes are **PENDING** until each phase lands; this file is updated as
> commits are made (matching the per-section discipline of the A1 PROMPTS log).

## How to read this log
- **Prompt** — the literal instruction given to the implementer.
- **Commit** — the resulting commit hash (subject `^(Phase \d+|...)`).
- **Human-judgment annotation** — the architect-decided, non-delegable call
  (CLAUDE.md §1.4 table) that gated or shaped the prompt.

## Phase 0 — Bootstrap
| Prompt | Commit | Human-judgment annotation |
|---|---|---|
| "Stand up the V3 scaffold (uv, ruff, pytest-cov fail_under=85, ≤150-LOC guard, CI, docs/ + ADR stubs)." | PENDING | Architect chose the gate thresholds and module boundaries before any code. |

## Phase 1 — Simulator from scratch
| Prompt | Commit | Human-judgment annotation |
|---|---|---|
| "Implement env units (house_map, raycast, kinematics, coverage, collision, reward, state, vacuum_env) TDD with hand-computed expectations." | PENDING | Architect fixed the MDP (state 20-dim, reward signs, 4-tuple, no Gym). |

## Phase 2 — DDPG from scratch
| Prompt | Commit | Human-judgment annotation |
|---|---|---|
| "Implement Actor (Tanh), Critic (state⊕action), ReplayBuffer, Gaussian noise, DDPGAgent (Polyak soft-update), Trainer — TDD." | PENDING | Architect chose Gaussian-not-OU (ADR-003), τ=0.005, LR split (ADR-007). |

## Phase 3 — Training + Results
| Prompt | Commit | Human-judgment annotation |
|---|---|---|
| "Fetch HouseExpo at pinned SHA, train 5 seeds, render learning_curve / critic_loss / trajectory, held-out generalization." | PENDING | Architect set seeds, episode budget, and the held-out split (ADR-008). |

## Phase 4 — Docs + Analysis + Gates
| Prompt | Commit | Human-judgment annotation |
|---|---|---|
| "Author architecture tests, ANALYSIS (3 questions), COST_ANALYSIS, QUALITY (ISO 25010), UX (§10 N/A), README, cover sheet; run final gates; tag v1.0.0." | PENDING | Architect signs off the self-grade (cover sheet only) and the submission. |
```

- [ ] **Step 4: Run test to verify it passes** —
```bash
uv run pytest "tests/architecture/test_prompts_doc_shape.py" -v
```
Expected: PASS — §1.4 framing, prompt/commit/annotation table, all five phase groups, and PENDING markers present.

- [ ] **Step 5: Commit** —
```bash
git add docs/shared/PROMPTS.md tests/architecture/test_prompts_doc_shape.py && git commit -m "Phase 4: PROMPTS.md — architect→implementer prompt log skeleton (CLAUDE.md §1.4)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 11: Finalize `README.md` with embedded figures
**Files:**
- Modify: `README.md`
- Test: `tests/architecture/test_readme_finalized.py`

- [ ] **Step 1: Write the failing test** — asserts the README embeds all three figures, documents the `uv` install/run + SDK usage, links the analysis docs, and keeps the honest `PENDING` convergence framing (spec §10).

```python
"""Doc contract (spec §7/§10, TODO T04-05): README is the submission-report
shell — embeds the three figures, the uv install/run + SDK usage, links the
analysis docs, and keeps the honest PENDING convergence framing.
"""

from __future__ import annotations

from pathlib import Path

_DOC = Path(__file__).resolve().parent.parent.parent / "README.md"


def test_readme_embeds_three_figures() -> None:
    text = _DOC.read_text(encoding="utf-8")
    assert "results/figures/learning_curve.png" in text
    assert "results/figures/critic_loss.png" in text
    assert "results/figures/trajectory.png" in text


def test_readme_has_install_run_sdk_and_links() -> None:
    text = _DOC.read_text(encoding="utf-8")
    assert "uv sync --dev" in text
    assert "RoboVacuumSDK" in text
    assert "docs/ANALYSIS.md" in text
    assert "docs/THEORY.md" in text
    assert "PENDING" in text
```

- [ ] **Step 2: Run test to verify it fails** —
```bash
uv run pytest "tests/architecture/test_readme_finalized.py" -v
```
Expected: FAIL — the current README does not yet embed `results/figures/critic_loss.png` / `trajectory.png` nor link `docs/ANALYSIS.md` + `docs/THEORY.md` together (assertion failure).

- [ ] **Step 3: Write minimal implementation** — append a **Deliverables (brief §7)** section to the existing `README.md` (do not rewrite the intro/install). Insert this block before the final references/footer:

```markdown
## 3. Deliverables (brief §7)

> **Status: training not yet run — all result numbers are `PENDING`** (spec §10
> honesty stance; no invented metrics). Figures regenerate via the render
> scripts below.

### 3.1 Quick run (uv only)

```bash
uv sync --dev
uv run python scripts/fetch_houseexpo.py      # pinned-SHA HouseExpo subset
uv run python -c "from src.sdk.sdk import RoboVacuumSDK; \
print(len(RoboVacuumSDK('config/config.yaml').train(seed=42)))"
uv run python scripts/render_learning_curve.py
uv run python scripts/render_critic_loss.py
uv run python scripts/render_trajectory.py
```

All business logic is reached only through `RoboVacuumSDK` (`src/sdk/sdk.py`):
`build_env`, `train`, `rollout`, `coverage_report`.

### 3.2 Learning curve (cumulative reward vs episode, mean ± CI over 5 seeds)
![Learning curve](results/figures/learning_curve.png)
*PENDING — regenerated by `scripts/render_learning_curve.py`.*

### 3.3 Critic loss (vs training step)
![Critic loss](results/figures/critic_loss.png)
*PENDING — regenerated by `scripts/render_critic_loss.py`.*

### 3.4 Trajectory over the 2D HouseExpo map (covered area shaded)
![Trajectory](results/figures/trajectory.png)
*PENDING — greedy rollout; proves wall-avoidance + smooth coverage.*

### 3.5 Held-out generalization
Train on `maps.train`, evaluate on `maps.holdout` (`apt_large`, `office`).
Coverage on held-out maps: **PENDING** (reported honestly per spec §10).

### 3.6 Deeper analysis
- `docs/THEORY.md` — DDPG objective, deterministic policy gradient, Polyak update.
- `docs/ANALYSIS.md` — the brief's three analysis questions.
- `docs/COST_ANALYSIS.md` — tiktoken + training runtime + compute envelope.
- `docs/QUALITY.md` — ISO/IEC 25010. `docs/UX.md` — §10 N/A (CLI + figures).
```

- [ ] **Step 4: Run test to verify it passes** —
```bash
uv run pytest "tests/architecture/test_readme_finalized.py" -v
```
Expected: PASS — three figures embedded, `uv sync --dev` + `RoboVacuumSDK` usage, both doc links, and `PENDING` framing present.

- [ ] **Step 5: Commit** —
```bash
git add README.md tests/architecture/test_readme_finalized.py && git commit -m "Phase 4: finalize README — embed 3 figures + SDK run + doc links + honest PENDING framing (spec §7/§10)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 12: FINAL GATES — coverage ≥85, ruff check, ruff format --check, file-size guard; cover sheet `adrl-001-ex05.pdf`; tag `v1.0.0`
**Files:**
- Create: `scripts/check_file_sizes.py`
- Create: `tests/architecture/test_final_gates.py`
- Create: `adrl-001-ex05.pdf` (from the official Moodle template)
- Test: `tests/architecture/test_final_gates.py`

- [ ] **Step 1: Write the failing test** — asserts the file-size guard script exists and reports green (every `.py` ≤150 LOC), and that the submission cover sheet `adrl-001-ex05.pdf` exists and carries the group code.

```python
"""Final-gate contract (CLAUDE.md §1/§6, spec §2, TODO T04-08/09): the
≤150-LOC guard runs green and the official cover sheet adrl-001-ex05.pdf exists.
The coverage/ruff/format gates are asserted by the Step-4 command sweep.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_file_size_guard_passes() -> None:
    script = _REPO_ROOT / "scripts" / "check_file_sizes.py"
    assert script.exists(), "scripts/check_file_sizes.py missing"
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(script)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"file-size guard failed:\n{result.stdout}\n{result.stderr}"


def test_submission_cover_sheet_exists() -> None:
    pdf = _REPO_ROOT / "adrl-001-ex05.pdf"
    assert pdf.exists(), "adrl-001-ex05.pdf (Moodle cover sheet) missing"
    assert pdf.stat().st_size > 0, "cover sheet is empty"
```

- [ ] **Step 2: Run test to verify it fails** —
```bash
uv run pytest "tests/architecture/test_final_gates.py" -v
```
Expected: FAIL — `scripts/check_file_sizes.py` does not exist yet and `adrl-001-ex05.pdf` is not yet produced (`AssertionError`).

- [ ] **Step 3: Write minimal implementation** — author `scripts/check_file_sizes.py` (mirrors A4; counts non-blank, non-comment lines, excludes vendored dirs):

```python
"""Fail if any committed .py file exceeds 150 lines of code (CLAUDE.md §1).

Counts lines excluding blank lines and pure-comment lines, so docstrings count
but `# divider` separators don't. Exit 1 if any file is over the limit.
"""

from __future__ import annotations

import sys
from pathlib import Path

LIMIT = 150
EXCLUDED_DIRS = {
    ".venv", ".git", "build", "dist", "__pycache__",
    ".ruff_cache", ".pytest_cache", "vendor", "data",
}


def count_loc(path: Path) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()
    loc = 0
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        loc += 1
    return loc


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    over: list[tuple[Path, int]] = []
    for path in root.rglob("*.py"):
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        loc = count_loc(path)
        if loc > LIMIT:
            over.append((path.relative_to(root), loc))
    if over:
        print(f"{len(over)} file(s) exceed {LIMIT} LOC:")
        for path, loc in over:
            print(f"  {path}: {loc} LOC")
        return 1
    print(f"all .py files <= {LIMIT} LOC")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Then produce the cover sheet from the official Moodle template (filled with group code `adrl-001`, assignment `ex05`, self-grade only on the PDF) and save it as `adrl-001-ex05.pdf` at the repo root:

```bash
# Fill the official Moodle ex05 cover-sheet template (group code adrl-001,
# self-grade ONLY on this PDF per spec §2) and export to the repo root.
# (Manual fill of the provided template; verify it is a non-empty PDF.)
test -s adrl-001-ex05.pdf && echo "cover sheet present" || echo "MISSING cover sheet"
```

- [ ] **Step 4: Run test to verify it passes** — run the file-size+cover-sheet test, then the full gate sweep:
```bash
uv run pytest "tests/architecture/test_final_gates.py" -v
uv run pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=85
uv run ruff check src/ tests/ scripts/
uv run ruff format --check src/ tests/ scripts/
uv run python scripts/check_file_sizes.py
```
Expected: PASS — `test_final_gates.py` green; coverage ≥85% (`fail_under=85`); `ruff check` reports zero violations; `ruff format --check` reports nothing to reformat; `check_file_sizes.py` prints `all .py files <= 150 LOC` and exits 0.

- [ ] **Step 5: Commit + tag** —
```bash
git add scripts/check_file_sizes.py tests/architecture/test_final_gates.py adrl-001-ex05.pdf && \
git commit -m "Phase 4: final gates green (cov>=85, ruff check+format, <=150 LOC) + cover sheet adrl-001-ex05.pdf

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>" && \
git tag -a v1.0.0 -m "RoboVacuumDDPG v1.0.0 — Assignment 5 submission (adrl-001-ex05)"
```
Expected: tag `v1.0.0` created on the final Phase-4 commit.
