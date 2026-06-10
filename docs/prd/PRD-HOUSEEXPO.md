# PRD-HOUSEEXPO — HouseExpo Data Adapter

> Component PRD for the HouseExpo map data layer. Realises the design spec
> (`docs/superpowers/specs/2026-06-10-robovacuum-ddpg-design.md`) §6 "Data —
> HouseExpo" and the `maps:` block of `config/config.yaml`.
> Status: DRAFT. Author lens: external researcher reading the brief cold,
> asking "where do the floor-plans come from and how do they become walls?".
> Owns: `scripts/fetch_houseexpo.py`, `data/maps/`, `src/env/house_map.py`.

## 1. Purpose

The simulator is graded on reading **real** floor-plans, not toy boxes. The
design spec §6 names exactly one source:

> `scripts/fetch_houseexpo.py` clones `github.com/TeaganLi/HouseExpo` at a
> pinned commit (full ~35k JSON **git-ignored**). Vendor **4–6 curated plans**
> (single-room → multi-room apartment) under `data/maps/`. Train on a subset;
> hold out 1–2 for a **generalization** evaluation. HouseExpo cited
> (Li et al. 2019, arXiv:1903.09845).

This PRD pins down the seam between *that raw dataset* and the rest of `src/`.
Three responsibilities, three artefacts, one invariant:

| Concern | Artefact | Boundary |
|---|---|---|
| **Fetch** the upstream dataset reproducibly | `scripts/fetch_houseexpo.py` | clone at pinned SHA → `data/raw/` (git-ignored) |
| **Vendor** a small curated, committed subset | `data/maps/*.json` | 4–6 plans, the only map bytes in the repo |
| **Parse** one JSON plan → geometry the env can raycast against | `src/env/house_map.py` | JSON → wall segments + free-space bounds |

`src/env/house_map.py` is the **only** module the rest of the simulator
(`raycast.py`, `collision.py`, `coverage.py`, `vacuum_env.py`) talks to for map
geometry. The fetch script and the raw dataset are *upstream* of the repo's
public surface — nothing under `src/` imports from `data/raw/`. This is the
adapter pattern: the env depends on a stable `HouseMap` value object, never on
HouseExpo's on-disk JSON shape.

## 2. Why a vendored subset (not the full ~35k dataset)

HouseExpo ships ~35,000 JSON floor-plans (GB-scale). Committing that is hostile
to the grader's clone and violates the repo's lightweight intent. The `.gitignore`
already encodes the decision:

```gitignore
# A5-specific: instructions/ is local-only ...
# HouseExpo full dataset (~35k JSON, ~GBs) — git-ignored; vendor only the
# curated subset under data/maps/
data/raw/
vendor/
```

So the contract is a deliberate split, mirrored from the spec:

- **`data/raw/`** — full clone target, **git-ignored**. Reproducible from
  `scripts/fetch_houseexpo.py` at the pinned SHA. Never committed.
- **`data/maps/`** — the **curated 4–6 plans**, the *only* map JSON committed to
  the repo. Hand-picked to span difficulty single-room → multi-room apartment so
  the graph in `results/figures/` and the held-out generalization test are
  meaningful, not a single trivial box.

This is not laziness — it is reproducibility. The grader can: (a) run the full
training/eval pipeline today on the vendored subset with a fresh clone, and
(b) re-run `scripts/fetch_houseexpo.py` to reconstruct `data/raw/` bit-for-bit
at the same SHA and re-curate if they wish. The committed subset and the
reconstructible full set are pinned to the same upstream commit, so they cannot
silently diverge.

## 3. Fetch script contract (`scripts/fetch_houseexpo.py`)

The fetch script is the reproducibility seam for the *raw* dataset. It is a
script (under `scripts/`, lint-clean per CLAUDE.md §5), NOT business logic — it
imports the SDK/config loader for paths only and writes to git-ignored
locations.

Behaviour:

1. Read the dataset source from `config/config.yaml` → `maps.dataset_repo`
   (`"https://github.com/TeaganLi/HouseExpo"`). Never hardcode the URL in the
   script body.
2. Clone (shallow is fine) into `data/raw/` — the git-ignored target.
3. **Pin the commit.** `config.maps.dataset_sha` ships as the sentinel
   `"PINNED_AT_FETCH"`; the script checks out a specific commit and **stamps the
   resolved 40-char SHA back into `config/config.yaml`** (replacing the
   sentinel), so the exact upstream state is recorded in version control even
   though the bytes are not. A second run with a real SHA checks out that SHA
   exactly (idempotent, deterministic).
4. Emit a short manifest (path + per-file count) so curation is auditable.

The script does **not** auto-curate `data/maps/` — selecting the 4–6 plans is a
human-decided requirements concern (CLAUDE.md §1.4: scope is non-delegable). The
curated subset is committed by hand from the raw clone.

## 4. Vendored subset (`data/maps/`)

The curated plans are addressed by the *logical names* in `config.maps`, which
the SDK resolves to `data/maps/<name>.json` under `config.paths.maps_dir`
(`"data/maps"`):

| Split | `config.maps` key | Names (logical) | Intent |
|---|---|---|---|
| Train | `maps.train` | `room_single`, `apt_small`, `apt_multi` | single-room → multi-room, increasing complexity |
| Holdout | `maps.holdout` | `apt_large`, `office` | unseen at train time → generalization eval |

That is **5 curated plans** total — inside the spec's "4–6 curated plans" band.
The names are config-driven: the loader never enumerates a hardcoded filename
list. Adding/removing a map is a `config.yaml` edit plus the corresponding
`data/maps/<name>.json`, no source change (CLAUDE.md §4, no hardcoded values).

The **train/holdout split** is the data-layer half of design-spec §6's
"generalization" requirement and ADR-008 (multi-seed eval + held-out
generalization): the DDPG agent trains only on `maps.train`; `maps.holdout`
plans are loaded **only** at evaluation to measure coverage on layouts the actor
never saw. The adapter's job is purely to surface both sets by name; the
trainer/evaluator (`src/services/trainer.py`, `RoboVacuumSDK.evaluate`) decides
which set to draw from.

## 5. Parse semantics (`src/env/house_map.py`)

The single transformation this module owns:

> **one HouseExpo JSON floor-plan → (`walls`, `bounds`, `free-space`)** —
> a `HouseMap` value object the env can raycast and collision-test against.

- **Wall segments** — the geometry `raycast.py` (ray–segment intersection) and
  `collision.py` (robot-radius vs segment) consume. Each wall is a 2D segment
  `((x1, y1), (x2, y2))` in metres, derived from the HouseExpo room contour /
  occupancy boundary. A map yields a *set* of wall segments (≥1).
- **Bounds** — the axis-aligned extent `(x_min, y_min, x_max, y_max)` of the
  free space, in metres. Used to size the coverage grid (`coverage.py`, cell
  edge `env.coverage_cell` = 0.10 m), to clamp `raycast.py` to `env.ray_max`
  (5.0 m), and to sample a valid spawn cell in `reset()`.
- **Free space** — the navigable interior (inside walls), so `reset()` can
  re-spawn the robot at a *random free cell* (spec §3) and `coverage.py` can
  enumerate the cleanable cells for the coverage %.

Unit conversion (HouseExpo stores plans at a metric/pixel resolution) lives
inside this module so callers receive metres directly — consistent with
`env.robot_radius` (0.17 m) and the rest of `config.env`. No raw-pixel values
leak past the adapter.

The module stays ≤150 LOC (CLAUDE.md §1); if JSON-shape handling and the
geometry derivation together approach the cap, the spec's A4 convention applies
— split a sibling `_house_map_parse.py` helper.

### 5.1 Determinism (the invariant we ship)

> Parsing the **same** `data/maps/<name>.json` twice MUST yield byte-identical
> wall segments and bounds — same count, same order, same coordinates.

(Canonical invariant — do not paraphrase.) Practically: segment ordering is
stabilised (sorted by endpoint coordinates), float conversion is fixed-precision,
and no set/dict iteration order leaks into the output list. This is load-bearing
because raycast distances feed the 16-dim lidar slice of the observation
(spec §3) and the replay buffer — a non-deterministic parse would make seeded
DDPG rollouts irreproducible and break the multi-seed reporting in ADR-008.

## 6. Acceptance criteria

A1. **Loader yields valid geometry.** For every name in `maps.train ∪
    maps.holdout`, `HouseMap` load returns **≥1 wall segment** and a finite
    `bounds` tuple with `x_max > x_min` and `y_max > y_min`. (Hand-checked
    against the vendored JSON in a unit test.)

A2. **Deterministic parse.** Loading the same map twice returns equal wall sets
    (same length, same ordering, same coordinates) — the §5.1 invariant,
    asserted directly.

A3. **Config-driven, no hardcoded list.** The set of available maps equals
    `config.maps.train + config.maps.holdout` resolved under
    `config.paths.maps_dir`; the loader contains no literal filename. Removing a
    name from config (or its JSON) is the only way to drop a map.

A4. **Subset size in band.** `data/maps/` contains 4–6 committed `*.json` plans
    (currently 5: `room_single`, `apt_small`, `apt_multi`, `apt_large`,
    `office`); `data/raw/` is git-ignored (asserted via `.gitignore`).

A5. **Train/holdout disjoint.** `set(maps.train) ∩ set(maps.holdout) == ∅`, so
    the generalization eval (ADR-008) never tests on a trained layout.

A6. **Fetch reproducibility.** After `scripts/fetch_houseexpo.py` runs,
    `config.maps.dataset_sha` is a concrete 40-char SHA (no longer the
    `"PINNED_AT_FETCH"` sentinel); re-running with that SHA checks out the same
    commit. (Smoke-level; the script's clone is not part of the ≥85% core
    coverage gate.)

## 7. References

- Design spec §6 "Data — HouseExpo" — fetch-at-pinned-SHA, 4–6 curated plans,
  train/holdout split, citation. The contract this PRD fills in.
- Design spec §3 (MDP) — `reset()` random free-cell spawn; lidar raycast to
  walls; consumers of `bounds` / `walls` / `free-space`.
- `config/config.yaml` → `maps` (`dataset_repo`, `dataset_sha`, `train`,
  `holdout`) and `paths.maps_dir`, `env.ray_max` / `coverage_cell` /
  `robot_radius` — the values this adapter is config-driven by.
- `src/env/house_map.py` — the parse module (spec §4 module map).
- `scripts/fetch_houseexpo.py` — the fetch/pin script (spec §6).
- ADR-005 — "HouseExpo adapter + pinned subset" (decision record for this layer).
- ADR-008 — "multi-seed eval + held-out generalization" (consumer of the split).
- CLAUDE.md §1 (≤150 LOC), §4 (no hardcoded values), §1.4 (scope/curation is
  human-decided).
- **Citation.** Li, T., Ho, D., Li, C., Zhu, D., Wang, C., Meng, M. Q.-H. (2019).
  *HouseExpo: A Large-scale 2D Indoor Layout Dataset for Learning-based Algorithms
  on Mobile Robots.* arXiv:1903.09845. Source repo `TeaganLi/HouseExpo`.
