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
| **Fetch** the upstream dataset reproducibly | `scripts/fetch_houseexpo.py` | download `HouseExpo/json.tar.gz` at pinned SHA → extract → `data/houseexpo_full/` (git-ignored) |
| **Vendor** a small curated, committed subset | `data/maps/*.json` | 5 real plans, the only map bytes in the repo |
| **Parse** one JSON plan → geometry the env can raycast against | `src/env/house_map.py` (+ `_house_map_geom.py`) | real JSON → wall segments + free-space bounds + boundary polygon |

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
# HouseExpo full dataset (~35k JSON + the json.tar.gz) — git-ignored; vendor
# only the curated subset under data/maps/
data/houseexpo_full/
data/raw/
vendor/
```

So the contract is a deliberate split, mirrored from the spec:

- **`data/houseexpo_full/`** — full extract target, **git-ignored**.
  Reproducible from `scripts/fetch_houseexpo.py` at the pinned SHA (downloads the
  upstream `HouseExpo/json.tar.gz` — all 35 126 real maps — and extracts to
  `data/houseexpo_full/json/`). Never committed.
- **`data/maps/`** — the **5 curated real plans**, the *only* map JSON committed
  to the repo, copied byte-for-byte from the extracted dataset. Hand-picked to
  span difficulty single-room → multi-room apartment (by `room_num`) so the graph
  in `results/figures/` and the held-out generalization test are meaningful, not
  a single trivial box.

This is not laziness — it is reproducibility. The grader can: (a) run the full
training/eval pipeline today on the vendored subset with a fresh clone, and
(b) re-run `scripts/fetch_houseexpo.py` to reconstruct `data/houseexpo_full/`
bit-for-bit at the same SHA and re-derive the curated subset. The committed
subset and the reconstructible full set are pinned to the same upstream commit,
so they cannot silently diverge — the curated `data/maps/<name>.json` are the
exact bytes the fetch script copies out of the extracted dataset.

## 3. Fetch script contract (`scripts/fetch_houseexpo.py`)

The fetch script is the reproducibility seam for the *raw* dataset. It is a
script (under `scripts/`, lint-clean per CLAUDE.md §5), NOT business logic — it
imports the SDK/config loader for paths only and writes to git-ignored
locations.

Behaviour (now a **real** fetch, no longer a silent no-op):

1. Read the dataset source from `config/config.yaml` → `maps.dataset_repo`,
   `maps.archive_path` (`"HouseExpo/json.tar.gz"`), `maps.dataset_sha`. Never
   hardcode the URL/path in the script body.
2. **Download the real archive.** Build the pinned
   `raw.githubusercontent.com/<owner/repo>/<sha>/HouseExpo/json.tar.gz` URL and
   download it (~25 MB) into `data/houseexpo_full/json.tar.gz` (git-ignored;
   skipped if already present). This is the upstream blob holding all 35 126 real
   maps. Equivalent to the repo's own `git clone … && tar -xvzf json.tar.gz`.
3. **Extract** the `*.json` members with Python's stdlib `tarfile` (a gzip tar —
   **no `7z` required**) into `data/houseexpo_full/json/<id>.json`.
4. **Pin the commit.** `config.maps.dataset_sha` is a concrete 40-char SHA
   (`45e2b25…`, the real HouseExpo HEAD at fetch); the script **stamps the
   resolved SHA back into `config/config.yaml`** so provenance is in version
   control even though the bytes are git-ignored. Re-running is idempotent.
5. **Copy the curated subset.** Using the human-decided `config.maps.curated_ids`
   mapping (logical name → real HouseExpo id), copy the 5 real `<id>.json` files
   into `data/maps/<logical-name>.json`, then print a manifest (sizes + counts).

The *selection* of which 5 real maps to curate is the human-decided requirements
concern (CLAUDE.md §1.4: scope is non-delegable) — it lives in
`config.maps.curated_ids`, hand-chosen to span `room_num` 1→8. The script only
*executes* that pinned mapping deterministically; it never invents the choice.

> `7z` is **not** needed: the dataset ships as a gzip tar (`json.tar.gz`), which
> `tarfile` handles natively. If a future HouseExpo release switches to `json.7z`,
> install p7zip (`brew install p7zip` / `apt-get install p7zip-full`) and extract
> with `7z x json.7z`, or `uv add py7zr` and swap the extractor — the curated
> bytes already committed under `data/maps/` keep the headline runs reproducible
> regardless.

## 4. Vendored subset (`data/maps/`)

The curated plans are addressed by the *logical names* in `config.maps`, which
the SDK resolves to `data/maps/<name>.json` under `config.paths.maps_dir`
(`"data/maps"`):

| Split | Logical name | Real HouseExpo id | `room_num` | bbox extent | walls |
|---|---|---|---|---|---|
| Train | `room_single` | `011bef0381ddca9381a7aeb1d0b0777d` | 1 | 6.8 × 4.3 m | 17 |
| Train | `apt_small` | `000514ade3bcc292a613a4c2755a5050` | 3 | 8.9 × 6.1 m | 37 |
| Train | `apt_multi` | `0016652bf7b3ec278d54e0ef94476eb8` | 5 | 10.1 × 7.6 m | 54 |
| Holdout | `apt_large` | `0099c392987b98355e984f7d06837dbc` | 6 | 10.9 × 7.3 m | 55 |
| Holdout | `office` | `041bd68db15bac8be2d96ca062c23d2f` | 8 | 5.0 × 10.1 m | 44 |

That is **5 curated REAL plans** total — inside the spec's "4–6 curated plans"
band — spanning `room_num` 1 → 8 (single-room → multi-room). Each is a real,
non-convex HouseExpo boundary (17–55 wall segments, not a 4-edge box). The
logical→id mapping lives in `config.maps.curated_ids`; the loader/SDK resolve
logical names to `data/maps/<name>.json` and never enumerate a hardcoded filename
list. Adding/removing a map is a `config.yaml` edit (`curated_ids` + `train`/
`holdout`) plus the corresponding `data/maps/<name>.json`, no source change
(CLAUDE.md §4, no hardcoded values).

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

Unit conversion lives inside this module so callers receive metres directly —
consistent with `env.robot_radius` (0.17 m) and the rest of `config.env`. The
**real** HouseExpo `verts` / `bbox` coordinates are already metric floats (e.g. a
plan spans `bbox.max = [6.9, 4.41]` → a ~7 m × 4 m room), so the documented scale
is `config.maps.coord_scale = 1.0` (identity). The factor is config-driven, not a
literal, so a pixel-resolution source could be rescaled without touching source
(CLAUDE.md §4). No raw coordinate leaks past the adapter un-scaled.

The module stays ≤150 LOC (CLAUDE.md §1); the JSON-shape handling and the
geometry derivation are split into a sibling `_house_map_geom.py` helper
(vertex extraction, polygon closure, wall derivation, bounds, point-in-polygon)
per the spec's A4 convention, leaving `house_map.py` a thin loader.

### 5.2 Real schema + boundary-only walls (the documented simplification)

The **real** HouseExpo JSON (confirmed against the pinned dataset, e.g.
`data/maps/room_single.json`, id `011bef0381…`) carries these keys:

| Key | Type | Use in the adapter |
|---|---|---|
| `id` | string | provenance only (not parsed into geometry) |
| `room_num` | int | curation signal (single-room → multi-room); not parsed |
| `bbox` | `{min:[x,y], max:[x,y]}` (metres) | free-space `bounds` |
| `verts` | list of `[x,y]` (metres, **open** contour) | exterior boundary → walls + `is_inside` polygon |
| `room_category` | `{room_type: [[x1,y1,x2,y2], …]}` | **NOT used** (see below) |

**Walls = the exterior boundary polygon.** The `verts` contour is the real,
non-convex floor-plan outline (17–55 vertices across the curated set). It is an
*open* ring (first vertex ≠ last), so the loader closes it (last→first edge) and
emits one wall segment per closed-ring edge. These are genuine HouseExpo
geometry, not synthetic boxes.

**`is_inside` is a proper point-in-polygon test** (ray-casting even-odd) against
that closed boundary — **not** the axis-aligned bbox. This is load-bearing: on a
non-convex plan (e.g. the L-shaped `office`, whose bbox *centre* is in a void),
the bbox would mark exterior void as "free". `coverage.py` enumerates cleanable
cells via `HouseMap.is_inside`, so point-in-polygon is what makes the coverage %
and the `coverage_target` honest on real layouts.

**`room_category` is deliberately NOT turned into interior walls.** Inspection of
the real dataset shows these boxes are *overlapping, nested semantic region
annotations* (e.g. `Living_Room` and `Dining_Room` share an identical box; some
boxes extend outside the house bbox with negative coords). They do **not** encode
clean interior wall or door segments. Per the brief and CLAUDE.md, we do **not
fabricate** interior walls from them — that would invent geometry the dataset
does not assert. The adapter therefore ships the **exterior boundary polygon as
the wall set** (real, non-convex geometry) plus point-in-polygon free space. This
is a documented simplification (also recorded in ADR-005): the maps are real and
non-convex, but interior partitions are not modelled because HouseExpo does not
provide them as wall geometry. If a future map source carries explicit interior
segments, they slot in behind the same `walls` contract with no consumer change.

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

A4. **Subset size in band, real geometry.** `data/maps/` contains 4–6 committed
    `*.json` plans (currently 5 REAL HouseExpo plans: `room_single`, `apt_small`,
    `apt_multi`, `apt_large`, `office`, each with 17–55 boundary wall segments —
    not a 4-edge box); `data/houseexpo_full/` is git-ignored (asserted via
    `.gitignore`).

A5. **Train/holdout disjoint.** `set(maps.train) ∩ set(maps.holdout) == ∅`, so
    the generalization eval (ADR-008) never tests on a trained layout.

A6. **Fetch reproducibility.** `config.maps.dataset_sha` is a concrete 40-char
    SHA (`45e2b25…`, no longer the `"PINNED_AT_FETCH"` sentinel); re-running
    `scripts/fetch_houseexpo.py` re-downloads/extracts the same pinned archive and
    re-stamps the same SHA (idempotent). (Smoke-level; the network fetch is not
    part of the ≥85% core coverage gate — the pure helpers `archive_url`,
    `extract_json`, `copy_curated_by_id`, `stamp_sha` are unit-tested.)

A7. **Point-in-polygon, not bbox.** `HouseMap.is_inside` is an even-odd
    point-in-polygon test over the exterior boundary, asserted on a real plan by a
    point that is inside the bbox but outside the polygon (`room_single` at
    `(0.5, 4.0)`), and on the L-shaped `office` whose bbox centre is exterior.

## 7. References

- Design spec §6 "Data — HouseExpo" — fetch-at-pinned-SHA, 4–6 curated plans,
  train/holdout split, citation. The contract this PRD fills in.
- Design spec §3 (MDP) — `reset()` random free-cell spawn; lidar raycast to
  walls; consumers of `bounds` / `walls` / `free-space`.
- `config/config.yaml` → `maps` (`dataset_repo`, `dataset_sha`, `archive_path`,
  `coord_scale`, `curated_ids`, `train`, `holdout`) and `paths.maps_dir`,
  `env.ray_max` / `coverage_cell` / `robot_radius` — the values this adapter is
  config-driven by.
- `src/env/house_map.py` (+ `src/env/_house_map_geom.py`) — the parse module and
  its geometry helper (spec §4 module map).
- `scripts/fetch_houseexpo.py` (+ `scripts/_fetch_archive.py`) — the real
  fetch/extract/pin script and its download helper (spec §6).
- ADR-005 — "HouseExpo adapter + pinned subset" (decision record for this layer).
- ADR-008 — "multi-seed eval + held-out generalization" (consumer of the split).
- CLAUDE.md §1 (≤150 LOC), §4 (no hardcoded values), §1.4 (scope/curation is
  human-decided).
- **Citation.** Li, T., Ho, D., Li, C., Zhu, D., Wang, C., Meng, M. Q.-H. (2019).
  *HouseExpo: A Large-scale 2D Indoor Layout Dataset for Learning-based Algorithms
  on Mobile Robots.* arXiv:1903.09845. Source repo `TeaganLi/HouseExpo`.
