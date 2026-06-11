# ADR-005 — HouseExpo Adapter + Pinned Curated Subset

| Field | Value |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-06-10 |
| **Deciders** | Architect (human, CLAUDE.md §1.4) + AI (implementer) |
| **Supersedes** | — |
| **Related** | ADR-001 (from-scratch env consumes these maps), ADR-004 (free-space bounds for the coverage grid), ADR-008 (held-out generalization), design spec §6, §4 |

## Context

The brief requires training on **real floor-plans**. The design spec (§6) fixes
the source as **HouseExpo** (Li et al. 2019, arXiv:1903.09845), via the
repository `github.com/TeaganLi/HouseExpo`. The full dataset is ~35k JSON files —
far too large to vendor into a course repo, and a moving target if pulled live.

Two problems follow: (1) **reproducibility** — a grader on a fresh checkout must
get the *same* maps we trained on, byte-for-byte; (2) **repo hygiene / V3 gates**
— 35k upstream JSON files would blow past sensible repo size and the third-party
content is not ours to lint or count toward coverage. We also need a clean
seam between "raw HouseExpo JSON" and "geometry the simulator understands"
(wall segments + free-space bounds), because `raycast.py`, `coverage.py`, and
`collision.py` all consume that geometry (design spec §4).

## Decision

Add `scripts/fetch_houseexpo.py` that **downloads the real HouseExpo dataset at a
pinned commit** — the upstream ships every map as `HouseExpo/json.tar.gz`, so the
script fetches that single ~25 MB blob from `raw.githubusercontent.com` at the
pinned SHA and extracts all 35 126 JSON maps with Python's stdlib `tarfile`
(**no `7z` needed** — it is a gzip tar). The resolved SHA
(`45e2b2505f6ea1fe49c0203f14efb7ce20b94e7c`, the real HouseExpo HEAD at fetch) is
stamped into `config/config.yaml#maps.dataset_sha`. The full ~35k JSON tree is
**git-ignored**; only a **curated subset of 5 REAL plans** (single-room →
multi-room apartment, `room_num` 1→8) is vendored under `data/maps/` (design spec
§6), copied byte-for-byte by the script via the `config.maps.curated_ids`
logical-name → real-id mapping.

**Adapter seam.** `src/env/house_map.py` (+ geometry helper `_house_map_geom.py`)
converts a *real* HouseExpo JSON plan into the simulator's geometry: **wall
segments + free-space bounds + boundary polygon**. Everything downstream (raycast
ray–segment intersection, coverage-grid sizing per ADR-004, collision tests,
random free-cell spawn per ADR-002) consumes only this adapter output — never raw
HouseExpo JSON. This keeps the dataset format isolated behind one ≤150-LOC module
and makes the loader unit-testable on the real schema (design spec §8: "HouseExpo
loader" is a tested deterministic unit).

**Walls = exterior boundary; `is_inside` = point-in-polygon (documented
simplification).** The real `verts` contour is an *open*, non-convex floor-plan
outline (17–55 vertices in the curated set); the loader closes it into a ring,
emits one wall per edge, and implements `is_inside` as a ray-casting
point-in-polygon test over that ring — **not** the axis-aligned bbox (so the
L-shaped `office`, whose bbox centre is exterior, exposes only real interior free
space to `coverage.py`). HouseExpo's `room_category` boxes are *overlapping,
nested semantic region annotations* (identical boxes for distinct rooms; some
with negative coords outside the house) — they are **not** clean interior wall or
door segments, so we **do not fabricate** interior walls from them. The maps are
real and non-convex, but interior partitions are not modelled because the dataset
does not provide them as wall geometry. This trade-off is documented here and in
PRD-HOUSEEXPO §5.2; a future source with explicit interior segments slots in
behind the same `walls` contract with no consumer change.

**Train / held-out split — from `config/config.yaml#maps` (CLAUDE.md §4):**

| Role | Config key | Value |
|---|---|---|
| dataset repo | `maps.dataset_repo` | `https://github.com/TeaganLi/HouseExpo` |
| pinned SHA | `maps.dataset_sha` | `45e2b2505f6ea1fe49c0203f14efb7ce20b94e7c` (stamped by `fetch_houseexpo.py`) |
| archive path | `maps.archive_path` | `HouseExpo/json.tar.gz` (gzip tar; no 7z) |
| coord scale | `maps.coord_scale` | `1.0` (real verts already metric) |
| curated ids | `maps.curated_ids` | logical name → real HouseExpo id (5 entries) |
| train maps | `maps.train` | `["room_single", "apt_small", "apt_multi"]` |
| held-out maps | `maps.holdout` | `["apt_large", "office"]` |
| maps dir | `paths.maps_dir` | `data/maps` |

We **train on the subset and hold out 1–2 plans** (`apt_large`, `office`) for a
**generalization** evaluation (design spec §6; consumed by ADR-008). HouseExpo
is cited (Li et al. 2019, arXiv:1903.09845) in `docs/THEORY.md` / README.

## Consequences

**Positive.**
- Reproducible: a pinned SHA + a vendored curated subset means a grader gets
  exactly our maps without cloning 35k files; `dataset_sha` records provenance.
- Repo stays lean and V3-clean: the bulk dataset is git-ignored, so it never
  pollutes the ≤150-LOC / coverage / ruff gates (the third-party JSON is data,
  not source).
- The `house_map.py` adapter is the single seam between dataset format and
  simulator geometry, so a different map source could be swapped behind the same
  wall-segment/bounds contract without touching raycast/coverage/collision.
- The explicit train/held-out split (`apt_large`, `office` held out) makes the
  generalization claim in ADR-008 honest — those plans are never seen in
  training.

**Negative.**
- A curated 4–6 plan subset is a small, hand-picked sample; results generalize
  to "HouseExpo-like apartments," not to arbitrary floor-plans. Documented as a
  threat to validity in `docs/ANALYSIS.md`.
- **Interior partitions are not modelled.** Walls are the exterior boundary
  polygon only; `room_category` is not fabricated into interior walls (it is not
  clean wall geometry). The robot navigates a single non-convex open interior, not
  a multi-room layout with doorways. This is the honest reading of what HouseExpo
  encodes as geometry; documented in PRD-HOUSEEXPO §5.2. A source carrying
  explicit interior segments would slot in behind the same `walls` contract.
- We depend on the upstream repo existing at fetch time; the pinned SHA and the
  vendored subset are the mitigation (if upstream disappears, the vendored
  subset still reproduces the headline runs).
- A non-trivial JSON-plan format means the adapter must handle whatever HouseExpo
  encodes (room polygons / bbox); bounded by keeping the parser in one tested
  ≤150-LOC module.

## Alternatives considered

| # | Alternative | Verdict | Why rejected |
|---|---|---|---|
| a | **Pinned clone + git-ignored full set + 4–6 vendored curated plans behind `house_map.py`** | **Chosen** | Reproducible, lean repo, clean adapter seam, supports held-out generalization. |
| b | Vendor the entire ~35k HouseExpo dataset | Rejected | Blows up repo size; third-party content distorts the V3 size/coverage gates; unnecessary for the experiments. |
| c | Fetch maps live at train time (no pin, no vendor) | Rejected | Non-reproducible — a grader could get a different upstream state; breaks the byte-for-byte determinism the rest of the test suite assumes. |
| d | Hand-author synthetic floor-plans instead of HouseExpo | Rejected | The brief/spec require *real* HouseExpo plans (design spec §6); synthetic maps lose the realism and the citable dataset. |
| e | Train and evaluate on the same maps (no held-out split) | Rejected | Can't make a generalization claim; ADR-008's held-out eval depends on `apt_large`/`office` never being trained on. |
| f | Derive interior walls from `room_category` boxes | Rejected | Those boxes are overlapping/nested semantic annotations (identical boxes for distinct rooms; negative coords outside the house), not clean wall segments — using them would **fabricate** geometry the dataset never asserts. We ship the real exterior boundary + point-in-polygon instead. |
| g | `is_inside` = axis-aligned bbox test | Rejected | A bbox marks exterior void as free on non-convex plans (the L-shaped `office` bbox centre is in a void); breaks the coverage % and `coverage_target`. Point-in-polygon over the real boundary is correct. |
| h | `git clone` the whole repo then `tar -xvzf` | Viable, not chosen | Equivalent result but pulls the full git history; downloading just the pinned `json.tar.gz` blob is leaner and the SHA still pins provenance. |

## Review trigger

Re-open if: the HouseExpo repo URL/commit changes or becomes unavailable;
the curated subset needs to grow/shrink; or `house_map.py`'s JSON schema
assumptions break on a new plan (then fix the adapter, not the consumers).
