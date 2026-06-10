# ADR-004 — Coverage-Grid Representation + Cleaning Radius

| Field | Value |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-06-10 |
| **Deciders** | Architect (human, CLAUDE.md §1.4) + AI (implementer) |
| **Supersedes** | — |
| **Related** | ADR-002 (pose), ADR-005 (free-space bounds from HouseExpo), ADR-006 (Δcoverage drives reward), design spec §3, §4 |

## Context

The whole point of a robotic vacuum is **area coverage**. The reward
(design spec §3, ADR-006) is `r = k_cov·(new cells cleaned) − k_col·(collision)
− k_step`, so the simulator must answer two questions every step, cheaply and
deterministically: (1) which area has the robot just cleaned, and (2) how many
*new* cells did this step add? We also need a coverage % for the
`coverage_report` SDK call and for the trajectory visualization
(`render_trajectory.py`) that shades covered area (design spec §4, §7).

This requires choosing a spatial representation. The two questions above must be
O(1)-ish per step (raycasting is already the per-step bottleneck, design spec
§10) and must produce hand-computable expectations for unit tests
(design spec §8: "coverage accounting" is a tested deterministic unit).

## Decision

Represent cleaned area as a **fixed-resolution 2-D occupancy grid** of boolean
"cleaned" cells, in `src/env/coverage.py`, sized to the map's free-space bounds
(provided by the HouseExpo adapter, ADR-005). The cleaning footprint is a
**disc of radius `clean_radius`** centered on the robot pose.

**Cell-cleaning rule.** After each `step()` integrates the new pose (ADR-002),
every grid cell whose center lies within `clean_radius` of the robot center is
marked cleaned. `Δcoverage` for the step = number of cells that flipped
`False → True` this step; this integer is what `reward.py` multiplies by `k_cov`
(ADR-006). The grid is **cleared on `reset()`** (per-episode coverage; design
spec §3).

**Constants — all from `config/config.yaml#env` (CLAUDE.md §4, no hardcoding):**

| Symbol | Config key | Value | Meaning |
|---|---|---|---|
| cell edge | `env.coverage_cell` | `0.10` m | coverage grid cell size |
| cleaning radius | `env.clean_radius` | `0.17` m | cleaning footprint disc radius |
| robot radius | `env.robot_radius` | `0.17` m | body radius (collision, separate concern) |

**Coverage %** = `cleaned_cells / total_free_cells`, where `total_free_cells` is
the count of grid cells inside the map free-space bounds (ADR-005). This is the
optional episode-termination "coverage target" signal (design spec §3) and the
number surfaced by `coverage_report`.

**Boundary with collision.** This ADR owns *cleaning* (coverage accounting).
**Collision** (robot-radius vs wall-segment intersection) is a separate concern
in `src/env/collision.py`. They share the `robot_radius` value but answer
different questions; do not conflate the cleaned-grid with the collision test.

## Consequences

**Positive.**
- A boolean grid makes Δcoverage a cheap set-membership count and makes coverage
  % a single division — both trivially unit-testable with hand-computed
  expectations (design spec §8).
- The dense per-step Δcoverage signal is what makes the reward non-sparse,
  directly mitigating the "reward sparsity → slow learning" risk (design spec
  §10) by feeding ADR-006's `k_cov·Δcells` term every step the robot reaches new
  area.
- Same grid backs the trajectory plot's shaded covered area (design spec §7),
  so visualization and reward read from one source of truth.
- Resolution (`coverage_cell = 0.10` m) is a config knob: a grader can coarsen
  the grid to trade fidelity for raycast/coverage speed on complex maps
  (design spec §10 mitigation).

**Negative.**
- Disc-vs-cell-center cleaning is an approximation: with `clean_radius = 0.17` m
  and `coverage_cell = 0.10` m, a cell is cleaned when its center enters the
  disc, so edges round to the nearest cell. Acceptable — the brief grades
  coverage *behavior*, not sub-centimeter cleaning geometry, and the same
  discretization applies to every agent and seed, so comparisons stay fair.
- Memory scales with map area / cell². For HouseExpo apartments at `0.10` m
  cells this is a small boolean array; the coarsening knob handles the worst
  case.

## Alternatives considered

| # | Alternative | Verdict | Why rejected |
|---|---|---|---|
| a | **Fixed-resolution boolean grid + disc cleaning footprint** | **Chosen** | O(1)-ish Δcoverage, hand-testable, config-tunable resolution, one source of truth for reward + plot. |
| b | Continuous swept-area polygon union | Rejected | Exact coverage but expensive polygon-union per step on the hot path; hard to unit-test with closed-form expectations; over-precise for the grade. |
| c | Visitation count grid (int, not boolean) | Rejected | The reward only needs *new* cells (`False→True`); counts add state with no signal benefit. A boolean flip is the minimal representation. |
| d | Point-coverage (only the cell under the robot center) | Rejected | Ignores the physical cleaning footprint (`clean_radius`), under-rewarding wide passes and distorting the coverage % the brief cares about. |
| e | Cleaning radius hardcoded equal to `robot_radius` in code | Rejected | Violates CLAUDE.md §4 (no hardcoding) and conflates cleaning with collision; both are independent config keys (`clean_radius`, `robot_radius`). |

## Review trigger

Re-open if: `coverage_cell` is retuned in a way that makes the disc/cell
approximation visibly distort coverage %; the coverage grid becomes a
performance bottleneck on the largest HouseExpo plan; or the reward design
(ADR-006) starts needing visitation counts rather than first-touch flips.
