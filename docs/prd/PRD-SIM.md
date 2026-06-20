# PRD-SIM — From-Scratch 2D Robotic-Vacuum Simulator (`VacuumEnv`)

**Status:** Draft v0.1 · **Owner:** RoboVacuumDDPG Working Group (`adrl-001`) · **Scope:** ADR-001 / ADR-002 / ADR-004 boundary
**Parent:** design spec `docs/superpowers/specs/2026-06-10-robovacuum-ddpg-design.md` §3 (MDP), §4 (`src/env/`), §8 (testing)
**Repo:** `github.com/adirelm/RoboVacuumDDPG`

## 1. Purpose

The design spec §1 mandates a **2D robotic-vacuum simulator built from
scratch** that reads real HouseExpo floor-plans and exposes a custom RL
environment to the DDPG agent. PRD-SIM is the contract for the `src/env/`
package: the eight focused modules (`house_map.py`, `raycast.py`,
`kinematics.py`, `coverage.py`, `collision.py`, `reward.py`, `state.py`,
`vacuum_env.py`) that together implement the MDP defined in spec §3.

The simulator is the *physical substrate* the agent acts on; PRD-DDPG and
PRD-HOUSEEXPO cover the learner and the map data respectively. PRD-SIM
owns the **unicycle kinematics**, **lidar raycasting**, **coverage grid +
cleaning radius**, **collision test**, **reward signal**, and the
**`VacuumEnv` 4-tuple `step` contract** — all under the hard ban on ready
simulation platforms (spec §1: *no Gymnasium, no Gazebo*).

This is the deterministic core. Per spec §8 it carries the ≥85% coverage
burden with hand-computed unit tests, because every downstream DDPG result
is only trustworthy if the physics underneath is provably correct.

## 2. Inputs / Output / Setup

### 2.1 Inputs

- **Map** — a HouseExpo-derived set of axis-described **wall segments**
  `[(x1,y1,x2,y2), …]` plus free-space bounds, produced by
  `src/env/house_map.py` from a vendored JSON plan in `data/maps/`
  (`config.maps.train = [room_single, apt_small, apt_multi]`,
  `holdout = [apt_large, office]`). The loader contract is owned by
  PRD-HOUSEEXPO; PRD-SIM consumes its segment/bounds output.
- **Action** `a = [throttle, steer] ∈ [−1, 1]²` (spec §3). Supplied by the
  agent each `step`; the simulator does **not** assume the action is
  pre-clipped (it defends, see FR-7).
- **Config** (`config/config.yaml`, via `src/utils/config_loader.py` — no
  hardcoded values, CLAUDE.md §4). The `env` and `reward` blocks:

  | key | value | meaning |
  |---|---|---|
  | `env.n_rays` | `16` | lidar ray count (ablation knob: 8 / 16 / 24) |
  | `env.ray_max` | `5.0` | max lidar range (m); distances normalized by this |
  | `env.dt` | `0.1` | kinematic integration timestep (s) |
  | `env.v_max` | `0.5` | max linear velocity (m/s); `v = throttle·v_max` |
  | `env.omega_max` | `1.5` | max angular velocity (rad/s); `ω = steer·omega_max` |
  | `env.robot_radius` | `0.17` | vacuum body radius (m) — collision test |
  | `env.clean_radius` | `0.17` | cleaning footprint radius (m) — coverage |
  | `env.coverage_cell` | `0.10` | coverage grid cell edge (m) |
  | `env.max_steps` | `1000` | steps per episode |
  | `reward.k_coverage` | `1.0` | `k_cov` reward per newly cleaned cell |
  | `reward.k_collision` | `10.0` | `k_col` penalty on a collision |
  | `reward.k_step` | `0.01` | `k_step` per-step time penalty |

- **Seed** (int, from `config.training.seeds = [42, 7, 123, 314, 271]`) —
  drives the random free-cell spawn in `reset()`; identical seed + identical
  map ⇒ identical episode rollout (FR-8 determinism).

### 2.2 Output

- **State / observation** — a `20`-dim normalized vector assembled by
  `src/env/state.py` (spec §3):
  - **16 lidar ray distances** (one per `env.n_rays`), each `∈ [0, 1]`
    after dividing the raycast hit distance by `env.ray_max`;
  - **current `(v, ω)`** normalized to `[−1, 1]` by `v_max` / `omega_max`;
  - a **heading cue** = `cos`/`sin` of the bearing to the nearest uncleaned
    cell in the robot frame (2 dims; a unit vector, no ±π wraparound).
  Dimension is `n_rays + 4` = **20** at `n_rays=16` (pinned; the heading cue is a
  fixed `cos`/`sin` unit vector emitted unconditionally by `state.py` — not a
  config-switchable representation).
- **Reward** (scalar float) from `src/env/reward.py`:
  `r = k_cov·(new cells cleaned this step) − k_col·(1 if collision else 0) − k_step`.
- **`done`** (bool) — `True` when `step_count == env.max_steps` or the
  optional coverage target is reached (spec §3).
- **`info`** (dict) — diagnostics: `coverage_pct`, `collision` flag, `pose
  (x, y, θ)`, `new_cells`, `step_count`. Never used by the agent's policy;
  consumed by the trainer/loggers and trajectory renderer.

### 2.3 Setup

`uv` only (CLAUDE.md §6). Build the environment through the single SDK
entry point (CLAUDE.md §3) — never instantiate `VacuumEnv` directly from a
CLI / notebook / script:

```python
from src.sdk.sdk import RoboVacuumSDK

sdk = RoboVacuumSDK("config/config.yaml")
env = sdk.build_env(map_name="room_single", seed=42)
state = env.reset()
state, reward, done, info = env.step([throttle, steer])
```

## 3. Component Architecture (spec §4)

Each module is a focused **≤150-LOC** unit (CLAUDE.md §1). If `vacuum_env.py`
approaches the cap, split helpers into a sibling `_*.py` module (spec §4
A4-convention).

| Module | Responsibility (this PRD) |
|---|---|
| `env/house_map.py` | HouseExpo JSON → wall segments + free-space bounds (loader owned by PRD-HOUSEEXPO; consumed here) |
| `env/kinematics.py` | unicycle pose integrator (§4.1) |
| `env/raycast.py` | ray–segment intersection → lidar distances (§4.2) |
| `env/coverage.py` | cleaned-cell grid + coverage % (§4.3) |
| `env/collision.py` | robot-radius vs wall-segment collision test (§4.4) |
| `env/reward.py` | `r = k_cov·Δcoverage − k_col·collision − k_step` (§4.5) |
| `env/state.py` | observation assembly: rays + `(v,ω)` + heading cue (§2.2) |
| `env/vacuum_env.py` | `VacuumEnv.reset()/step()` 4-tuple orchestration (§4.6) |

### 3.1 Unicycle kinematics (`kinematics.py`)

The robot is a unicycle (ADR-002). From action `[throttle, steer]` and the
config maxima:

```
v = throttle · env.v_max          # linear velocity, m/s
ω = steer    · env.omega_max      # angular velocity, rad/s
```

Pose `(x, y, θ)` is integrated forward by one timestep `Δt = env.dt`
(explicit Euler, as written in spec §3):

```
x_{t+1} = x_t + v · cos(θ_t) · Δt
y_{t+1} = y_t + v · sin(θ_t) · Δt
θ_{t+1} = θ_t + ω · Δt
```

`θ` is wrapped to `(−π, π]`. The integrator is **pure** (no I/O, no global
state) so it can be hand-verified in tests (§5.2).

### 3.2 Lidar raycasting (`raycast.py`)

The robot emits `env.n_rays` rays evenly spaced over `2π`, each offset from
the robot heading `θ`. Ray *i* points along angle
`φ_i = θ + 2π·i / n_rays`. Each ray is the parametric segment
`P(t) = origin + t·d`, `t ∈ [0, ray_max]`, `d = (cos φ_i, sin φ_i)`.

For every wall segment `A→B`, solve the **ray–segment intersection**: find
`t` (along the ray) and `u ∈ [0, 1]` (along the wall) where the two lines
meet, using the 2-D cross-product form:

```
r = d                       (ray direction, unit)
s = B − A                   (wall direction)
rxs = r × s                 (scalar 2-D cross)
if rxs == 0: parallel  → no hit (skip; handle collinear as no-hit)
t = ((A − origin) × s) / rxs
u = ((A − origin) × r) / rxs
hit ⇔ (0 ≤ t ≤ ray_max) ∧ (0 ≤ u ≤ 1)
```

The ray distance is `min` over all hitting segments of `t`; if no wall is
hit within range, the distance is clamped to `ray_max`. The reported
observation value is `t / ray_max ∈ [0, 1]`. The segment math is a tight
per-segment Python loop (not numpy-vectorized); the gradient update — not
raycasting — dominates training wall-clock, so the loop is kept simple (spec §10 perf risk).

### 3.3 Coverage grid + cleaning radius (`coverage.py`)

Free space is discretized into a square grid of cell edge
`env.coverage_cell = 0.10` m. A boolean `cleaned` grid is sized from the
map's free-space bounds. On each step, any cell **whose center lies within
`env.clean_radius = 0.17` m of the robot center** is marked cleaned. The
**number of newly cleaned cells this step** (`Δcells`) feeds the reward;
`coverage_pct = cleaned.sum() / reachable_cells` is reported in `info`.
The grid is per-episode and is cleared on `reset()` (spec §3).

> **Note (consistency flag — see §8):** spec §3 reward text says
> `k_cov·(new cells cleaned)`, while `config.reward.k_coverage` is
> commented `*Δcells`. These are the same quantity (`Δcells` = newly
> cleaned cells); PRD-SIM treats `Δcoverage ≡ Δcells` (integer count of
> newly-True cells), which is the unit the acceptance tests in §5.4 assume.

### 3.4 Collision (`collision.py`)

The robot body is a disc of radius `env.robot_radius = 0.17` m. A
collision occurs when the **shortest distance from the robot center to any
wall segment** is `< robot_radius`. The point-to-segment distance projects
the center onto the segment, clamps the projection parameter to `[0, 1]`,
and measures the Euclidean distance to that clamped point. A collision (a)
raises the `k_col` penalty in the reward and (b) is exposed in `info`. The
spec leaves the post-collision pose policy to implementation; PRD-SIM
specifies **reject-the-move** (revert to the pre-step pose, zero the
velocity for the state) so the robot cannot tunnel through a wall — a
hand-computed test pins this (§5.5).

### 3.5 Reward (`reward.py`)

Pure function of `(Δcells, collided)`:

```
r = k_cov · Δcells − k_col · (1 if collided else 0) − k_step
  = 1.0 · Δcells − 10.0 · collided − 0.01
```

with `k_cov, k_col, k_step` read from `config.reward`. **No reward
constant is hardcoded** (CLAUDE.md §4). Sign conventions: coverage is the
only positive term; collision and per-step time are penalties.

### 3.6 `VacuumEnv` 4-tuple step contract (`vacuum_env.py`)

`VacuumEnv` orchestrates the modules above. **No Gymnasium** (spec §3, §8;
ADR-001) — it is a plain class with a deliberately Gym-shaped but
gym-free API:

```python
class VacuumEnv:
    def reset(self) -> state: ...
    def step(self, action) -> tuple[state, reward, done, info]: ...
```

- **`reset()`** — re-spawns the robot at a **random free cell** (seeded),
  resets pose `(x, y, θ)`, zeroes `(v, ω)`, clears the per-episode coverage
  grid, resets `step_count`, returns the initial **state** only.
- **`step(action)`** — single transition, in this exact order:
  1. clip `action` to `[−1, 1]²` (FR-7 defensiveness);
  2. integrate kinematics (`kinematics.py`) to a *candidate* pose;
  3. collision test on the candidate (`collision.py`); if collided,
     revert pose (§3.4) and set `collided = True`;
  4. update coverage at the resolved pose (`coverage.py`) → `Δcells`;
  5. compute `reward` (`reward.py`) from `(Δcells, collided)`;
  6. raycast at the resolved pose (`raycast.py`) and assemble the next
     **state** (`state.py`);
  7. increment `step_count`; set `done` if `step_count == env.max_steps`
     or coverage target reached;
  8. return `(state, reward, done, info)`.

The 4-tuple ordering `(state, reward, done, info)` is fixed by spec §3 and
verified by an architecture/contract test (§5.6). It is deliberately the
classic-Gym 4-tuple (not the 5-tuple) so no Gymnasium shim is implied.

## 4. Functional Requirements

- **FR-1 No ready platforms.** No `gymnasium` and no `gazebo` import
  anywhere under `src/`. An AST architecture test (spec §8) scans `src/`
  and fails on any such import. `VacuumEnv` subclasses nothing from a sim
  framework. (ADR-001.)
- **FR-2 Unicycle integrator.** `kinematics.py` implements the explicit
  Euler unicycle update of §3.1 exactly, as a pure function of
  `(pose, action, dt, v_max, omega_max)`; `θ` wrapped to `(−π, π]`.
- **FR-3 Lidar raycast.** `raycast.py` returns `n_rays` distances via
  ray–segment intersection (§3.2), each clamped to `ray_max` when no wall
  is hit, normalized by `ray_max` for the observation.
- **FR-4 Coverage accounting.** `coverage.py` marks cells within
  `clean_radius` of the robot center and reports the integer `Δcells` and
  fractional `coverage_pct`; grid cleared on `reset()`.
- **FR-5 Collision test.** `collision.py` flags a collision iff the
  point-to-segment distance from the robot center to any wall is
  `< robot_radius`; the move is rejected on collision (§3.4).
- **FR-6 Reward signal.** `reward.py` returns
  `k_cov·Δcells − k_col·collided − k_step` with all weights from config;
  signs as in §3.5.
- **FR-7 Defensive action handling.** `step` clips out-of-range actions to
  `[−1, 1]²` before integration (the Tanh-bounded actor should already be
  in range, but the env must not trust its caller).
- **FR-8 Determinism.** Same `(map, seed)` ⇒ identical sequence of
  `(state, reward, done)` across runs (the only randomness is the seeded
  spawn). Verified in `tests/`.
- **FR-9 Config single-source.** Every numeric constant above is read from
  `config/config.yaml` via `config_loader`; PRD-SIM introduces **no new
  literals** (CLAUDE.md §4). The `n_rays` ablation (8 / 16 / 24) works by
  config edit alone, no source change.
- **FR-10 SDK single entry.** The env is reachable only via
  `RoboVacuumSDK.build_env(...)`; no CLI / notebook / script imports
  `src/env/` directly (CLAUDE.md §3, enforced by an architecture test).

## 5. Acceptance Criteria & Hand-Computed Tests (TDD, spec §8)

All tests are RED→GREEN→REFACTOR (CLAUDE.md §2). The deterministic core
carries the **≥85% coverage** gate (`fail_under=85`); ruff clean; every
file ≤150 LOC.

### 5.1 AC-coverage / AC-lint / AC-size
- **AC-1** `uv run pytest tests/ --cov=src --cov-report=term-missing`
  reports ≥85% on `src/env/`.
- **AC-2** `uv run ruff check src/ tests/ scripts/` → zero violations.
- **AC-3** No file in `src/env/` exceeds 150 LOC.

### 5.2 Kinematics (hand-computed)
- **K-1** Pure rotation: pose `(0, 0, 0)`, action `[0, 1]`,
  `dt=0.1`, `omega_max=1.5` ⇒ `ω = 1.5`, expected
  `θ = 0 + 1.5·0.1 = 0.15` rad, `x = y = 0` (since `v = 0`).
- **K-2** Pure translation along +x: pose `(0, 0, 0)`, action `[1, 0]`,
  `v = 1·0.5 = 0.5`, ⇒ `x = 0 + 0.5·cos(0)·0.1 = 0.05`, `y = 0`,
  `θ = 0`.
- **K-3** Translation at heading `θ = π/2`: pose `(0, 0, π/2)`,
  action `[1, 0]` ⇒ `x = 0.05·cos(π/2) ≈ 0`, `y = 0.05·sin(π/2) = 0.05`.
- **K-4** `θ`-wrap: starting `θ = 3.10` rad, action `[0, 1]` over enough
  steps crosses `+π` and wraps into `(−π, π]`.

### 5.3 Raycast (hand-computed)
- **R-1** Single wall ahead: robot at `(0, 0)` heading `0`, vertical wall
  segment `(2, −1)→(2, 1)`. The forward ray (`φ=0`) hits at `t = 2.0`;
  normalized distance `2.0 / ray_max = 2.0 / 5.0 = 0.4`.
- **R-2** No wall within range: same robot, wall at `x = 6 > ray_max=5` ⇒
  forward ray clamps to `5.0`, normalized `1.0`.
- **R-3** Oblique hit: robot at `(0,0)` heading `0`, wall
  `(3, 0)→(3, 4)`; a ray at `φ = π/4` hits the wall `x=3` at
  `t = 3 / cos(π/4) = 3√2 ≈ 4.2426 ≤ 5`, hit point `(3, 3)` with
  `u = 0.75 ∈ [0,1]`.
- **R-4** Parallel / collinear wall ⇒ `rxs = 0` ⇒ no hit (distance clamps
  to `ray_max`).

### 5.4 Coverage (hand-computed)
- **C-1** First step from a fresh grid (`coverage_cell=0.10`,
  `clean_radius=0.17`): the disc of radius `0.17` m around the robot
  covers all cells whose centers are within `0.17` m; the count of those
  cells equals `Δcells` on step 1.
- **C-2** Re-cleaning already-clean cells yields `Δcells = 0` (so the
  coverage reward term is `0`) — the robot is not rewarded for revisiting.
- **C-3** `coverage_pct` is monotonically non-decreasing within an episode
  and resets to `0` after `reset()`.

### 5.5 Collision (hand-computed)
- **X-1** Center at `(0, 0)`, wall `(0.10, −1)→(0.10, 1)`: distance to
  wall `= 0.10 < robot_radius=0.17` ⇒ collision `True`.
- **X-2** Center at `(0, 0)`, wall `(0.20, −1)→(0.20, 1)`: distance
  `= 0.20 > 0.17` ⇒ collision `False`.
- **X-3** Move-rejection: a `step` whose candidate pose collides leaves the
  reported pose equal to the pre-step pose (no tunneling).

### 5.6 Reward & env contract
- **W-1** `Δcells=3, collided=False` ⇒ `r = 1.0·3 − 0 − 0.01 = 2.99`.
- **W-2** `Δcells=0, collided=True` ⇒ `r = 0 − 10.0 − 0.01 = −10.01`.
- **W-3** `step` returns a **4-tuple** `(state, reward, done, info)` in
  that order; `state` has length `n_rays + 2 + heading_dim`; every ray
  component `∈ [0, 1]`; `(v, ω)` components `∈ [−1, 1]`.
- **W-4** `done` is `True` exactly when `step_count == env.max_steps`
  (=`1000`) or the coverage target is reached.
- **W-5 (architecture)** AST scan: no `gymnasium` import under `src/`
  (FR-1); CLI / notebook import only the SDK (FR-10); all `env` numerics
  resolve from `config.yaml` (FR-9).
- **W-6 (determinism)** Two `reset()→N×step()` rollouts with the same
  `(map, seed)` produce identical `(state, reward, done)` sequences (FR-8).

## 6. Out of Scope

- **DDPG learner** (Actor/Critic/Polyak/Gaussian noise, replay, trainer) —
  see PRD-DDPG / spec §5.
- **HouseExpo loading internals** (`scripts/fetch_houseexpo.py`, pinned
  SHA, JSON→segment parsing) — see PRD-HOUSEEXPO / spec §6. PRD-SIM only
  consumes the loader's segment/bounds output.
- **Figures & visualization** (`learning_curve.png`, `critic_loss.png`,
  `render_trajectory.py`) — spec §7. The simulator only emits the `info`
  fields those renderers read.
- **3-D physics, real sensor noise models, dynamic obstacles, multi-robot.**
  The model is the unicycle of ADR-002 with ideal lidar; no friction,
  inertia, or wheel-slip.
- **Gymnasium / Gazebo / SB3 compatibility shims** — explicitly banned
  (spec §1, ADR-001).

## 7. References

- Design spec §3 (MDP: action/state/reward/episode, NO-Gym mandate)
- Design spec §4 (`src/env/` module map), §8 (TDD gates), §10 (perf risks)
- `config/config.yaml` — `env`, `reward`, `training` blocks
- CLAUDE.md §1 (≤150 LOC), §2 (TDD ≥85%), §3 (SDK single entry), §4 (no
  hardcoded values), §6 (uv)
- ADR-001 (simulator-from-scratch boundary), ADR-002 (unicycle model),
  ADR-004 (coverage-grid + cleaning radius), ADR-006 (reward shaping)
- HouseExpo: Li et al. 2019, arXiv:1903.09845

## 8. Inconsistencies noticed vs spec / config

- **Reward unit wording.** Spec §3 writes `k_cov·(new cells cleaned)`;
  `config.reward.k_coverage` comment writes `k_coverage*Δcells`. Same
  quantity — `Δcells` *is* the count of newly cleaned cells. PRD-SIM uses
  `Δcoverage ≡ Δcells` (integer) to remove ambiguity; no contradiction,
  just unified notation (§3.3).
- **State dimensionality (pinned).** Spec §3 fixes the heading cue as a fixed
  2-component `cos`/`sin` unit vector (emitted unconditionally by `state.py`, no
  config switch), so
  `state_dim = n_rays + 2 (v, ω) + 2 = 20` at `n_rays=16` (12/20/28 for 8/16/24
  rays). `env/state.py` builds it; `test_state_dim_is_20` pins it.
- No value, filename, module name, or hyperparameter in this PRD departs
  from the spec or `config.yaml`.
