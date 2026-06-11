# PRD — RoboVacuumDDPG (Assignment 5 product requirements)

The project-wide requirements document for Assignment 5 of the Bar-Ilan
*Vibe Coding & RL* workshop (Lecture 09, DDPG; brief
`EX05-DDPG-Robot-Simulator.pdf`, lecturer Dr. Yoram Segal). The single
source of truth every doc must agree with verbatim is the design spec
`docs/superpowers/specs/2026-06-10-robovacuum-ddpg-design.md`. Per-component
design lives in `PLAN.md` (C4 + UML + ADRs); the literal prompts and AI
workflow live in `docs/PROMPTS.md`; the phased task list with
definition-of-done lives in `TODO.md`; the DDPG math lives in
`docs/THEORY.md`; the three analysis answers live in `docs/ANALYSIS.md`;
the per-domain requirements split across
`docs/prd/PRD-{SIM,DDPG,HOUSEEXPO}.md`.

> Teaching artefact, **not a production cleaning-robot controller**. The
> agent learns continuous navigation/coverage in a from-scratch 2D
> simulator over real HouseExpo floor-plans; the "wall avoidance" and
> "coverage" it demonstrates are simulator-internal, not validated on
> physical hardware. See §6 for the full honest-limitations list and §7
> for the deliberate decision to claim no numeric self-grade in the
> public repo.

---

## 1. Vision

### 1.1 Headline goal

Per the brief, build **from scratch** (a) a 2D robotic-vacuum simulator
that reads real **HouseExpo** floor-plan JSON, and (b) a **DDPG** agent
(Lecture 09) that learns continuous navigation and area coverage. The
brief's central emphasis is the *hands-on DDPG implementation* — the
summary must point at our own code lines for the **Actor**, the
**Critic**, the **Polyak soft-target update**, and the **Gaussian
exploration noise**. Ambition is excellence-maxed: full V3 compliance
plus research extras.

**Hard ban (brief §"דרישת חובה").** No ready simulation platforms —
**no Gymnasium, no Gazebo** — and (by the "show your Polyak code lines"
demand) **no ready RL library** (no Stable-Baselines3); DDPG is
implemented from scratch in PyTorch. An AST architecture test forbids any
`gymnasium` import under `src/`.

### 1.2 Two artefacts, one SDK

The submission ships two built components behind one Python facade
(`RoboVacuumSDK`):

1. **VacuumEnv** — a from-scratch 2D simulator (no Gymnasium / Gazebo)
   exposing a custom `reset()` and `step(action) -> (state, reward, done,
   info)` 4-tuple over a unicycle-kinematic robot moving across a
   HouseExpo floor-plan. It assembles state from a raycast lidar
   (`env/raycast.py`), a unicycle pose integrator (`env/kinematics.py`),
   a cleaned-cell coverage grid (`env/coverage.py`), a robot-radius vs
   wall-segment collision test (`env/collision.py`), the reward function
   (`env/reward.py`), and observation assembly (`env/state.py`).
2. **DDPGAgent** — a from-scratch actor-critic agent (`ddpg/agent.py`)
   with a Tanh-bounded deterministic **Actor** (`model/actor.py`), a
   state⊕action **Critic** (`model/critic.py`), a uniform experience
   replay buffer (`ddpg/replay_buffer.py`), **Gaussian** exploration
   noise (`ddpg/noise.py`, brief-mandated — not Ornstein-Uhlenbeck), and
   **Polyak soft-target updates** `θ_target = τ·θ + (1−τ)·θ_target` for
   both actor and critic targets.

The custom training loop (collect → store → update → log) lives in
`services/trainer.py`; cost/runtime accounting in `cost/meter.py`;
config access through `utils/config_loader.py`. Every business-logic
path is reachable through `RoboVacuumSDK` (`build_env`, `train`,
`evaluate`, `rollout`, `coverage_report`) — CLI, notebook, and scripts
import only the SDK.

Plus the deliverables: the two required graphs
(`results/figures/learning_curve.png`,
`results/figures/critic_loss.png`), the trajectory visualization
(`render_trajectory.py`), `docs/ANALYSIS.md` (the brief's three analysis
questions), and `docs/THEORY.md` (DDPG math).

### 1.3 In scope

- **From-scratch 2D simulator** (spec §3, §4) — `VacuumEnv` with custom
  `reset()` / `step(action) -> (state, reward, done, info)`, **no
  Gymnasium / Gazebo** (spec §1 hard ban; AST test under `src/`).
- **Unicycle kinematic model** (spec §3) — action `a ∈ [−1,1]²` =
  `[throttle, steer]`; linear velocity `v = throttle·v_max`, angular
  velocity `ω = steer·omega_max`; pose integrated
  `x += v·cosθ·dt; y += v·sinθ·dt; θ += ω·dt` with `v_max = 0.5 m/s`,
  `omega_max = 1.5 rad/s`, `dt = 0.1 s` (config `env`).
- **20-dim normalized state** (spec §3) — **16 lidar ray distances**
  (raycast to walls, normalized by `ray_max = 5.0 m`) + current `(v, ω)`
  + a heading cue (unit vector / angle in robot frame) to the nearest
  uncleaned cell. Ray count is a config knob (`env.n_rays`, default 16,
  ablation values 8 / 16 / 24).
- **Reward shaping** (spec §3, config `reward`) —
  `r = k_coverage·(new cells cleaned) − k_collision·(collision) − k_step`
  with `k_coverage = 1.0`, `k_collision = 10.0`, `k_step = 0.01`.
- **DDPG from scratch in PyTorch** (spec §5) — Tanh-bounded Actor,
  state⊕action Critic, uniform replay buffer, Gaussian exploration
  noise, Polyak soft-target updates; γ, τ, learning rates, batch size,
  buffer size, hidden sizes all in config `ddpg` / `noise`.
- **HouseExpo data adapter** (spec §6) — `scripts/fetch_houseexpo.py`
  clones `github.com/TeaganLi/HouseExpo` at a pinned commit (full ~35k
  JSON git-ignored); 4–6 curated plans vendored under `data/maps/`,
  with a train subset and a held-out generalization subset.
- **Two required graphs + trajectory viz** (spec §7) —
  `learning_curve.png`, `critic_loss.png`, and a colored robot-path
  trajectory over the 2D JSON map with covered area shaded.
- **Three analysis answers** in `docs/ANALYSIS.md` (spec §7).
- **DDPG theory** in `docs/THEORY.md` (spec §7) with LaTeX + citations.
- **Sensor-resolution ablation** (spec §3) — the `env.n_rays` knob
  (8 / 16 / 24) enables a lidar-resolution study.

### 1.4 Out of scope

- **Gymnasium / Gazebo** (spec §1 hard ban) — the simulator is custom;
  an AST test forbids any `gymnasium` import under `src/`.
- **Any ready RL library** (Stable-Baselines3 / RLlib) — the brief's
  "show your Polyak code lines" demand requires DDPG from scratch, so
  no library implements the actor/critic/soft-update/noise for us.
- **Ornstein-Uhlenbeck exploration noise** — the brief mandates
  **Gaussian** noise (spec §5, ADR-003); OU is explicitly not shipped.
- **Physical hardware / real-robot validation** — the robot, lidar,
  collisions, and coverage are simulator-internal; no claim of
  sim-to-real transfer is made.
- **The full ~35k-plan HouseExpo dataset in the repo** — the full clone
  is git-ignored; only the 4–6 curated plans are vendored (spec §6).
- **Discrete control / DQN / PPO as the shipped agent** — the shipped
  agent is DDPG; DQN and PPO appear only as comparison points in the
  `docs/ANALYSIS.md` "why DDPG not DQN/PPO" answer.

---

## 2. Objectives & KPIs

Quantitative success criteria (every claim in §5 acceptance criteria
links back to one of these KPIs). All headline numbers report mean ± CI
across the config `training.seeds = [42, 7, 123, 314, 271]`.

### 2.1 Algorithmic objectives

- **O1 — Coverage**. The trained DDPG policy achieves a meaningfully
  higher final coverage % (cleaned cells ÷ free cells over the per-episode
  coverage grid) than a random-action baseline on the held-out maps,
  reported as mean ± CI across the five seeds. Coverage % is computed by
  `env/coverage.py` and surfaced via `RoboVacuumSDK.coverage_report`.
- **O2 — Learning-curve trend**. Cumulative episode reward shows an
  upward trend over `training.episodes = 500`, reported as mean ± CI over
  the five seeds. This is the `learning_curve.png` deliverable.
- **O3 — Critic stability**. Critic loss vs training step does not
  diverge — the target-network + Polyak soft-update mechanism
  (`τ = 0.005`) keeps the TD target from chasing the online critic. This
  is the `critic_loss.png` deliverable and the empirical basis for
  analysis answer (3).
- **O4 — Actor bound**. The Actor output is Tanh-bounded to exactly
  `[−1, 1]²` for every state, so `[throttle, steer]` is always a legal
  unicycle command (no clipping needed downstream).
- **O5 — Soft-target correctness**. Polyak averaging
  `θ_target = τ·θ + (1−τ)·θ_target` holds for both actor and critic
  targets, verified symbolically against the closed form on a synthetic
  step (`τ = 0.005`).
- **O6 — Generalization**. The policy is trained on the `maps.train`
  subset (`room_single`, `apt_small`, `apt_multi`) and evaluated on the
  held-out `maps.holdout` subset (`apt_large`, `office`); the held-out
  coverage gap is reported honestly, not hidden.

### 2.2 Engineering KPIs

- **E1**. ≥ 85 % `pytest` coverage on `src/` (`fail_under = 85`;
  CLAUDE.md hard rule), enforced from day 1 in CI.
- **E2**. Zero ruff violations: `uv run ruff check src/ tests/ scripts/`
  (CLAUDE.md hard rule).
- **E3**. Every `.py` ≤ 150 LOC, tests included (CLAUDE.md hard rule).
- **E4**. `uv sync --dev && uv run pytest` reproduces a green build on a
  clean checkout (uv only — no pip / conda / venv / requirements.txt).
- **E5**. CI green on every commit from the bootstrap commit forward
  (`.github/workflows/ci.yml`).
- **E6**. Single-source config: every algorithm-relevant parameter
  (DDPG hyperparams, noise schedule, env/sim constants, reward weights,
  seeds, map lists) lives in `config/config.yaml` and is read only via
  `src/utils/config_loader.py`.

### 2.3 Deliverable KPIs

- **D1**. `results/figures/learning_curve.png` exists (>1 KB) and shows
  cumulative reward vs episode as a mean ± CI envelope over the five
  seeds (rendered by `render_learning_curve.py`).
- **D2**. `results/figures/critic_loss.png` exists (>1 KB) and shows
  critic loss vs training step (rendered by `render_critic_loss.py`).
- **D3**. Trajectory visualization exists (rendered by
  `render_trajectory.py`): the robot path drawn in colour over the 2D
  HouseExpo JSON map, covered area shaded, proving wall-avoidance and
  smooth continuous coverage; optional short MP4/GIF animation.
- **D4**. `docs/ANALYSIS.md` answers the brief's three analysis
  questions (see §2.4) with seed/episode-count citations, never bare
  adjectives.
- **D5**. `docs/THEORY.md` derives the DDPG objective, deterministic
  policy gradient, critic TD target, Polyak update, and exploration
  noise with LaTeX + citations (incl. HouseExpo, Li et al. 2019,
  arXiv:1903.09845).
- **D6**. A numeric final coverage % (mean ± CI across the five seeds)
  for the train subset **and** the held-out subset is reported in
  `docs/ANALYSIS.md`.
- **D7**. Cost / runtime accounting (`cost/meter.py`,
  `docs/COST_ANALYSIS.md`): training wall-clock and the honest
  convergence statement (full / partial), reported like A4.

### 2.4 The three analysis answers (brief, spec §7)

`docs/ANALYSIS.md` answers, verbatim to the brief:

1. **Why DDPG, not DQN/PPO** — deterministic physical motors +
   continuous `[throttle, steer]` control + off-policy replay-buffer
   reuse of collected experience.
2. **Effect of removing Gaussian exploration noise early** — the
   coverage map collapses to a narrow repeated path (the deterministic
   actor stops exploring the floor).
3. **How target networks + soft updates prevent critic collapse** — the
   slowly-tracked target (`τ = 0.005`) stops the TD target from chasing
   the online critic, which is what keeps the `critic_loss.png` curve
   from diverging.

---

## 3. Functional requirements

Each `F#` names the spec §-id it satisfies inline. The full
brief-§ → F# → test mapping lives in `docs/TRACE.md`.

### 3.1 Simulator core (spec §3, §4)

- **F1 (Unicycle kinematics, spec §3)**. `env/kinematics.py` integrates
  the unicycle model: `v = throttle·v_max`, `ω = steer·omega_max`,
  `x += v·cosθ·dt; y += v·sinθ·dt; θ += ω·dt`, reading `v_max`,
  `omega_max`, `dt` from config `env`. Hand-computed expectations for a
  one-step integration are the unit test.
- **F2 (Raycast lidar, spec §3, §4)**. `env/raycast.py` casts
  `env.n_rays` (default 16) rays via ray–segment intersection against
  the wall segments and returns distances normalized by `ray_max = 5.0`.
  Ray count is a config knob (8 / 16 / 24) for the sensor-resolution
  ablation.
- **F3 (Coverage grid, spec §3, §4)**. `env/coverage.py` maintains a
  cleaned-cell grid (`coverage_cell = 0.10 m` edge) over the free-space
  bounds and computes coverage %; the cleaning footprint is
  `clean_radius = 0.17 m`. `reset()` clears the per-episode grid.
- **F4 (Collision, spec §3, §4)**. `env/collision.py` tests the robot
  body (`robot_radius = 0.17 m`) against wall segments and returns a
  collision flag consumed by the reward and the episode logic.
- **F5 (Reward, spec §3)**. `env/reward.py` computes
  `r = k_coverage·Δcells − k_collision·collision − k_step` with
  `k_coverage = 1.0`, `k_collision = 10.0`, `k_step = 0.01` from config
  `reward`. Sign of each term is asserted (coverage positive, collision
  and step negative).
- **F6 (State assembly, spec §3)**. `env/state.py` assembles the
  normalized observation: 16 lidar distances (÷`ray_max`) + current
  `(v, ω)` + heading cue (unit vector / angle in robot frame) to the
  nearest uncleaned cell.
- **F7 (VacuumEnv, no Gymnasium, spec §3, §4)**. `env/vacuum_env.py`
  exposes custom `reset()` (re-spawn robot at a random free cell, clear
  coverage grid) and `step(action) -> (state, reward, done, info)`;
  episodes end at `env.max_steps = 1000` or an optional coverage target.
  It does **not** subclass or import `gymnasium`.
- **F8 (HouseExpo loader, spec §6)**. `env/house_map.py` parses a
  HouseExpo JSON floor-plan into wall segments + free-space bounds.

### 3.2 DDPG networks & agent (spec §5)

- **F9 (Actor, spec §5.1)**. `model/actor.py` is an MLP
  (`ddpg.hidden_sizes = [256, 256]`) with a **Tanh**-bounded output →
  deterministic action in exactly `[−1, 1]²`. The summary cites the
  exact line numbers.
- **F10 (Critic, spec §5.1)**. `model/critic.py` is an MLP taking state
  **and** action (`state ⊕ action`) → scalar Q-value. The summary cites
  the exact line numbers.
- **F11 (Replay buffer, spec §5)**. `ddpg/replay_buffer.py` is a uniform
  experience replay of capacity `ddpg.buffer_size = 1_000_000`; sampling
  returns correctly-shaped batches of `ddpg.batch_size = 128`.
- **F12 (Gaussian noise, spec §5.4, ADR-003)**. `ddpg/noise.py` adds
  **Gaussian** noise to the actor action during collection, with
  `noise.sigma_start = 0.2`, `noise.sigma_end = 0.05`, decayed over
  `noise.sigma_decay_steps = 50_000`. Seeded sampling is deterministic.
- **F13 (Polyak soft-target update, spec §5.2)**. `ddpg/agent.py`
  performs `θ_target = τ·θ + (1−τ)·θ_target` (`ddpg.tau = 0.005`) for
  **both** actor and critic targets after each update. The summary cites
  the exact line numbers; the math is asserted against the closed form.
- **F14 (DDPG update step, spec §5)**. `ddpg/agent.py` performs one
  finite DDPG update: critic TD target uses `ddpg.gamma = 0.99` and the
  target networks; critic optimized at `lr_critic = 1e-3`, actor at
  `lr_actor = 1e-4` (actor slower than critic — standard DDPG); gradient
  norm clipped at `ddpg.grad_clip = 1.0`. `ddpg.warmup_steps = 1000`
  random actions precede learning. `act()` returns a legal action.

### 3.3 Training service & SDK (spec §4)

- **F15 (Trainer, spec §4)**. `services/trainer.py` runs the custom
  training loop: collect (with Gaussian noise) → store in replay → update
  → log, for `training.episodes = 500`, returning a full per-episode
  history; final greedy evaluation runs via `RoboVacuumSDK.evaluate` (no
  in-loop eval cadence). No `gymnasium`.
- **F16 (SDK single entry, spec §2, §4)**. `sdk/sdk.py` exposes
  `RoboVacuumSDK` with `build_env`, `train`, `evaluate`, `rollout`,
  `coverage_report` as the single business-logic entry point; CLI,
  notebook, and scripts import only the SDK.
- **F17 (Config single-source, CLAUDE.md §4, spec §2)**.
  `src/utils/config_loader.py` loads `config/config.yaml` once and is the
  only path by which `src/` reads algorithm-relevant parameters; no `.py`
  under `src/` opens `config/config.yaml` directly.

### 3.4 Data adapter (spec §6)

- **F18 (HouseExpo fetch, spec §6)**. `scripts/fetch_houseexpo.py` clones
  `maps.dataset_repo` (`github.com/TeaganLi/HouseExpo`) at a pinned commit
  (`maps.dataset_sha`, stamped at fetch; full ~35k JSON git-ignored) and
  vendors 4–6 curated plans under `data/maps` (`paths.maps_dir`). Train on
  `maps.train` (`room_single`, `apt_small`, `apt_multi`); hold out
  `maps.holdout` (`apt_large`, `office`).

### 3.5 Deliverable generators (spec §7)

- **F19 (Learning-curve render, spec §7)**.
  `render_learning_curve.py` writes `results/figures/learning_curve.png`
  (cumulative reward vs episode, mean ± CI over the five seeds).
- **F20 (Critic-loss render, spec §7)**. `render_critic_loss.py` writes
  `results/figures/critic_loss.png` (critic loss vs training step).
- **F21 (Trajectory render, spec §7)**. `render_trajectory.py` draws the
  robot path in colour over the 2D HouseExpo JSON map with covered area
  shaded; optional short MP4/GIF animation.

### 3.6 Architecture & no-Gym enforcement (spec §3, §8)

- **F22 (No-Gymnasium AST test, spec §3, §8)**. An architecture test
  walks the AST of every module under `src/` and asserts no
  `Import` / `ImportFrom` node names `gymnasium` (AST-level, not grep —
  grep false-positives on comments/strings).
- **F23 (Actor-bound architecture test, spec §8)**. An architecture test
  asserts the Actor output lies in `[−1, 1]` for sampled states.
- **F24 (Soft-update architecture test, spec §8)**. An architecture test
  asserts the agent's target update is Polyak (matches the `τ` closed
  form), not a hard copy.
- **F25 (SDK single-entry architecture test, spec §8)**. An architecture
  test asserts CLI / notebook / scripts import only `RoboVacuumSDK`, not
  the inner env/agent modules directly.

---

## 4. Non-functional requirements

Inherited from `CLAUDE.md` Hard Constraints, plus A5-specific additions.

- **N1**. Every `.py` file ≤ 150 LOC (hard, CI-enforced; CLAUDE.md §1).
  If `vacuum_env.py` or `agent.py` approach 150 LOC, split helpers into a
  sibling `_*.py` module (A4 convention, spec §4).
- **N2**. TDD: tests written before implementation (RED→GREEN→REFACTOR);
  coverage **≥ 85 %** on `src/` (`fail_under = 85`); enforced from day 1
  in CI (CLAUDE.md §2).
- **N3**. Zero ruff violations on `src/`, `tests/`, `scripts/`
  (CLAUDE.md §6).
- **N4**. **uv-only** dependency management; `uv sync --dev` reproduces
  the environment from `uv.lock`. No pip, no conda, no requirements.txt
  (CLAUDE.md §7).
- **N5**. CI green from the bootstrap commit forward. The
  `.github/workflows/ci.yml` runs `uv sync --dev`, `uv run ruff check`,
  and `uv run pytest --cov=src --cov-fail-under=85` on every push.
- **N6**. No hardcoded algorithm parameters. All DDPG hyperparameters
  (`gamma`, `tau`, `lr_actor`, `lr_critic`, `batch_size`, `buffer_size`,
  `hidden_sizes`, `grad_clip`, `warmup_steps`), the noise schedule
  (`type`, `sigma_start`, `sigma_end`, `sigma_decay_steps`), env/sim
  constants (`n_rays`, `ray_max`, `dt`, `v_max`, `omega_max`,
  `robot_radius`, `clean_radius`, `coverage_cell`, `max_steps`), reward
  weights (`k_coverage`, `k_collision`, `k_step`), training settings
  (`episodes`, `seeds`), and map lists live in
  `config/config.yaml` (CLAUDE.md §4). Local UI/plot styling literals
  (matplotlib alpha / fontsize / dpi) stay local.
- **N7**. Deterministic seeds. Training and evaluation seed Python
  `random`, `numpy`, and `torch` from `training.seeds = [42, 7, 123,
  314, 271]`. Known non-determinism (CUDA / MPS kernels) named in the
  README.
- **N8**. Single SDK entry. All business logic is reachable through
  `RoboVacuumSDK`; CLI / notebook / scripts import only the SDK, no logic
  in UIs (CLAUDE.md §3; F16, F25).
- **N9**. No-Gymnasium architecture invariant. No `.py` under `src/`
  imports `gymnasium` (AST test, F22; spec §3, §8) — and no SB3 / RLlib
  anywhere (the agent is from scratch, spec §1).
- **N10**. Version starts `1.0.0`, kept in sync across `src/__init__.py`
  `__version__`, `config.version`, and `pyproject.toml` (CLAUDE.md;
  spec §2).
- **N11**. No PII / secrets in tracked files; `.env-example` shipped; the
  forbidden-PII / secret grep guard reports zero hits (spec §2, V3
  scaffold).
- **N12**. The full HouseExpo dataset (~35k JSON) is git-ignored; only
  the 4–6 curated plans under `data/maps` are tracked, with the pinned
  `maps.dataset_sha` recorded (spec §6).

---

## 5. Acceptance criteria

### 5.1 Per-requirement DoD

| Req | Acceptance criterion | Evidence pointer |
|-----|---------------------|------------------|
| F1  | One-step unicycle integration matches hand-computed pose for given `throttle, steer` using config `v_max`/`omega_max`/`dt` | `tests/test_kinematics.py::test_one_step_integration` |
| F2  | Raycast returns `env.n_rays` distances; a ray to a known wall segment matches the analytic intersection, normalized by `ray_max` | `tests/test_raycast.py::test_ray_segment_distance` |
| F3  | Coverage grid (`coverage_cell` edge) increments cleaned cells under the `clean_radius` footprint; coverage % matches a hand-counted grid; `reset()` clears it | `tests/test_coverage.py::test_coverage_accounting` |
| F4  | Robot at `robot_radius` overlapping a wall segment returns collision `True`; clear of all segments returns `False` | `tests/test_collision.py::test_collision_flag` |
| F5  | `reward(...)` equals `k_coverage·Δcells − k_collision·collision − k_step` symbolically; coverage term positive, collision and step terms negative | `tests/test_reward.py::test_reward_form_and_signs` |
| F6  | `state()` returns a normalized vector with 16 lidar distances + `(v, ω)` + heading cue; lidar entries ∈ `[0, 1]` | `tests/test_state.py::test_state_shape_and_norm` |
| F7  | `step(action)` returns a `(state, reward, done, info)` 4-tuple; `done` fires at `max_steps`; `reset()` re-spawns on a free cell | `tests/test_vacuum_env.py::test_reset_step_contract` |
| F8  | `house_map` parses a HouseExpo JSON into wall segments + free-space bounds with hand-checked counts | `tests/test_house_map.py::test_parse_segments` |
| F9  | Actor output ∈ `[−1, 1]²` for sampled states (Tanh bound); shape == 2 | `tests/test_actor.py::test_tanh_bounds` |
| F10 | Critic accepts `(state, action)` and returns a scalar Q of correct shape | `tests/test_critic.py::test_output_shape` |
| F11 | Replay buffer stores transitions and samples a batch of `batch_size` with correct shapes | `tests/test_replay_buffer.py::test_sample_shapes` |
| F12 | Seeded Gaussian noise is deterministic; σ decays from `sigma_start` toward `sigma_end` over `sigma_decay_steps` | `tests/test_noise.py::test_gaussian_seed_and_decay` |
| F13 | After `soft_update`, target params equal `τ·θ + (1−τ)·θ_target` for actor and critic (`τ = 0.005`) | `tests/test_agent.py::test_polyak_soft_update` |
| F14 | One `update()` step runs finite (no NaN/Inf) on a synthetic batch; gradient norm respects `grad_clip`; uses `gamma` in the TD target | `tests/test_agent.py::test_finite_update_step` |
| F15 | A short training smoke run (reduced episodes) completes, populates the replay buffer past `warmup_steps`, and logs reward + critic loss | `tests/test_trainer.py::test_training_smoke` |
| F16 | `RoboVacuumSDK` exposes `build_env`, `train`, `evaluate`, `rollout`, `coverage_report`; a smoke call returns expected shapes | `tests/test_sdk.py::test_sdk_surface` |
| F17 | Mutating the loaded config dict does not leak across reload; no `.py` under `src/` opens `config/config.yaml` outside the loader (AST scan) | `tests/architecture/test_config_single_source.py::test_loader_is_single_source` |
| F18 | `fetch_houseexpo` resolves `maps.train` / `maps.holdout` plan names to vendored files under `paths.maps_dir`; pinned SHA recorded | `tests/test_fetch_houseexpo.py::test_curated_subset` |
| F19 | `results/figures/learning_curve.png` exists, non-empty (>1 KB), shows mean ± CI over the five seeds | `tests/test_learning_curve.py::test_png_exists` |
| F20 | `results/figures/critic_loss.png` exists, non-empty (>1 KB) | `tests/test_critic_loss.py::test_png_exists` |
| F21 | Trajectory render writes a non-empty image of the path over the JSON map with covered area shaded | `tests/test_trajectory.py::test_trajectory_image` |
| F22 | No `Import` / `ImportFrom` `gymnasium` node anywhere under `src/` (AST walk, NOT grep) | `tests/architecture/test_no_gymnasium.py::test_no_gym_import_ast` |
| F23 | Actor action ∈ `[−1, 1]` for a random batch of states | `tests/architecture/test_actor_bounds.py::test_action_in_range` |
| F24 | Agent target update is Polyak (`τ` closed form), not a hard copy | `tests/architecture/test_soft_update_is_polyak.py::test_polyak_not_hardcopy` |
| F25 | CLI / notebook / scripts import only `RoboVacuumSDK`, not inner env/agent modules | `tests/architecture/test_sdk_single_entry.py::test_only_sdk_imported` |

### 5.2 Project-level DoD

- Clean checkout: `uv sync --dev && uv run pytest` is green, ≥ 85 %
  coverage on the deterministic core.
- `uv run ruff check src/ tests/ scripts/` → zero violations.
- Every `.py` ≤ 150 LOC.
- Running the SDK training + render path reproduces
  `results/figures/learning_curve.png`, `results/figures/critic_loss.png`,
  the trajectory visualization, and all headline numbers in
  `docs/ANALYSIS.md`.
- `docs/ANALYSIS.md` answers the three brief analysis questions and
  cites **seed**, **episode count**, and **mean ± CI** for every numeric
  claim — no bare adjectives like "converges fast".
- The summary points at exact code lines for the Actor (`model/actor.py`),
  Critic (`model/critic.py`), Polyak soft-update (`ddpg/agent.py`), and
  Gaussian noise (`ddpg/noise.py`).
- Every `F#` row in §5.1 has an evidence pointer; `docs/TRACE.md` covers
  every brief §-id → at least one `F#` and at least one test.
- CI is green on the merge commit; the no-Gymnasium AST test passes; the
  forbidden-PII / secret grep guard reports zero hits.

---

## 6. Dependencies & honest limitations

### 6.1 External dependencies

- **HouseExpo (Li et al. 2019, arXiv:1903.09845).**
  `scripts/fetch_houseexpo.py` clones `github.com/TeaganLi/HouseExpo` at a
  pinned commit (`maps.dataset_sha` stamped at fetch). The full ~35k-plan
  dataset is git-ignored; 4–6 curated plans (single-room → multi-room
  apartment) are vendored under `data/maps`. If the upstream repo or a
  pinned plan moves, the fetch script re-pins and the vendored subset is
  the artefact of record.
- **PyTorch.** The Actor, Critic, replay buffer, Gaussian noise, and
  Polyak update are hand-written on top of PyTorch tensors/optimizers;
  **no** Stable-Baselines3 / RLlib provides the RL algorithm (spec §1).
- **uv.** The only dependency manager; `uv sync --dev` reproduces the
  environment from `uv.lock`.

### 6.2 Honest limitations

The submission ships as a pedagogical artefact. These caveats are called
out head-on so the `docs/ANALYSIS.md` results are read in the right frame.

- **L1 — Simulator, not hardware.** The robot, lidar, collisions, and
  coverage are simulator-internal. "Wall avoidance" and "coverage %" are
  properties of the 2D `VacuumEnv`, not of a physical vacuum; no
  sim-to-real transfer is claimed.
- **L2 — Curated map subset.** Only 4–6 HouseExpo plans are vendored
  (train: `room_single`, `apt_small`, `apt_multi`; holdout: `apt_large`,
  `office`). Coverage and generalization numbers are over this small
  subset, not the full ~35k-plan distribution.
- **L3 — DDPG instability is a known failure mode.** DDPG is sensitive to
  hyperparameters; target nets, soft updates (`τ = 0.005`), gradient
  clipping (`grad_clip = 1.0`), and seeded multi-run reporting mitigate
  but do not eliminate run-to-run variance. The `critic_loss.png` curve
  is reported as-is, including any instability.
- **L4 — Reward is hand-designed.** `k_coverage = 1.0`,
  `k_collision = 10.0`, `k_step = 0.01` are author choices, not learned;
  an inverse-RL formulation is future work, not A5 scope.
- **L5 — Compute-bounded training.** Episodes are time-boxed
  (`episodes = 500`, `max_steps = 1000`); if convergence is only partial
  within the compute budget, it is reported honestly (like A4) rather
  than overstated.
- **L6 — Raycasting performance.** Raycast cost grows with map complexity
  and ray count; the mitigations (vectorized segment math, capped ray
  count via `n_rays`, coarse coverage grid via `coverage_cell`) trade
  fidelity for speed and are documented in ADR-001/ADR-004.
- **L7 — Reproducibility has known gaps.** CPU seeding is enforced (N7);
  CUDA / MPS kernel non-determinism can still introduce run-to-run drift
  and is named in the README's reproducibility caveats.

---

## 7. Honest framing — no numeric self-grade in the public repo

The architect's standing decision (Assignment 1 lesson, locked across all
subsequent assignments): **honest scope beats inflated certainty.** A1's
over-confidence in declaring "all gates passed" without surfacing the
hidden limitations cost credibility, and the lecturer's A1 feedback made
the cost explicit. A5 inherits that lesson — and since the brief does not
request a self-grade in the repository, the **public repo claims no
numeric score**. The self-grade appears **only on the Moodle cover sheet**
(`adrl-001-ex05.pdf`, the official template; group code `adrl-001`), never
in the tracked source — exactly as in A4.

- **No numeric self-grade in the repo.** The brief asks for the spec §7
  deliverables — the two graphs, the trajectory visualization, the three
  analysis answers, the DDPG theory — not a 0–100 self-assessment in the
  source tree. Assigning ourselves a number in the repo would only anchor
  the grader; the honest-limitations list (§6.2) is the credibility
  signal A1 feedback rewarded. The numeric self-grade lives solely on the
  Moodle cover sheet PDF.
- **What is honestly strong.** From-scratch DDPG with exact code-line
  pointers for the Actor, Critic, Polyak soft-update, and Gaussian noise
  (the brief's central demand); a from-scratch simulator with no
  Gymnasium / Gazebo (AST-enforced); deterministic sim units tested
  against hand-computed expectations; multi-seed evaluation over
  `[42, 7, 123, 314, 271]` with mean ± CI; V3 code gates green
  (≤150 LOC, ≥85 % coverage, zero Ruff, single SDK entry, config
  single-source).
- **What is honestly bounded.** Every one of L1–L7 (§6.2) is a real
  reduction in claim strength — simulator-not-hardware, curated map
  subset, DDPG instability, hand-designed reward, compute-bounded
  training, raycasting performance, reproducibility gaps — surfaced
  head-on rather than buried.
- **Why not overstate.** §5 acceptance criteria are quantitative and
  testable; §3 functional coverage is exhaustive against the spec; the
  multi-seed gates and the no-Gymnasium / Polyak architecture tests are
  stricter than the brief requires.

This section is the contract the submission writes with itself **before**
seeing the grade — not retroactive cover after.

---

## 8. Milestones

Phased delivery; each phase ends green (CI, coverage, ruff, ≤150 LOC) and
leaves an auditable artefact. Definition-of-done detail lives in `TODO.md`.

- **Phase 0 — Bootstrap.** V3 scaffold: `uv` project, `config/config.yaml`
  (version `1.0.0`), `src/utils/config_loader.py`, `.env-example`, CI
  (`.github/workflows/ci.yml`), `RoboVacuumSDK` skeleton, ADR stubs
  (ADR-001…008, spec §9). DoD: empty-but-green test suite, ruff clean.
- **Phase 1 — Simulator core (F1–F8).** Kinematics, raycast, coverage,
  collision, reward, state assembly, `VacuumEnv`, HouseExpo loader —
  TDD against hand-computed expectations; the no-Gymnasium AST test
  (F22) passes. DoD: deterministic sim units ≥ 85 % covered.
- **Phase 2 — DDPG networks & agent (F9–F14).** Actor (Tanh bound),
  Critic (state⊕action), replay buffer, Gaussian noise, Polyak
  soft-update, finite update step. DoD: Polyak math + actor bounds +
  critic shape tests green (F13, F23, F24); exact code-line pointers
  recorded.
- **Phase 3 — Trainer, SDK & data (F15–F18).** Custom training loop,
  `RoboVacuumSDK` surface, `fetch_houseexpo.py` + vendored curated plans.
  DoD: training smoke test green; SDK single-entry test green (F25).
- **Phase 4 — Deliverables (F19–F21, D1–D7).** `learning_curve.png`,
  `critic_loss.png`, trajectory visualization; `docs/ANALYSIS.md` (three
  answers + coverage numbers), `docs/THEORY.md`, `docs/COST_ANALYSIS.md`;
  optional sensor-resolution ablation via `n_rays`. DoD: figures exist
  (>1 KB), analysis numbers cite seed/episode/CI.
- **Phase 5 — Submission.** `docs/TRACE.md` full coverage, README, tag
  `1.0.0`, share repo read access with the lecturer's GitHub handle `@rmisegal`, fill the
  Moodle cover sheet `adrl-001-ex05.pdf` (numeric self-grade on the PDF
  only). DoD: clean-checkout green build; §5.2 project-level DoD met.

---

## 9. References

### Brief & lecture

- Assignment 5 brief `EX05-DDPG-Robot-Simulator.pdf`, Bar-Ilan *Vibe
  Coding & RL* workshop, Lecture 09 (DDPG), lecturer Dr. Yoram Segal.
- Design spec (single source of truth):
  `docs/superpowers/specs/2026-06-10-robovacuum-ddpg-design.md`.

### Papers (anchor set — full list in `docs/THEORY.md`)

- Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa,
  Y., Silver, D., Wierstra, D. (2016). *Continuous Control with Deep
  Reinforcement Learning (DDPG).* ICLR 2016, arXiv:1509.02971.
- Silver, D., Lever, G., Heess, N., Degris, T., Wierstra, D., Riedmiller,
  M. (2014). *Deterministic Policy Gradient Algorithms.* ICML 2014.
- Li, T., Ho, D., Li, C., Zhu, D., Wang, C., Meng, M. Q.-H. (2019).
  *HouseExpo: A Large-scale 2D Indoor Layout Dataset for Learning-based
  Algorithms on Mobile Robots.* arXiv:1903.09845.

### Project documents

- `CLAUDE.md` — global coding standards + §1.4 architect/implementer
  contract.
- `PLAN.md` — C4 + UML architecture + ADRs (ADR-001 simulator-from-scratch
  boundary; ADR-002 unicycle kinematic model; ADR-003 Gaussian (not OU)
  noise; ADR-004 coverage-grid + cleaning radius; ADR-005 HouseExpo
  adapter + pinned subset; ADR-006 reward shaping; ADR-007 network sizing
  + soft-update τ; ADR-008 multi-seed eval + held-out generalization).
- `docs/prd/PRD-SIM.md`, `docs/prd/PRD-DDPG.md`,
  `docs/prd/PRD-HOUSEEXPO.md` — per-domain requirements detail.
- `TODO.md` — phased task list with definition-of-done.
- `docs/THEORY.md` — DDPG objective, deterministic policy gradient,
  critic TD target, Polyak update, exploration noise (LaTeX + citations).
- `docs/ANALYSIS.md` — the three brief analysis answers + multi-seed
  coverage numbers (mean ± CI).
- `docs/COST_ANALYSIS.md` — runtime / cost accounting + honest
  convergence statement.
- `docs/QUALITY.md` — ISO 25010 quality model.
- `docs/UX.md` — CLI / figures (spec §10).
- `docs/PROMPTS.md` — the literal prompts and AI-workflow narrative.
- `docs/TRACE.md` — bidirectional brief §-id ↔ F# ↔ test mapping.
