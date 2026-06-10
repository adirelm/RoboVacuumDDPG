# TODO — phased task list (A5: RoboVacuumDDPG — DDPG Robotic-Vacuum Simulator)

Task tracker for Assignment 5. See `docs/PRD.md` for scope and `docs/PLAN.md`
for architecture context, both of which must agree verbatim with the design
spec (`docs/superpowers/specs/2026-06-10-robovacuum-ddpg-design.md`, the single
source of truth). Every row below states **who** owns it (always the solo
developer in the architect role — see §1), **what** the DoD is (§2), and
**which phase + acceptance criterion** it lives under (§3).

## §1 Ownership statement (CLAUDE.md §1.4)

Solo project. Every task below is owned end-to-end by the solo developer in
the **architect** role (scope, MDP design, reward shaping, hyperparameter
choices, acceptance criteria, theory cross-references, sign-off). The AI acts
as **implementer** against an approved PRD/PLAN/spec edit — never as
decision-maker. No per-task hand-off — ownership is stated once here rather
than per row.

The architect-decided commitments that frame the whole tracker (all drawn
verbatim from the design spec + `config/config.yaml`):

- **Hard ban (spec §1, brief "דרישת חובה").** No ready simulation platform —
  **no Gymnasium, no Gazebo** — and no ready RL library (**no SB3**). The
  2D simulator and the DDPG agent are both implemented **from scratch** in
  PyTorch. This is a human-decided, non-negotiable boundary (ADR-001).
- **MDP contract (spec §3).** Action `a ∈ [−1,1]²` = `[throttle, steer]`
  driving a unicycle model (`v = throttle·v_max`, `ω = steer·omega_max`,
  pose integrated `x+=v·cosθ·Δt; y+=v·sinθ·Δt; θ+=ω·Δt`). State 20-dim,
  all normalized: **16 lidar ray distances** (`/ray_max`) + current `(v, ω)`
  + a heading cue to the nearest uncleaned cell. Ray count is a config knob
  (8/16/24) for a sensor-resolution ablation. Reward
  `r = k_coverage·Δcells − k_collision·hit − k_step`.
- **DDPG hyperparameters (spec §5, config `ddpg:`/`noise:`).** `γ=0.99`,
  `τ=0.005` (Polyak), `lr_actor=1.0e-4`, `lr_critic=1.0e-3`,
  `batch_size=128`, `buffer_size=1000000`, `hidden_sizes=[256,256]`,
  `grad_clip=1.0`, `warmup_steps=1000`. Gaussian noise (NOT OU):
  `sigma_start=0.2`, `sigma_end=0.05`, `sigma_decay_steps=50000`. Every value
  lives in `config/config.yaml`; none is hardcoded in `src/`.
- **Multi-seed protocol (config `training.seeds`).** Train across
  **5 seeds {42, 7, 123, 314, 271}**; report cumulative-reward learning curve
  as **mean ± 95% CI** over those seeds. Single-shot numbers are forbidden.
- **HouseExpo data discipline (spec §6, config `maps:`).**
  `scripts/fetch_houseexpo.py` clones `github.com/TeaganLi/HouseExpo` at a
  **pinned commit** (`dataset_sha` stamped at fetch; full ~35k JSON
  git-ignored). Vendor **4–6 curated plans** under `data/maps/`. Train on
  `["room_single", "apt_small", "apt_multi"]`; hold out
  `["apt_large", "office"]` for a **generalization** evaluation. HouseExpo
  cited (Li et al. 2019, arXiv:1903.09845).
- **Single SDK entry (spec §2/§4).** `RoboVacuumSDK` (`src/sdk/sdk.py`) is the
  single business-logic entry point — `build_env`, `train`, `evaluate`,
  `rollout`, `coverage_report`. CLI and notebook import **only** the SDK.
- **Architecture invariants (spec §8).** AST test forbids any `gymnasium`
  import under `src/`; actor action ∈ [−1,1]; soft-update is Polyak; SDK
  single-entry; config single-source. These are properties, not one-off tasks
  (see §3.6).

## §2 Definition of Done (DoD)

Every build task is **done** only when **all five** hold:

1. **Behaviour implemented** in `src/` matching the spec module path
   (spec §4) and the PRD requirement id (column `req`). The function/class
   signature lines up with whatever the spec or ADR named — renames are
   architect-level changes and need a spec/PRD edit first.
2. **Test asserting the behaviour** in `tests/` — the test must fail without
   the implementation and pass with it (TDD RED→GREEN→REFACTOR). For the
   deterministic sim units (raycast geometry, kinematics integration,
   coverage accounting, collision, reward signs, HouseExpo loader) the test
   carries a **hand-computed expectation**, not a self-referential snapshot.
   A single happy-path test is not enough when the behaviour has a
   conditional branch (e.g. collision vs no-collision); cover both sides.
3. **Quality gates green** locally and in CI:
   - `uv run ruff check src/ tests/ scripts/` → zero violations
   - every `.py` ≤150 LOC (tests included; enforced by the pre-commit hook
     from T00-02)
   - `uv run pytest --cov=src` ≥85% statement coverage (`fail_under=85`;
     enforced once Phase 1 code lands — Phase 0 is docs+scaffold so coverage
     is N/A)
   - no PII matches against the deny-list (see CLAUDE.md)
4. **Evidence pointer** recorded in the commit message body — file path /
   test id / chart path / notebook cell / commit hash. Bare "done" is not
   acceptable. The evidence pointer is what `docs/TRACE.md` (added in Phase 4)
   will index.
5. **Commit lands** under the matching phase, subject line matching regex
   `^(Phase \d+|Phase 0 bootstrap|chore: bootstrap)` and trailer
   `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

**Honesty stance.** The self-grade lives **only** on the Moodle cover sheet
PDF (`adrl-001-ex05.pdf`), per spec §2. Internal docs describe what was built
and its honest limitations — not what grade it will earn. If training
convergence is partial within the compute budget, report it honestly (spec
§10) rather than masking it. Per-task evidence pointers describe what runs and
how it was measured — never "looks good" or "works well".

**Failure modes that do not pass DoD** (record these honestly in TRACE rather
than masking them):

- A test that passes only because the assertion was weakened mid-task.
- A coverage number ≥85% achieved by deleting hard-to-test branches.
- A raycast/kinematics test whose "expectation" is just the function's own
  output captured once (no independent hand computation).
- A learning-curve caption that omits seed list, sample size, or the
  mean ± 95% CI band.
- A generalization claim made on a train map instead of a held-out map.
- An actor that can emit actions outside [−1,1] (Tanh bound not asserted).

## §3 Build phases — status + evidence-pointer template

| # | Phase | Status | Evidence pointer template |
|---|---|---|---|
| **0** | **Bootstrap** (repo skeleton, pyproject, uv, ruff, pytest-cov ≥85, CI, ≤150-LOC guardrail, docs/ scaffold, ADR-001..008 stubs) | ☐ | `pyproject.toml`, `.github/workflows/ci.yml`, `.env-example`, `docs/PRD.md`, `docs/PLAN.md`, `docs/adr/ADR-001-...md`, commit `Phase 0 bootstrap` |
| 1 | **Simulator from scratch** (house_map, raycast, kinematics, coverage, collision, reward, state, vacuum_env — each TDD) | ☐ | `src/env/{house_map,raycast,kinematics,coverage,collision,reward,state,vacuum_env}.py` + matching `tests/env/test_*.py`, commit `Phase 1` |
| 2 | **DDPG from scratch** (actor Tanh, critic, replay_buffer, noise Gaussian, agent Polyak, trainer loop — each TDD) | ☐ | `src/model/{actor,critic}.py`, `src/ddpg/{replay_buffer,noise,agent}.py`, `src/services/trainer.py` + matching tests, commit `Phase 2` |
| 3 | **Training + Results** (fetch_houseexpo pinned, curated subset, multi-seed train, learning_curve.png, critic_loss.png, trajectory viz, held-out generalization) | ☐ | `scripts/fetch_houseexpo.py`, `data/maps/`, `results/figures/{learning_curve,critic_loss}.png`, `scripts/{render_learning_curve,render_critic_loss,render_trajectory}.py`, commit `Phase 3` |
| 4 | **Docs + Analysis** (THEORY, ANALYSIS 3 questions, COST_ANALYSIS, QUALITY ISO 25010, README report, cover sheet) | ☐ | `docs/{THEORY,ANALYSIS,COST_ANALYSIS,QUALITY}.md`, `README.md`, `adrl-001-ex05.pdf`, commit `Phase 4` |

Phase gates (all must be green before a phase is marked ✅):
ruff zero violations · every `.py` ≤150 LOC · coverage ≥85% on the
deterministic core · uv-only · `RoboVacuumSDK` is single business-logic entry ·
notebook is a *consumer* of the SDK (no parallel implementation) · learning
curve reported mean ± 95% CI across 5 seeds {42, 7, 123, 314, 271} · no
`gymnasium` import anywhere in `src/` · actor action ∈ [−1,1] · soft-update is
Polyak · no PII matches in tree.

## §3.0 Phase 0 — Bootstrap

Goal: stand up the V3 scaffold (same as A4) so every later phase inherits the
gates for free — `uv` only, TDD, ≤150 LOC/file, ≥85% coverage
(`fail_under=85`), 0 Ruff violations, no hardcoded values (all in
`config/config.yaml`), single `RoboVacuumSDK` entry, `.env-example`, CI,
version `1.0.0`. Scaffold `docs/` and stub all eight ADRs (spec §9).

| id | content | activeForm | phase | LOC-Δ | req | acceptance |
|----|---------|------------|-------|-------|-----|------------|
| T00-01 | Initialise repo skeleton (`pyproject.toml` version `1.0.0`, `uv.lock`, ruff config, pytest config, `.gitignore` ignoring full HouseExpo dump, `.env-example`, CI workflow) | Initialising repo skeleton | 0 | +200 | TR-bootstrap | `uv run pytest` and `uv run ruff check src/ tests/ scripts/` both exit 0 on empty suite |
| T00-02 | Add `tool.coverage` `fail_under=85`, ruff line-length, and ≤150-LOC pre-commit hook script (tests included in the count) | Adding coverage gate and 150-LOC pre-commit hook | 0 | +40 | (gate) | Hook rejects a deliberately bloated `.py` file in CI dry-run |
| T00-03 | Scaffold `docs/` (PRD, PLAN, TODO, README, THEORY, ANALYSIS, COST_ANALYSIS, QUALITY, UX, adr/, prd/, shared/, diagrams/) + dedicated `docs/prd/PRD-{SIM,DDPG,HOUSEEXPO}.md` (spec §2) | Scaffolding docs/ tree | 0 | +0 | (docs) | All top-level docs files exist; `docs/prd/` has the three split PRDs; `adr/` contains ADR-001..008 stubs |
| T00-04 | Author `config/config.yaml` loader `src/utils/config_loader.py` + assert every spec §5/§3 knob is read from config (no literals in `src/`) | Implementing config loader | 0 | +90 | (no-hardcode) | `test_config_loader_exposes_ddpg_env_reward_blocks` reads γ/τ/lr/n_rays/k_coverage from YAML |
| T00-05 | Draft `docs/adr/ADR-001-simulator-from-scratch.md` (no Gym/Gazebo/SB3 boundary, spec §1/§9) | Drafting ADR-001 (from-scratch boundary) | 0 | +120 docs | ADR-001 | ADR has Context / Decision / Consequences; names the AST no-gym test as enforcement |
| T00-06 | Draft remaining ADR stubs: ADR-002 unicycle kinematics · ADR-003 Gaussian (not OU) noise · ADR-004 coverage-grid + cleaning radius · ADR-005 HouseExpo adapter + pinned subset · ADR-006 reward shaping · ADR-007 network sizing + soft-update τ · ADR-008 multi-seed + held-out generalization (spec §9) | Drafting ADR-002..008 stubs | 0 | +280 docs | ADR-002..008 | Seven ADR files exist, each with Context / Decision / Consequences |
| T00-07 | CLAUDE.md inheritance (≤150 LOC, ruff, pytest ≥85%, uv-only, commit-subject regex, PII deny-list) carried into A5 repo | Importing CLAUDE.md inheritance | 0 | +0 | (gates) | `grep -E "150\|ruff\|85%\|uv\|Phase " CLAUDE.md` returns ≥6 matches |
| T00-08 | Stub `src/sdk/sdk.py` `RoboVacuumSDK` with the five method signatures (`build_env`, `train`, `evaluate`, `rollout`, `coverage_report`) raising `NotImplementedError` | Stubbing RoboVacuumSDK facade | 0 | +60 | (sdk) | `test_sdk_exposes_five_public_methods` passes against stubs |

## §3.1 Phase 1 — Simulator from scratch (each unit TDD)

Goal: build the 2D robotic-vacuum simulator (spec §3/§4) as eight focused
≤150-LOC `src/env/` units, each with its own deterministic test carrying a
hand-computed expectation. `VacuumEnv` exposes
`reset() / step(action) -> (state, reward, done, info)` — a **4-tuple**, with
**NO** `gymnasium` import (spec §3/§8). If `vacuum_env.py` approaches 150 LOC,
split helpers into a sibling `_*.py` module (A4 convention, spec §4).

| id | content | activeForm | phase | LOC-Δ | req | acceptance |
|----|---------|------------|-------|-------|-----|------------|
| T01-01 | Implement `src/env/house_map.py` — HouseExpo JSON → wall segments + free-space bounds | Implementing HouseExpo map loader | 1 | +130 | SIM-map | `test_house_map_parses_json_to_wall_segments_and_bounds` with a tiny hand-built JSON fixture |
| T01-02 | Implement `src/env/raycast.py` — ray–segment intersection → lidar distances (vectorized segment math, spec §10) | Implementing raycaster | 1 | +130 | SIM-raycast | `test_raycast_hits_known_wall_at_expected_distance` (hand-computed geometry: axis-aligned wall, 45° ray) + `test_ray_misses_returns_ray_max` |
| T01-03 | Implement `src/env/kinematics.py` — unicycle pose integrator (`x+=v·cosθ·Δt; y+=v·sinθ·Δt; θ+=ω·Δt`, `v=throttle·v_max`, `ω=steer·omega_max`) | Implementing unicycle kinematics | 1 | +90 | SIM-kinematics | `test_kinematics_straight_line_advances_by_v_dt` + `test_pure_rotation_changes_only_theta` (hand-computed with `dt=0.1`, `v_max=0.5`, `omega_max=1.5`) |
| T01-04 | Implement `src/env/coverage.py` — cleaned-cell grid (`coverage_cell=0.10`, `clean_radius=0.17`) + coverage % + `Δcells` per step | Implementing coverage grid | 1 | +120 | SIM-coverage | `test_new_cell_marked_once` (re-cleaning a cell yields Δcells=0) + `test_coverage_percent_matches_grid_count` |
| T01-05 | Implement `src/env/collision.py` — robot-radius (`robot_radius=0.17`) vs wall-segment collision test | Implementing collision test | 1 | +100 | SIM-collision | `test_robot_overlapping_wall_returns_collision` + `test_clear_space_returns_no_collision` (point-to-segment distance vs radius, hand-computed) |
| T01-06 | Implement `src/env/reward.py` — `r = k_coverage·Δcells − k_collision·hit − k_step` (k_coverage=1.0, k_collision=10.0, k_step=0.01 from config) | Implementing reward function | 1 | +80 | SIM-reward | `test_reward_signs_match_config` (new-cell positive, collision strongly negative, idle step costs `−k_step`) + `tests/architecture/test_reward_formula.py` asserts term structure + signs |
| T01-07 | Implement `src/env/state.py` — observation assembly: 16 lidar distances (`/ray_max`) + `(v, ω)` + heading cue to nearest uncleaned cell; all normalized | Implementing state assembler | 1 | +110 | SIM-state | `test_state_dim_is_20_for_16_rays` + `test_all_state_components_normalized` (lidar ∈[0,1], heading cue unit vector) |
| T01-08 | Implement `src/env/vacuum_env.py` — `VacuumEnv.reset()` (random free-cell respawn, clears coverage grid) + `step(action) -> (state, reward, done, info)` 4-tuple; ends at `max_steps` (=1000) or optional coverage target; **NO gym** | Implementing VacuumEnv | 1 | +140 | SIM-env | `test_reset_returns_state_vector` + `test_step_returns_4_tuple` + `test_episode_terminates_at_max_steps` |
| T01-09 | Add `tests/architecture/test_no_gym_import.py` — AST walk of `src/` rejecting any `import gymnasium` / `from gymnasium import …` (grep is insufficient — it misses aliased imports), spec §8 | Adding no-gym AST architecture test | 1 | +70 test | ARCH-no-gym | Test fails if any `src/**/*.py` AST contains `Import(name='gymnasium')` or `ImportFrom(module='gymnasium')` |
| T01-10 | Add `tests/env/test_env_smoke.py` — full random-policy rollout from `reset()` to `done=True`, asserting state shape, reward finiteness, and coverage monotonic non-decreasing | Adding env smoke test | 1 | +80 test | SIM-env | Random-policy rollout terminates within `max_steps` and coverage % never decreases |

## §3.2 Phase 2 — DDPG from scratch (each unit TDD)

Goal: implement DDPG (spec §5) as focused `src/model/` + `src/ddpg/` +
`src/services/` units. The brief code-requirement checklist (spec §5) is the
acceptance surface: Actor (Tanh-bounded deterministic action), Critic (takes
state AND action), **Polyak** soft-target update on both targets, **Gaussian**
exploration noise. THEORY.md (Phase 4) must cite the exact line numbers for the
Actor, Critic, Polyak update, and Gaussian noise. **NO SB3** — from scratch in
PyTorch (spec §1).

| id | content | activeForm | phase | LOC-Δ | req | acceptance |
|----|---------|------------|-------|-------|-----|------------|
| T02-01 | Implement `src/model/actor.py` — Actor MLP (`hidden_sizes=[256,256]`) → **Tanh**-bounded action ∈ [−1,1]² | Implementing Tanh-bounded Actor | 2 | +90 | DDPG-actor | `test_actor_output_shape_is_2` + `tests/architecture/test_actor_bounds.py` asserts every output ∈ [−1,1] over random+extreme inputs |
| T02-02 | Implement `src/model/critic.py` — Critic MLP (state ⊕ action → scalar Q) | Implementing Critic | 2 | +90 | DDPG-critic | `test_critic_takes_state_and_action_returns_scalar_q` (output shape `[batch, 1]`) |
| T02-03 | Implement `src/ddpg/replay_buffer.py` — uniform experience replay (`buffer_size=1000000`), `add` + `sample(batch_size=128)` | Implementing replay buffer | 2 | +90 | DDPG-buffer | `test_replay_sample_shapes` (state/action/reward/next_state/done tensor shapes) + `test_buffer_overwrites_at_capacity` |
| T02-04 | Implement `src/ddpg/noise.py` — Gaussian exploration noise (`sigma_start=0.2 → sigma_end=0.05` over `sigma_decay_steps=50000`); NOT OU (spec §5, ADR-003) | Implementing Gaussian noise | 2 | +80 | DDPG-noise | `test_gaussian_noise_seeded_reproducible` (same seed → same sample) + `test_sigma_decays_to_floor` |
| T02-05 | Implement `src/ddpg/agent.py` — `DDPGAgent`: `act()` (actor + noise during collection, clipped to [−1,1]), `update()` (critic TD target with `γ=0.99` + deterministic policy gradient, `grad_clip=1.0`), **Polyak** `soft_update(τ=0.005)` `θ_target = τ·θ + (1−τ)·θ_target` for **both** actor & critic targets | Implementing DDPGAgent (act/update/Polyak) | 2 | +140 | DDPG-agent | `test_one_finite_update_step` (no NaN; critic + actor losses finite) + `tests/architecture/test_soft_update_is_polyak.py` asserts the `τ·θ + (1−τ)·θ_target` form on a hand-set toy param |
| T02-06 | Implement `src/services/trainer.py` — custom training loop: collect (with `warmup_steps=1000` random actions) → store → `update()` → log per-episode reward + per-step critic loss | Implementing DDPG trainer loop | 2 | +130 | DDPG-trainer | `test_trainer_runs_one_short_episode_cycle` (smoke: warmup honored, buffer fills, one update fires, logs emitted) |
| T02-07 | Implement `src/cost/meter.py` — tiktoken/runtime cost accounting (spec §4/§11) feeding `docs/COST_ANALYSIS.md` | Implementing cost meter | 2 | +90 | COST-meter | `test_cost_meter_accumulates_runtime_and_tokens` |
| T02-08 | Wire `RoboVacuumSDK` real implementations — `build_env` (Phase 1 `VacuumEnv`), `train` (trainer), `evaluate`, `rollout`, `coverage_report`; replace Phase 0 `NotImplementedError` stubs | Wiring RoboVacuumSDK methods | 2 | +120 | SDK-impl | `tests/architecture/test_sdk_single_entry.py`: CLI + notebook import only `RoboVacuumSDK`; `grep` finds no direct `src.ddpg`/`src.env` import outside SDK + tests |
| T02-09 | Add `tests/architecture/test_no_sb3_import.py` — AST walk of `src/` rejecting `stable_baselines3` import (spec §1 "no ready RL library") | Adding no-SB3 AST architecture test | 2 | +60 test | ARCH-no-sb3 | Test fails if any `src/**/*.py` AST imports `stable_baselines3` |

## §3.3 Phase 3 — Training + Results

Goal: fetch the pinned HouseExpo subset, train DDPG across the 5 seeds, and
produce the two required graphs + trajectory visualization + held-out
generalization eval (spec §6/§7). Report honestly if convergence is partial
within the compute budget (spec §10).

| id | content | activeForm | phase | LOC-Δ | req | acceptance |
|----|---------|------------|-------|-------|-----|------------|
| T03-01 | Implement `scripts/fetch_houseexpo.py` — clone `github.com/TeaganLi/HouseExpo` at a **pinned commit**, stamp `dataset_sha` into config, leave the full ~35k JSON **git-ignored** (spec §6) | Implementing pinned HouseExpo fetcher | 3 | +110 | DATA-fetch | Script clones at a fixed SHA; `dataset_sha` no longer reads `PINNED_AT_FETCH`; full dump matches `.gitignore` |
| T03-02 | Vendor **4–6 curated plans** under `data/maps/` (single-room → multi-room apartment): train `["room_single", "apt_small", "apt_multi"]`, holdout `["apt_large", "office"]` (config `maps:`) | Vendoring curated HouseExpo subset | 3 | +0 (data) | DATA-curate | All five named maps present under `data/maps/`; `house_map.py` loads each without error |
| T03-03 | Implement `scripts/train_multi_seed.py` (or `RoboVacuumSDK.train` driver) — train across **5 seeds {42, 7, 123, 314, 271}** (config `training.seeds`, `episodes=500`, `eval_every=25`) on the train maps; persist per-seed reward arrays + critic-loss arrays | Running multi-seed training | 3 | +120 | TRAIN-multiseed | Five per-seed result artifacts exist; each carries reward-vs-episode + critic-loss-vs-step arrays |
| T03-04 | Implement `scripts/render_learning_curve.py` → `results/figures/learning_curve.png` — cumulative reward vs episode, **mean ± 95% CI over the 5 seeds** (spec §7) | Rendering learning_curve.png | 3 | +90 | RES-learning-curve | `results/figures/learning_curve.png` exists; caption names the 5 seeds, episode count, and the mean ± 95% CI band |
| T03-05 | Implement `scripts/render_critic_loss.py` → `results/figures/critic_loss.png` — critic loss vs training step (spec §7) | Rendering critic_loss.png | 3 | +80 | RES-critic-loss | `results/figures/critic_loss.png` exists; caption names seed(s) + step count |
| T03-06 | Implement `scripts/render_trajectory.py` — draw robot path in colour over the 2D JSON map, covered area shaded, proving wall-avoidance + smooth continuous coverage; optional short MP4/GIF (spec §7) | Rendering trajectory visualization | 3 | +120 | RES-trajectory | Trajectory PNG (and optional MP4/GIF) exists; path stays inside walls; shaded coverage region visible |
| T03-07 | Held-out **generalization** eval — run the trained policy on the holdout maps `["apt_large", "office"]`; report coverage % vs the train maps (spec §6, ADR-008) | Running held-out generalization eval | 3 | +90 | RES-generalization | Coverage % reported on `apt_large` + `office` (held-out, never trained on) alongside train-map coverage |
| T03-08 | Add `tests/test_results_artifacts.py` — assert both required figures exist after a short training run AND that the learning-curve artifact carries the per-seed arrays (not just a rendered image) | Adding results-artifact test | 3 | +80 test | RES-graphs | Test fails if `learning_curve.png` or `critic_loss.png` missing, or if per-seed reward arrays not extractable |

## §3.4 Phase 4 — Docs + Analysis

Goal: author the analysis docs (spec §7), the ISO-25010 quality write-up, the
README report, and the Moodle cover sheet. THEORY must cite the exact `src/`
line numbers for the Actor / Critic / Polyak update / Gaussian noise (spec §5).

| id | content | activeForm | phase | LOC-Δ | req | acceptance |
|----|---------|------------|-------|-------|-----|------------|
| T04-01 | Author `docs/THEORY.md` — DDPG objective, deterministic policy gradient, critic TD target (`γ=0.99`), Polyak update (`τ=0.005`), Gaussian exploration — LaTeX + citations; cross-ref table citing exact line numbers in `src/model/actor.py`, `src/model/critic.py`, `src/ddpg/agent.py` (Polyak), `src/ddpg/noise.py` (spec §5/§7) | Authoring THEORY.md | 4 | +240 docs | DOC-theory | All equations render in LaTeX; cross-ref table names exact `src/` module + line range per equation; DDPG citation present |
| T04-02 | Author `docs/ANALYSIS.md` — answer the brief's **3 analysis questions** (spec §7): (1) why DDPG not DQN/PPO (deterministic physical motors + continuous control + dataset reuse); (2) effect of removing Gaussian exploration noise early (coverage map collapses to a narrow path); (3) how target networks + soft updates prevent critic collapse | Authoring ANALYSIS.md (3 questions) | 4 | +200 docs | DOC-analysis | All three questions answered with evidence cross-links to figures/results; no question stubbed |
| T04-03 | Author `docs/COST_ANALYSIS.md` — tiktoken cl100k headline + chars/bytes appendix, AI-tooling + training-runtime cost table, **cost envelope** (architect-decided spend cap vs running total); consumes `src/cost/meter.py` | Authoring COST_ANALYSIS.md | 4 | +180 docs | DOC-cost | Headline number is tiktoken; appendix has chars + bytes; cost envelope section names cap + actual spend |
| T04-04 | Author `docs/QUALITY.md` — **ISO/IEC 25010** quality model write-up (functional suitability, performance efficiency, reliability, maintainability, portability) + honest self-assessment with bounded limitations | Authoring QUALITY.md (ISO 25010) | 4 | +200 docs | DOC-quality | Each named ISO-25010 characteristic has a paragraph with evidence pointer; honest-limitations section present |
| T04-05 | Author `README.md` report — install/run via `uv`, SDK usage, both figures embedded, trajectory viz, held-out generalization result, honest convergence framing (spec §10) | Authoring README report | 4 | +200 docs | DOC-readme | README boots a reader from clone → `uv sync` → train → figures; embeds `learning_curve.png` + `critic_loss.png` + trajectory |
| T04-06 | Author `docs/shared/PROMPTS.md` — verbatim prompts (architect → implementer trail per CLAUDE.md §1.4) mapped to commit hashes with human-judgment annotations | Authoring PROMPTS.md | 4 | +200 docs | §1.4 | Every prompt mapped to a commit hash; architect decisions annotated |
| T04-07 | Author `notebooks/analysis.ipynb` — consumes `RoboVacuumSDK` only; LaTeX theory blocks precede plots; renders learning curve + critic loss + trajectory from saved artifacts | Authoring analysis notebook | 4 | +200 nb | DOC-nb | Notebook imports only the SDK (no parallel impl); executes top-to-bottom |
| T04-08 | Run final gate sweep — ruff clean, all `.py` ≤150 LOC, coverage ≥85%, notebook executes top-to-bottom, no PII matches | Running final gate sweep | 4 | 0 | (gates) | `make check` (or scripted equivalent) exits 0; `grep -E "REDACTED-NAME\|REDACTED-HANDLE\|REDACTED\|REDACTED-ID\|GoogleDrive-REDACTED-HANDLE"` returns zero matches |
| T04-09 | Export submission PDF `adrl-001-ex05.pdf` (official Moodle template; self-grade lives only on the PDF, spec §2) | Exporting submission cover sheet | 4 | +0 | (submission) | `adrl-001-ex05.pdf` exists; carries group code `adrl-001`; self-grade only on the PDF |
| T04-10 | Invite `rmisegal` as read-only collaborator on `github.com/adirelm/RoboVacuumDDPG`; record invitation ID in `docs/SUBMISSION.md` (spec §2) | Inviting rmisegal as read-only collaborator | 4 | +0 | (submission) | Invitation sent with **Read** role to `@rmisegal`; invitation ID recorded |
| T04-11 | Tag submission commit `assignment-5`, push branch | Tagging submission and pushing branch | 4 | +0 | (submission) | Tag pushed |

## §3.5 Cross-phase invariants (recheck at every phase boundary)

These are not single tasks — they are properties the codebase must hold at the
end of every phase. The phase is only ✅ when every invariant below is
satisfied. They are the architecture-test invariants called out in spec §8.

| invariant | how it is checked | first phase enforced |
|---|---|---|
| Every `.py` ≤150 LOC (tests included) | pre-commit hook (T00-02) + CI | 0 |
| Ruff clean | `uv run ruff check src/ tests/ scripts/` exits 0 | 0 |
| Coverage ≥85% on deterministic core | `uv run pytest --cov=src --cov-report=term-missing` (`fail_under=85`) | 1 |
| Config single-source (no hardcoded RL/env/reward values in `src/`) | `src/utils/config_loader.py` is the only reader; spot-check grep for literal γ/τ/k_* in `src/` | 0 |
| No `gymnasium` import anywhere in `src/` | `tests/architecture/test_no_gym_import.py` AST walk (T01-09) | 1 |
| No `stable_baselines3` import anywhere in `src/` | `tests/architecture/test_no_sb3_import.py` AST walk (T02-09) | 2 |
| Actor action ∈ [−1,1] (Tanh-bounded) | `tests/architecture/test_actor_bounds.py` (T02-01) over random + extreme inputs | 2 |
| Soft-update is Polyak `θ_target = τ·θ + (1−τ)·θ_target` (both targets, τ=0.005) | `tests/architecture/test_soft_update_is_polyak.py` (T02-05) on a hand-set toy param | 2 |
| `RoboVacuumSDK` is single business-logic entry | `tests/architecture/test_sdk_single_entry.py` (T02-08); CLI + notebook import only the SDK | 2 |
| Reward formula matches spec `r = k_coverage·Δcells − k_collision·hit − k_step` | `tests/architecture/test_reward_formula.py` (T01-06) AST check on term structure + signs | 1 |
| Learning curve reported mean ± 95% CI across 5 seeds {42, 7, 123, 314, 271} | `results/figures/learning_curve.png` caption + per-seed arrays in artifact (T03-04, T03-08) | 3 |
| Both required figures exist (`learning_curve.png`, `critic_loss.png`) | `tests/test_results_artifacts.py` (T03-08) | 3 |
| Held-out generalization reported on holdout maps (not train maps) | coverage % on `apt_large` + `office` (T03-07) | 3 |
| HouseExpo cloned at a pinned SHA; full dump git-ignored | `dataset_sha` != `PINNED_AT_FETCH` (T03-01); `.gitignore` covers the full clone | 3 |
| Cost envelope recorded in `docs/COST_ANALYSIS.md` | T04-03 acceptance check | 4 |
| Notebook is a consumer, not a parallel impl | `grep -E "^class \|^def " notebooks/analysis.ipynb` returns only thin wrappers | 4 |
| No PII matches against deny-list | `grep -E "REDACTED-NAME\|REDACTED-HANDLE\|REDACTED\|REDACTED-ID\|GoogleDrive-REDACTED-HANDLE"` returns zero matches in tree (the literal pattern lives only in this TODO row and in CLAUDE.md) | 0 |
| Commit subject matches `^(Phase \d+\|Phase 0 bootstrap\|chore: bootstrap)` | git log walk in T04-08 final sweep | 0 |
