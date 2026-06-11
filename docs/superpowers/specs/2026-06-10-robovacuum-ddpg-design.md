# RoboVacuumDDPG — Design Spec (single source of truth)

> Bar-Ilan *Vibe Coding & RL* workshop — **Assignment 5** (Lecture 09, DDPG).
> Brief: `EX05-DDPG-Robot-Simulator.pdf`. This spec is the contract every other
> doc (PRD / PLAN / TODO / THEORY / ADRs / README) must agree with verbatim.
> Date 2026-06-10. Approved in brainstorming.

## 1. Goal & graded scope

Build, **from scratch**, (a) a 2D robotic-vacuum simulator that reads real
**HouseExpo** floor-plans and (b) a **DDPG** agent that learns continuous
navigation/coverage. The brief's central emphasis is the *hands-on DDPG
implementation* — the summary must point at our own code lines for the Actor,
Critic, Polyak soft-target update, and Gaussian exploration noise. **Ambition:
excellence-maxed** (full V3 compliance + research extras; target ~90+).

**Hard ban (brief §"דרישת חובה"):** no ready simulation platforms —
**no Gymnasium, no Gazebo** — and (by the "show your Polyak code lines" demand)
**no ready RL library** (no SB3); DDPG is implemented from scratch in PyTorch.

## 2. Repo & conventions

- New public repo **github.com/adirelm/RoboVacuumDDPG**, group code `adrl-001`,
  share read access with the lecturer's GitHub handle `@rmisegal`. Submission cover sheet
  `adrl-001-ex05.pdf` (official Moodle template; self-grade only on the PDF).
- Same V3 scaffold as A4: `uv` only · TDD (RED→GREEN→REFACTOR) · every `.py`
  **≤150 LOC** (tests included) · **≥85%** coverage (`fail_under=85`) · **0** Ruff
  violations · no hardcoded values (all in `config/config.yaml`) · single **SDK**
  entry point (`RoboVacuumSDK`) · `.env-example` · CI · version starts `1.0.0`.
- `docs/`: PRD, PLAN (C4 + UML + ADRs), TODO (phased + DoD), THEORY (DDPG math),
  ANALYSIS, COST_ANALYSIS, QUALITY (ISO 25010), UX (CLI/figures, §10), PROMPTS;
  dedicated `docs/prd/PRD-{SIM,DDPG,HOUSEEXPO}.md`; ADRs in `docs/adr/`.

## 3. MDP definition

- **Action** `a ∈ [−1,1]²` = `[throttle, steer]`. Unicycle model: linear velocity
  `v = throttle·V_MAX`, angular velocity `ω = steer·Ω_MAX`; pose integrated
  `x+=v·cosθ·Δt; y+=v·sinθ·Δt; θ+=ω·Δt`. Actor output is **Tanh**-bounded → exactly [−1,1].
- **State** = **20-dim** (all normalized): **16 lidar ray distances** (raycast to
  walls, /ray_max) + current `(v, ω)` (2) + a **heading cue** = the `cos`/`sin` of
  the bearing to the nearest uncleaned cell in the robot frame (2). The heading cue
  is a **unit vector** (not a raw angle) on purpose — it is continuous with no ±π
  wraparound, which DDPG's MLP learns far more stably. With `n_rays` as a config
  knob (8/16/24) the dim is `n_rays + 4` (so 12/20/28); the default 16 → **20**.
- **Reward** `r = k_cov·(new cells cleaned) − k_col·(collision) − k_step`.
  Defaults in config (e.g. k_cov=1.0, k_col=10.0, k_step=0.01).
- **Episode**: ends at `max_steps` or optional coverage target; `reset()` re-spawns
  the robot at a random free cell, clears the per-episode coverage grid.
- **No Gymnasium**: custom `VacuumEnv.reset()/step(action)->(state,reward,done,info)`;
  an AST architecture test forbids any `gymnasium` import under `src/`.

## 4. Module architecture (each a focused ≤150-LOC unit)

```
src/
├── sdk/sdk.py            # RoboVacuumSDK — single business-logic entry (build_env, train, evaluate, rollout, coverage_report)
├── env/
│   ├── house_map.py      # HouseExpo JSON → wall segments + free-space bounds
│   ├── raycast.py        # ray–segment intersection → lidar distances
│   ├── kinematics.py     # unicycle pose integrator
│   ├── coverage.py       # cleaned-cell grid + coverage %
│   ├── collision.py      # robot-radius vs wall-segment collision test
│   ├── reward.py         # r = k_cov·Δcoverage − k_col·collision − k_step
│   ├── state.py          # observation assembly (rays + speed + heading cue)
│   └── vacuum_env.py      # VacuumEnv: reset/step (4-tuple), NO gym
├── model/
│   ├── actor.py          # Actor MLP → Tanh-bounded action
│   └── critic.py         # Critic MLP (state⊕action → Q)
├── ddpg/
│   ├── replay_buffer.py  # uniform experience replay
│   ├── noise.py          # Gaussian exploration noise (brief-mandated)
│   └── agent.py          # DDPGAgent: act(), Polyak soft-update(τ), update()
├── services/
│   └── trainer.py        # custom training loop: collect → store → update → log
├── cost/meter.py         # tiktoken/runtime cost accounting (§11)
└── utils/config_loader.py
```
If `vacuum_env.py` or `agent.py` approach 150 LOC, split helpers into a sibling
`_*.py` module (A4 convention).

## 5. DDPG specifics (brief code-requirement checklist)

1. **Actor-Critic**: `model/actor.py` (Tanh-bounded deterministic action),
   `model/critic.py` (takes state AND action). Doc must cite exact line numbers.
2. **Soft Target Updates**: `ddpg/agent.py` performs Polyak averaging
   `θ_target = τ·θ + (1−τ)·θ_target` for both actor & critic targets; doc cites the lines.
3. **Hyperparameters** (all in config): `lr_actor`, `lr_critic`, `γ` (≈0.99),
   `τ` (=0.005), `noise_std` (initial Gaussian σ + decay), batch size, buffer size,
   hidden sizes. Doc justifies each.
4. **Exploration Noise**: **Gaussian** added to the actor action during collection
   (`ddpg/noise.py`); doc explains where, the initial variance, and γ/τ choices.

## 6. Data — HouseExpo

`scripts/fetch_houseexpo.py` clones `github.com/TeaganLi/HouseExpo` at a pinned
commit (full ~35k JSON **git-ignored**). Vendor **4–6 curated plans**
(single-room → multi-room apartment) under `data/maps/`. Train on a subset; hold
out 1–2 for a **generalization** evaluation. HouseExpo cited (Li et al. 2019,
arXiv:1903.09845).

## 7. Deliverables

- **Two required graphs**: `results/figures/learning_curve.png` (cumulative reward
  vs episode, mean±CI over seeds) and `results/figures/critic_loss.png` (critic
  loss vs training step). Scripts: `render_learning_curve.py`, `render_critic_loss.py`.
- **Trajectory visualization**: `render_trajectory.py` draws the robot path in
  colour over the 2D JSON map (covered area shaded), proving wall-avoidance +
  smooth continuous coverage; optional short MP4/GIF animation.
- **`docs/ANALYSIS.md`** answers the brief's **3 analysis questions**: (1) why DDPG
  not DQN/PPO (deterministic physical motors + continuous control + dataset reuse);
  (2) effect of removing Gaussian exploration noise early (coverage map collapses
  to a narrow path); (3) how target networks + soft updates prevent critic collapse.
- **`docs/THEORY.md`**: DDPG objective, deterministic policy gradient, GAE-free
  critic TD target, Polyak update, exploration — with LaTeX + citations.

## 8. Testing (TDD) & gates

Deterministic sim units (raycast geometry, kinematics integration, coverage
accounting, collision, reward signs, HouseExpo loader) → unit tests with hand-
computed expectations. DDPG units (replay sampling shapes, Gaussian noise
seeding, **Polyak math**, actor Tanh-bounds, critic output shape, one finite
update step) → unit tests. Architecture tests: **no `gymnasium` import** in
`src/`, actor action ∈[−1,1], soft-update is Polyak, SDK single-entry (CLI/notebook
import only the SDK), config single-source. Training = short smoke test.
**≥85% coverage** on the deterministic core; ruff clean; all files ≤150 LOC.

## 9. Planned ADRs

ADR-001 simulator-from-scratch boundary (no Gym/Gazebo) · ADR-002 unicycle
kinematic model · ADR-003 Gaussian (not OU) exploration noise · ADR-004
coverage-grid representation + cleaning radius · ADR-005 HouseExpo adapter +
pinned subset · ADR-006 reward shaping (coverage/collision/step) · ADR-007
network sizing + soft-update τ · ADR-008 multi-seed eval + held-out generalization.

## 10. Risks / open items

- Raycasting performance on complex maps (mitigate: vectorized segment math, cap
  ray count, coarse coverage grid). · Reward sparsity → slow learning (mitigate:
  dense coverage delta + step cost). · DDPG instability (mitigate: target nets,
  soft updates, gradient clip, seeded multi-run reporting). · Compute budget for
  training (time-box episodes; report honestly like A4 if convergence is partial).
