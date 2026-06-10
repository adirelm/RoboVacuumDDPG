# PRD-DDPG — Deep Deterministic Policy Gradient Agent

Status: Draft v0.2
Owner: human architect (per CLAUDE.md §1.4)
Master reference: design spec `docs/superpowers/specs/2026-06-10-robovacuum-ddpg-design.md`
§5 (DDPG specifics) + §4 (module architecture). Group code `adrl-001`, repo
`github.com/adirelm/RoboVacuumDDPG`.

## 1. Purpose

DDPG is the learning core of Assignment 5. It consumes the `(state, reward,
done, info)` 4-tuple from the from-scratch simulator (`PRD-SIM`,
`src/env/vacuum_env.py`) over HouseExpo floor-plans (`PRD-HOUSEEXPO`), and
outputs a **deterministic** continuous-control policy that drives the
unicycle vacuum to cover floor area while avoiding walls.

Satisfies the brief (`EX05-DDPG-Robot-Simulator.pdf`, Lecture 09, DDPG) and
its central "show your own code lines" demand: the submission summary must
point at OUR lines for the **Actor**, the **Critic**, the **Polyak
soft-target update**, and the **Gaussian exploration noise** (spec §5).

**Hard ban (spec §1, brief "דרישת חובה").** DDPG is implemented **from
scratch in PyTorch** — no SB3, no RLlib, no ready RL library. The simulator
side bans Gymnasium and Gazebo (`PRD-SIM`). DQN and PPO are out of scope as
the primary optimizer; the *why-not* is argued in §4 and in `docs/ANALYSIS.md`
question 1.

## 2. Functional Requirements

All values live in `config/config.yaml` per CLAUDE.md §4, read through
`src/utils/config_loader.py`. No hyperparameter is hardcoded in `src/`.

| ID | Requirement | Value | Config key |
|----|-------------|-------|------------|
| F5.1 | Discount factor `γ` | **0.99** | `ddpg.gamma` |
| F5.2 | Polyak soft-update coefficient `τ` | **0.005** (brief example value) | `ddpg.tau` |
| F5.3 | Actor learning rate | **1e-4** | `ddpg.lr_actor` |
| F5.4 | Critic learning rate | **1e-3** (faster than actor — standard DDPG) | `ddpg.lr_critic` |
| F5.5 | Minibatch size | **128** | `ddpg.batch_size` |
| F5.6 | Replay buffer capacity | **1 000 000** | `ddpg.buffer_size` |
| F5.7 | Hidden layer sizes (actor & critic) | **[256, 256]** | `ddpg.hidden_sizes` |
| F5.8 | Gradient-norm clip | **1.0** | `ddpg.grad_clip` |
| F5.9 | Warmup (random actions before learning) | **1000** steps | `ddpg.warmup_steps` |
| F5.10 | Exploration noise type | **Gaussian** (NOT Ornstein–Uhlenbeck) | `noise.type` |
| F5.11 | Initial noise std `σ_start` | **0.2** | `noise.sigma_start` |
| F5.12 | Final noise std `σ_end` | **0.05** | `noise.sigma_end` |
| F5.13 | Noise decay horizon | **50 000** steps | `noise.sigma_decay_steps` |
| F5.14 | Action space | continuous `[throttle, steer] ∈ [−1,1]²` | `env.*` (v_max, omega_max) |
| F5.15 | Actor output bound | **Tanh** → exactly [−1,1] | (architecture, `model/actor.py`) |
| F5.16 | Critic input | **state ⊕ action** → scalar Q | (architecture, `model/critic.py`) |
| F5.17 | Seeds per evaluation | **5** — `[42, 7, 123, 314, 271]` | `training.seeds` |
| F5.18 | Training episodes / eval cadence | **500 episodes**, eval every **25** | `training.episodes`, `training.eval_every` |

**Citation discipline.** `τ = 0.005`, `γ = 0.99`, the actor-slower-than-critic
LR split (`1e-4` / `1e-3`), uniform replay, and the deterministic target-policy
TD target are all from Lillicrap et al., *Continuous control with deep
reinforcement learning*, **arXiv:1509.02971** (ICLR 2016) — the DDPG paper.
The deterministic policy gradient itself is Silver et al., *Deterministic
Policy Gradient Algorithms*, **ICML 2014**. The one deliberate *deviation*
from the paper is the **noise model**: the brief mandates **Gaussian**
exploration noise, whereas Lillicrap et al. used an Ornstein–Uhlenbeck
process; the simpler uncorrelated Gaussian is the modern default (e.g. TD3,
Fujimoto et al. arXiv:1802.09477) and is recorded in **ADR-003**. The decision
is the human architect's per CLAUDE.md §1.4.

## 3. Algorithm — losses, gradients, and the four checkpoints

### 3.1 Critic TD target and loss (no GAE — off-policy, single-step)

For a minibatch `{(s, a, r, s′, d)}` of size `batch_size` (F5.5) sampled
uniformly from the replay buffer (F5.6), the target uses the **target actor**
`μ′` and **target critic** `Q′`:

```
y = r + γ · (1 − d) · Q′( s′, μ′(s′) )
L_critic(φ) = (1/N) · Σ ( Q_φ(s, a) − y )²
```

`γ = 0.99` (F5.1). `y` is treated as a constant (no gradient flows through the
target networks). This is the standard DDPG single-step TD target — there is
**no GAE and no λ** (that is a PPO construct; DDPG is off-policy actor-critic).

### 3.2 Deterministic policy gradient (actor loss)

The actor `μ_θ` is updated to maximize the current critic's Q-value of its own
action — i.e. minimize the negated Q:

```
L_actor(θ) = − (1/N) · Σ Q_φ( s, μ_θ(s) )
```

Gradients backprop through the critic into the actor (deterministic policy
gradient, Silver 2014 / Lillicrap 2016). Both losses are clipped to
`grad_clip = 1.0` (F5.8) by global norm before each optimizer step.

### 3.3 Polyak soft target update (CHECKPOINT 2)

After every gradient step, both target networks track their online networks
by Polyak averaging with `τ = 0.005` (F5.2):

```
θ_target ← τ · θ + (1 − τ) · θ_target        # target actor
φ_target ← τ · φ + (1 − τ) · φ_target        # target critic
```

Implemented in `src/ddpg/agent.py` as an in-place `soft_update(online,
target, tau)` over `zip(target.parameters(), online.parameters())`. The
submission summary cites the **exact line numbers** of this method (spec §5.2).
Acceptance test in §5.1 verifies the math element-wise.

### 3.4 The brief's FOUR code-requirement checkpoints

The brief grades the *hands-on DDPG implementation* against four points; this
PRD pins each to a file and an acceptance test.

| # | Checkpoint | Location (spec §4) | Verified by |
|---|------------|--------------------|-------------|
| 1 | **Actor–Critic networks** | `src/model/actor.py` (Tanh-bounded deterministic action), `src/model/critic.py` (state ⊕ action → scalar Q) | §5.2 (Tanh bounds), §5.3 (critic shape) |
| 2 | **Soft Target Updates** | `src/ddpg/agent.py` Polyak `θ_t ← τθ + (1−τ)θ_t` for actor & critic targets | §5.1 (Polyak math) |
| 3 | **Hyperparameters** | `config/config.yaml` (`ddpg.*`, `noise.*`) loaded via `config_loader` — F5.1–F5.13 | §2 table; config single-source test |
| 4 | **Exploration noise** | `src/ddpg/noise.py` Gaussian σ-schedule added to actor action during collection | §5.4 (decay), ADR-003 |

The doc must cite the literal line numbers for checkpoints 1 and 2 once the
code is GREEN (spec §5.1–§5.2). Until then the locations above are the contract.

## 4. Why DDPG (not DQN, not PPO)

- **DQN:** discrete-action only; `argmax_a Q(s,a)` cannot be enumerated over a
  continuous `[throttle, steer] ∈ [−1,1]²` motor command. Discretizing the
  action grid throws away the smooth control the physical vacuum needs.
- **PPO:** on-policy and **stochastic**; throws away each rollout after one
  update and cannot reuse the HouseExpo experience the way an off-policy
  replay buffer can. The vacuum's motors are deterministic actuators — a
  deterministic target policy is the natural fit.
- **DDPG:** off-policy, **deterministic** actor with a Q-critic — continuous
  control + sample-efficient replay reuse + Polyak-stabilized targets. Matches
  Lillicrap et al., 2016 and Lecture 09.

This is `docs/ANALYSIS.md` question 1 (deterministic physical motors +
continuous control + dataset reuse).

## 5. Acceptance Criteria

Per CLAUDE.md §2 (TDD) and spec §8, the DDPG units have hand-checkable
acceptance tests. Tests 5.1–5.3 are **blocking** (deterministic, must pass on
every commit); 5.4–5.5 are blocking shape/behavior checks; 5.6 is the
convergence-quality gate evaluated over the final 20% of training across the
5 seeds (F5.17).

### 5.1 Polyak math (blocking)
Given online params all `= 1.0`, target params all `= 0.0`, and `τ = 0.005`,
one `soft_update` yields every target param `= 0.005` exactly (within fp
tolerance). A second call yields `0.005·1 + 0.995·0.005 = 0.009975`. Asserts
the update direction (target moves *toward* online) and the τ-weighting.

### 5.2 Tanh action bounds (blocking)
For 1000 random states (and for adversarially large pre-activations), the
actor output is element-wise within `[−1, 1]` inclusive — the Tanh squashing
guarantee (F5.15). Architecture test (spec §8) asserts the same on a separate
batch: `(action >= -1).all() and (action <= 1).all()`.

### 5.3 Critic output shape (blocking)
`critic(state_batch, action_batch)` returns shape `(batch_size, 1)`; the
concatenation is `state ⊕ action` (F5.16), so the critic's first linear layer
has `in_features == state_dim + action_dim` (action_dim = 2).

### 5.4 Finite, seeded update + noise schedule (blocking)
- One full `agent.update(batch)` step produces **finite** (no NaN/Inf)
  critic loss and actor loss, and finite gradients after `grad_clip = 1.0`.
- Replay sampling with a fixed seed returns reproducible index batches.
- Gaussian noise: `σ` decays linearly from `σ_start = 0.2` to `σ_end = 0.05`
  over `sigma_decay_steps = 50 000` (F5.11–F5.13) and is clamped at `σ_end`
  thereafter; the noisy action is re-clipped to `[−1, 1]`.

### 5.5 No-gym / single-entry architecture (blocking)
- AST test: **no `gymnasium` import** anywhere under `src/` (spec §8).
- SDK single-entry: the CLI / notebook / render scripts import only
  `RoboVacuumSDK` (CLAUDE.md §3); no DDPG logic leaks into a UI module.
- Every `.py` (agent, actor, critic, buffer, noise) is **≤150 LOC**; if
  `agent.py` nears the cap, split helpers into a sibling `_*.py` (spec §4).

### 5.6 Convergence quality (gate, advisory-then-blocking)
Evaluated over the final 20% of the 500 training episodes (F5.18), averaged
across the 5 seeds (F5.17), via greedy (noise-off) rollouts:
- **Critic loss converges:** rolling `critic_loss` slope `≤ 0` over the final
  window (proves the target-network + soft-update stabilization, ANALYSIS
  question 3). Surfaced in `results/figures/critic_loss.png`.
- **Coverage / reward rises:** mean cumulative reward in the final window is
  above the warmup-baseline window; rising coverage % across episodes.
  Surfaced in `results/figures/learning_curve.png` (mean ± CI over seeds).
- Removing Gaussian noise early collapses the coverage map to a narrow path
  (ANALYSIS question 2) — a documented ablation, not a pass condition.

Per CLAUDE.md §1.4, if compute time-boxing leaves convergence partial, the
shortfall is reported **honestly** (A4 precedent, spec §10) rather than masked.

## 6. Implementation Path — from scratch in PyTorch

No SB3. The agent is assembled from four focused ≤150-LOC modules (spec §4):

- `src/model/actor.py` — Actor MLP `state → [256,256] → Tanh` → action ∈[−1,1]².
- `src/model/critic.py` — Critic MLP `(state ⊕ action) → [256,256] → 1` (scalar Q).
- `src/ddpg/replay_buffer.py` — uniform experience replay, capacity 1e6 (F5.6),
  returns batched `(s, a, r, s′, d)` tensors.
- `src/ddpg/noise.py` — Gaussian σ-schedule (F5.10–F5.13); brief-mandated.
- `src/ddpg/agent.py` — `DDPGAgent`: `act(state, explore)` (adds noise during
  collection), `update(batch)` (critic TD + deterministic policy gradient +
  grad-clip), `soft_update(τ)` Polyak averaging; holds online + target copies
  of actor and critic.
- `src/services/trainer.py` — custom training loop: collect → store → (after
  `warmup_steps`) sample → `update` → `soft_update` → log. **No Gym loop.**

All reachable only through `RoboVacuumSDK` (`src/sdk/sdk.py`):
`build_env / train / evaluate / rollout / coverage_report`.

## 7. References

- Design spec (single source of truth):
  `docs/superpowers/specs/2026-06-10-robovacuum-ddpg-design.md` §5 (DDPG
  specifics + 4-checkpoint checklist), §4 (modules), §8 (tests).
- Config: `config/config.yaml` (`ddpg`, `noise`, `env`, `reward`, `training`).
- Standards: `CLAUDE.md` (≤150 LOC, ≥85% coverage, 0 Ruff, no hardcoded values,
  uv-only, SDK single entry, §1.4 architect↔implementer contract).
- Brief: `EX05-DDPG-Robot-Simulator.pdf` (Lecture 09, DDPG).
- Lillicrap, Hunt, Pritzel, Heess, Erez, Tassa, Silver, Wierstra, 2016,
  "Continuous control with deep reinforcement learning", **arXiv:1509.02971**
  (ICLR 2016) — source of DDPG, `τ`-soft updates, actor/critic LR split.
- Silver, Lever, Heess, Degris, Wierstra, Riedmiller, 2014, "Deterministic
  Policy Gradient Algorithms", **ICML 2014** — deterministic policy gradient.
- Fujimoto, van Hoof, Meger, 2018, "Addressing Function Approximation Error in
  Actor-Critic Methods" (TD3), **arXiv:1802.09477** — supports uncorrelated
  Gaussian exploration over OU.
- Sibling PRDs: `docs/prd/PRD-SIM.md` (VacuumEnv 4-tuple, no Gym),
  `docs/prd/PRD-HOUSEEXPO.md` (map adapter). HouseExpo: Li et al., 2019,
  **arXiv:1903.09845**.
- ADRs: ADR-003 (Gaussian not OU exploration), ADR-007 (network sizing +
  soft-update τ), ADR-008 (multi-seed eval + held-out generalization).
