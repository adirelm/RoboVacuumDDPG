# ADR-001 — Simulator-From-Scratch Boundary (no Gymnasium, no Gazebo)

| Field | Value |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-06-10 |
| **Deciders** | Architect (human, CLAUDE.md §1.4) + AI (implementer) |
| **Supersedes** | — |
| **Related** | ADR-002 (unicycle model), ADR-004 (coverage grid), ADR-005 (HouseExpo adapter), ADR-006 (reward shaping); design spec §1, §3, §4, §8 |

## Context

The Assignment 5 brief (Lecture 09, DDPG; `EX05-DDPG-Robot-Simulator.pdf`)
states a hard mandatory requirement (brief §"דרישת חובה"): **no ready
simulation platforms** — explicitly **no Gymnasium and no Gazebo** — and, by
the brief's "show your Polyak code lines" demand, **no ready RL library** (no
Stable-Baselines3). The design spec (`docs/superpowers/specs/2026-06-10-robovacuum-ddpg-design.md`
§1) restates this verbatim: the simulator and the DDPG agent are both built
**from scratch** in PyTorch.

A typical RL submission reaches for `gymnasium.Env` to get `reset()` / `step()`
plumbing, vectorized wrappers, and a registry for free. That is exactly the
shortcut the brief forbids. The pull toward it is real: Gym's 4/5-tuple
contract is so standard that contributors import it reflexively, and a single
stray `import gymnasium` under `src/` would void the mandatory requirement and
the grade attached to it.

We therefore need (a) a concrete in-repo environment contract that mimics the
*shape* a grader expects (so the agent code reads like standard RL) without
*depending* on any banned platform, and (b) a mechanical gate that makes an
accidental banned import impossible to merge.

## Decision

Implement a custom environment, `VacuumEnv`, in `src/env/vacuum_env.py`, with a
hand-rolled control loop and **no dependency on Gymnasium, Gazebo, or any RL
library**. The public surface deliberately mirrors the familiar Gym signature
so the agent and trainer code stay idiomatic:

```python
# src/env/vacuum_env.py
class VacuumEnv:
    def reset(self) -> state: ...
    def step(self, action) -> (state, reward, done, info)   # 4-tuple, NO gym
```

- `reset()` re-spawns the robot at a random free cell and clears the
  per-episode coverage grid (design spec §3).
- `step(action)` integrates the unicycle one `dt`, raycasts the 16 lidar rays,
  computes the reward, and returns the standard 4-tuple `(state, reward, done,
  info)`. The 4-tuple shape is a *convention we re-implement ourselves*, not an
  inherited base class.
- The environment is assembled from the focused `src/env/` units listed in
  design spec §4 (`house_map`, `raycast`, `kinematics`, `coverage`,
  `collision`, `reward`, `state`), each a ≤150-LOC module per CLAUDE.md §1.
- The DDPG stack (`src/model/`, `src/ddpg/`) is likewise from scratch in
  PyTorch: Actor, Critic, replay buffer, Gaussian noise, and the Polyak
  soft-update all live in our own code so the summary can cite exact line
  numbers (design spec §1, §5).

**Enforcement (the gate that makes the ban real).** An AST-based architecture
test asserts that **no `gymnasium` import appears anywhere under `src/`**
(design spec §3, §8). The test walks every `.py` module under `src/`, parses
it, and fails if any `Import` / `ImportFrom` node names `gymnasium` (or
`gym`). The same architecture-test family also asserts SDK single-entry (CLI
and notebooks import only `RoboVacuumSDK`) and config single-source. These
tests run in CI on every push, so a regression cannot land silently.

## Consequences

**Positive.**
- The mandatory brief requirement is satisfied *and* enforced — not merely
  promised in prose. A grader can run the AST test and watch it pass.
- Every line of physics, sensing, and RL math is ours, so the summary points
  at our own code for the Actor, Critic, Polyak soft-target update, and
  Gaussian exploration noise (design spec §1) — the brief's central emphasis.
- The 4-tuple `reset()`/`step()` shape keeps the trainer (`src/services/trainer.py`)
  and SDK (`src/sdk/sdk.py`) readable to anyone who knows Gym, with zero
  banned dependency.
- Full control over determinism: seeded resets, hand-computed unit-test
  expectations for raycast geometry, kinematics, coverage, collision, and
  reward signs (design spec §8).

**Negative.**
- We re-implement plumbing Gym would give for free (episode bookkeeping, seed
  handling, the 4-tuple contract). Bounded by the ≤150-LOC-per-module rule and
  the small `src/env/` decomposition; if `vacuum_env.py` approaches 150 LOC we
  split helpers into a sibling `_*.py` module (design spec §4 / A4 convention).
- No off-the-shelf vectorized envs, wrappers, or monitoring; we provide only
  what training and evaluation actually use.
- We own correctness of the physics and sensing. Mitigated by deterministic
  unit tests with hand-computed expectations (design spec §8).

## Alternatives considered

| # | Alternative | Verdict | Why rejected |
|---|---|---|---|
| a | Use `gymnasium.Env` as the base and wrap our dynamics | Rejected | Directly violates the brief's mandatory "no Gymnasium" requirement; voids the grade attached to it. |
| b | Use Gazebo / a ROS-backed physics sim for realism | Rejected | Explicitly banned ("no Gazebo"); also vastly over-scoped for a 2D coverage task and not reproducible on a grader's machine. |
| c | Build the env from scratch but train with Stable-Baselines3 DDPG | Rejected | The brief's "show your Polyak code lines" demand requires a from-scratch agent; SB3 hides the soft-update and exploration code we must cite. |
| d | Subclass nothing but copy Gym's exact API names from the package | Rejected | Importing names from `gymnasium` to "match the API" still trips the AST ban; we re-implement the *convention*, not the dependency. |
| e | From-scratch env + from-scratch PyTorch DDPG, AST-gated against banned imports | **Chosen** | Satisfies the mandatory ban, keeps all RL math citable, stays inside the ≤150-LOC / SDK-single-entry / config-single-source gates. |

## Review trigger

Re-open this ADR if: the brief's banned-platform list changes; any
`src/` module needs a banned import for a capability we cannot reasonably
hand-roll (record the justification here before relaxing the AST gate); or the
SDK single-entry / config single-source invariants are amended.
