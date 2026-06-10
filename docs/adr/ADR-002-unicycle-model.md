# ADR-002 — Unicycle Kinematic Model for Continuous Control

| Field | Value |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-06-10 |
| **Deciders** | Architect (human, CLAUDE.md §1.4) + AI (implementer) |
| **Supersedes** | — |
| **Related** | ADR-001 (simulator from scratch), ADR-003 (Gaussian noise on `[throttle, steer]`), ADR-007 (network output dim), design spec §3 |

## Context

DDPG is a continuous-action algorithm; the brief's whole reason for choosing it
(design spec §7 analysis Q1) is **deterministic physical motors + continuous
control**. We need a motion model that (a) exposes a low-dimensional continuous
action, (b) maps cleanly onto a Tanh-bounded actor output, and (c) is cheap and
deterministic enough for hand-computed unit tests (design spec §8).

A real differential-drive vacuum is controlled by two wheel velocities, but the
brief and design spec frame the action at the higher level of *throttle and
steering*. We must pick the exact action semantics, the pose integrator, and
the velocity caps — and they must match `config/config.yaml` and the state
vector (`(v, ω)` are part of the 20-dim observation, design spec §3).

## Decision

Adopt the **unicycle (differential-drive abstraction) kinematic model** with a
2-D continuous action and explicit Euler pose integration, exactly as design
spec §3 specifies.

**Action** `a ∈ [−1, 1]² = [throttle, steer]`. The actor (`src/model/actor.py`)
is **Tanh**-bounded so its output is exactly `[−1, 1]` (design spec §3, §5).

**Velocity mapping** (`src/env/kinematics.py`):

```
v = throttle · V_MAX        # linear velocity
ω = steer    · Ω_MAX        # angular velocity
```

**Pose integration** (forward Euler, one step per `dt`):

```
x += v·cos(θ)·Δt
y += v·sin(θ)·Δt
θ += ω·Δt
```

**Constants — all from `config/config.yaml#env` (CLAUDE.md §4, no hardcoding):**

| Symbol | Config key | Value | Meaning |
|---|---|---|---|
| `V_MAX` | `env.v_max` | `0.5` m/s | max linear velocity = `throttle · v_max` |
| `Ω_MAX` | `env.omega_max` | `1.5` rad/s | max angular velocity = `steer · omega_max` |
| `Δt` | `env.dt` | `0.1` s | integration timestep |
| `robot_radius` | `env.robot_radius` | `0.17` m | body radius (collision, ADR-004 boundary) |

The current `(v, ω)` produced by this model is fed back into the observation
vector (design spec §3) so the policy is aware of its own motion state.

## Consequences

**Positive.**
- The 2-D `[throttle, steer]` action is the minimal continuous control that
  still lets the robot navigate and turn — a natural fit for a 2-output
  Tanh actor (ADR-007) and for Gaussian exploration noise added per-component
  in `[−1, 1]` (ADR-003).
- Closed-form, deterministic integration makes kinematics unit-testable with
  hand-computed expectations (design spec §8): given a fixed action and pose,
  the next pose is exact.
- Cheap per-step cost (a handful of trig ops), so raycasting — not motion — is
  the per-step bottleneck (design spec §10), and we can afford `max_steps=1000`.
- Velocity caps live in config, so a grader can retune `v_max` / `omega_max`
  without touching source (CLAUDE.md §4).

**Negative.**
- Forward Euler accumulates integration error at large `Δt`; with `Δt = 0.1` s
  and the modest `v_max = 0.5` m/s the per-step displacement is ≤ 0.05 m
  (well under the `robot_radius = 0.17` m), so tunneling through walls is
  unlikely and is independently caught by the collision test (ADR-004 / §4).
- The unicycle ignores wheel slip, acceleration limits, and inertia. Acceptable:
  the brief grades coverage-navigation learning, not high-fidelity dynamics, and
  the simplification keeps the model fully testable.

## Alternatives considered

| # | Alternative | Verdict | Why rejected |
|---|---|---|---|
| a | **Unicycle model, `[throttle, steer]`, Euler integration** | **Chosen** | Matches design spec §3 verbatim; minimal continuous action; Tanh-friendly; deterministically testable. |
| b | Explicit differential-drive with two wheel-velocity actions `[v_L, v_R]` | Rejected | Same expressive power once mapped, but a less intuitive action space and an extra mixing step; the spec fixes `[throttle, steer]`. |
| c | Holonomic point-mass (`[vx, vy]`, no heading) | Rejected | Lets the robot strafe arbitrarily — unrealistic for a vacuum and removes the heading dynamics that make the lidar/heading-cue state meaningful. |
| d | Second-order dynamics (acceleration-controlled, inertia) | Rejected | Adds state dimensions and tuning with no grading benefit; harder to unit-test; over-scoped for a 2-D coverage task. |
| e | Higher-order integrator (RK4) | Rejected | Unnecessary accuracy at `Δt = 0.1` s and `v_max = 0.5` m/s; Euler error is already below the collision-detection margin. Revisit only if integration drift shows up in trajectory plots. |

## Review trigger

Re-open if: the action semantics in design spec §3 change; integration drift
becomes visible in `render_trajectory.py` output; or `v_max`/`omega_max`/`dt`
are retuned in a way that makes per-step displacement approach `robot_radius`
(then reconsider RK4 or sub-stepping).
