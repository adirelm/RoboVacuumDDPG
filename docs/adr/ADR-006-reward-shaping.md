# ADR-006 — Reward Shaping (coverage / collision / step)

| Field | Value |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-06-10 |
| **Deciders** | Architect (human, CLAUDE.md §1.4) + AI (implementer) |
| **Supersedes** | — |
| **Related** | ADR-004 (Δcoverage source), ADR-002 (collision via pose/radius), ADR-003 (γ context), design spec §3, §7 |

## Context

The agent must learn to **cover** the floor while **avoiding walls**, and do so
**efficiently** (not dawdle). The design spec (§3) fixes the reward form:

```
r = k_cov · (new cells cleaned)  −  k_col · (collision)  −  k_step
```

This ADR locks the *semantics, signs, and magnitudes* of each term, the source
of each signal, and ties them to the analysis the brief requires. Reward sparsity
is a named risk (design spec §10): a coverage-only-at-episode-end signal would
make DDPG learn slowly. We must keep the signal **dense** while still penalizing
collisions and wasted steps.

## Decision

Implement the reward in `src/env/reward.py` as a per-step sum of three terms,
with **all coefficients in `config/config.yaml#reward`** (CLAUDE.md §4, no
hardcoding):

```
r_t = k_coverage · Δcoverage_t          # dense progress signal (+)
    − k_collision · 1[collision_t]       # safety penalty       (−)
    − k_step                             # per-step time cost   (−)
```

| Term | Config key | Value | Source / meaning |
|---|---|---|---|
| `k_coverage` | `reward.k_coverage` | `1.0` | × `Δcoverage` = new cells flipped `False→True` this step (ADR-004) |
| `k_collision` | `reward.k_collision` | `10.0` | × `1` if `collision.py` flags a wall-segment hit this step (ADR-002 pose, `robot_radius`) |
| `k_step` | `reward.k_step` | `0.01` | constant per-step subtraction (time/efficiency pressure) |

- **`Δcoverage`** comes straight from the coverage grid (ADR-004): the integer
  count of cells newly cleaned this step. This is the **dense** term that
  rewards continuous progress and is the primary defense against reward sparsity
  (design spec §10).
- **Collision** is a binary event from `src/env/collision.py` (robot-radius vs
  wall-segment). The `10.0 : 1.0` ratio means one wall hit costs as much as
  cleaning ~10 fresh cells — a deliberately stiff penalty so the policy learns
  to avoid walls early.
- **`k_step = 0.01`** is a small constant drag so the agent prefers covering new
  area over idling, without overwhelming the coverage signal.

These signs and magnitudes are exactly the config defaults (`k_coverage=1.0`,
`k_collision=10.0`, `k_step=0.01`), matching design spec §3. The reward function
is a deterministic unit (design spec §8: "reward signs" is a tested expectation)
— given a step's Δcoverage and collision flag, `r_t` is exact and hand-checkable.

## Consequences

**Positive.**
- Dense `k_coverage·Δcoverage` term keeps the reward informative on (almost)
  every step the robot reaches new area, directly mitigating the sparsity risk
  (design spec §10) and giving DDPG's critic a smooth target to fit.
- The stiff collision penalty (`10×` the per-cell reward) makes wall-avoidance a
  first-class learned behavior, which the trajectory plot (design spec §7) is
  meant to demonstrate.
- All three coefficients live in config, so reward ablations (e.g. raise/lower
  `k_collision`, drop `k_step`) are config edits a grader can run — no source
  change (CLAUDE.md §4).
- Clean, signed, hand-computable form makes the "reward signs" unit test trivial
  and unambiguous (design spec §8).

**Negative.**
- Three coefficients are a small reward-engineering surface; mis-balancing them
  (e.g. `k_step` too large) can make the agent freeze. Bounded by keeping them
  in config and justifying each in `docs/ANALYSIS.md`.
- A pure coverage-delta term gives no explicit reward for *returning home* or
  for smoothness; the brief grades coverage + wall-avoidance, so we keep the
  reward minimal rather than over-shaping it.
- The collision penalty does not terminate the episode (the robot is blocked,
  not destroyed); whether collisions also set `done` is left to `vacuum_env.py`
  per design spec §3's `max_steps`/coverage-target termination, not to this ADR.

## Alternatives considered

| # | Alternative | Verdict | Why rejected |
|---|---|---|---|
| a | **`k_cov·Δcoverage − k_col·collision − k_step` with config coefficients** | **Chosen** | Exactly the spec form (§3); dense, signed, hand-testable, fully config-tunable. |
| b | Sparse terminal reward (coverage % only at episode end) | Rejected | Maximally sparse — the exact failure mode design spec §10 warns about; starves DDPG's critic of signal. |
| c | Coverage reward without any step cost | Rejected | Removes efficiency pressure; agent can wander indefinitely within `max_steps` with no penalty for slow coverage. |
| d | Hardcode coefficients in `reward.py` | Rejected | Violates CLAUDE.md §4 (no hardcoded rewards); blocks the reward ablations ANALYSIS needs. |
| e | Add curiosity / smoothness / return-to-base shaping terms | Rejected | Over-shapes beyond the brief's coverage+avoidance objective; adds tuning surface and confounds the analysis. Deferred unless coverage stalls. |

## Review trigger

Re-open if: coverage stalls or the agent learns to freeze (re-balance `k_step`
vs `k_coverage`); the collision penalty proves too soft/stiff in trajectory
plots; or design spec §3 changes the reward form. Log any coefficient retune in
`docs/ANALYSIS.md` with the seed evidence behind it.
