# ADR-003 — Gaussian (not OU) Exploration Noise

| Field | Value |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-06-10 |
| **Deciders** | Architect (human, CLAUDE.md §1.4) + AI (implementer) |
| **Supersedes** | — |
| **Related** | ADR-002 (action space `[throttle, steer]`), ADR-007 (τ / network sizing), design spec §3, §5, §7 |

## Context

DDPG learns a **deterministic** policy; without injected noise the actor would
output the same action for the same state and never explore. The classic DDPG
paper used **Ornstein-Uhlenbeck (OU)** temporally-correlated noise. However, the
Assignment 5 brief **mandates Gaussian** exploration noise (design spec §5
item 4: "Exploration Noise: **Gaussian** added to the actor action during
collection"), and design spec §7 analysis Q2 specifically asks us to demonstrate
"the effect of removing Gaussian exploration noise early (coverage map collapses
to a narrow path)." So the *type* is fixed by the brief; what this ADR decides is
**where** noise is added, the **schedule**, and the exact config-driven values.

## Decision

Implement zero-mean **Gaussian** exploration noise in `src/ddpg/noise.py` and add
it to the actor's action **only during environment collection** (not during the
Polyak target evaluation or during deterministic evaluation rollouts).

**Where.** In the collection loop (`src/services/trainer.py` via the agent's
`act()`), the exploration action is:

```
a = clip( actor(s) + N(0, σ²) ,  −1, +1 )
```

The clip keeps the noised action inside the `[−1, 1]²` bound that the unicycle
mapping (ADR-002) and the Tanh actor (ADR-007) assume. During the first
`warmup_steps` the agent emits **uniform random** actions instead (no learning),
then switches to noised-actor actions.

**Schedule — linear σ decay, all values from `config/config.yaml#noise`
(CLAUDE.md §4, no hardcoding):**

| Symbol | Config key | Value | Meaning |
|---|---|---|---|
| type | `noise.type` | `gaussian` | brief-mandated noise family |
| `σ_start` | `noise.sigma_start` | `0.2` | initial std added to actions in `[−1, 1]` |
| `σ_end` | `noise.sigma_end` | `0.05` | floor std after decay |
| decay horizon | `noise.sigma_decay_steps` | `50000` | steps over which σ anneals `σ_start → σ_end` |
| warmup | `ddpg.warmup_steps` | `1000` | uniform-random steps before noised policy |

Decay is enforced relative to global environment-step count so the schedule is
seed-stable and reproducible. Noise sampling is seeded (design spec §8:
"Gaussian noise seeding" is a unit test) so a run is exactly reproducible.

## Consequences

**Positive.**
- Satisfies the brief's mandatory Gaussian requirement and gives us the exact
  code lines to cite in the summary (design spec §1, §5).
- Decaying σ (`0.2 → 0.05`) gives broad early exploration then exploitation —
  directly supporting analysis Q2's experiment: kill σ early and the coverage
  map collapses to a narrow path (design spec §7).
- Per-component independent Gaussian on a 2-D action is trivial to seed and
  unit-test (`np.random.default_rng(seed).normal`), satisfying design spec §8's
  noise-seeding test.
- Clipping to `[−1, 1]` keeps the unicycle velocity mapping (ADR-002) valid for
  every collected action.

**Negative.**
- Independent per-step Gaussian is temporally *uncorrelated*, so in principle it
  explores inertial systems less efficiently than OU. For our first-order
  unicycle (no momentum) this is a non-issue, and the brief forbids OU anyway.
- The linear decay introduces two tuned values (`sigma_decay_steps`, `sigma_end`);
  both live in config and are justified in `docs/ANALYSIS.md`, not buried in code.

## Alternatives considered

| # | Alternative | Verdict | Why rejected |
|---|---|---|---|
| a | **Decaying zero-mean Gaussian, clipped, on collection only** | **Chosen** | Brief-mandated type; simple to seed/test; supports the analysis-Q2 ablation; bounds the action. |
| b | Ornstein-Uhlenbeck (OU) temporally-correlated noise | Rejected | Explicitly *not* what the brief mandates ("Gaussian, not OU"); design spec §5 fixes Gaussian. |
| c | Constant-σ Gaussian (no decay) | Rejected | Never settles into exploitation; degrades late-training coverage; can't show the clean σ-decay story in ANALYSIS Q2. |
| d | Parameter-space / NoisyNet exploration | Rejected | Over-engineered for this task, harder to seed deterministically, and not the brief-mandated mechanism. |
| e | ε-greedy / discretized actions | Rejected | Throws away the continuous-control rationale that motivated DDPG (design spec §7 Q1). |

## Review trigger

Re-open if: the brief's mandated noise family changes; the σ schedule needs to
become non-linear to converge; or analysis Q2 shows the σ-decay horizon
(`50000` steps) is mistuned relative to the `500`-episode × `1000`-step budget.
