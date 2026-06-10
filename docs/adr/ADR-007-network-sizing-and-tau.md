# ADR-007 — Network Sizing + Soft-Update τ

| Field | Value |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-06-10 |
| **Deciders** | Architect (human, CLAUDE.md §1.4) + AI (implementer) |
| **Supersedes** | — |
| **Related** | ADR-002 (action dim), ADR-003 (exploration), ADR-008 (multi-seed eval), design spec §3, §5, §7 |

## Context

DDPG is two networks — a deterministic **Actor** `μ(s)` and a **Critic** `Q(s,a)`
— each shadowed by a slowly-tracking **target** network. The brief's central
code-requirement checklist (design spec §5) calls out exactly these four things
the summary must cite by line number: Actor-Critic structure, **soft target
updates (Polyak)**, the hyperparameters, and Gaussian noise (ADR-003). This ADR
fixes the **network sizes**, the **learning rates**, the **discount γ**, and the
**Polyak coefficient τ**, and explains why each value (all from config) is what
it is. Design spec §7 analysis Q3 specifically asks us to explain how target
networks + soft updates prevent critic collapse — so τ is not just a number, it's
the centerpiece of a graded analysis question.

The state is 20-dim (design spec §3: 16 lidar rays + `(v, ω)` + a heading cue to
the nearest uncleaned cell), and the action is 2-dim `[throttle, steer]`
(ADR-002). The networks must map those dimensions and the actor must be
Tanh-bounded to `[−1, 1]²`.

## Decision

**Architecture.**

- **Actor** (`src/model/actor.py`): MLP `state → hidden → hidden → 2`, final
  activation **Tanh** so the action is exactly `[−1, 1]²` (design spec §3, §5).
- **Critic** (`src/model/critic.py`): MLP taking **state ⊕ action** →
  `hidden → hidden → 1` scalar Q-value (design spec §4, §5: the critic takes
  state AND action).
- Both have target twins; **Polyak soft-update** in `src/ddpg/agent.py`:

```
θ_target ← τ · θ + (1 − τ) · θ_target      # for BOTH actor and critic targets
```

**Hyperparameters — all from `config/config.yaml#ddpg` (CLAUDE.md §4):**

| Symbol | Config key | Value | Justification |
|---|---|---|---|
| hidden sizes | `ddpg.hidden_sizes` | `[256, 256]` | Standard DDPG two-layer width; ample capacity for a 20-dim state without overfitting. |
| `lr_actor` | `ddpg.lr_actor` | `1.0e-4` | Actor steps **slower** than critic — standard DDPG, keeps the policy from chasing a noisy Q. |
| `lr_critic` | `ddpg.lr_critic` | `1.0e-3` | Critic steps **faster** (10×) so the value estimate leads the policy. |
| `γ` (gamma) | `ddpg.gamma` | `0.99` | Long horizon — coverage credit must propagate across a 1000-step episode. |
| `τ` (tau) | `ddpg.tau` | `0.005` | Brief's example value; slow target tracking ⇒ stable TD targets (analysis Q3). |
| batch size | `ddpg.batch_size` | `128` | Standard minibatch for off-policy updates from replay. |
| buffer size | `ddpg.buffer_size` | `1000000` | Large uniform replay (`src/ddpg/replay_buffer.py`) for decorrelated samples. |
| grad clip | `ddpg.grad_clip` | `1.0` | Gradient-norm clip for stability (design spec §10). |
| warmup | `ddpg.warmup_steps` | `1000` | Random actions before learning starts (seeds the buffer; see ADR-003). |

**Why τ = 0.005 matters (analysis Q3).** The TD target uses the *target*
networks: `y = r + γ · Q_target(s', μ_target(s'))`. Because τ is tiny, the
target tracks the online critic only ~0.5% per update, so `y` changes slowly —
the critic is not chasing a moving target it just created. This is exactly the
mechanism that prevents the critic collapse / divergence that a hard target copy
(or no target net) would cause. The Polyak math is unit-tested (design spec §8:
"Polyak math" is a tested expectation) and an architecture test asserts the
update *is* Polyak.

## Consequences

**Positive.**
- `[256, 256]` Actor and Critic are the well-trodden DDPG sizes; we inherit a
  large body of working precedent rather than tuning width from scratch.
- The `1e-4` actor / `1e-3` critic split (critic 10× faster) is the standard
  DDPG asymmetry that keeps the value estimate ahead of the policy.
- `τ = 0.005` (the brief's example) gives the slow, stable target tracking that
  ADR-008's multi-seed runs need to be comparable, and it is the concrete answer
  to analysis Q3.
- Every value is in config, so a grader can run an ablation (e.g. τ=0.01, hidden
  `[128,128]`) without touching source (CLAUDE.md §4); the sensor-resolution
  ablation (8/16/24 rays, design spec §3) likewise just changes the actor input
  width via `env.n_rays`.

**Negative.**
- Two `256×256` MLPs plus their targets are more parameters than a coverage task
  strictly needs; the per-step forward cost is modest on CPU but is the agent's
  main compute term. Bounded by `grad_clip` and the small action/state dims.
- Small τ means *slow* target adaptation — early training is more cautious than
  with a larger τ. Accepted: stability is the priority the brief's analysis Q3
  rewards, and the learning curve (design spec §7) reports convergence honestly.
- Hidden width and τ are coupled tuning knobs; if a seed diverges we adjust in
  config and log it (per ADR-008 failure discipline), not in code.

## Alternatives considered

| # | Alternative | Verdict | Why rejected |
|---|---|---|---|
| a | **`[256,256]` Actor/Critic, lr 1e-4/1e-3, γ=0.99, τ=0.005 (config-driven)** | **Chosen** | Standard DDPG sizing; brief's τ example; answers analysis Q3; fully config-tunable. |
| b | Hard target update (periodic full copy, DQN-style) | Rejected | Defeats the brief's *soft*-update (Polyak) requirement (design spec §5); abrupt target jumps are exactly what destabilizes the critic (analysis Q3). |
| c | No target networks (bootstrap off the online critic) | Rejected | Known to diverge — the failure mode analysis Q3 is meant to explain; brief mandates soft target updates. |
| d | Larger nets (`[400,300]` original-paper sizes) | Rejected | More parameters/compute for no expected gain on a 20-dim state; `[256,256]` is the modern default. Held as an ablation only. |
| e | Equal actor/critic learning rates | Rejected | Loses the critic-leads-actor asymmetry that stabilizes DDPG; the 10× split is standard practice. |

## Review trigger

Re-open if: a canonical seed (ADR-008) diverges and a τ / lr / width retune is
needed; the sensor-resolution ablation (`env.n_rays` 8/16/24) shifts the actor
input width enough to warrant resizing; or design spec §5's hyperparameter list
changes. Log any retune in `docs/ANALYSIS.md`.
