# ADR-008 — Multi-Seed Evaluation + Held-Out Generalization

| Field | Value |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-06-10 |
| **Deciders** | Architect (human, CLAUDE.md §1.4) + AI (implementer) |
| **Supersedes** | — |
| **Related** | ADR-005 (held-out maps), ADR-007 (training hyperparameters), design spec §6, §7, §8 |

## Context

A single DDPG run is not evidence — DDPG is notoriously seed-sensitive, and one
lucky/unlucky seed can paint a misleading learning curve. The deliverables
(design spec §7) require a `learning_curve.png` showing **cumulative reward vs
episode, mean ± CI over seeds**, which only makes sense with a fixed, plural seed
set. Separately, the brief/spec require a **generalization** evaluation on maps
the agent never trained on (design spec §6: hold out 1–2 plans). This ADR fixes
(a) the canonical seed set and how every headline number is reported, and (b) the
train/held-out protocol that makes the generalization claim honest.

The A3/A4 lineage already established that 3 seeds give only *directional*
results; the design spec config commits to a five-seed set, which we adopt
verbatim.

## Decision

**Canonical seeds — from `config/config.yaml#training.seeds` (CLAUDE.md §4,
single source):**

```
seeds = [42, 7, 123, 314, 271]      # |seeds| = 5
```

Every headline number — the `learning_curve.png` band, the held-out coverage %,
any table in `docs/ANALYSIS.md` — is reported as **mean ± 95% CI over these five
seeds**. Exploratory runs may use any seed; they just cannot be cited.

**Training schedule — from `config/config.yaml#training`:**

| Knob | Config key | Value |
|---|---|---|
| episodes | `training.episodes` | `500` |
| seeds | `training.seeds` | `[42, 7, 123, 314, 271]` |

> **Simplification (implemented).** No in-loop periodic-eval cadence was wired up:
> `Trainer.train` returns a full per-episode history (reward, coverage, critic
> loss) and final greedy evaluation runs through `RoboVacuumSDK.evaluate`, so the
> earlier `training.eval_every` knob was dropped from config rather than left dead.

**Held-out generalization (design spec §6, maps from ADR-005).** The agent
trains on **`room_single` only** — the canonical `train()` driver uses the first
`maps.train` entry (`maps.train[0]`); `apt_small`/`apt_multi` are configured but
reserved for future multi-map training (see `docs/ANALYSIS.md`). It is evaluated
on **`maps.holdout = ["apt_large", "office"]`**, which are never seen during
training. The held-out coverage % (mean ± 95% CI over the five seeds, computed by
`scripts/evaluate.py`, which loads each trained checkpoint and fails loudly if one
is missing) is the generalization headline.

**Evaluation is deterministic.** During eval rollouts the Gaussian exploration
noise (ADR-003) is **off** — the policy acts greedily (`a = μ(s)`, clipped) — so
the curve reflects the learned policy, not exploration. The two required figures
are produced by `render_learning_curve.py` (cumulative reward vs episode, mean ±
CI) and `render_critic_loss.py` (critic loss vs training step), per design spec
§7. The `render_trajectory.py` plot is rendered on the **training** map
`room_single` (a qualitative view of the trained policy's coverage sweep);
generalization is reported **numerically** via the held-out coverage table in
`docs/ANALYSIS.md`, not via a held-out trajectory figure.

**Failure discipline.** If a canonical seed diverges or NaN-crashes, we **do not
silently drop it** — the run is re-launched with a logged justification, and the
honest-reporting stance from design spec §10 applies (if convergence is partial,
say so).

## Consequences

**Positive.**
- Five seeds with mean ± 95% CI turn the learning curve into defensible evidence
  rather than an anecdote; the CI band is exactly what `learning_curve.png`
  requires (design spec §7).
- A strict never-trained-on held-out set (`apt_large`, `office`) makes the
  generalization claim real — the held-out coverage table (mean ± 95% CI over the
  five seeds) is the quantitative evidence.
- Seeds and schedule live in config (single source), so a grader reproduces the
  exact reported runs from a fresh checkout.
- Deterministic (noise-off) eval cleanly separates "what the policy learned" from
  "what exploration stumbled into," matching ADR-003's collection-only noise.

**Negative.**
- Five seeds × 500 episodes × 1000 steps is the dominant compute cost of the
  project. Accepted; the trustworthiness of the headline numbers is downstream of
  it, and episodes are time-boxed with honest partial-convergence reporting
  (design spec §10).
- A 2-plan held-out set is a small generalization sample; we report it as
  "generalizes to HouseExpo-like unseen apartments," not a universal claim
  (threat to validity noted in `docs/ANALYSIS.md`).
- Re-running a failed seed (rather than dropping it) costs extra compute, but
  preserves the integrity of the |seeds| = 5 reporting contract.

## Alternatives considered

| # | Alternative | Verdict | Why rejected |
|---|---|---|---|
| a | **5 fixed seeds `[42,7,123,314,271]`, mean ± 95% CI, strict held-out maps** | **Chosen** | Plural enough for an honest CI band; reproducible from config; held-out split makes generalization real. |
| b | Single seed (one run) | Rejected | No CI possible; DDPG seed-variance makes one run misleading; can't produce the required mean±CI learning curve. |
| c | 3 seeds | Rejected | A3/A4 experience showed 3 seeds give only directional results; CI too wide to support headline claims. |
| d | Train and evaluate on the same maps | Rejected | No generalization claim possible; defeats design spec §6's held-out requirement. |
| e | Silently drop a diverging seed to clean up the curve | Rejected | Dishonest; breaks the |seeds| = 5 contract. We re-run with a logged reason instead. |
| f | 10+ seeds | Rejected | Marginal CI tightening for ~2× the dominant compute cost; over-spend against the §1.4 cost-budget envelope. |

## Review trigger

Re-open if: a canonical seed repeatedly diverges (revisit ADR-007 τ/lr or the
seed set); the held-out maps need to change (coordinate with ADR-005); or the
episode/eval schedule is retuned. Record any deviation and the eventual numbers
in `docs/ANALYSIS.md`.
