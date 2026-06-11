# PROMPTS — RoboVacuumDDPG (architect → implementer trail)

> Evidence for the **Human ↔ AI Responsibility Contract** (CLAUDE.md §1.4): the
> developer is the architect (decides PRD/architecture/acceptance/sign-off), the
> AI is the implementer (code against an approved spec). Each row records the
> verbatim prompt, the representative commits it produced, and the human-judgment
> call attached (matching the per-section discipline of the A1 PROMPTS log). The
> full commit trail is `git log --oneline`; the SHAs below are each phase's
> landing commits.

## How to read this log
- **Prompt** — the literal instruction given to the implementer.
- **Commit** — the resulting commit hash (subject `^(Phase \d+|...)`).
- **Human-judgment annotation** — the architect-decided, non-delegable call
  (CLAUDE.md §1.4 table) that gated or shaped the prompt.

## Phase 0 — Bootstrap
| Prompt | Commit | Human-judgment annotation |
|---|---|---|
| "Stand up the V3 scaffold (uv, ruff, pytest-cov fail_under=85, ≤150-LOC guard, CI, docs/ + ADR stubs)." | `be42a8a`→`e4aa521` (config loader `894a4c7`, size guard `a4c5642`, CI `4ee1679`) | Architect chose the gate thresholds and module boundaries before any code. |

## Phase 1 — Simulator from scratch
| Prompt | Commit | Human-judgment annotation |
|---|---|---|
| "Implement env units (house_map, raycast, kinematics, coverage, collision, reward, state, vacuum_env) TDD with hand-computed expectations." | `8c6a8f0`→`c28f3f1` (real HouseExpo schema `c7d2646`) | Architect fixed the MDP (state 20-dim, reward signs, 4-tuple, no Gym). |

## Phase 2 — DDPG from scratch
| Prompt | Commit | Human-judgment annotation |
|---|---|---|
| "Implement Actor (Tanh), Critic (state⊕action), ReplayBuffer, Gaussian noise, DDPGAgent (Polyak soft-update), Trainer — TDD." | `5fe5053`→`60c5e78`; Trainer `340aeea`; sigma-decay fix `79080b8` | Architect chose Gaussian-not-OU (ADR-003), τ=0.005, LR split (ADR-007). |

## Phase 3 — Training + Results
| Prompt | Commit | Human-judgment annotation |
|---|---|---|
| "Fetch HouseExpo at pinned SHA, train 5 seeds, render learning_curve / critic_loss / trajectory, held-out generalization." | SDK `0bffe2c`, fetch `9382d02`/`40cab6f`, renders `41e2a7f`→`810f028`; real 5-seed run `bd8db36` | Architect set seeds, episode budget, and the held-out split (ADR-008). |

## Phase 4 — Docs + Analysis + Gates
| Prompt | Commit | Human-judgment annotation |
|---|---|---|
| "Author architecture tests, ANALYSIS (3 questions), COST_ANALYSIS, QUALITY (ISO 25010), UX (§10 N/A), README, cover sheet; run final gates; tag v1.0.0." | tests `4675b79`/`866bb75`, docs `99f3900`→`a4f7a41`; finalized `c2bddb2`/`4ef7571`; sensitivity `9444baa`; CI PII-skip `a65bc6b` | Architect signs off the self-grade (cover sheet only) and the submission. |
