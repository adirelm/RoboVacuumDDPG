# PROMPTS — RoboVacuumDDPG (architect → implementer trail)

> Evidence for the **Human ↔ AI Responsibility Contract** (CLAUDE.md §1.4): the
> developer is the architect (decides PRD/architecture/acceptance/sign-off), the
> AI is the implementer (code against an approved spec). Each row records the
> verbatim prompt, the representative commits it produced, and the human-judgment
> call attached (matching the per-section discipline of the A1 PROMPTS log). The
> full commit trail is `git log --oneline`; the SHAs below are each phase's
> landing commits, recorded against the repository's canonical (PII-scrubbed)
> history — every SHA resolves in a fresh clone, and an architecture test
> asserts it stays that way.

## How to read this log
- **Prompt** — the literal instruction given to the implementer.
- **Commit** — the resulting commit hash (subject `^(Phase \d+|...)`).
- **Human-judgment annotation** — the architect-decided, non-delegable call
  (CLAUDE.md §1.4 table) that gated or shaped the prompt.

## Phase 0 — Bootstrap
| Prompt | Commit | Human-judgment annotation |
|---|---|---|
| "Stand up the V3 scaffold (uv, ruff, pytest-cov fail_under=85, ≤150-LOC guard, CI, docs/ + ADR stubs)." | `2b77b5b`→`3032504` (config loader `18b8ff4`, size guard `4b981cc`, CI `88ac398`) | Architect chose the gate thresholds and module boundaries before any code. |

## Phase 1 — Simulator from scratch
| Prompt | Commit | Human-judgment annotation |
|---|---|---|
| "Implement env units (house_map, raycast, kinematics, coverage, collision, reward, state, vacuum_env) TDD with hand-computed expectations." | `03b3694`→`dea91f8` (real HouseExpo schema `ce8261c`) | Architect fixed the MDP (state 20-dim, reward signs, 4-tuple, no Gym). |

## Phase 2 — DDPG from scratch
| Prompt | Commit | Human-judgment annotation |
|---|---|---|
| "Implement Actor (Tanh), Critic (state⊕action), ReplayBuffer, Gaussian noise, DDPGAgent (Polyak soft-update), Trainer — TDD." | `b1d7beb`→`b74cc73`; Trainer `aedfea7`; sigma-decay fix `179ff5b` | Architect chose Gaussian-not-OU (ADR-003), τ=0.005, LR split (ADR-007). |

## Phase 3 — Training + Results
| Prompt | Commit | Human-judgment annotation |
|---|---|---|
| "Fetch HouseExpo at pinned SHA, train 5 seeds, render learning_curve / critic_loss / trajectory, held-out generalization." | SDK `1579d9a`, fetch `ae86e91`/`ee8ed86`, renders `a8d3d4c`→`d92c565`; real 5-seed run `29a8d80` | Architect set seeds, episode budget, and the held-out split (ADR-008). |

## Phase 4 — Docs + Analysis + Gates
| Prompt | Commit | Human-judgment annotation |
|---|---|---|
| "Author architecture tests, ANALYSIS (3 questions), COST_ANALYSIS, QUALITY (ISO 25010), UX (§10 N/A), README, cover sheet; run final gates; tag v1.0.0." | tests `6cab8f1`/`1b2aa6a`, docs `1b8eed1`→`19edbdd`; finalized `ecade36`/`477646d`; sensitivity `99e1142`; CI PII-skip `4e752dc` | Architect signs off the self-grade (cover sheet only) and the submission. |
