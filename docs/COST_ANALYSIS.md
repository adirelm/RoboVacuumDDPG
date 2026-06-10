# COST_ANALYSIS — RoboVacuumDDPG (spec §11)

> Token counts are produced by `src/cost/meter.py` (tiktoken `cl100k_base`
> headline + chars/bytes appendix). Training-runtime and dollar numbers are
> **PENDING** until the seeded run + the AI-tooling tally complete — no
> invented spend (spec §10 honesty stance). The architect owns the spend cap
> (CLAUDE.md §1.4 cost-budget row).

## 1. Headline — tiktoken (cl100k_base)

Total prompt/response tokens across the build, counted by
`src/cost/meter.py` with the `cl100k_base` encoder.

| Channel | Tokens (tiktoken cl100k_base) |
|---|---|
| Architect → implementer prompts | PENDING |
| Implementer responses | PENDING |
| **Total** | PENDING |

## 2. Appendix — chars & bytes

Encoder-independent fallback (same corpus as §1), reported for reproducibility.

| Measure | Value |
|---|---|
| Characters | PENDING |
| Bytes (UTF-8) | PENDING |

## 3. AI-tooling cost

| Item | Unit | Qty | Cost |
|---|---|---|---|
| Claude Code session(s) | session | PENDING | PENDING |
| Token spend (from §1 × rate) | USD | PENDING | PENDING |
| **AI-tooling subtotal** | USD | — | PENDING |

## 4. Training runtime & compute envelope

`training.episodes = 500`, `training.seeds = [42, 7, 123, 314, 271]` (5 seeds),
`env.max_steps = 1000`. Compute is **CPU/laptop-class** (no GPU assumed); the
DDPG nets are small (`hidden_sizes = [256, 256]`).

| Quantity | Value |
|---|---|
| Episodes per seed | 500 |
| Seeds | 5 |
| Max steps / episode | 1000 |
| Wall-clock per seed | PENDING |
| Total wall-clock (5 seeds) | PENDING |
| Peak RSS | PENDING |
| Device | PENDING (CPU model) |

## 5. Cost envelope — architect spend cap vs running total

The architect set a spend cap (CLAUDE.md §1.4 *cost-budget envelope* row).

| | Amount |
|---|---|
| Architect-decided cap (USD) | PENDING |
| Running total (AI-tooling + compute) | PENDING |
| Headroom remaining | PENDING |

If the running total approaches the cap, training is time-boxed and any
partial convergence is reported honestly (spec §10), not masked.
