# COST_ANALYSIS — RoboVacuumDDPG (spec §11)

> Token counts are produced by `src/cost/meter.py` (tiktoken `cl100k_base`
> headline + chars/bytes appendix). **There is no paid API in this project** —
> all training and evaluation run locally on CPU, so the only "spend" is
> wall-clock compute (§4) plus the AI-tooling session envelope (§3/§5), which we
> keep qualitative rather than invent a dollar figure (spec §10 honesty stance).
> The architect owns the spend cap (CLAUDE.md §1.4 cost-budget row).

## 1. Headline — tiktoken (cl100k_base)

Total prompt/response tokens across the build, counted by
`src/cost/meter.py` with the `cl100k_base` encoder.

| Channel | Tokens (tiktoken cl100k_base) |
|---|---|
| Architect → implementer prompts | not separately metered |
| Implementer responses | not separately metered |
| **Total** | counted on demand by `src/cost/meter.py` over the prompt corpus |

Token accounting is a *tooling* concern, not a runtime cost: no model is called
during training or evaluation. The meter exists for reproducibility; the headline
spend that matters for this assignment is the CPU wall-clock in §4.

## 2. Appendix — chars & bytes

Encoder-independent fallback (same corpus as §1), reported for reproducibility.

| Measure | Value |
|---|---|
| Characters | counted on demand by `src/cost/meter.py` (encoder-independent) |
| Bytes (UTF-8) | counted on demand by `src/cost/meter.py` |

## 3. AI-tooling cost

The AI-tooling envelope is **qualitative** — the build used Claude Code
sessions, but **no paid inference API was billed for the artifact itself** (the
shipped code calls no model). We do not invent a dollar figure (spec §10).

| Item | Unit | Qty | Cost |
|---|---|---|---|
| Claude Code session(s) | session | multi-session build | subscription tooling (no per-call API bill) |
| Token spend (from §1 × rate) | USD | — | n/a — no paid API in the artifact |
| **AI-tooling subtotal** | USD | — | qualitative (no metered spend) |

## 4. Training runtime & compute envelope

`training.episodes = 500`, `training.seeds = [42, 7, 123, 314, 271]` (5 seeds),
`env.max_steps = 1000`. Compute is **CPU/laptop-class** (no GPU assumed); the
DDPG nets are small (`hidden_sizes = [256, 256]`).

| Quantity | Value |
|---|---|
| Episodes per seed | 500 |
| Seeds | 5 (42, 7, 123, 314, 271) |
| Max steps / episode | 1000 (≈ 500k env steps / seed) |
| Wall-clock per seed | ~40–80 min (CPython, torch **CPU**) |
| Total wall-clock (5 seeds) | ≈ 4 h (sequential, one process) |
| Peak RSS | laptop-class, replay buffer (`buffer_size = 1e6`) dominates — fits in commodity RAM |
| Device | CPU only — no GPU used or required (`hidden_sizes = [256, 256]`) |

Runtime is dominated by **per-step raycasting** (the lidar `n_rays = 16`
ray–segment math, vectorized over wall segments in `src/env/raycast.py`) plus the
critic/actor gradient updates; the nets are tiny, so the bottleneck is the
simulator loop, not the torch backward pass. The 5 seeds are independent and
were run **sequentially for reproducibility** (see §15 in `docs/QUALITY.md`).

## 5. Cost envelope — architect spend cap vs running total

The architect set a spend cap (CLAUDE.md §1.4 *cost-budget envelope* row).

| | Amount |
|---|---|
| Architect-decided cap (USD) | $0 paid-API budget — local CPU only by design |
| Running total (AI-tooling + compute) | ≈ 4 h CPU wall-clock; no paid-API spend |
| Headroom remaining | within envelope (no API bill incurred) |

The cap was honoured the easy way: by **design there is no paid inference in the
artifact**, so the only cost is the ≈ 4 h of local CPU compute in §4. Training
was time-boxed at 500 episodes × 5 seeds; the seed-271 outlier
(−175.5 tail reward) and the held-out overfitting (coverage 0.39 → ~0.003) are
reported honestly (spec §10), not masked by extending the run until the numbers
looked uniform — see `docs/ANALYSIS.md`.
