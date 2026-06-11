# COST_ANALYSIS — RoboVacuumDDPG (spec §11)

> This project bills **no paid inference API**: the shipped artifact calls no
> model, so the only real "spend" is local CPU wall-clock (§4) plus the
> qualitative AI-tooling session envelope (§3/§5). `src/cost/meter.py` is a
> dependency-free **`RuntimeMeter`** (wall-clock timer + step/episode counters)
> the training loop polls to keep the runtime cost honest — it is *not* a token
> counter, and we do not invent a dollar figure (spec §10 honesty stance). The
> architect↔implementer prompt volume is not separately token-metered (no paid
> API to bill); instead we report the concrete **artifact corpus size** (§1/§2)
> and the **CPU runtime** (§4). The architect owns the spend cap (CLAUDE.md §1.4).

## 1. Headline — artifact corpus size (concrete)

The measurable "input volume" of the committed artifact, counted
encoder-independently from `git ls-files`. The `~tokens` column is a transparent
`chars / 4` estimate — no tiktoken dependency is shipped, so we do **not** claim a
specific BPE token count.

| Corpus | Files | Lines | Chars | ~Tokens (chars/4) |
|---|---:|---:|---:|---:|
| `src/` + `scripts/` (.py) | 37 | 1,718 | 60,974 | ~15.2k |
| `tests/` (.py) | 48 | 2,260 | 77,623 | ~19.4k |
| `docs/` (.md) + README | 24 | 8,989 | 424,820 | ~106.2k |
| **Total artifact** | **109** | **12,967** | **563,417** | **~140.9k** |

No model is called during training or evaluation, so there is no per-call token
bill; the spend that matters for this assignment is the CPU wall-clock in §4.

## 2. Appendix — chars & bytes

Encoder-independent byte sizes of the same corpus (chars are in §1), for
reproducibility.

| Corpus | Chars | Bytes (UTF-8) |
|---|---:|---:|
| `src/` + `scripts/` (.py) | 60,974 | 61,030 |
| `tests/` (.py) | 77,623 | 77,704 |
| `docs/` (.md) + README | 424,820 | 429,947 |
| **Total** | **563,417** | **568,681** |

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
