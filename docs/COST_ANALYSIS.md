# COST_ANALYSIS — RoboVacuumDDPG (spec §11)

> This project bills **no paid inference API**: the shipped artifact calls no
> model, so the only real "spend" is local CPU wall-clock (§4) plus the
> qualitative AI-tooling session envelope (§3/§5). `src/cost/meter.py` is a
> dependency-free **`RuntimeMeter`** (wall-clock timer + step/episode counters) —
> a unit-tested instrument *available* to the training loop, but it is **not
> currently wired into the multi-seed training driver**, so the §4 per-seed
> wall-clock figures are **estimates**, not meter readings. It is *not* a token
> counter, and we do not invent a dollar figure (spec §10 honesty stance). The
> architect↔implementer prompt volume is not separately token-metered (no paid
> API to bill); instead we report the concrete **artifact corpus size** (§1/§2)
> and the **(estimated) CPU runtime** (§4). The architect owns the spend cap
> (CLAUDE.md §1.4).

## 1. Headline — artifact corpus size (concrete)

The measurable "input volume" of the committed artifact, counted
encoder-independently from `git ls-files`. The `~tokens` column is a transparent
`chars / 4` estimate — no tiktoken dependency is shipped, so we do **not** claim a
specific BPE token count. Lines/chars/tokens are rounded to ≈3 significant
figures, measured at the **v1.0.0 release tree** (HEAD `925effd`) via `git ls-files`
over tracked `.py` (src + scripts / tests) and `.md` (docs recursive + README);
re-measure with the same command set if the tree moves.

| Corpus | Files | Lines | Chars | ~Tokens (chars/4) |
|---|---:|---:|---:|---:|
| `src/` + `scripts/` (.py) | 49 | ≈2,500 | ≈92.9k | ~23.2k |
| `tests/` (.py) | 58 | ≈2,850 | ≈100k | ~25.1k |
| `docs/` (.md, recursive) + README | 26 | ≈9,400 | ≈458k | ~115k |
| **Total artifact** | **133** | **≈14.8k** | **≈651k** | **~163k** |

No model is called during training or evaluation, so there is no per-call token
bill; the spend that matters for this assignment is the CPU wall-clock in §4.

## 2. Appendix — chars & bytes

Encoder-independent byte sizes of the same corpus (chars are in §1), for
reproducibility (same rounding/measurement note as §1). The corpus is ~99 % ASCII
(a few math glyphs like τ/γ/≈/→ add <1 %), so chars ≈ bytes at this rounding.

| Corpus | Chars | Bytes (UTF-8) |
|---|---:|---:|
| `src/` + `scripts/` (.py) | ≈92.9k | ≈92.9k |
| `tests/` (.py) | ≈100k | ≈100k |
| `docs/` (.md, recursive) + README | ≈458k | ≈458k |
| **Total** | **≈651k** | **≈651k** |

## 3. AI-tooling cost

The AI-tooling envelope is **qualitative** — the build used Claude Code
sessions, but **no paid inference API was billed for the artifact itself** (the
shipped code calls no model). We do not invent a dollar figure (spec §10).

| Item | Unit | Qty | Cost |
|---|---|---|---|
| Claude Code session(s) | session | multi-session build | subscription tooling (no per-call API bill) |
| Token spend (from §1 × rate) | USD | — | n/a — no paid API in the artifact |
| **AI-tooling subtotal** | USD | — | qualitative (no metered spend) |

**Development-cost dimension (qualitative — not instrumented).** For a Vibe-Coding
project the larger cost is *development effort*, not runtime. We did **not**
instrument dev-hours or token usage, so we report this honestly rather than
inventing a figure: the project was built across multiple Claude Code sessions on
a flat subscription (no metered per-call API bill), under the architect↔implementer
loop (CLAUDE.md §1.4). An **AI-rework tax** genuinely exists — a real share of
effort went to *verifying and redoing* AI output (e.g. a multi-model review that
caught and corrected a held-out-evaluation defect, and that reverted one
false-positive fix) — and it is acknowledged here, not silently counted as zero.
The ≤150-LOC / ≥85%-coverage / zero-Ruff gates acted as a verification-cost
discipline that bounded that rework. A precise dev-hours/token number is the
architect's to record on the cover sheet (§1.4).

## 4. Training runtime & compute envelope

`training.episodes = 500`, `training.seeds = [42, 7, 123, 314, 271]` (5 seeds),
`env.max_steps = 1000`. Compute is **CPU/laptop-class** (no GPU assumed); the
DDPG nets are small (`hidden_sizes = [256, 256]`).

| Quantity | Value |
|---|---|
| Episodes per seed | 500 |
| Seeds | 5 (42, 7, 123, 314, 271) |
| Max steps / episode | 1000 (≈ 500k env steps / seed) |
| Wall-clock per seed | ~40–80 min (CPython, torch **CPU**) — **estimated, not metered** |
| Total wall-clock (5 seeds) | ≈ 4 h (sequential, one process) — **estimated** |
| Peak RSS | laptop-class, replay buffer (`buffer_size = 1e6`) dominates — fits in commodity RAM |
| Device | CPU only — no GPU used or required (`hidden_sizes = [256, 256]`) |

Runtime is dominated by the **DDPG gradient update** (`agent.update`: the
critic + actor backward pass, ≈ 4 ms/step measured), which runs on ~99.8% of
steps (warmup is only 1000 of ~500k). The simulator `env.step` — whose per-step
hotspot is the `n_rays = 16` per-segment lidar raycasting in
`src/env/raycast.py` — is ≈ 0.1–0.3 ms/step (~14–43× cheaper), so it does **not**
dominate wall-clock. The 5 seeds are independent and were run **sequentially for
reproducibility** (see §15 in `docs/QUALITY.md`).

> **Caveat (honesty, spec §10):** the per-seed/total wall-clock above is an
> **estimate** from observed runs, **not** a `RuntimeMeter` reading — the meter
> (`src/cost/meter.py`) is implemented and unit-tested but is not yet wired into
> `scripts/train.py` / the multi-seed driver, and no timing field is persisted in
> `results/history/*.json`. Treat ≈4 h as order-of-magnitude, not measured.

**Where it stops being laptop-cheap (scaling threshold).** Cost scales roughly
**linearly** in the episode×seed budget — doubling seeds or episodes ≈ doubles the
≈4 h. The laptop-CPU envelope holds only while (a) the nets stay tiny
(`[256,256]`, the ≈4 ms gradient step is the bottleneck) and (b) maps stay small.
It stops being cheap when larger maps + higher `n_rays` push `env.step` up, or
multi-map training multiplies the run. A GPU buys little there (the small nets
underuse it); the cheap win is parallelizing the **embarrassingly-parallel seeds**
across processes (one per seed), cutting the ≈4 h to ≈ one seed's wall-clock
(`docs/QUALITY.md` §15).

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
(−175.5 tail reward) and the partial held-out generalization (≈0.39 on the training map → 0.10–0.18 on
unseen maps, at a real collision cost) are
reported honestly (spec §10), not masked by extending the run until the numbers
looked uniform — see `docs/ANALYSIS.md`.
