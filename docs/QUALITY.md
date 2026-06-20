# QUALITY — RoboVacuumDDPG (ISO/IEC 25010 product-quality model)

> The eight **ISO/IEC 25010** product-quality characteristics, each with a
> concrete evidence pointer into this repo and an honest bound. Gates:
> `fail_under=85` coverage, every `.py` ≤ 150 LOC, zero ruff violations.

## 1. Functional Suitability
The simulator emits a custom `(state, reward, done, info)` 4-tuple and the
DDPG agent learns continuous coverage control. Evidence: `src/env/vacuum_env.py`,
`src/ddpg/agent.py`; hand-computed unit tests in `tests/` (kinematics K-1..K-4,
Polyak math, Tanh bounds, critic shape).

## 2. Performance Efficiency
Small MLPs (`hidden_sizes = [256, 256]`), per-segment ray–segment math
(`src/env/raycast.py`), coarse coverage grid (`coverage_cell = 0.10`). Runtime
envelope tracked in `docs/COST_ANALYSIS.md` §4 (~40–80 min/seed CPU, ≈ 4 h for
all 5 seeds).

**§15 — Parallelism (honest framing).** Training is **single-process, CPU-only**.
The dominant per-step training cost is the **DDPG gradient update**
(`agent.update`: critic + actor backward, ≈ 4 ms/step measured), which runs on
~99.8% of steps (warmup is only 1000 of ~500k). The simulator `env.step`
(≈ 0.1–0.3 ms/step) is ~14–43× cheaper; within it the per-segment lidar
raycasting (`src/env/raycast.py`) is the hotspot, but it does not dominate
wall-clock. The five seeds `[42, 7, 123, 314, 271]` are **embarrassingly parallel** —
each is an independent run writing its own `results/history/seed_*.json`, sharing
no state — and could trivially be fanned out across processes (e.g.
`multiprocessing` / one process per seed) to cut the ≈ 4 h wall-clock to roughly
one seed's time. We deliberately left them **sequential** for reproducibility and
simplicity: a single deterministic process per seed makes the seeded results easy
to reproduce and audit, and the ≈ 4 h budget did not justify the added
orchestration. Within a single episode the step loop is inherently sequential
(each pose depends on the last), so the realistic parallelism axis is *across
seeds*, not within a run.

## 3. Compatibility
Pure Python + PyTorch/NumPy/Matplotlib; no Gymnasium/Gazebo/SB3 coupling
(architecture test `tests/architecture/test_no_gymnasium_import.py`). Reads the
upstream HouseExpo JSON format unchanged (`src/env/house_map.py`).

## 4. Usability
Single entry point `RoboVacuumSDK` (`src/sdk/sdk.py`); a Pygame live viewer
(`scripts/play.py` — train/play/drive) plus the batch CLI + static figures
(see `docs/UX.md`, which maps all 10 Nielsen heuristics). One-command install
(`uv sync --dev`) and run.

## 5. Reliability
Seeded determinism across 5 seeds `[42, 7, 123, 314, 271]`; gradient-norm clip
`grad_clip = 1.0`; target networks + Polyak soft updates prevent critic
collapse (`docs/ANALYSIS.md` Q3). Collision revert keeps the robot inside walls
(`src/env/collision.py`).

## 6. Security
No secrets in source; `.env-example` documents required env without values; no
PII in committed files (final-gate grep). No network calls at train/eval time
(dataset fetched offline by `scripts/fetch_houseexpo.py` at a pinned SHA).

## 7. Maintainability
Every `.py` ≤ **150** LOC (`scripts/check_file_sizes.py`); zero **ruff**
violations; ≥85% coverage (`fail_under=85`); strict module boundaries
(env / model / ddpg / services / sdk); ADR-001..008 record every architecture
decision.

## 8. Portability
`uv`-only, `requires-python >= 3.11`, CPU-class compute (no GPU required);
config-driven (`config/config.yaml`) so the `n_rays` ablation (8/16/24) and all
hyperparameters change without editing source.

## 9. Honest limitations
- **Single-map training generalizes only partially.** Trained on `room_single`
  only, the policy reaches ≈ 0.39 coverage / 691 ± 443 reward there and transfers
  partially to the held-out maps — `apt_large` 10.4% ± 5.7%, `office` 17.9% ± 7.9%
  coverage (mean ± 95% CI over the five seeds) — but at a real collision cost
  (≈ 162 / ≈ 508 bumps per episode), so it has **not** learned transferable
  wall-avoidance; the map-agnostic state carries no global map identity. See
  `docs/ANALYSIS.md`.
- **Seed sensitivity.** Seed 271 did not lock in (−175.5 tail reward vs 760–1052
  for the other four), widening the across-seed σ; reported, not dropped.
- Raycasting cost grows with map complexity; mitigated by capped ray count and
  a coarse coverage grid, but very large held-out plans may slow eval.
- Security scope is "no secrets / no PII / offline" — there is no adversarial
  threat model; this is a research artifact, not a deployed service.
