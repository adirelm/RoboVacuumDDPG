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
Small MLPs (`hidden_sizes = [256, 256]`), vectorized ray–segment math
(`src/env/raycast.py`), coarse coverage grid (`coverage_cell = 0.10`). Runtime
envelope tracked in `docs/COST_ANALYSIS.md` §4.

## 3. Compatibility
Pure Python + PyTorch/NumPy/Matplotlib; no Gymnasium/Gazebo/SB3 coupling
(architecture test `tests/architecture/test_no_gymnasium_import.py`). Reads the
upstream HouseExpo JSON format unchanged (`src/env/house_map.py`).

## 4. Usability
Single entry point `RoboVacuumSDK` (`src/sdk/sdk.py`); CLI + figures, no GUI
(see `docs/UX.md`). One-command install (`uv sync --dev`) and run.

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
- Convergence numbers are **PENDING** until the seeded run completes; partial
  convergence will be reported honestly (spec §10), not masked.
- Raycasting cost grows with map complexity; mitigated by capped ray count and
  a coarse coverage grid, but very large held-out plans may slow eval.
- Security scope is "no secrets / no PII / offline" — there is no adversarial
  threat model; this is a research artifact, not a deployed service.
