# CLAUDE.md — Global Coding Standards for RoboVacuumDDPG (Assignment 5)

## Project Context

RoboVacuumDDPG is a reinforcement-learning project for the Bar-Ilan *Vibe Coding
& RL* workshop (Lecture 09, DDPG). It builds — **from scratch** — a 2D robotic-
vacuum simulator that reads real **HouseExpo** floor-plans, and a **DDPG** agent
(actor-critic, Polyak soft-target updates, Gaussian exploration noise, replay
buffer) that learns continuous navigation + area coverage. Single source of
truth: `docs/superpowers/specs/2026-06-10-robovacuum-ddpg-design.md`.

## Human ↔ AI Responsibility Contract (§1.4)

The developer is **architect** (non-delegable: requirements/PRD, architecture/ADRs,
test acceptance criteria, final review + commit intent, self-grade, cost budget);
the AI is **implementer** (code against an approved spec, refactor within an API,
test scaffolding from a spec, docstring drafts, lint/format fixes, doc freshness).
Any AI change that alters a human-decided concern needs explicit sign-off first.

## Hard Constraints (apply to ALL files)

1. **File size ≤150 LOC** per `.py` (excl. blanks/comments), **tests included**.
   Split into modules before hitting the cap; never compress to dodge it.
2. **TDD** — RED→GREEN→REFACTOR. Tests before implementation. **≥85% coverage**
   (`fail_under=85`). Run: `uv run pytest tests/ --cov=src --cov-report=term-missing`.
3. **OOP / SDK single entry** — all business logic reachable via `RoboVacuumSDK`;
   CLI / notebook / scripts import only the SDK (no logic in UIs). No duplication (DRY).
4. **No hardcoded values** — every algorithm-relevant parameter (DDPG hyperparams,
   noise, env/sim constants, reward weights, seeds, map lists) lives in
   `config/config.yaml` via the config loader. Local UI/plot styling literals stay local.
5. **Zero Ruff violations** — `uv run ruff check src/ tests/ scripts/`.
6. **uv only** — `uv sync --dev`, `uv run …`. No pip / conda / venv / requirements.txt.

## Algorithm Requirements (brief Lecture-09 / ex05)

- **Simulator from scratch** — **NO Gymnasium, NO Gazebo** (hard ban; AST test
  forbids `gymnasium` import under `src/`). Custom `VacuumEnv.reset()/step()` returns
  a `(state, reward, done, info)` 4-tuple. Reads HouseExpo JSON maps.
- **DDPG from scratch** — no SB3/RLlib. The summary must point at OUR code lines for:
  the **Actor** (Tanh-bounded deterministic action), the **Critic** (state ⊕ action),
  the **Polyak soft-target update** (`θ_t ← τθ + (1−τ)θ_t`), and the **Gaussian
  exploration noise**.
- **Action**: continuous `[throttle, steer] ∈ [−1,1]²` (unicycle model).
- **State**: 16 lidar ray distances + speed `(v,ω)` + heading cue (normalized).
- **Reward**: `+k_cov·Δcoverage − k_col·collision − k_step`.
- Fixed-by-brief intent: Gaussian noise (not OU), soft updates τ=0.005, discount γ.

## Deliverables (brief §3)

`results/figures/learning_curve.png` (cumulative reward vs episode) +
`critic_loss.png` (critic loss vs step); a **trajectory visualization** (colored
path over the JSON map, covered area shaded; optional animation);
`docs/ANALYSIS.md` answering the 3 required questions; `docs/THEORY.md` (DDPG math).

## Version Control

- New public repo `github.com/adirelm/RoboVacuumDDPG`. Branch: `main`.
- Group code `adrl-001`; share read access with the lecturer's GitHub handle `@rmisegal`.
- Version starts `1.0.0` (`src/__init__.py` `__version__` + `config.version` + `pyproject`).
- Prompt log in `docs/PROMPTS.md`. Meaningful commits; tag major versions.
- Submission cover sheet `adrl-001-ex05.pdf` via the official Moodle template
  (self-grade ONLY on the PDF; the public repo claims no numeric self-grade).

## Config Structure

`config/config.yaml`: `ddpg`, `noise`, `env`, `reward`, `training`, `maps`, `paths`,
`logging`. Accessed via `src/utils/config_loader.py`.
