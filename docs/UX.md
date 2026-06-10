# UX — RoboVacuumDDPG (§10)

## 1. Verdict — §10 Not Applicable (no GUI)
Submission-guideline §10 (UX/UI) is **N/A** for this assignment: RoboVacuumDDPG
ships **no interactive GUI**. The deliverable surface is a command-line workflow
plus **static analysis figures**, which is the appropriate interface for a
from-scratch RL training/evaluation artifact (the brief grades the DDPG code +
analysis, not a UI). This mirrors the A4 precedent and is recorded here so the
N/A is an explicit, evidenced decision rather than an omission.

## 2. The interaction surface (CLI + static figures)
- **Single entry point** `RoboVacuumSDK` (`src/sdk/sdk.py`): `build_env`,
  `train`, `rollout`, `coverage_report`, `evaluate`, `trajectory`, `map_walls`.
- **CLI / scripts** drive training and rendering:
  `scripts/train.py`, `scripts/evaluate.py`, `scripts/render_learning_curve.py`,
  `scripts/render_critic_loss.py`, `scripts/render_trajectory.py`,
  `scripts/fetch_houseexpo.py`.
- **Static figures** under `results/figures/`: `learning_curve.png`,
  `critic_loss.png`, and the trajectory visualization over the 2D HouseExpo map.

## 3. What a GUI would have added (and why it is out of scope)
A live Pygame-style viewer (as in A1's DroneRL) would let a user watch the
vacuum sweep in real time, but it adds zero grading value here and would risk
the ≤150-LOC and single-entry constraints. The static trajectory figure already
proves wall-avoidance + smooth continuous coverage (spec §7), so a GUI is
deliberately out of scope. Should real-time visualization ever be wanted, it
would attach to `RoboVacuumSDK.rollout()` without touching the env/agent layers.
