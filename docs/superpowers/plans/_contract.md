# RoboVacuumDDPG — Interface Contract (signatures are LAW for the plan)

Every plan task MUST use these exact module paths, class/function names, and
signatures. No deviation, no invented names. Types: `np.ndarray` (float32),
`torch.Tensor`, plain tuples. Config is a nested dict from `load_config()`.

## config — `src/utils/config_loader.py`
- `load_config(path: str | None = None) -> dict` — parse `config/config.yaml`; default path = repo `config/config.yaml`. Caches.
- `get(section: str) -> dict` — return a top-level block (`ddpg`,`noise`,`env`,`reward`,`training`,`maps`,`paths`,`logging`); raise `KeyError` if missing.

## env geometry & dynamics
- `src/env/house_map.py`
  - `Segment = tuple[float, float, float, float]`  # (x1, y1, x2, y2)
  - `@dataclass HouseMap: walls: list[Segment]; bounds: tuple[float,float,float,float]`  # (xmin,ymin,xmax,ymax)
  - `load_house_map(path: str) -> HouseMap` — parse a HouseExpo JSON → wall segments + bounds.
  - `HouseMap.is_inside(self, x: float, y: float) -> bool`
- `src/env/raycast.py`
  - `cast_ray(x: float, y: float, angle: float, walls: list[Segment], max_range: float) -> float` — distance to nearest wall along `angle`, capped at `max_range`.
  - `cast_lidar(x: float, y: float, theta: float, n_rays: int, walls: list[Segment], max_range: float) -> np.ndarray` — shape `(n_rays,)`, raw distances (not normalized), rays evenly spaced over 2π relative to `theta`.
- `src/env/kinematics.py`
  - `step_unicycle(pose: tuple[float,float,float], throttle: float, steer: float, v_max: float, omega_max: float, dt: float) -> tuple[float,float,float]` — returns new `(x, y, theta)`; `theta` wrapped to (−π, π].
- `src/env/collision.py`
  - `collides(x: float, y: float, robot_radius: float, walls: list[Segment]) -> bool` — True if the robot disc intersects any wall segment.
- `src/env/coverage.py`
  - `class CoverageGrid: __init__(self, bounds, cell_size: float, clean_radius: float)`
  - `mark(self, x: float, y: float) -> int` — mark cells within `clean_radius`; return count of NEWLY cleaned cells.
  - `fraction(self) -> float` — cleaned free-cells / total free-cells in `[0,1]`.
  - `nearest_uncleaned_bearing(self, x: float, y: float, theta: float) -> tuple[float, float]` — `(cos, sin)` of bearing to nearest uncleaned cell **in the robot frame**; `(0.0, 0.0)` if fully cleaned.
  - `reset(self) -> None`
- `src/env/reward.py`
  - `compute_reward(new_cells: int, collision: bool, k_coverage: float, k_collision: float, k_step: float) -> float` — `k_coverage*new_cells − k_collision*collision − k_step`.
- `src/env/state.py`
  - `assemble_state(lidar: np.ndarray, v: float, omega: float, heading_cos: float, heading_sin: float, ray_max: float, v_max: float, omega_max: float) -> np.ndarray` — normalized float32 vector, shape `(n_rays + 4,)`: `lidar/ray_max` ⊕ `v/v_max` ⊕ `omega/omega_max` ⊕ `heading_cos` ⊕ `heading_sin`. At `n_rays=16` → shape `(20,)`.
- `src/env/vacuum_env.py`
  - `class VacuumEnv:`
    - `__init__(self, house_map: HouseMap, cfg: dict, seed: int | None = None)` — stores config; builds `CoverageGrid`; sets `self.action_dim = 2`, `self.state_dim = n_rays + 4`.
    - `reset(self) -> np.ndarray` — random free spawn, clear coverage, return state.
    - `step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]` — clip action to [−1,1]; integrate kinematics; on collision: revert move + collision=True; mark coverage; build reward + next state; `done` at `max_steps` or coverage target; `info` has `{"coverage","collision","pose"}`.

## model (PyTorch)
- `src/model/actor.py`
  - `class Actor(nn.Module): __init__(self, state_dim: int, action_dim: int, hidden_sizes: list[int])`
  - `forward(self, state: torch.Tensor) -> torch.Tensor` — output `torch.tanh(...)`, shape `(B, action_dim)`, values ∈ (−1, 1).
- `src/model/critic.py`
  - `class Critic(nn.Module): __init__(self, state_dim: int, action_dim: int, hidden_sizes: list[int])`
  - `forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor` — shape `(B, 1)`; concatenates state⊕action at the first layer.

## ddpg
- `src/ddpg/replay_buffer.py`
  - `class ReplayBuffer: __init__(self, capacity: int, state_dim: int, action_dim: int, seed: int | None = None)`
  - `add(self, s: np.ndarray, a: np.ndarray, r: float, s2: np.ndarray, done: bool) -> None`
  - `sample(self, batch_size: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]` — `(s, a, r, s2, done)` as float32 tensors; `r`,`done` shape `(B,1)`.
  - `__len__(self) -> int`
- `src/ddpg/noise.py`
  - `class GaussianNoise: __init__(self, action_dim: int, sigma_start: float, sigma_end: float, decay_steps: int, seed: int | None = None)`
  - `sample(self) -> np.ndarray` — `(action_dim,)` ~ `N(0, sigma_t²)`.
  - `decay(self) -> None` — linearly step `sigma` from `sigma_start` toward `sigma_end` over `decay_steps`.
  - `sigma` (property/attr) — current std.
- `src/ddpg/agent.py`
  - `class DDPGAgent: __init__(self, state_dim: int, action_dim: int, cfg: dict, seed: int | None = None)` — builds actor/critic + targets (hard-copied), Adam optimizers (`lr_actor`,`lr_critic`), `ReplayBuffer`, `GaussianNoise`.
  - `act(self, state: np.ndarray, explore: bool = True) -> np.ndarray` — actor(state) (+ noise if explore), clipped to [−1,1], shape `(action_dim,)`.
  - `store(self, s, a, r, s2, done) -> None` — forward to buffer.
  - `update(self) -> dict` — one gradient step if `len(buffer) >= batch_size` else `{}`; returns `{"critic_loss": float, "actor_loss": float}`. TD target uses target nets + `gamma`; critic MSE; actor = −mean(Q(s, actor(s))); grad-clip `grad_clip`; then `soft_update()`.
  - `soft_update(self) -> None` — Polyak `θ_t ← τ·θ + (1−τ)·θ_t` for both targets.

## services
- `src/services/trainer.py`
  - `class Trainer: __init__(self, env: VacuumEnv, agent: DDPGAgent, cfg: dict)`
  - `train(self, episodes: int) -> list[dict]` — per-episode loop (collect→store→update); returns history list of `{"episode","reward","critic_loss","coverage","steps"}`; warmup random actions for `warmup_steps`.

## sdk — single entry (`src/sdk/sdk.py`)
- `class RoboVacuumSDK: __init__(self, config_path: str | None = None)`
  - `build_env(self, map_name: str, seed: int | None = None) -> VacuumEnv`
  - `train(self, seed: int, map_name: str | None = None) -> list[dict]` — build env+agent+trainer, run `training.episodes`, return history.
  - `rollout(self, agent: DDPGAgent, env: VacuumEnv, max_steps: int | None = None) -> list[tuple[float,float]]` — greedy (explore=False) path of `(x,y)` poses for trajectory viz.
  - `coverage_report(self, agent: DDPGAgent, env: VacuumEnv) -> dict` — `{"coverage","steps","collisions"}`.

UIs / scripts / notebook import ONLY `RoboVacuumSDK`. No `gymnasium` anywhere under `src/`.

## Contract amendments (2026-06-10, post plan-reconcile — these are LAW too)

- `DDPGAgent.save(self, path: str) -> None` / `DDPGAgent.load(self, path: str) -> None` —
  persist/restore actor+critic+target `state_dict`s (torch.save/load, `weights_only=True`
  on load). **Required** so trained policies are reloaded for the trajectory figure + eval
  (not run untrained).
- `RoboVacuumSDK.evaluate(self, checkpoint_path: str, map_name: str, seed: int | None = None) -> dict` —
  build env+agent, `agent.load(checkpoint_path)`, greedy rollout, return
  `{"coverage","steps","collisions"}`. (Resolves spec §4 listing `evaluate`.)
- `RoboVacuumSDK.map_walls(self, map_name: str) -> list[Segment]` — read-only map geometry
  for `render_trajectory.py` to draw walls through the SDK (no `src.env` import in scripts).
- `Trainer.train` history dict additionally carries a **step-level** `critic_losses: list[float]`
  (one entry per gradient update) so `critic_loss.png` is step-granular (spec §7), plus the
  per-episode mean. `scripts/train.py` saves `agent.save(...)` weights alongside the history JSON.
- **conftest is incremental**: each phase that edits `tests/conftest.py` APPENDS fixtures
  (`cfg`, `house_map`, `tiny_map`) — never replaces/drops earlier ones. Canonical map fixture
  name is `house_map`; `tiny_map` is an alias of the same 4-wall room.
- **`scripts/check_file_sizes.py` is authored ONCE in Phase 0** (scans `src/ tests/ scripts/`,
  exposes `count_loc()`+`scan_dirs()`); Phase 4 only *runs* it, never recreates it.
- The no-RL-library ban is gated: the architecture test forbids `gymnasium`/`gym` **and**
  `stable_baselines3`/`rllib`/`ray.rllib` imports under `src/`.
- Already present from scaffold (do NOT re-create): `docs/THEORY.md`, `.env-example`,
  `src/cost/meter.py` is a small `RuntimeMeter` (wall-clock + step/episode counters) authored in P0.
- `config.env.coverage_target: float` (add to config; default `0.9`) — `VacuumEnv.step` sets
  `done` at `coverage_target` OR `max_steps`; replaces the hardcoded full-coverage check.
