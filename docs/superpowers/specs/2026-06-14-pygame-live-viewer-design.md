# Design — Pygame Live Viewer (RoboVacuumDDPG)

> Approved 2026-06-14. A single-window Pygame app that visualizes the 2D vacuum
> env in three modes (watch-training / play-trained / manual-drive), built within
> the existing V3 constraints (SDK single-entry, ≤150 LOC/file, ≥85% coverage,
> no hardcoded values, ruff clean).

## 1. Goal
Give the project an interactive GUI (the brief permits "a short animation … or
sketch the path over the JSON map"). Primary mode is the **live training
watcher**; secondary modes are **play** (greedy trained policy) and **drive**
(human control). Read the robot motion, covered area, lidar, a live reward
curve, and a training HUD.

## 2. Modes
- **TRAIN** (default): runs the DDPG training loop live; renders the current
  episode's motion + a live per-episode reward curve + HUD (episode, step, σ,
  buffer size, running reward).
- **PLAY**: loads the committed demo checkpoint and steps the greedy policy
  (`explore=False`); restarts the episode on `done`.
- **DRIVE**: advances the env with a keyboard action (↑/↓ throttle, ←/→ steer).

Controls: `[T]/[P]/[D]` mode, `space` pause, `r` reset, `+/-` speed,
`[L]` lidar toggle, `[C]` coverage toggle, `tab` cycle map, `s` cycle seed,
`esc` quit.

## 3. Architecture
Business logic (`env`/`ddpg`/`services`) stays reachable **only through the
SDK**. The GUI is a presentation layer that imports **only `src.sdk`** (+ pygame
/ stdlib). The real-time Pygame loop lives in `scripts/play.py` (excluded from
`--cov=src`); everything in `src/gui` is unit-tested headlessly via
`SDL_VIDEODRIVER=dummy`.

```
scripts/play.py              # thin entry: pygame.init + window + while-loop + display.flip
  imports: from src.gui.app import App            (ONLY)

src/gui/                     # presentation; imports ONLY src.sdk (+ pygame/stdlib + intra-gui)
  app.py        # App: owns SDK + session; update()/draw(surface)/handle(event); mode/speed/toggles
  transform.py  # pure world(m)↔screen(px) with aspect-preserving letterbox
  env_view.py   # draw walls/robot/heading/trail/covered-cells/lidar onto a Surface
  curve_view.py # draw the live per-episode reward curve onto a Surface
  hud.py        # text HUD (mode, coverage%, step, reward, episode, σ, controls)
  input_map.py  # pure: pygame key → Command / manual action vector
  palette.py    # local RGB style constants (CLAUDE.md permits visual-styling literals)

src/services/live_session.py # LiveSession: per-step driver for one mode (train/play/drive)
src/services/trainer.py      # + Trainer.step(state) extracted so train() and LiveSession share it (DRY)
src/sdk/sdk.py               # + live_session(map, seed, mode, checkpoint_path) accessor
```

### 3.1 Architecture-test change (human-decided, approved)
`tests/architecture/test_sdk_single_entry.py` is extended to:
- allow `scripts/*.py` to import `src.gui` **in addition to** `src.sdk`;
- **assert** every `src/gui/*.py` imports only `src.sdk` (+ pygame/stdlib/intra-gui)
  — never `src.env`/`src.ddpg`/`src.services` directly.

This preserves the real invariant (no business logic in UIs) while admitting a
presentation layer.

## 4. Data flow
`sdk.live_session(map_name, seed, mode, checkpoint_path=None) -> LiveSession`.

`LiveSession`:
- `.meta -> dict`: `{bounds, walls, cell_size, clean_radius, robot_radius, n_rays, ray_max}` (static draw metadata).
- `.step(action=None) -> Frame`: advance ONE step. TRAIN ignores `action` and runs
  the shared `Trainer.step` (select→env.step→store→[warmup? noise.decay+update]); PLAY ignores
  `action` and steps greedy; DRIVE uses the given `[throttle, steer]`. Auto-resets to a new
  episode on `done`.
- `.reset() -> None`: restart the current episode/agent state.

`Frame` (frozen, in `live_session.py`):
`mode, pose(x,y,θ), lidar_endpoints[(x,y)…], new_cells[(x,y)…], coverage, reward,
collision, episode, step, sigma, buffer_size, done`.

The App keeps the static `walls`/`cell_size` from `.meta`, accumulates `new_cells`
into a covered set and recent poses into a trail, appends per-episode reward for
the curve, and on each tick calls `session.step()` `gui.train_steps_per_frame`
times (speed), then `draw(surface)`.

`LiveSession` lives in `src/services`, so it MAY use `src.env` (e.g. `cast_lidar`
to compute `lidar_endpoints`); the GUI never does.

### 4.1 Reproducibility guard
`Trainer.step(state)` is the single per-step routine; `train()` loops it and so
does `LiveSession` (train mode) — DRY. A test asserts `train()` returns
bit-identical history for a fixed seed before/after the refactor, so the
committed 5-seed metrics still reproduce.

## 5. Config (`config/config.yaml` — new `gui` block)
```yaml
gui:
  window_width: 900
  window_height: 620
  fps: 60
  train_steps_per_frame: 4    # env/training steps advanced per rendered frame (speed)
  trail_length: 400           # max path-trail points kept
  demo_checkpoint: "assets/demo_policy.pt"
```
Exact RGB colours + line widths stay in `src/gui/palette.py` (visual design,
explicitly allowed by CLAUDE.md §4).

## 6. Demo checkpoint
`assets/demo_policy.pt` (~1.1 MB, copy of the trained `seed_42.pt`) is committed
(not under the git-ignored `results/`). PLAY mode loads it; if absent it falls
back to a fresh agent with a "no checkpoint" badge so the GUI still runs.

## 7. Dependencies
`pygame` added to `pyproject [project].dependencies` (runtime). Headless tests
set `SDL_VIDEODRIVER=dummy` / `SDL_AUDIODRIVER=dummy` so Pygame creates
Surfaces/fonts with no display (works on the CI Linux runner).

## 8. Testing (≥85% on src/)
- `transform.py`: world↔screen round-trip + letterbox math (pure).
- `env_view`/`curve_view`/`hud`: render to an offscreen Surface headlessly; assert
  no error + a representative pixel/extent; HUD font init guarded.
- `input_map.py`: pure key→command / action mapping table.
- `app.py`: `update/draw/handle` against a headless Surface + a fake/real short session.
- `live_session.py`: short train/play/drive runs yield well-formed Frames; meta keys.
- `trainer.py`: reproducibility guard (train() unchanged).
- `scripts/play.py`: only the window loop (unmeasured); a `runpy` import-smoke test
  in the existing `test_scripts_runnable` set (guarded to not open a window).

Screenshots for §10 are generated headlessly (`pygame.image.save` of composed
Surfaces for each mode/state) into `assets/screenshots/`.

## 9. V3 §10 (UI/UX) — now in-scope
`docs/UX.md` is rewritten from "N/A" to: usability criteria, **Nielsen's 10
heuristics mapped to the GUI**, a controls reference, accessibility notes, and
**screenshots of each mode/state** (TRAIN/PLAY/DRIVE, paused, no-checkpoint).
The audit doc §10 flips N/A → ✅.

## 10. Build order (incremental, each independently testable)
1. `palette` + `transform` (+ tests).
2. `Trainer.step` refactor + reproducibility guard.
3. `LiveSession` + `Frame` + `sdk.live_session` (+ tests).
4. `env_view` (walls/robot/trail/coverage/lidar) (+ headless tests).
5. `curve_view` + `hud` (+ headless tests).
6. `input_map` + `App` (+ tests); architecture-test extension.
7. `scripts/play.py` window loop; `config.gui` block; `pygame` dep; demo checkpoint.
8. Screenshots + `docs/UX.md` (§10) + audit-doc update.

## 11. Non-goals (YAGNI)
No menus/file dialogs (CLI args + key cycling), no sound, no multi-window, no
networked/remote viewer, no live hyperparameter editing, no recording-to-video
(static screenshots only).
