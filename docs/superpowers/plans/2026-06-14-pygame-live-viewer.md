# Pygame Live Viewer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or executing-plans. Steps use checkbox (`- [ ]`) tracking. Source of truth: `docs/superpowers/specs/2026-06-14-pygame-live-viewer-design.md`.

**Goal:** A 3-mode (train/play/drive) single-window Pygame viewer of the 2D vacuum env, within all V3 gates.

**Architecture:** Presentation layer `src/gui/*` imports ONLY `src.sdk`; business logic streamed per-step by `src/services/live_session.py` via `sdk.live_session()`; the real-time loop lives in `scripts/play.py` (unmeasured). All `src/gui` render code is unit-tested headlessly (`SDL_VIDEODRIVER=dummy`).

**Tech Stack:** Python 3.11, pygame, numpy, torch, uv, pytest, ruff.

---

## Phase 0 — Dependency + headless test harness

### Task 0.1 — Add pygame + headless conftest
**Files:** Modify `pyproject.toml`; Create `tests/unit/conftest_gui.py` helper (or extend `tests/conftest.py`).
- [ ] `uv add pygame` (lands in `[project].dependencies`); run `uv sync --dev`.
- [ ] In `tests/conftest.py` set headless drivers at import time:
  `os.environ.setdefault("SDL_VIDEODRIVER", "dummy"); os.environ.setdefault("SDL_AUDIODRIVER", "dummy")`.
- [ ] Add a session fixture `gui_surface` returning a `pygame.Surface((900, 620))` after `pygame.init()`.
- [ ] Verify: `uv run python -c "import os;os.environ['SDL_VIDEODRIVER']='dummy';import pygame;pygame.init();print(pygame.Surface((10,10)))"` prints a Surface.

---

## Phase 1 — Pure helpers (palette, transform)

### Task 1.1 — `src/gui/palette.py` (local style constants)
**Files:** Create `src/gui/palette.py`, `src/gui/__init__.py`; Test `tests/unit/test_palette.py`.
- [ ] Define RGB tuples + widths: `BG, WALL, ROBOT, HEADING, TRAIL, COVERED, LIDAR, TEXT, BADGE` and `WALL_W, TRAIL_W`.
- [ ] Test asserts every colour is a 3-tuple of ints in 0..255 (catches typos).

### Task 1.2 — `src/gui/transform.py` (world↔screen)
**Files:** Create `src/gui/transform.py`; Test `tests/unit/test_transform.py`.
- [ ] `class View(bounds, win_w, win_h, pad=20)` computing an aspect-preserving scale + offset (letterbox) so the map bounds fill the window minus pad.
- [ ] `to_screen(x, y) -> (px, py)` with y flipped (screen y grows down); `scale_len(m) -> px`.
- [ ] Tests (pure): bounds corners map inside `[0, win]`; aspect preserved (`scale_x == scale_y`); a known midpoint maps to the expected pixel; `to_screen` round-trips via an inverse within 1e-6.

---

## Phase 2 — Trainer step refactor (DRY + reproducibility)

### Task 2.1 — Extract `Trainer.step`
**Files:** Modify `src/services/trainer.py`; Test `tests/unit/test_trainer_smoke.py` (+ new reproducibility test).
- [ ] Add `def step(self, state) -> tuple[np.ndarray, dict, bool]:` doing exactly today's per-step body: `_select_action`, `env.step`, `agent.store`, `_global_step += 1`, post-warmup `noise.decay()` + `update()` (collect critic loss), return `(next_state, info_with_reward_collision_coverage_critic_loss, done)`.
- [ ] Rewrite `_run_episode` to loop `self.step(state)` (identical RNG/update order).
- [ ] **Reproducibility guard** `test_train_history_unchanged_for_seed`: build env+agent+Trainer (seed 0, tiny synthetic map, 3 episodes), capture `train(3)`; assert the rewards/critic_loss list equals a hard-coded expected snapshot (computed once from the pre-refactor code) — guarantees the committed metrics still reproduce.
- [ ] Keep `trainer.py` ≤150 LOC (extract nothing else; `step` ~18 lines).

---

## Phase 3 — LiveSession + Frame + SDK accessor

### Task 3.1 — `Frame` + `LiveSession` (train/play/drive)
**Files:** Create `src/services/live_session.py`; Test `tests/unit/test_live_session.py`.
- [ ] `@dataclass(frozen=True) class Frame:` fields per spec §4 (`mode, pose, lidar_endpoints, new_cells, coverage, reward, collision, episode, step, sigma, buffer_size, done`).
- [ ] `class LiveSession(sdk_cfg, house_map, mode, seed, checkpoint_path=None)`: builds `VacuumEnv`, `DDPGAgent`, and (train) a `Trainer`; loads checkpoint for play when present (`has_checkpoint` flag for the badge).
- [ ] `.meta` property: `{bounds, walls, cell_size, clean_radius, robot_radius, n_rays, ray_max}`.
- [ ] `.step(action=None) -> Frame`: train → `Trainer.step`; play → greedy `agent.act(explore=False)` + `env.step`; drive → `env.step(action)`. Compute `lidar_endpoints` from pose via `cast_lidar`; compute `new_cells` as the cell-centres cleaned this step (diff the coverage `cleaned` mask before/after). Auto-reset env on `done` and bump `episode`.
- [ ] `.reset()`.
- [ ] Tests: each mode runs 5 steps → Frames have right types/shapes; `coverage` in [0,1]; `lidar_endpoints` length == n_rays; drive uses the supplied action (pose changes deterministically vs a no-op).

### Task 3.2 — `sdk.live_session()`
**Files:** Modify `src/sdk/sdk.py`; Test `tests/integration/test_sdk.py`.
- [ ] `def live_session(self, map_name, seed, mode, checkpoint_path=None) -> LiveSession:` builds the house map via `load_house_map(self._map_path(map_name))` and returns `LiveSession(self.cfg, house_map, mode, seed, checkpoint_path)`.
- [ ] Keep `sdk.py` ≤150 LOC (delegate; method ~4 lines + docstring).
- [ ] Test: `sdk.live_session("room_single", 42, "play").step()` returns a Frame with `mode == "play"`.

---

## Phase 4 — env_view (the map render)

### Task 4.1 — `src/gui/env_view.py`
**Files:** Create `src/gui/env_view.py`; Test `tests/unit/test_env_view.py`.
- [ ] `draw_env(surface, view, walls, covered, trail, robot_pose, robot_radius, cell_size, lidar_endpoints, show_lidar, show_coverage)`:
  fills BG; draws covered cells (small rects, if `show_coverage`); trail polyline; lidar rays (if `show_lidar`); walls (black lines); robot disc + heading line. Pure pixel ops on the given Surface.
- [ ] Tests (headless): on a 900×620 Surface with a 4-wall square + one covered cell + a 2-pt trail + a pose, `draw_env` runs without error; the robot centre pixel ≈ ROBOT colour; a BG corner pixel == BG; toggling `show_coverage=False` leaves the covered-cell pixel == BG.

---

## Phase 5 — curve_view + hud

### Task 5.1 — `src/gui/curve_view.py`
**Files:** Create `src/gui/curve_view.py`; Test `tests/unit/test_curve_view.py`.
- [ ] `draw_curve(surface, rect, rewards, *, rolling=10)`: plot per-episode rewards inside `rect` with autoscale + a zero line + a rolling-mean line; no-op on empty list.
- [ ] Tests: empty list → no error; a known list draws a non-BG pixel inside the rect; autoscale maps min/max to rect bottom/top.

### Task 5.2 — `src/gui/hud.py`
**Files:** Create `src/gui/hud.py`; Test `tests/unit/test_hud.py`.
- [ ] `class Hud:` lazily inits `pygame.font.Font(None, size)`; `draw(surface, frame, mode, speed, no_checkpoint)` blits lines: mode, coverage%, step, reward, episode, σ, buffer, the controls hint, and a "no checkpoint — untrained" badge when `no_checkpoint`.
- [ ] Tests (headless, `pygame.font.init()`): `Hud().draw(surface, frame, "train", 4, False)` runs; text region has non-BG pixels; badge path runs when `no_checkpoint=True`.

---

## Phase 6 — input_map + App + architecture test

### Task 6.1 — `src/gui/input_map.py` (pure)
**Files:** Create `src/gui/input_map.py`; Test `tests/unit/test_input_map.py`.
- [ ] `command_for(key) -> str | None` mapping pygame keys → `{"quit","pause","reset","mode_train","mode_play","mode_drive","speed_up","speed_down","toggle_lidar","toggle_coverage","cycle_map","cycle_seed"}`.
- [ ] `drive_action(pressed_keys) -> [throttle, steer]` from the arrow keys (clamped to [-1,1]).
- [ ] Tests: `K_SPACE→"pause"`, `K_t→"mode_train"`, unknown key → None; up+right pressed → throttle>0 and steer>0.

### Task 6.2 — `src/gui/app.py` (orchestrator)
**Files:** Create `src/gui/app.py`; Test `tests/unit/test_app.py`.
- [ ] `class App(maps, seed, checkpoint_path, cfg_path=None)`: builds `RoboVacuumSDK`; `_new_session()` from current map/seed/mode; holds `mode, speed, paused, show_lidar, show_coverage, covered set, trail, episode_rewards, last_frame`.
- [ ] `update()`: if not paused, advance `speed` steps via `session.step(drive_action)`; accumulate covered/trail (cap `trail_length`)/episode_rewards (append on `done`).
- [ ] `draw(surface)`: compose `env_view` + `curve_view` (right/bottom panel `rect`) + `hud`.
- [ ] `handle_command(cmd)`: mutate state / rebuild session for mode/map/seed/reset.
- [ ] Tests (headless): construct `App(["room_single"], 42, None)`; `update()` then `draw(surface)` runs; `handle_command("mode_play")` switches mode + rebuilds; `"toggle_lidar"` flips the flag; pause stops advancement (`step` count unchanged).

### Task 6.3 — Extend `test_sdk_single_entry`
**Files:** Modify `tests/architecture/test_sdk_single_entry.py`.
- [ ] Allow `scripts/*.py` to import `src.gui` in addition to `src.sdk`.
- [ ] Add a check: every `src/gui/*.py` AST imports only `src.sdk` (+ pygame/stdlib/`src.gui.*`), asserting NO `src.env|src.ddpg|src.services` import.
- [ ] Run the architecture suite green.

---

## Phase 7 — Entry script, config, dep, demo checkpoint

### Task 7.1 — `config.gui` block
**Files:** Modify `config/config.yaml`; Test `tests/architecture/test_config_single_source.py`.
- [ ] Add the `gui:` block (spec §5). Extend the config-single-source test to assert the 6 gui keys exist.

### Task 7.2 — `scripts/play.py` (window loop)
**Files:** Create `scripts/play.py`; Test `tests/unit/test_scripts_runnable.py`.
- [ ] sys.path bootstrap (mirror the other scripts); `from src.gui.app import App`.
- [ ] `main()`: parse `--map/--seed/--checkpoint` (argparse, defaults from cfg/`assets/demo_policy.pt`); read `gui` cfg via the App's sdk; `pygame.init()`, `display.set_mode((w,h))`, `Clock`; while-loop: pump events → `app.handle_command`/quit; `app.update()`; `app.draw(screen)`; `display.flip()`; `clock.tick(fps)`. Guard the loop behind `if __name__ == "__main__"` so `runpy` import-smoke doesn't open a window.
- [ ] Add `"play"` to the runnable-scripts parametrize (import resolves, no window).

### Task 7.3 — Demo checkpoint + gitignore
**Files:** Create `assets/demo_policy.pt`; Modify `.gitignore` if needed.
- [ ] `cp results/checkpoints/seed_42.pt assets/demo_policy.pt`; confirm it is NOT ignored (`git check-ignore` empty) and ~1.1 MB.

---

## Phase 8 — §10 docs + screenshots + audit

### Task 8.1 — Headless screenshots
**Files:** Create `scripts/capture_screenshots.py` (dev tool); Create `assets/screenshots/*.png`.
- [ ] Build an `App`, drive a few steps per mode, `app.draw(surface)`, `pygame.image.save(surface, "assets/screenshots/<mode>.png")` for train/play/drive + paused + no-checkpoint. (Headless via dummy driver.)

### Task 8.2 — Rewrite `docs/UX.md` (§10 now in-scope)
**Files:** Modify `docs/UX.md`; Modify `README.md` (§ usage + GUI screenshot); Modify `instructions/assignment-5/submission_guidelines_audit.md` (§10 N/A → ✅).
- [ ] Usability criteria, Nielsen's 10 heuristics mapped to the GUI, controls reference, accessibility notes, embedded screenshots. README §2 gets a `uv run python scripts/play.py` quickstart + one screenshot. Flip the audit §10 verdict.

---

## Final gates (run before declaring done)
- [ ] `uv run ruff check src/ tests/ scripts/` → All checks passed!
- [ ] `uv run ruff format --check src/ tests/ scripts/` → all formatted
- [ ] `uv run python scripts/check_file_sizes.py` → all ≤150 LOC
- [ ] `uv run pytest tests/ --cov=src` → ≥85%, green
- [ ] Manual: `uv run python scripts/play.py` opens, cycles modes, drives, quits cleanly (local, non-headless).

## Self-review notes
- Spec coverage: every spec section maps to a task (§3→Ph1/3/4/5/6, §4→Ph3, §5→Ph7.1, §6→Ph7.3, §7→Ph0, §8→all tests, §9→Ph8, §10→build order).
- Reproducibility (spec §4.1) is Task 2.1's guard.
- All new `.py` ≤150 LOC by construction (each module is one focused responsibility).
