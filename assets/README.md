# assets/

Committed binary/visual assets used by the GUI and the §10 docs.

- **`demo_policy.pt`** — a copy of the trained `results/checkpoints/seed_42.pt`
  (the per-seed checkpoints under `results/` are git-ignored, so this committed
  copy lets PLAY mode show a real trained policy on a fresh clone). It is the
  default `gui.demo_checkpoint` in `config/config.yaml`. Regenerate with
  `uv run python scripts/train.py` (seed 42), then copy the checkpoint here.
- **`screenshots/`** — the four GUI mode captures (`train`, `play`, `drive`,
  `play_no_checkpoint`) embedded in `docs/UX.md`; regenerate headlessly with
  `uv run python scripts/capture_screenshots.py`.
