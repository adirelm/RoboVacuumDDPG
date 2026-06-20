"""Cached YAML config loader for RoboVacuumDDPG — single source of truth.

Contract (docs/superpowers/plans/_contract.md):
  load_config(path: str | None = None) -> dict   # parse config/config.yaml; caches
  get(section: str) -> dict                       # top-level block; KeyError if missing
"""

from __future__ import annotations

import copy
import logging
from functools import cache
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "config.yaml"


@cache
def _load_cached(cfg_path: str) -> dict:
    path = Path(cfg_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path}. Expected config/config.yaml at repo root.")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} did not parse to a dict.")
    return data


def load_config(path: str | None = None) -> dict:
    """Cache the YAML parse; return a fresh deepcopy each call so caller mutations never poison the cache."""
    cfg_path = path if path is not None else str(_DEFAULT_CONFIG_PATH)
    return copy.deepcopy(_load_cached(cfg_path))


def get(section: str) -> dict:
    """Return a caller-local deepcopy of a top-level config block; raise KeyError if missing."""
    cfg = load_config()
    if section not in cfg:
        raise KeyError(f"Config section '{section}' not found.")
    return cfg[section]


def setup_logging(cfg: dict) -> None:
    """Apply `config.logging.level` to the root logger — the single log-level knob.

    Wired from `RoboVacuumSDK.__init__`, so every script / GUI / notebook that
    goes through the SDK honours the documented `logging.level` config value.
    """
    logging.getLogger().setLevel(cfg["logging"]["level"])
