"""Unit tests for the cached YAML config loader (contract: src/utils/config_loader.py)."""

from __future__ import annotations

import logging

import pytest

from src.utils.config_loader import get, load_config, setup_logging


def test_load_config_returns_dict_with_version() -> None:
    cfg = load_config()
    assert isinstance(cfg, dict)
    assert cfg["version"] == "1.0.1"


def test_load_config_is_cached_equal_value() -> None:
    # Repeated loads must be value-equal (parse is cached) but NOT the same object:
    # each call returns its own deepcopy so caller mutations cannot leak.
    first, second = load_config(), load_config()
    assert first == second
    assert first is not second


def test_load_config_returns_isolated_copy() -> None:
    # Mutating a returned config must not poison the shared cached parse.
    c = load_config()
    original_gamma = c["ddpg"]["gamma"]
    c["ddpg"]["gamma"] = 999.0
    fresh = load_config()
    assert fresh["ddpg"]["gamma"] == original_gamma
    assert fresh["ddpg"]["gamma"] != 999.0


def test_get_returns_isolated_copy() -> None:
    # get(...) must likewise hand back a caller-local copy.
    block = get("ddpg")
    original_gamma = block["gamma"]
    block["gamma"] = 999.0
    assert get("ddpg")["gamma"] == original_gamma


def test_get_ddpg_block_has_tau() -> None:
    ddpg = get("ddpg")
    assert isinstance(ddpg, dict)
    assert ddpg["tau"] == 0.005
    assert ddpg["gamma"] == 0.99


def test_get_env_block_has_n_rays() -> None:
    env = get("env")
    assert env["n_rays"] == 16
    assert env["ray_max"] == 5.0


def test_get_missing_section_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        get("missing")


def test_load_config_explicit_path(tmp_path) -> None:
    custom = tmp_path / "c.yaml"
    custom.write_text('version: "9.9.9"\nddpg: {tau: 1.0}\n', encoding="utf-8")
    data = load_config(str(custom))
    assert data["version"] == "9.9.9"
    assert data["ddpg"]["tau"] == 1.0


def test_load_config_missing_file_raises_filenotfound(tmp_path) -> None:
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError):
        load_config(str(missing))


def test_load_config_non_dict_yaml_raises_valueerror(tmp_path) -> None:
    scalar = tmp_path / "scalar.yaml"
    scalar.write_text("just-a-string\n", encoding="utf-8")
    with pytest.raises(ValueError, match="did not parse to a dict"):
        load_config(str(scalar))


def test_setup_logging_applies_config_level() -> None:
    # EXT-1: the documented config.logging.level knob actually sets the root level.
    root = logging.getLogger()
    prev = root.level
    try:
        setup_logging({"logging": {"level": "WARNING"}})
        assert root.level == logging.WARNING
        setup_logging({"logging": {"level": "DEBUG"}})
        assert root.level == logging.DEBUG
    finally:
        root.setLevel(prev)
