"""Unit tests for the cached YAML config loader (contract: src/utils/config_loader.py)."""

from __future__ import annotations

import pytest

from src.utils.config_loader import get, load_config


def test_load_config_returns_dict_with_version() -> None:
    cfg = load_config()
    assert isinstance(cfg, dict)
    assert cfg["version"] == "1.0.0"


def test_load_config_is_cached_same_object() -> None:
    assert load_config() is load_config()


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
