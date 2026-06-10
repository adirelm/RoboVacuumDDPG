"""Verify the shared conftest fixtures (cfg dict + synthetic 4-wall HouseMap)."""

from __future__ import annotations


def test_cfg_fixture_is_config_dict(cfg) -> None:
    assert isinstance(cfg, dict)
    assert cfg["version"] == "1.0.0"
    assert cfg["ddpg"]["tau"] == 0.005  # noqa: PLR2004
    assert cfg["env"]["n_rays"] == 16  # noqa: PLR2004


def test_house_map_has_four_walls(house_map) -> None:
    assert len(house_map.walls) == 4  # noqa: PLR2004
    for seg in house_map.walls:
        assert len(seg) == 4  # noqa: PLR2004
        assert all(isinstance(c, float) for c in seg)


def test_house_map_bounds(house_map) -> None:
    assert house_map.bounds == (0.0, 0.0, 4.0, 4.0)


def test_house_map_is_inside_interior(house_map) -> None:
    assert house_map.is_inside(2.0, 2.0) is True
    assert house_map.is_inside(0.0, 0.0) is False
    assert house_map.is_inside(5.0, 2.0) is False
    assert house_map.is_inside(2.0, -1.0) is False
