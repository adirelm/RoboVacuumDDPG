from src.env.house_map import HouseMap


def test_tiny_map_is_closed_box(tiny_map):
    assert isinstance(tiny_map, HouseMap)
    assert len(tiny_map.walls) == 4
    assert tiny_map.bounds == (0.0, 0.0, 4.0, 4.0)
    # centre of the box is inside; far outside corner is not
    assert tiny_map.is_inside(2.0, 2.0) is True
    assert tiny_map.is_inside(10.0, 10.0) is False
