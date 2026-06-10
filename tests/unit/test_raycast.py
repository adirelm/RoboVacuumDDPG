import math

import numpy as np

from src.env.raycast import cast_lidar, cast_ray

RAY_MAX = 5.0


def test_east_ray_hits_wall_at_two():
    # square room half-width 2 centred at origin; east wall x=2 from (2,-2)->(2,2)
    walls = [(2.0, -2.0, 2.0, 2.0)]
    d = cast_ray(0.0, 0.0, 0.0, walls, RAY_MAX)
    assert math.isclose(d, 2.0, abs_tol=1e-9)


def test_no_wall_within_range_clamps_to_max():
    walls = [(6.0, -1.0, 6.0, 1.0)]  # x=6 > ray_max=5
    d = cast_ray(0.0, 0.0, 0.0, walls, RAY_MAX)
    assert math.isclose(d, RAY_MAX, abs_tol=1e-9)


def test_oblique_hit_at_three_root_two():
    walls = [(3.0, 0.0, 3.0, 4.0)]
    d = cast_ray(0.0, 0.0, math.pi / 4, walls, RAY_MAX)
    assert math.isclose(d, 3.0 * math.sqrt(2.0), abs_tol=1e-6)  # 4.2426


def test_parallel_wall_is_no_hit():
    # ray along +x; wall also horizontal (parallel) -> rxs == 0 -> clamp
    walls = [(1.0, 1.0, 4.0, 1.0)]
    d = cast_ray(0.0, 0.0, 0.0, walls, RAY_MAX)
    assert math.isclose(d, RAY_MAX, abs_tol=1e-9)


def test_nearest_of_two_walls():
    walls = [(4.0, -2.0, 4.0, 2.0), (2.0, -2.0, 2.0, 2.0)]
    d = cast_ray(0.0, 0.0, 0.0, walls, RAY_MAX)
    assert math.isclose(d, 2.0, abs_tol=1e-9)


def test_cast_lidar_shape_and_ray_zero_matches_cast_ray():
    walls = [(2.0, -2.0, 2.0, 2.0)]
    theta = 0.0
    out = cast_lidar(0.0, 0.0, theta, 16, walls, RAY_MAX)
    assert isinstance(out, np.ndarray)
    assert out.shape == (16,)
    # ray 0 points along phi = theta -> east -> 2.0
    assert math.isclose(out[0], 2.0, abs_tol=1e-9)
    # all distances within [0, ray_max]
    assert np.all(out >= 0.0) and np.all(out <= RAY_MAX + 1e-9)
