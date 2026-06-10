import numpy as np

from src.env.state import assemble_state

RAY_MAX, V_MAX, OMEGA_MAX = 5.0, 0.5, 1.5


def test_state_dim_is_20_at_16_rays():
    lidar = np.full(16, 5.0, dtype=np.float32)
    s = assemble_state(lidar, 0.0, 0.0, 1.0, 0.0, RAY_MAX, V_MAX, OMEGA_MAX)
    assert isinstance(s, np.ndarray)
    assert s.shape == (20,)
    assert s.dtype == np.float32


def test_components_are_normalized():
    lidar = np.full(16, 5.0, dtype=np.float32)  # at max range -> 1.0
    s = assemble_state(lidar, V_MAX, OMEGA_MAX, 0.6, 0.8, RAY_MAX, V_MAX, OMEGA_MAX)
    assert np.allclose(s[:16], 1.0)  # lidar / ray_max
    assert np.isclose(s[16], 1.0)  # v / v_max
    assert np.isclose(s[17], 1.0)  # omega / omega_max
    assert np.isclose(s[18], 0.6)  # heading_cos passthrough
    assert np.isclose(s[19], 0.8)  # heading_sin passthrough


def test_ray_slice_in_unit_interval_and_speed_in_signed_unit():
    lidar = np.array([0.0, 2.5, 5.0] + [1.0] * 13, dtype=np.float32)
    s = assemble_state(lidar, -V_MAX, -OMEGA_MAX, 1.0, 0.0, RAY_MAX, V_MAX, OMEGA_MAX)
    assert np.all(s[:16] >= 0.0) and np.all(s[:16] <= 1.0)
    assert np.isclose(s[0], 0.0) and np.isclose(s[1], 0.5)
    assert np.isclose(s[16], -1.0) and np.isclose(s[17], -1.0)  # signed unit


def test_dim_scales_with_n_rays():
    lidar = np.full(8, 1.0, dtype=np.float32)
    s = assemble_state(lidar, 0.0, 0.0, 1.0, 0.0, RAY_MAX, V_MAX, OMEGA_MAX)
    assert s.shape == (12,)  # n_rays + 4
