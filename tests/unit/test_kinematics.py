import math

from src.env.kinematics import step_unicycle

V_MAX, OMEGA_MAX, DT = 0.5, 1.5, 0.1


def test_pure_translation_east():
    x, y, th = step_unicycle((0.0, 0.0, 0.0), 1.0, 0.0, V_MAX, OMEGA_MAX, DT)
    assert math.isclose(x, 0.05, abs_tol=1e-12)  # v_max * dt
    assert math.isclose(y, 0.0, abs_tol=1e-12)
    assert math.isclose(th, 0.0, abs_tol=1e-12)


def test_pure_rotation():
    x, y, th = step_unicycle((0.0, 0.0, 0.0), 0.0, 1.0, V_MAX, OMEGA_MAX, DT)
    assert math.isclose(x, 0.0, abs_tol=1e-12)
    assert math.isclose(y, 0.0, abs_tol=1e-12)
    assert math.isclose(th, 0.15, abs_tol=1e-12)  # omega_max * dt


def test_translation_at_heading_north():
    x, y, th = step_unicycle((0.0, 0.0, math.pi / 2), 1.0, 0.0, V_MAX, OMEGA_MAX, DT)
    assert math.isclose(x, 0.0, abs_tol=1e-9)
    assert math.isclose(y, 0.05, abs_tol=1e-12)
    assert math.isclose(th, math.pi / 2, abs_tol=1e-12)


def test_theta_wraps_into_minus_pi_pi():
    # start near +pi, rotate positive until it crosses; result stays in (-pi, pi]
    th = 3.10
    for _ in range(10):
        _, _, th = step_unicycle((0.0, 0.0, th), 0.0, 1.0, V_MAX, OMEGA_MAX, DT)
    assert -math.pi < th <= math.pi


def test_negative_throttle_moves_backward():
    x, _, _ = step_unicycle((0.0, 0.0, 0.0), -1.0, 0.0, V_MAX, OMEGA_MAX, DT)
    assert math.isclose(x, -0.05, abs_tol=1e-12)
