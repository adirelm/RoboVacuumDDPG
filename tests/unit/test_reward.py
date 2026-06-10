import math

from src.env.reward import compute_reward

K_COV, K_COL, K_STEP = 1.0, 10.0, 0.01


def test_positive_reward_on_coverage():
    r = compute_reward(3, False, K_COV, K_COL, K_STEP)
    assert math.isclose(r, 2.99, abs_tol=1e-9)  # 1.0*3 - 0 - 0.01


def test_negative_reward_on_collision():
    r = compute_reward(0, True, K_COV, K_COL, K_STEP)
    assert math.isclose(r, -10.01, abs_tol=1e-9)  # 0 - 10.0 - 0.01


def test_sign_flips_with_collision():
    clean = compute_reward(3, False, K_COV, K_COL, K_STEP)
    crash = compute_reward(3, True, K_COV, K_COL, K_STEP)
    assert clean > 0.0
    assert crash < 0.0  # 3 - 10 - 0.01 = -7.01
    assert math.isclose(crash, -7.01, abs_tol=1e-9)


def test_idle_step_cost_only():
    r = compute_reward(0, False, K_COV, K_COL, K_STEP)
    assert math.isclose(r, -0.01, abs_tol=1e-9)
