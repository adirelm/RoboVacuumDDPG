import numpy as np

from src.ddpg.noise import GaussianNoise


def test_sample_shape_is_action_dim():
    noise = GaussianNoise(action_dim=2, sigma_start=0.2, sigma_end=0.05, decay_steps=50000, seed=0)
    sample = noise.sample()
    assert sample.shape == (2,)


def test_decay_moves_sigma_from_start_toward_end_linearly_and_monotonic():
    start, end, steps = 0.2, 0.05, 100
    noise = GaussianNoise(action_dim=2, sigma_start=start, sigma_end=end, decay_steps=steps, seed=0)
    assert noise.sigma == start
    prev = noise.sigma
    for k in range(1, steps + 1):
        noise.decay()
        assert noise.sigma <= prev + 1e-12  # monotonic non-increasing
        expected = start + (end - start) * min(k / steps, 1.0)
        assert abs(noise.sigma - expected) < 1e-9  # linear
        prev = noise.sigma
    assert abs(noise.sigma - end) < 1e-9
    noise.decay()  # clamp past horizon
    assert abs(noise.sigma - end) < 1e-9


def test_same_seed_same_samples():
    n1 = GaussianNoise(action_dim=2, sigma_start=0.2, sigma_end=0.05, decay_steps=50000, seed=42)
    n2 = GaussianNoise(action_dim=2, sigma_start=0.2, sigma_end=0.05, decay_steps=50000, seed=42)
    s1 = np.stack([n1.sample() for _ in range(5)])
    s2 = np.stack([n2.sample() for _ in range(5)])
    assert np.array_equal(s1, s2)
