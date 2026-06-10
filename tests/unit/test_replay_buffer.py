import numpy as np
import torch

from src.ddpg.replay_buffer import ReplayBuffer


def _dummy(state_dim, action_dim):
    s = np.ones(state_dim, dtype=np.float32)
    a = np.full(action_dim, 0.5, dtype=np.float32)
    s2 = np.zeros(state_dim, dtype=np.float32)
    return s, a, s2


def test_len_grows_then_caps_at_capacity():
    buf = ReplayBuffer(capacity=3, state_dim=20, action_dim=2, seed=0)
    assert len(buf) == 0
    s, a, s2 = _dummy(20, 2)
    for _ in range(5):
        buf.add(s, a, 1.0, s2, False)
    assert len(buf) == 3  # capped


def test_sample_returns_five_float32_tensors_with_right_shapes():
    state_dim, action_dim, batch = 20, 2, 8
    buf = ReplayBuffer(capacity=100, state_dim=state_dim, action_dim=action_dim, seed=1)
    s, a, s2 = _dummy(state_dim, action_dim)
    for i in range(50):
        buf.add(s, a, float(i), s2, bool(i % 2))
    out = buf.sample(batch)
    assert len(out) == 5
    bs, ba, br, bs2, bd = out
    for t in out:
        assert isinstance(t, torch.Tensor)
        assert t.dtype == torch.float32
    assert bs.shape == (batch, state_dim)
    assert ba.shape == (batch, action_dim)
    assert br.shape == (batch, 1)
    assert bs2.shape == (batch, state_dim)
    assert bd.shape == (batch, 1)


def test_same_seed_same_sample_indices():
    state_dim, action_dim = 20, 2
    s, a, s2 = _dummy(state_dim, action_dim)
    b1 = ReplayBuffer(capacity=100, state_dim=state_dim, action_dim=action_dim, seed=7)
    b2 = ReplayBuffer(capacity=100, state_dim=state_dim, action_dim=action_dim, seed=7)
    for i in range(50):
        b1.add(s, a, float(i), s2, False)
        b2.add(s, a, float(i), s2, False)
    r1 = b1.sample(8)[2]
    r2 = b2.sample(8)[2]
    assert torch.equal(r1, r2)
