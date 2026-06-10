import torch

from src.model.critic import Critic


def test_critic_forward_shape():
    torch.manual_seed(0)
    state_dim, action_dim = 20, 2
    critic = Critic(state_dim, action_dim, [256, 256])
    batch = 32
    state = torch.randn(batch, state_dim)
    action = torch.randn(batch, action_dim)
    q = critic.forward(state, action)
    assert q.shape == (batch, 1)
    assert torch.all(torch.isfinite(q))


def test_critic_first_layer_concatenates_state_and_action():
    # in_features of the first linear must be state_dim + action_dim (§5.3, F5.16).
    critic = Critic(20, 2, [256, 256])
    first = next(m for m in critic.modules() if isinstance(m, torch.nn.Linear))
    assert first.in_features == 22
    last = [m for m in critic.modules() if isinstance(m, torch.nn.Linear)][-1]
    assert last.out_features == 1
