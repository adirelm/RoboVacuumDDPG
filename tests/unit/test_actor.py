import torch

from src.model.actor import Actor


def test_actor_forward_shape_and_bounds():
    torch.manual_seed(0)
    state_dim, action_dim = 20, 2
    actor = Actor(state_dim, action_dim, [256, 256])
    batch = 64
    state = torch.randn(batch, state_dim)
    action = actor.forward(state)
    assert action.shape == (batch, action_dim)
    assert torch.all(action >= -1.0)
    assert torch.all(action <= 1.0)


def test_actor_adversarial_preactivation_stays_bounded():
    # Huge inputs must still saturate within [-1, 1] via tanh (PRD-DDPG §5.2).
    actor = Actor(20, 2, [256, 256])
    state = torch.full((1000, 20), 1.0e6)
    action = actor.forward(state)
    assert torch.all(action >= -1.0)
    assert torch.all(action <= 1.0)
    assert action.shape == (1000, 2)


def test_actor_hidden_sizes_respected():
    actor = Actor(20, 2, [256, 256])
    linears = [m for m in actor.modules() if isinstance(m, torch.nn.Linear)]
    assert linears[0].in_features == 20
    assert linears[0].out_features == 256
    assert linears[-1].out_features == 2
