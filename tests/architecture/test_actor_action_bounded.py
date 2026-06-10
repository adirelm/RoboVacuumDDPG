"""Architecture contract (PRD-DDPG §5.2, spec §5.1): Actor output is Tanh-
bounded to [-1, 1] for all inputs, including adversarially large states.
"""

from __future__ import annotations

import torch

from src.model.actor import Actor
from src.utils.config_loader import get, load_config


def _build_actor() -> tuple[Actor, int]:
    load_config()
    env = get("env")
    state_dim = int(env["n_rays"]) + 4
    hidden = list(get("ddpg")["hidden_sizes"])
    return Actor(state_dim=state_dim, action_dim=2, hidden_sizes=hidden), state_dim


def test_actor_output_within_unit_box_random() -> None:
    torch.manual_seed(0)
    actor, state_dim = _build_actor()
    states = torch.randn(1000, state_dim)
    actions = actor(states)
    assert actions.shape == (1000, 2)
    assert torch.all(actions >= -1.0)
    assert torch.all(actions <= 1.0)


def test_actor_output_within_unit_box_adversarial() -> None:
    actor, state_dim = _build_actor()
    states = torch.full((64, state_dim), 1.0e6)
    actions = actor(states)
    assert torch.all(actions >= -1.0)
    assert torch.all(actions <= 1.0)
    assert torch.all(torch.isfinite(actions))
