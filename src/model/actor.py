"""Actor MLP: state -> Tanh-bounded deterministic action in (-1, 1).

PRD-DDPG checkpoint 1 (Actor-Critic). Tanh guarantees the action stays in
[-1, 1] (acceptance test §5.2). No hardcoded sizes: hidden_sizes is injected.
"""

from __future__ import annotations

import torch
from torch import nn


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        in_features = state_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden))
            layers.append(nn.ReLU())
            in_features = hidden
        layers.append(nn.Linear(in_features, action_dim))
        self.body = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.body(state))
