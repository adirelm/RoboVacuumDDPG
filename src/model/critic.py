"""Critic MLP: (state, action) -> scalar Q. PRD-DDPG checkpoint 1, §5.3.

State and action are concatenated at the first layer (F5.16), so the first
linear has in_features == state_dim + action_dim; output is shape (B, 1).
"""

from __future__ import annotations

import torch
from torch import nn


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        in_features = state_dim + action_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden))
            layers.append(nn.ReLU())
            in_features = hidden
        layers.append(nn.Linear(in_features, 1))
        self.body = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.body(torch.cat([state, action], dim=1))
