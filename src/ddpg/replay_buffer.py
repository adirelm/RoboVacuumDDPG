"""Uniform experience replay (PRD-DDPG F5.6). Pre-allocated float32 ring buffer.

sample() returns (s, a, r, s2, done) as float32 tensors; r and done are (B, 1).
A seeded numpy Generator makes index draws reproducible (acceptance §5.4).
"""

from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int, seed: int | None = None):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._idx = 0
        self._size = 0

    def add(self, s: np.ndarray, a: np.ndarray, r: float, s2: np.ndarray, done: bool) -> None:
        i = self._idx
        self.states[i] = s
        self.actions[i] = a
        self.rewards[i, 0] = r
        self.next_states[i] = s2
        self.dones[i, 0] = float(done)
        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        idx = self._rng.integers(0, self._size, size=batch_size)
        return (
            torch.as_tensor(self.states[idx], dtype=torch.float32),
            torch.as_tensor(self.actions[idx], dtype=torch.float32),
            torch.as_tensor(self.rewards[idx], dtype=torch.float32),
            torch.as_tensor(self.next_states[idx], dtype=torch.float32),
            torch.as_tensor(self.dones[idx], dtype=torch.float32),
        )

    def __len__(self) -> int:
        return self._size
