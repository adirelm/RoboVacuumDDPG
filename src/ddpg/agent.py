"""DDPGAgent: act / update / Polyak soft_update (PRD-DDPG §3, checkpoints 1-2).

Online + hard-copied target nets; Adam optimizers (lr_actor, lr_critic);
ReplayBuffer; GaussianNoise. update() does the single-step TD critic loss,
the deterministic-policy-gradient actor loss, grad-norm clip, then soft_update.
save/load persist actor+critic+target state_dicts (contract amendment F6/F30).
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from src.ddpg.noise import GaussianNoise
from src.ddpg.replay_buffer import ReplayBuffer
from src.model.actor import Actor
from src.model.critic import Critic


class DDPGAgent:
    """DDPG agent: online + hard-copied target Actor/Critic, replay buffer, Gaussian noise.

    `act` selects a (optionally noise-perturbed) bounded action; `update` runs one
    TD critic step + deterministic-policy-gradient actor step + Polyak `soft_update`.
    """

    def __init__(self, state_dim: int, action_dim: int, cfg: dict, seed: int | None = None):
        if seed is not None:
            torch.manual_seed(seed)
        d = cfg["ddpg"]
        n = cfg["noise"]
        self.gamma = d["gamma"]
        self.tau = d["tau"]
        self.batch_size = d["batch_size"]
        self.grad_clip = d["grad_clip"]
        self.action_dim = action_dim
        hidden = d["hidden_sizes"]
        self.actor = Actor(state_dim, action_dim, hidden)
        self.critic = Critic(state_dim, action_dim, hidden)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=d["lr_actor"])
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=d["lr_critic"])
        self.buffer = ReplayBuffer(d["buffer_size"], state_dim, action_dim, seed)
        self.noise = GaussianNoise(action_dim, n["sigma_start"], n["sigma_end"], n["sigma_decay_steps"], seed)

    def act(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """Return the deterministic actor action, clipped to [-1, 1].

        With `explore=True`, additive Gaussian noise is added before clipping
        (data collection); `explore=False` gives the greedy policy mu(s).
        """
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            action = self.actor(s).squeeze(0).numpy()
        if explore:
            action = action + self.noise.sample()
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def store(self, s, a, r, s2, done) -> None:
        """Add a transition (s, a, r, s', done) to the replay buffer."""
        self.buffer.add(s, a, r, s2, done)

    def update(self) -> dict:
        """One DDPG learning step: TD critic loss, DPG actor loss, clip, soft-update.

        Returns the critic/actor loss dict, or {} when the buffer is below
        `batch_size` (no update performed yet).
        """
        if len(self.buffer) < self.batch_size:
            return {}
        s, a, r, s2, done = self.buffer.sample(self.batch_size)
        with torch.no_grad():
            target_q = self.critic_target(s2, self.actor_target(s2))
            y = r + self.gamma * (1.0 - done) * target_q
        critic_loss = nn.functional.mse_loss(self.critic(s, a), y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_opt.step()
        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_opt.step()
        self.soft_update()
        return {"critic_loss": float(critic_loss.item()), "actor_loss": float(actor_loss.item())}

    def soft_update(self) -> None:
        """Polyak-average each target net toward its online net: pt <- tau*po + (1-tau)*pt."""
        for online, target in (
            (self.actor, self.actor_target),
            (self.critic, self.critic_target),
        ):
            with torch.no_grad():
                for po, pt in zip(online.parameters(), target.parameters(), strict=True):
                    pt.mul_(1.0 - self.tau).add_(self.tau * po)

    def save(self, path: str) -> None:
        """Persist actor + critic + target state_dicts (contract amendment F6/F30)."""
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_target": self.critic_target.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """Restore actor + critic + target state_dicts (weights_only=True)."""
        ckpt = torch.load(path, weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
