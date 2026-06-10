"""Custom DDPG training loop: collect -> store -> update -> log.

No Gymnasium loop. Honors ddpg.warmup_steps (random actions, no learning) and
returns a per-episode history list for the renderers/SDK. Each record carries a
step-level critic_losses list (contract F29) plus the per-episode mean.
"""

from __future__ import annotations

import numpy as np

from src.ddpg.agent import DDPGAgent
from src.env.vacuum_env import VacuumEnv


class Trainer:
    def __init__(self, env: VacuumEnv, agent: DDPGAgent, cfg: dict) -> None:
        self.env = env
        self.agent = agent
        self.cfg = cfg
        self.warmup_steps = int(cfg["ddpg"]["warmup_steps"])
        self.action_dim = env.action_dim
        self._global_step = 0
        self._rng = np.random.default_rng(0)

    def _select_action(self, state: np.ndarray) -> np.ndarray:
        if self._global_step < self.warmup_steps:
            return self._rng.uniform(-1.0, 1.0, self.action_dim).astype(np.float32)
        return self.agent.act(state, explore=True)

    def _run_episode(self, episode: int) -> dict:
        state = self.env.reset()
        total_reward = 0.0
        critic_losses: list[float] = []
        steps = 0
        coverage = 0.0
        done = False
        while not done:
            action = self._select_action(state)
            next_state, reward, done, info = self.env.step(action)
            self.agent.store(state, action, reward, next_state, done)
            self._global_step += 1
            if self._global_step >= self.warmup_steps:
                self.agent.noise.decay()  # anneal exploration sigma once per learning step
                metrics = self.agent.update()
                if metrics:
                    critic_losses.append(metrics["critic_loss"])
            state = next_state
            total_reward += reward
            steps += 1
            coverage = info["coverage"]
        mean_loss = float(np.mean(critic_losses)) if critic_losses else 0.0
        return {
            "episode": episode,
            "reward": float(total_reward),
            "critic_loss": mean_loss,
            "critic_losses": critic_losses,
            "coverage": float(coverage),
            "steps": steps,
        }

    def train(self, episodes: int) -> list[dict]:
        return [self._run_episode(ep) for ep in range(episodes)]
