"""The single business-logic entry point for RoboVacuumDDPG.

UIs / scripts / notebooks import ONLY this class — never src.env / src.ddpg /
src.services directly (CLAUDE.md §3, PRD-SIM FR-10).
"""

from __future__ import annotations

from src.ddpg.agent import DDPGAgent
from src.env.house_map import Segment, load_house_map
from src.env.vacuum_env import VacuumEnv
from src.services.trainer import Trainer
from src.utils.config_loader import load_config


class RoboVacuumSDK:
    def __init__(self, config_path: str | None = None) -> None:
        self.config_path = config_path
        self.cfg = load_config(config_path)

    def _map_path(self, map_name: str) -> str:
        return f"{self.cfg['paths']['maps_dir']}/{map_name}.json"

    def build_env(self, map_name: str, seed: int | None = None) -> VacuumEnv:
        house_map = load_house_map(self._map_path(map_name))
        return VacuumEnv(house_map, self.cfg, seed=seed)

    def map_walls(self, map_name: str) -> list[Segment]:
        # Read-only geometry accessor so the trajectory renderer can draw walls
        # through the single SDK entry point (scripts never import src.env).
        return load_house_map(self._map_path(map_name)).walls

    def train(self, seed: int, map_name: str | None = None, checkpoint_path: str | None = None) -> list[dict]:
        name = map_name or self.cfg["maps"]["train"][0]
        env = self.build_env(name, seed=seed)
        agent = DDPGAgent(env.state_dim, env.action_dim, self.cfg, seed=seed)
        trainer = Trainer(env, agent, self.cfg)
        history = trainer.train(self.cfg["training"]["episodes"])
        if checkpoint_path is not None:
            agent.save(checkpoint_path)  # F6/F30: persist trained weights for eval/trajectory
        return history

    def rollout(
        self, agent: DDPGAgent, env: VacuumEnv, max_steps: int | None = None
    ) -> list[tuple[float, float]]:
        limit = max_steps if max_steps is not None else self.cfg["env"]["max_steps"]
        state = env.reset()
        path: list[tuple[float, float]] = []
        for _ in range(limit):
            action = agent.act(state, explore=False)
            state, _reward, done, info = env.step(action)
            x, y, _theta = info["pose"]
            path.append((float(x), float(y)))
            if done:
                break
        return path

    def coverage_report(self, agent: DDPGAgent, env: VacuumEnv) -> dict:
        state = env.reset()
        steps = 0
        collisions = 0
        info = {"coverage": 0.0, "collision": False}
        done = False
        while not done:
            action = agent.act(state, explore=False)
            state, _reward, done, info = env.step(action)
            steps += 1
            if info["collision"]:
                collisions += 1
        return {
            "coverage": float(info["coverage"]),
            "steps": steps,
            "collisions": collisions,
        }

    def evaluate(self, checkpoint_path: str, map_name: str, seed: int | None = None) -> dict:
        # Contract amendment: build env+agent, load trained weights, greedy report.
        env = self.build_env(map_name, seed=seed)
        agent = DDPGAgent(env.state_dim, env.action_dim, self.cfg, seed=seed)
        agent.load(checkpoint_path)
        return self.coverage_report(agent, env)

    def trajectory(
        self, map_name: str, checkpoint_path: str | None = None, seed: int | None = None
    ) -> list[tuple[float, float]]:
        # Greedy (x, y) path for the trajectory figure; loads trained weights when
        # given (F6/F30) so the figure is the trained policy, not a fresh agent.
        env = self.build_env(map_name, seed=seed)
        agent = DDPGAgent(env.state_dim, env.action_dim, self.cfg, seed=seed)
        if checkpoint_path is not None:
            agent.load(checkpoint_path)
        return self.rollout(agent, env)
