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
    """Single business-logic entry point: build envs, train, evaluate, and render-data.

    Every UI / script / notebook imports only this class; it owns the wiring of
    `VacuumEnv`, `DDPGAgent`, and `Trainer` so no business logic leaks into callers.
    """

    def __init__(self, config_path: str | None = None) -> None:
        self.config_path = config_path
        self.cfg = load_config(config_path)

    def _map_path(self, map_name: str) -> str:
        return f"{self.cfg['paths']['maps_dir']}/{map_name}.json"

    def build_env(self, map_name: str, seed: int | None = None) -> VacuumEnv:
        """Construct a `VacuumEnv` over the named HouseExpo map (optionally seeded)."""
        house_map = load_house_map(self._map_path(map_name))
        return VacuumEnv(house_map, self.cfg, seed=seed)

    def map_walls(self, map_name: str) -> list[Segment]:
        """Return the map's wall segments so renderers draw geometry via the SDK only."""
        return load_house_map(self._map_path(map_name)).walls

    def train(self, seed: int, map_name: str | None = None, checkpoint_path: str | None = None) -> list[dict]:
        """Train one seed on `map_name` (default first train map); save weights if a path is given.

        Returns the per-episode history list.
        """
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
        """Run one greedy episode and return the robot's (x, y) path."""
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
        """Run one greedy episode; return {coverage, steps, collisions}."""
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

    def evaluate(self, checkpoint_path: str | None, map_name: str, seed: int | None = None) -> dict:
        """Greedy coverage report on `map_name`, loading weights when given (None = fresh agent)."""
        agent, env = self._agent_env(map_name, checkpoint_path, seed)
        return self.coverage_report(agent, env)

    def _agent_env(
        self, map_name: str, checkpoint_path: str | None = None, seed: int | None = None
    ) -> tuple[DDPGAgent, VacuumEnv]:
        """Build env + agent for `map_name`, loading trained weights when given."""
        env = self.build_env(map_name, seed=seed)
        agent = DDPGAgent(env.state_dim, env.action_dim, self.cfg, seed=seed)
        if checkpoint_path is not None:
            agent.load(checkpoint_path)
        return agent, env

    def coverage_grid(
        self,
        map_name: str,
        checkpoint_path: str | None = None,
        seed: int | None = None,
        max_steps: int | None = None,
    ) -> dict:
        """Greedy rollout; return the cumulative cleaned-cell grid + extent + walls.

        Feeds render_coverage_heatmap.py the "coverage map" (Q2) through the SDK
        so scripts never import src.env directly.
        """
        agent, env = self._agent_env(map_name, checkpoint_path, seed)
        self.rollout(agent, env, max_steps=max_steps)
        cov = env.coverage
        xmax = cov.xmin + cov.nx * cov.cell_size
        ymax = cov.ymin + cov.ny * cov.cell_size
        return {
            "cleaned": cov.cleaned.tolist(),
            "free": cov.free.tolist(),
            "extent": [cov.xmin, xmax, cov.ymin, ymax],
            "walls": [tuple(w) for w in env.house_map.walls],
            "coverage": float(cov.fraction()),
        }

    def trajectory(
        self, map_name: str, checkpoint_path: str | None = None, seed: int | None = None
    ) -> list[tuple[float, float]]:
        """Greedy (x, y) path for the trajectory figure, loading trained weights when given (F6/F30)."""
        agent, env = self._agent_env(map_name, checkpoint_path, seed)
        return self.rollout(agent, env)
