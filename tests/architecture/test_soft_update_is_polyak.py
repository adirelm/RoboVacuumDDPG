"""Architecture contract (PRD-DDPG §5.1, spec §5.2): soft_update() is Polyak
averaging  theta_target <- tau*theta + (1-tau)*theta_target  for BOTH targets.

Hand-computed: online=1.0, target=0.0, tau -> target == tau after one call;
second call -> tau*1 + (1-tau)*tau.
"""

from __future__ import annotations

import torch

from src.ddpg.agent import DDPGAgent
from src.utils.config_loader import get, load_config


def _build_agent() -> tuple[DDPGAgent, float]:
    cfg = load_config()
    tau = float(get("ddpg")["tau"])
    state_dim = int(get("env")["n_rays"]) + 4
    agent = DDPGAgent(state_dim=state_dim, action_dim=2, cfg=cfg, seed=0)
    return agent, tau


def _set_all(module: torch.nn.Module, value: float) -> None:
    with torch.no_grad():
        for p in module.parameters():
            p.fill_(value)


def _assert_all_close(module: torch.nn.Module, value: float) -> None:
    for p in module.parameters():
        assert torch.allclose(p, torch.full_like(p, value), atol=1e-7), (
            f"param not ~{value}: got {p.flatten()[0].item()}"
        )


def test_soft_update_moves_targets_toward_online_by_tau() -> None:
    agent, tau = _build_agent()
    for online, target in (
        (agent.actor, agent.actor_target),
        (agent.critic, agent.critic_target),
    ):
        _set_all(online, 1.0)
        _set_all(target, 0.0)
    agent.soft_update()
    _assert_all_close(agent.actor_target, tau)
    _assert_all_close(agent.critic_target, tau)
    agent.soft_update()
    expected = tau * 1.0 + (1.0 - tau) * tau
    _assert_all_close(agent.actor_target, expected)
    _assert_all_close(agent.critic_target, expected)
