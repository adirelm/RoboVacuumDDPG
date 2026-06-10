import numpy as np
import torch

from src.ddpg.agent import DDPGAgent

CFG = {
    "ddpg": {
        "gamma": 0.99,
        "tau": 0.005,
        "lr_actor": 1.0e-4,
        "lr_critic": 1.0e-3,
        "batch_size": 16,
        "buffer_size": 1000,
        "hidden_sizes": [32, 32],
        "grad_clip": 1.0,
    },
    "noise": {"sigma_start": 0.2, "sigma_end": 0.05, "sigma_decay_steps": 1000},
}


def test_act_returns_action_dim_clipped_to_bounds():
    agent = DDPGAgent(state_dim=20, action_dim=2, cfg=CFG, seed=0)
    state = np.ones(20, dtype=np.float32)
    a = agent.act(state, explore=True)
    assert a.shape == (2,)
    assert np.all(a >= -1.0) and np.all(a <= 1.0)
    a_greedy = agent.act(state, explore=False)
    assert a_greedy.shape == (2,)
    assert np.all(a_greedy >= -1.0) and np.all(a_greedy <= 1.0)


def test_update_returns_finite_losses_after_prefilled_buffer():
    agent = DDPGAgent(state_dim=20, action_dim=2, cfg=CFG, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(CFG["ddpg"]["batch_size"] + 5):
        s = rng.standard_normal(20).astype(np.float32)
        a = rng.uniform(-1, 1, 2).astype(np.float32)
        s2 = rng.standard_normal(20).astype(np.float32)
        agent.store(s, a, float(rng.standard_normal()), s2, bool(rng.integers(2)))
    out = agent.update()
    assert set(out) == {"critic_loss", "actor_loss"}
    assert np.isfinite(out["critic_loss"])
    assert np.isfinite(out["actor_loss"])


def test_update_returns_empty_when_buffer_below_batch():
    agent = DDPGAgent(state_dim=20, action_dim=2, cfg=CFG, seed=0)
    s = np.zeros(20, dtype=np.float32)
    agent.store(s, np.zeros(2, dtype=np.float32), 0.0, s, False)
    assert agent.update() == {}


def test_soft_update_is_exact_polyak_with_hand_set_weights():
    # PRD-DDPG §5.1: online=1.0, target=0.0, tau=0.005 -> target == 0.005 exactly;
    # a second call -> 0.005*1 + 0.995*0.005 = 0.009975.
    agent = DDPGAgent(state_dim=20, action_dim=2, cfg=CFG, seed=0)
    with torch.no_grad():
        for p in agent.actor.parameters():
            p.fill_(1.0)
        for p in agent.actor_target.parameters():
            p.fill_(0.0)
        for p in agent.critic.parameters():
            p.fill_(1.0)
        for p in agent.critic_target.parameters():
            p.fill_(0.0)
    agent.soft_update()
    for p in agent.actor_target.parameters():
        assert torch.allclose(p, torch.full_like(p, 0.005), atol=1e-9)
    for p in agent.critic_target.parameters():
        assert torch.allclose(p, torch.full_like(p, 0.005), atol=1e-9)
    agent.soft_update()
    for p in agent.actor_target.parameters():
        assert torch.allclose(p, torch.full_like(p, 0.009975), atol=1e-9)


def test_save_then_load_restores_weights(tmp_path):
    agent = DDPGAgent(state_dim=20, action_dim=2, cfg=CFG, seed=0)
    with torch.no_grad():
        for p in agent.actor.parameters():
            p.fill_(0.42)
    ckpt = tmp_path / "agent.pt"
    agent.save(str(ckpt))
    fresh = DDPGAgent(state_dim=20, action_dim=2, cfg=CFG, seed=1)
    fresh.load(str(ckpt))
    for p in fresh.actor.parameters():
        assert torch.allclose(p, torch.full_like(p, 0.42), atol=1e-9)
