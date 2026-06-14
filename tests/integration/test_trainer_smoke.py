from src.ddpg.agent import DDPGAgent
from src.env.vacuum_env import VacuumEnv
from src.services.trainer import Trainer
from src.utils.config_loader import load_config


def _tiny_cfg() -> dict:
    cfg = load_config()
    cfg["env"]["max_steps"] = 5  # tiny episode for speed
    cfg["ddpg"]["warmup_steps"] = 2  # learning starts mid-episode
    cfg["ddpg"]["batch_size"] = 2  # update can fire on a tiny buffer
    return cfg


def test_train_one_episode_returns_history_with_required_keys(tiny_map):
    cfg = _tiny_cfg()
    env = VacuumEnv(tiny_map, cfg, seed=0)
    agent = DDPGAgent(env.state_dim, env.action_dim, cfg, seed=0)
    history = Trainer(env, agent, cfg).train(episodes=1)

    assert isinstance(history, list) and len(history) == 1
    record = history[0]
    # Contract amendment F29: step-level critic_losses list alongside per-episode mean.
    assert set(record) == {"episode", "reward", "critic_loss", "critic_losses", "coverage", "steps"}
    assert record["episode"] == 0
    assert record["steps"] == 5
    assert isinstance(record["reward"], float)
    assert isinstance(record["critic_losses"], list)
    assert 0.0 <= record["coverage"] <= 1.0


def test_train_honors_warmup_no_update_before_threshold(tiny_map):
    cfg = _tiny_cfg()
    cfg["env"]["max_steps"] = 1  # 1 step < warmup_steps=2 => no update
    env = VacuumEnv(tiny_map, cfg, seed=0)
    agent = DDPGAgent(env.state_dim, env.action_dim, cfg, seed=0)
    history = Trainer(env, agent, cfg).train(episodes=1)

    # No learning step fired during warmup => critic_loss defaults to 0.0.
    assert history[0]["critic_loss"] == 0.0
    assert history[0]["critic_losses"] == []
    assert history[0]["steps"] == 1


def test_step_returns_next_state_info_done(tiny_map):
    # The shared per-step routine used by both train() and the live GUI session.
    cfg = _tiny_cfg()
    env = VacuumEnv(tiny_map, cfg, seed=0)
    agent = DDPGAgent(env.state_dim, env.action_dim, cfg, seed=0)
    trainer = Trainer(env, agent, cfg)
    state = env.reset()
    next_state, info, done = trainer.step(state)
    assert next_state.shape == state.shape
    assert set(info) >= {"reward", "coverage", "collision", "pose", "action", "critic_loss"}
    assert isinstance(done, bool)


def test_train_is_deterministic_across_two_runs(tiny_map):
    # Same seed + setup => bit-identical history. Guards the step() refactor against
    # introducing nondeterminism and confirms the committed metrics reproduce on a
    # given machine. (Exact float values are not pinned across OS/BLAS — torch CPU.)
    cfg = _tiny_cfg()
    cfg["env"]["max_steps"] = 8

    def run() -> list[dict]:
        env = VacuumEnv(tiny_map, cfg, seed=0)
        agent = DDPGAgent(env.state_dim, env.action_dim, cfg, seed=0)
        return Trainer(env, agent, cfg).train(episodes=3)

    a, b = run(), run()
    for ra, rb in zip(a, b, strict=True):
        assert ra["reward"] == rb["reward"]
        assert ra["coverage"] == rb["coverage"]
        assert ra["steps"] == rb["steps"]
        assert ra["critic_losses"] == rb["critic_losses"]


def test_noise_sigma_decays_after_warmup_steps(tiny_map):
    # sigma must shrink during training: the trainer must call noise.decay() per
    # post-warmup step (otherwise sigma stays pinned at sigma_start forever).
    cfg = _tiny_cfg()
    cfg["env"]["max_steps"] = 20
    cfg["ddpg"]["warmup_steps"] = 2
    cfg["noise"]["sigma_decay_steps"] = 50  # short horizon so decay is visible
    env = VacuumEnv(tiny_map, cfg, seed=0)
    agent = DDPGAgent(env.state_dim, env.action_dim, cfg, seed=0)
    sigma_start = agent.noise.sigma_start
    Trainer(env, agent, cfg).train(episodes=3)  # >> warmup steps total
    assert agent.noise.sigma < sigma_start
