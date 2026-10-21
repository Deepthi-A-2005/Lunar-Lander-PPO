"""Automated tests for the PPO implementation. Run: pytest -v"""
import math

import numpy as np
import torch

from config import PPOConfig
from gae import compute_gae
from networks import Actor, Critic
from ppo import PPOAgent, RolloutBuffer, set_global_seeds


def test_gae_matches_hand_calculated_example():
    rewards = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    values = np.array([0.2, 0.3, 0.4], dtype=np.float32)
    dones = np.array([False, False, False])
    truncations = np.array([False, False, False])
    gamma, lam, bootstrap = 0.9, 0.8, 0.5
    advantages, returns = compute_gae(rewards, values, dones, truncations, bootstrap, gamma, lam)
    delta2 = 1.0 + 0.9 * bootstrap - 0.4
    a2 = delta2
    delta1 = 1.0 + 0.9 * 0.4 - 0.3
    a1 = delta1 + 0.9 * 0.8 * a2
    delta0 = 1.0 + 0.9 * 0.3 - 0.2
    a0 = delta0 + 0.9 * 0.8 * a1
    assert math.isclose(float(a0), 2.37752, rel_tol=1e-6)
    assert math.isclose(float(a1), 1.816, rel_tol=1e-6)
    assert math.isclose(float(a2), 1.05, rel_tol=1e-6)
    np.testing.assert_allclose(returns, advantages + values, rtol=1e-6)


def test_gae_terminal_and_truncation_handling():
    rewards = np.array([1.0, 1.0], dtype=np.float32)
    values = np.array([0.2, 0.3], dtype=np.float32)
    gamma, lam = 0.99, 0.95
    dones = np.array([True, False])
    adv_terminal, _ = compute_gae(rewards, values, dones, np.array([False, False]), 7.0, gamma, lam)
    assert math.isclose(float(adv_terminal[0]), 1.0 - 0.2, rel_tol=1e-6)  # no bootstrap
    trunc = np.array([False, True])
    adv_trunc, _ = compute_gae(rewards[:1].copy(), values[:1].copy(), np.array([False]), trunc[:1].copy(), 0.4, gamma, lam)
    assert math.isclose(float(adv_trunc[0]), 1.0 + gamma * 0.4 - 0.2, rel_tol=1e-6)  # bootstrap V(s_T)


def test_ppo_probability_ratio_and_clipping():
    logp_old = torch.log(torch.tensor(0.25))
    logp_new = torch.log(torch.tensor(0.4))
    ratio = torch.exp(logp_new - logp_old)
    assert math.isclose(float(ratio), 1.6, rel_tol=1e-6)
    adv = torch.tensor(1.0)
    eps = 0.2
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * adv
    assert math.isclose(float(clipped), 1.2, rel_tol=1e-6)
    assert float(torch.min(unclipped, clipped)) == float(clipped)
    neg_adv = torch.tensor(-1.0)
    unclipped_neg = ratio * neg_adv
    clipped_neg = torch.clamp(ratio, 1 - eps, 1 + eps) * neg_adv
    assert float(torch.min(unclipped_neg, clipped_neg)) == float(unclipped_neg)


def test_actor_critic_output_shapes():
    actor, critic = Actor(), Critic()
    obs = torch.zeros(16, 8)
    dist = actor(obs)
    assert dist.probs.shape == (16, 4)
    assert torch.allclose(dist.probs.sum(dim=-1), torch.ones(16), atol=1e-6)
    assert critic(obs).shape == (16,)


def test_finite_losses_and_gradients():
    cfg = PPOConfig(total_timesteps=10, num_envs=2, rollout_steps=8, minibatch_size=8, update_epochs=1)
    agent = PPOAgent(cfg)
    buffer = RolloutBuffer(2, 8, 8)
    rng = np.random.default_rng(0)
    for _ in range(8):
        obs = rng.standard_normal((2, 8)).astype(np.float32)
        actions, log_probs, values = agent.select_action(obs)
        buffer.add(obs, actions, log_probs, np.ones(2, dtype=np.float32), values, np.array([False, False]), np.array([False, False]))
    buffer.compute_returns_and_advantages(np.zeros(2, dtype=np.float32), cfg.gamma, cfg.gae_lambda)
    metrics = agent.update(buffer)
    for name, value in metrics.items():
        assert math.isfinite(value), name
    for param in list(agent.actor.parameters()) + list(agent.critic.parameters()):
        assert torch.isfinite(param).all()


def test_parameters_change_after_update():
    cfg = PPOConfig(num_envs=2, rollout_steps=16, minibatch_size=16, update_epochs=2)
    agent = PPOAgent(cfg)
    before = [p.detach().clone() for p in agent.actor.parameters()]
    buffer = RolloutBuffer(2, 16, 8)
    rng = np.random.default_rng(1)
    for _ in range(16):
        obs = rng.standard_normal((2, 8)).astype(np.float32)
        actions, log_probs, values = agent.select_action(obs)
        buffer.add(obs, actions, log_probs, rng.standard_normal(2).astype(np.float32), values, np.array([False, False]), np.array([False, False]))
    buffer.compute_returns_and_advantages(np.zeros(2, dtype=np.float32), cfg.gamma, cfg.gae_lambda)
    agent.update(buffer)
    after = list(agent.actor.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_checkpoint_roundtrip(tmp_path):
    cfg = PPOConfig()
    agent = PPOAgent(cfg)
    path = tmp_path / "agent.pt"
    agent.save_checkpoint(str(path), {"purpose": "test"})
    fresh = PPOAgent(cfg)
    fresh.load_checkpoint(str(path), resume_optimizer=False)
    obs = np.linspace(-1, 1, 8, dtype=np.float32)
    assert agent.deterministic_action(obs) == fresh.deterministic_action(obs)


def test_reproducible_evaluation():
    from envs import make_env
    set_global_seeds(123)
    agent = PPOAgent(PPOConfig())
    returns_a, returns_b = [], []
    for run in (returns_a, returns_b):
        for offset in range(3):
            env = make_env(seed=10_000 + offset)
            obs, _ = env.reset(seed=10_000 + offset)
            terminated = truncated = False
            total = 0.0
            while not (terminated or truncated):
                action = agent.deterministic_action(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                total += reward
            run.append(total)
            env.close()
    assert returns_a == returns_b


def test_success_rate_calculation():
    from envs import is_landing_success
    outcomes = [(True, False, 150.0), (True, False, 80.0), (False, True, 120.0), (True, False, 260.0), (True, False, -50.0)]
    successes = sum(1 for terminated, truncated, ret in outcomes if is_landing_success(terminated, truncated, ret))
    assert math.isclose(successes / len(outcomes), 0.4)


def test_advantage_normalization():
    cfg = PPOConfig(num_envs=4, rollout_steps=8)
    buffer = RolloutBuffer(4, 8, 8)
    rng = np.random.default_rng(2)
    buffer.obs = rng.standard_normal((8, 4, 8)).astype(np.float32)
    buffer.actions = rng.integers(0, 4, (8, 4)).astype(np.int64)
    buffer.log_probs = rng.standard_normal((8, 4)).astype(np.float32)
    buffer.rewards = rng.standard_normal((8, 4)).astype(np.float32)
    buffer.values = rng.standard_normal((8, 4)).astype(np.float32)
    buffer.terminated = np.zeros((8, 4), dtype=bool)
    buffer.truncated = np.zeros((8, 4), dtype=bool)
    buffer.compute_returns_and_advantages(np.zeros(4, dtype=np.float32), cfg.gamma, cfg.gae_lambda)
    flat = buffer.advantages.reshape(-1)
    assert abs(float(flat.mean())) < 1e-5
    assert abs(float(flat.std()) - 1.0) < 1e-2
