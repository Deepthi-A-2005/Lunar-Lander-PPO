"""Proximal Policy Optimization — implemented from scratch.

No Stable-Baselines3, no CleanRL, no RL libraries in the learning path.
Only PyTorch (autodiff) and Gymnasium are required.

Pipeline per iteration:
  1. Collect rollout_steps x num_envs on-policy transitions.
  2. Generalized Advantage Estimation per trajectory.
  3. Advantage normalization across the whole batch.
  4. update_epochs x minibatch Adam passes with the clipped surrogate.
  5. Gradient-norm clipping at 0.5 (all actor + critic parameters).
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from config import PPOConfig
from networks import Actor, Critic
from gae import compute_gae

EPS = 1e-8


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class RolloutBuffer:
    """Stores a full rollout across N vectorized environments."""

    def __init__(self, num_envs: int, rollout_steps: int, obs_dim: int):
        self.obs = np.zeros((rollout_steps, num_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((rollout_steps, num_envs), dtype=np.int64)
        self.log_probs = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.rewards = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.values = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.terminated = np.zeros((rollout_steps, num_envs), dtype=bool)
        self.truncated = np.zeros((rollout_steps, num_envs), dtype=bool)
        self.advantages = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.returns = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.ptr = 0

    def add(self, obs, action, log_prob, reward, value, terminated, truncated):
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.rewards[i] = reward
        self.values[i] = value
        self.terminated[i] = terminated
        self.truncated[i] = truncated
        self.ptr += 1

    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda):
        num_envs = self.obs.shape[1]
        for env_idx in range(num_envs):
            adv, ret = compute_gae(
                self.rewards[:, env_idx],
                self.values[:, env_idx],
                self.terminated[:, env_idx],
                self.truncated[:, env_idx],
                last_value[env_idx],
                gamma,
                gae_lambda,
            )
            self.advantages[:, env_idx] = adv
            self.returns[:, env_idx] = ret
        flat = self.advantages.reshape(-1)
        self.advantages = ((flat - flat.mean()) / (flat.std() + EPS)).reshape(self.advantages.shape)


class PPOAgent:
    def __init__(self, cfg: PPOConfig, device: str = "cpu"):
        set_global_seeds(cfg.seed)
        self.cfg = cfg
        self.device = torch.device(device)
        self.actor = Actor().to(self.device)
        self.critic = Critic().to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=cfg.learning_rate,
            eps=1e-5,
        )
        self.total_steps = 0
        self.iteration = 0

    @torch.no_grad()
    def select_action(self, obs: np.ndarray):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        dist = self.actor(obs_t)
        action = dist.sample()
        value = self.critic(obs_t)
        return action.cpu().numpy(), dist.log_prob(action).cpu().numpy(), value.cpu().numpy()

    @torch.no_grad()
    def deterministic_action(self, obs: np.ndarray) -> int:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return int(self.actor.deterministic(obs_t).item())

    def update(self, buffer: RolloutBuffer):
        cfg = self.cfg
        obs = torch.as_tensor(buffer.obs.reshape(-1, 8), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(buffer.actions.reshape(-1), dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(buffer.log_probs.reshape(-1), dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(buffer.advantages.reshape(-1), dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(buffer.returns.reshape(-1), dtype=torch.float32, device=self.device)
        old_values = torch.as_tensor(buffer.values.reshape(-1), dtype=torch.float32, device=self.device)

        batch_size = obs.shape[0]
        minibatch_size = min(cfg.minibatch_size, batch_size)
        metrics = {k: 0.0 for k in ["actor_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "explained_variance"]}
        updates = 0

        for _ in range(cfg.update_epochs):
            permutation = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch_size):
                index = permutation[start:start + minibatch_size]
                dist = self.actor(obs[index])
                log_prob = dist.log_prob(actions[index])
                entropy = dist.entropy().mean()
                value = self.critic(obs[index])

                log_ratio = log_prob - old_log_probs[index]
                ratio = log_ratio.exp()
                adv = advantages[index]

                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef) * adv
                actor_loss = -torch.min(unclipped, clipped).mean()

                # PPO-style clipped value loss (same epsilon)
                value_clipped = old_values[index] + torch.clamp(value - old_values[index], -cfg.clip_coef, cfg.clip_coef)
                value_loss_unclipped = (value - returns[index]) ** 2
                value_loss_clipped = (value_clipped - returns[index]) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                loss = actor_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                params = list(self.actor.parameters()) + list(self.critic.parameters())
                nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = (torch.abs(ratio - 1.0) > cfg.clip_coef).float().mean()
                    var_returns = torch.var(returns[index])
                    explained_variance = 1.0 - torch.var(returns[index] - value) / (var_returns + 1e-8)

                metrics["actor_loss"] += actor_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += entropy.item()
                metrics["approx_kl"] += approx_kl.item()
                metrics["clip_fraction"] += clip_fraction.item()
                metrics["explained_variance"] += explained_variance.item()
                updates += 1

        for key in metrics:
            metrics[key] /= max(updates, 1)
        self.iteration += 1
        return metrics

    def save_checkpoint(self, path: str, meta: dict | None = None) -> None:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "iteration": self.iteration,
            "config": self.cfg.as_dict(),
            "meta": meta or {},
        }
        torch.save(payload, path_obj)

    def load_checkpoint(self, path: str, resume_optimizer: bool = True) -> dict:
        payload = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(payload["actor_state_dict"])
        self.critic.load_state_dict(payload["critic_state_dict"])
        if resume_optimizer and "optimizer_state_dict" in payload:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.total_steps = payload.get("total_steps", 0)
        self.iteration = payload.get("iteration", 0)
        return payload.get("meta", {})
