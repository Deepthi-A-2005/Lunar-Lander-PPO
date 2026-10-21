"""Experiment configuration for PPO LunarLander training.

Hyperparameters live in a dataclass, optionally overridden by a YAML file
(configs/default.yaml) or by command-line flags. All paths are relative, so
the project runs unchanged from a fresh folder on any machine.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class PPOConfig:
    environment: str = "LunarLander-v3"
    seed: int = 42
    total_timesteps: int = 1_200_000
    num_envs: int = 16
    rollout_steps: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 256
    target_eval_return: float = 260.0
    eval_interval: int = 25_000
    eval_episodes: int = 10
    eval_seed_start: int = 10_000
    early_stop_patience: int = 6
    save_dir: str = "models"
    log_dir: str = "logs"
    plot_dir: str = "plots"

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.rollout_steps

    def as_dict(self):
        return dataclasses.asdict(self)


def load_config(config_path=None, overrides=None):
    cfg = PPOConfig()
    if config_path and Path(config_path).exists():
        if yaml is None:
            raise RuntimeError("PyYAML is required to load a YAML config")
        with open(config_path, "r") as fh:
            data = yaml.safe_load(fh) or {}
        known = {f.name for f in dataclasses.fields(PPOConfig)}
        for key, value in data.items():
            if key in known:
                setattr(cfg, key, value)
    for key, value in (overrides or {}).items():
        if value is not None:
            setattr(cfg, key, value)
    return cfg
