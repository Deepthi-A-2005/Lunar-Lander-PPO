"""Actor and Critic networks for PPO.

Both are two-hidden-layer MLPs with tanh nonlinearities and orthogonal
initialization — the architecture that made PPO famous for control tasks.
"""
import numpy as np
import torch
import torch.nn as nn


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    """Stochastic policy: observation -> logits over 4 discrete actions."""

    def __init__(self, obs_dim: int = 8, act_dim: int = 4, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, act_dim), std=0.01),
        )

    def forward(self, x):
        logits = self.net(x)
        return torch.distributions.Categorical(logits=logits)

    @torch.no_grad()
    def deterministic(self, x):
        return self.forward(x).probs.argmax(dim=-1)


class Critic(nn.Module):
    """State-value function: observation -> V(s)."""

    def __init__(self, obs_dim: int = 8, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
