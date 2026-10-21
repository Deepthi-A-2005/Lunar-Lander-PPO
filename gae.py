"""Generalized Advantage Estimation (Schulman et al., 2016).

delta_t = r_t + gamma * V(s_t+1) - V(s_t)          (TD residual)
A_t     = delta_t + gamma * lambda * A_t+1          (recursive GAE)

Handles both terminal states (no bootstrap) and Gymnasium time-limit
truncations (bootstrap V of the final observation), which is the detail
most implementations get wrong.
"""
import numpy as np


def compute_gae(rewards, values, dones, truncations, last_value,
                gamma=0.99, lam=0.95):
    """Single-trajectory GAE.

    rewards, values:   (T,) float arrays
    dones:             (T,) bool — True on true termination (bootstrap = 0)
    truncations:       (T,) bool — True on 1000-step time limit (bootstrap V)
    last_value:        V(s_T), used only when the final step was truncated
    returns:           (advantages, returns=advantages+values)
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        if dones[t]:
            non_terminal = 0.0
            bootstrap = 0.0
        else:
            non_terminal = 1.0
            bootstrap = values[t + 1] if t + 1 < T else float(last_value)
        delta = rewards[t] + gamma * bootstrap - values[t]
        last_gae = delta + gamma * lam * non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns
