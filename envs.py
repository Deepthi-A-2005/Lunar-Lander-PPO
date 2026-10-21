"""Gymnasium environment construction.

LunarLander-v3 with Box2D physics. Rendering is fully decoupled from
training: training uses headless SyncVectorEnv instances; the dashboard
creates its own env with render_mode="rgb_array".
"""
from __future__ import annotations

import gymnasium as gym


def make_env(seed: int | None = None, render_mode: str | None = None):
    env = gym.make("LunarLander-v3", render_mode=render_mode)
    if seed is not None:
        env.reset(seed=int(seed))
        env.action_space.seed(int(seed))
        env.observation_space.seed(int(seed))
    return env


def make_vector_envs(num_envs: int, seed: int):
    def build(index):
        def thunk():
            env = gym.make("LunarLander-v3")
            env.reset(seed=seed + index)
            env.action_space.seed(seed + index)
            return env
        return thunk
    return gym.vector.SyncVectorEnv([build(i) for i in range(num_envs)])


def is_landing_success(terminated: bool, truncated: bool, episode_return: float) -> bool:
    """Success definition shared by evaluation and training.

    A successful landing terminates (not the 1000-step time limit) with a
    return of at least 100. Crash termination subtracts 100, so landings
    reliably clear this threshold on LunarLander-v3.
    """
    return bool(terminated) and not bool(truncated) and episode_return >= 100.0


def env_metadata():
    env = gym.make("LunarLander-v3")
    meta = {
        "environment": "LunarLander-v3",
        "gymnasium_version": gym.__version__,
        "observation_space": str(env.observation_space),
        "action_space": str(env.action_space),
        "max_episode_steps": env.spec.max_episode_steps,
    }
    env.close()
    return meta
