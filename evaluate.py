"""Deterministic evaluation of a saved PPO checkpoint.

    python evaluate.py --checkpoint models/ppo_lander_best.pt --episodes 100 --seed 10000 --deterministic --csv logs/eval_100.csv --json logs/eval_100.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from config import load_config
from envs import is_landing_success, make_env
from ppo import PPOAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PPO LunarLander checkpoint")
    parser.add_argument("--checkpoint", type=str, default="models/ppo_lander_best.pt")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--stochastic", dest="deterministic", action="store_false")
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--json", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    agent = PPOAgent(cfg)
    agent.load_checkpoint(args.checkpoint, resume_optimizer=False)

    rows = []
    for i in range(args.episodes):
        seed = args.seed + i
        env = make_env(seed=seed)
        obs, _ = env.reset(seed=seed)
        total, length = 0.0, 0
        terminated = truncated = False
        while not (terminated or truncated):
            if args.deterministic:
                action = agent.deterministic_action(obs)
            else:
                action = int(env.action_space.sample())
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            length += 1
        landed = is_landing_success(terminated, truncated, total)
        rows.append({"episode": i, "seed": seed, "return": total, "length": length,
                     "terminated": terminated, "truncated": truncated, "landed": landed})
        env.close()

    returns = np.array([row["return"] for row in rows], dtype=np.float64)
    lengths = np.array([row["length"] for row in rows], dtype=np.float64)
    successes = sum(1 for row in rows if row["landed"])
    summary = {
        "checkpoint": args.checkpoint,
        "episodes": args.episodes,
        "seed_start": args.seed,
        "deterministic": bool(args.deterministic),
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "median_return": float(np.median(returns)),
        "min_return": float(returns.min()),
        "max_return": float(returns.max()),
        "success_rate": successes / len(rows),
        "successes": int(successes),
        "mean_length": float(lengths.mean()),
    }
    print(json.dumps(summary, indent=2))

    if args.csv:
        import pandas as pd
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.csv, index=False)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({"summary": summary, "episodes": rows}, indent=2))


if __name__ == "__main__":
    main()
