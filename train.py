"""Train PPO on LunarLander-v3.

Example (VS Code integrated terminal, Windows):

    python train.py --config configs/default.yaml

Resumes automatically from models/ppo_lander_latest.pt with --resume.
Writes logs/training_log.csv, logs/episode_log.csv, logs/eval_log.csv,
checkpoints to models/, and figures to plots/.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from checkpoint import CheckpointManager
from config import load_config
from envs import is_landing_success, make_env, make_vector_envs
from ppo import PPOAgent, RolloutBuffer


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on LunarLander-v3")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--clip-coef", type=float, default=None)
    parser.add_argument("--gae-lambda", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--update-epochs", type=int, default=None)
    parser.add_argument("--target-eval-return", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


@torch.no_grad()
def evaluate_policy(agent, episodes, seed_start, deterministic=True):
    returns, lengths, successes = [], [], 0
    for i in range(episodes):
        seed = seed_start + i
        env = make_env(seed=seed)
        obs, _ = env.reset(seed=seed)
        total, length = 0.0, 0
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.deterministic_action(obs) if deterministic else int(env.action_space.sample())
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            length += 1
        returns.append(total)
        lengths.append(length)
        if is_landing_success(terminated, truncated, total):
            successes += 1
        env.close()
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "median_return": float(np.median(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
        "success_rate": successes / max(episodes, 1),
        "mean_length": float(np.mean(lengths)),
    }


def main():
    args = parse_args()
    overrides = {
        "total_timesteps": args.total_timesteps,
        "num_envs": args.num_envs,
        "learning_rate": args.learning_rate,
        "clip_coef": args.clip_coef,
        "gae_lambda": args.gae_lambda,
        "gamma": args.gamma,
        "rollout_steps": args.rollout_steps,
        "minibatch_size": args.minibatch_size,
        "update_epochs": args.update_epochs,
        "target_eval_return": args.target_eval_return,
        "seed": args.seed,
    }
    cfg = load_config(args.config, overrides)

    for directory in (cfg.save_dir, cfg.log_dir, cfg.plot_dir):
        Path(directory).mkdir(parents=True, exist_ok=True)

    training_log = Path(cfg.log_dir) / "training_log.csv"
    episode_log = Path(cfg.log_dir) / "episode_log.csv"
    eval_log = Path(cfg.log_dir) / "eval_log.csv"
    checkpoints = CheckpointManager(cfg.save_dir)

    agent = PPOAgent(cfg)
    if args.resume and checkpoints.latest_path().exists():
        agent.load_checkpoint(str(checkpoints.latest_path()), resume_optimizer=True)
        print(f"Resumed from {checkpoints.latest_path()} at step {agent.total_steps}")

    vec_envs = make_vector_envs(cfg.num_envs, cfg.seed)
    obs, _ = vec_envs.reset()

    env_returns = np.zeros(cfg.num_envs, dtype=np.float32)
    env_lengths = np.zeros(cfg.num_envs, dtype=np.int64)
    episode_returns = deque(maxlen=100)
    episode_counter = 0
    best_eval = -float("inf")
    evals_without_improvement = 0
    last_eval_step = 0
    start_time = time.time()
    print(f"Training PPO on {cfg.environment} | envs={cfg.num_envs} | batch={cfg.batch_size} | budget={cfg.total_timesteps} steps")

    while agent.total_steps < cfg.total_timesteps:
        buffer = RolloutBuffer(cfg.num_envs, cfg.rollout_steps, 8)
        for _ in range(cfg.rollout_steps):
            actions, log_probs, values = agent.select_action(obs)
            next_obs, rewards, terminated, truncated, _ = vec_envs.step(actions)
            buffer.add(obs, actions, log_probs, rewards, values, terminated, truncated)
            obs = next_obs
            agent.total_steps += cfg.num_envs
            env_returns += rewards
            env_lengths += 1
            for i in range(cfg.num_envs):
                if terminated[i] or truncated[i]:
                    episode_counter += 1
                    episode_returns.append(float(env_returns[i]))
                    pd.DataFrame([{"iteration": agent.iteration, "total_steps": agent.total_steps, "episode_return": float(env_returns[i]), "episode_length": int(env_lengths[i])}]).to_csv(episode_log, mode="a", header=not episode_log.exists(), index=False)
                    env_returns[i] = 0.0
                    env_lengths[i] = 0

        last_value = agent.critic(torch.as_tensor(obs, dtype=torch.float32)).detach().cpu().numpy()
        buffer.compute_returns_and_advantages(last_value, cfg.gamma, cfg.gae_lambda)
        metrics = agent.update(buffer)

        if agent.iteration % 5 == 0:
            elapsed = time.time() - start_time
            mean_ret = float(np.mean(episode_returns)) if episode_returns else 0.0
            print(f"iter {agent.iteration:4d} | steps {agent.total_steps:8d} | mean20 {mean_ret:+7.1f} | actor {metrics['actor_loss']:+.4f} | value {metrics['value_loss']:.4f} | entropy {metrics['entropy']:.3f} | kl {metrics['approx_kl']:.5f} | {elapsed:6.1f}s")

        pd.DataFrame([{
            "iteration": agent.iteration,
            "total_steps": agent.total_steps,
            "actor_loss": metrics["actor_loss"],
            "value_loss": metrics["value_loss"],
            "entropy": metrics["entropy"],
            "approx_kl": metrics["approx_kl"],
            "clip_fraction": metrics["clip_fraction"],
            "explained_variance": metrics["explained_variance"],
            "elapsed_seconds": time.time() - start_time,
        }]).to_csv(training_log, mode="a", header=not training_log.exists(), index=False)

        if agent.total_steps - last_eval_step >= cfg.eval_interval:
            last_eval_step = agent.total_steps
            result = evaluate_policy(agent, cfg.eval_episodes, cfg.eval_seed_start, deterministic=True)
            print(f"EVAL @ step {agent.total_steps}: mean {result['mean_return']:.1f} +/- {result['std_return']:.1f} | success {result['success_rate']*100:.0f}%")
            pd.DataFrame([{"total_steps": agent.total_steps, **result}]).to_csv(eval_log, mode="a", header=not eval_log.exists(), index=False)
            checkpoints.save(agent, "latest", {"eval_mean": result["mean_return"], "step": agent.total_steps})
            if result["mean_return"] > best_eval:
                best_eval = result["mean_return"]
                evals_without_improvement = 0
                checkpoints.save(agent, "best", {"eval_mean": best_eval, "step": agent.total_steps, "config": cfg.as_dict()})
            else:
                evals_without_improvement += 1
            if best_eval >= cfg.target_eval_return:
                print(f"Early stopping: target eval return {cfg.target_eval_return} reached (best {best_eval:.1f}).")
                break
            if evals_without_improvement >= cfg.early_stop_patience and agent.total_steps > 400_000:
                print("Plateau stop: patience exhausted. Best checkpoint preserved.")
                break

    vec_envs.close()
    runtime = time.time() - start_time
    summary = {"best_eval_return": best_eval, "total_steps": agent.total_steps, "runtime_seconds": runtime, "episodes": episode_counter}
    Path(cfg.log_dir, "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    try:
        from visualization import make_plots
        make_plots(cfg.log_dir, cfg.plot_dir)
        print(f"Plots written to {cfg.plot_dir}/")
    except Exception as exc:
        print(f"Plot generation skipped: {exc}")


if __name__ == "__main__":
    main()
