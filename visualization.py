"""Matplotlib figures rendered from raw CSV logs (headless Agg backend)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BACKGROUND, PANEL, TEXT = "#0b1730", "#111f3a", "#e8eefb"
GRID, CYAN, AMBER, GREEN, RED, VIOLET = "#22345c", "#43d6ff", "#ffc95e", "#3ce68f", "#ff6a5e", "#b49aff"


def _style(ax):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.6)


def _moving_average(values, window=50):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def make_plots(log_dir="logs", plot_dir="plots"):
    log_dir, plot_dir = Path(log_dir), Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    episode_log, eval_log, training_log = log_dir / "episode_log.csv", log_dir / "eval_log.csv", log_dir / "training_log.csv"

    if episode_log.exists():
        df = pd.read_csv(episode_log)
        fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=BACKGROUND)
        ax.plot(df["total_steps"], df["episode_return"], color="#3f6ea8", linewidth=0.6, alpha=0.75, label="raw returns")
        smooth = _moving_average(df["episode_return"].to_numpy(), 50)
        if len(smooth):
            offset = len(df) - len(smooth)
            ax.plot(df["total_steps"].iloc[offset:], smooth, color=CYAN, linewidth=1.8, label="smoothed (window 50)")
        ax.axhline(200, color=AMBER, linestyle="--", linewidth=1, alpha=0.9, label="solved = 200")
        ax.axhline(260, color=GREEN, linestyle="--", linewidth=1, alpha=0.9, label="target = 260")
        ax.set_xlabel("environment steps"); ax.set_ylabel("episode return")
        ax.set_title("Episode return vs environment steps")
        ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
        _style(ax); fig.tight_layout(); fig.savefig(plot_dir / "01_episode_return.png", dpi=150)
        figures.append("01_episode_return.png"); plt.close(fig)

    if eval_log.exists():
        ev = pd.read_csv(eval_log)
        fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=BACKGROUND)
        ax.errorbar(ev["total_steps"], ev["mean_return"], yerr=ev["std_return"], color=CYAN,
                    ecolor="#2b4a73", elinewidth=1.4, marker="o", markersize=4, capsize=3, label="eval mean +/- std")
        ax.axhline(200, color=AMBER, linestyle="--", linewidth=1, label="solved = 200")
        ax.axhline(260, color=GREEN, linestyle="--", linewidth=1, label="target = 260")
        ax.set_xlabel("environment steps"); ax.set_ylabel("evaluation mean return")
        ax.set_title("Held-out evaluation performance")
        ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
        _style(ax); fig.tight_layout(); fig.savefig(plot_dir / "02_eval_performance.png", dpi=150)
        figures.append("02_eval_performance.png"); plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=BACKGROUND)
        ax.plot(ev["total_steps"], 100 * ev["success_rate"], color=GREEN, marker="o", markersize=4, linewidth=1.6, label="landing success rate")
        ax.axhline(90, color=AMBER, linestyle="--", linewidth=1, label="90% goal")
        ax.set_ylim(0, 105); ax.set_ylabel("success rate (%)"); ax.set_xlabel("environment steps")
        ax.set_title("Landing success rate on held-out seeds")
        ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
        _style(ax); fig.tight_layout(); fig.savefig(plot_dir / "03_success_rate.png", dpi=150)
        figures.append("03_success_rate.png"); plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.4, 4.6), facecolor=BACKGROUND)
        bars = ax.bar(["initial", "final"], [ev["mean_return"].iloc[0], ev["mean_return"].iloc[-1]], color=[RED, GREEN], width=0.45)
        ax.axhline(0, color=GRID, linewidth=0.8)
        ax.set_ylabel("evaluation mean return"); ax.set_title("Initial vs final evaluation return")
        for bar in bars:
            ax.annotate(f"{bar.get_height():+.1f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom" if bar.get_height() >= 0 else "top", color=TEXT, fontsize=10)
        _style(ax); fig.tight_layout(); fig.savefig(plot_dir / "04_initial_vs_final.png", dpi=150)
        figures.append("04_initial_vs_final.png"); plt.close(fig)

    if training_log.exists():
        tl = pd.read_csv(training_log)
        fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=BACKGROUND)
        ax.plot(tl["total_steps"], tl["actor_loss"], color=VIOLET, linewidth=1.2, label="actor loss")
        ax.plot(tl["total_steps"], tl["value_loss"], color="#ff7a8a", linewidth=1.2, label="critic loss")
        ax.set_xlabel("environment steps"); ax.set_ylabel("loss"); ax.set_title("PPO losses")
        ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
        _style(ax); fig.tight_layout(); fig.savefig(plot_dir / "05_losses.png", dpi=150)
        figures.append("05_losses.png"); plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=BACKGROUND)
        ax.plot(tl["total_steps"], tl["entropy"], color=CYAN, linewidth=1.4)
        ax.set_xlabel("environment steps"); ax.set_ylabel("policy entropy (nats)"); ax.set_title("Policy entropy during training")
        _style(ax); fig.tight_layout(); fig.savefig(plot_dir / "06_entropy.png", dpi=150)
        figures.append("06_entropy.png"); plt.close(fig)

    return figures
