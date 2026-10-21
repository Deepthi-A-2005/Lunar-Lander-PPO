"""Streamlit research dashboard for the LunarLander PPO project.

    streamlit run app.py

Lives entirely on top of the Python training/evaluation pipeline — training
logic stays in ppo.py / train.py. Environments are created once per Streamlit
session and always closed; session_state guards prevent duplicate envs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st

from config import load_config
from envs import is_landing_success, make_env
from ppo import PPOAgent

st.set_page_config(page_title="LunarLander PPO — Research Dashboard", layout="wide")


@st.cache_resource
def get_config():
    return load_config("configs/default.yaml")


def get_agent(cfg):
    if "agent" not in st.session_state:
        agent = PPOAgent(cfg)
        best = Path(cfg.save_dir) / "ppo_lander_best.pt"
        if best.exists():
            agent.load_checkpoint(str(best), resume_optimizer=False)
            st.session_state.agent_source = "best checkpoint (" + str(best) + ")"
        else:
            st.session_state.agent_source = "untrained — no checkpoint found, run train.py"
        st.session_state.agent = agent
    return st.session_state.agent


def get_sim_env():
    if "sim_env" not in st.session_state:
        st.session_state.sim_env = make_env(seed=st.session_state.get("sim_seed", 7), render_mode="rgb_array")
        st.session_state.sim_obs, _ = st.session_state.sim_env.reset(seed=st.session_state.get("sim_seed", 7))
        st.session_state.sim_return = 0.0
        st.session_state.sim_done = False
    return st.session_state.sim_env


cfg = get_config()
st.sidebar.title("LunarLander PPO")
page = st.sidebar.radio("Section", ["Overview", "Live Simulation", "Training Monitor", "Scientific Results", "Model Evaluation", "Automated Tests"])
st.sidebar.caption("Python 3.11 · PyTorch · Gymnasium · Box2D")

if page == "Overview":
    st.title("LunarLander PPO — Research Dashboard")
    st.caption("PPO actor-critic, implemented from scratch. Environment: Gymnasium LunarLander-v3 (Box2D).")
    agent = get_agent(cfg)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Algorithm", "PPO")
    col2.metric("Environment", "LunarLander-v3")
    col3.metric("Policy source", st.session_state.agent_source.split(" — ")[0])
    col4.metric("Total steps", f"{agent.total_steps:,}")
    st.markdown("Use the sidebar pages: **Live Simulation** renders the real rgb_array frames, **Training Monitor** tails logs/*.csv, **Scientific Results** shows plots/*.png, **Model Evaluation** re-runs deterministic eval on held-out seeds.")

elif page == "Live Simulation":
    st.header("Live Simulation")
    agent = get_agent(cfg)
    env = get_sim_env()
    col_view, col_ctrl = st.columns([2, 1])
    frame = env.render()
    col_view.image(frame, caption="Gymnasium rgb_array — real Box2D physics", use_container_width=True)
    with col_ctrl:
        st.subheader("Episode telemetry")
        st.write(f"Return: **{st.session_state.sim_return:+.1f}**")
        st.write(f"Action: {env.action_space.sample() if False else st.session_state.get('last_action', '—')}")
        col_a, col_b, col_c = st.columns(3)
        if col_a.button("Start"):
            st.session_state.sim_running = True
        if col_b.button("Pause"):
            st.session_state.sim_running = False
        if col_c.button("Reset"):
            obs, _ = env.reset()
            st.session_state.sim_obs = obs
            st.session_state.sim_return = 0.0
            st.session_state.sim_done = False
        speed = st.slider("Frames per refresh", 1, 10, 2)
    if st.session_state.get("sim_running") and not st.session_state.sim_done:
        for _ in range(speed):
            action = agent.deterministic_action(st.session_state.sim_obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            st.session_state.sim_obs = obs
            st.session_state.sim_return += reward
            st.session_state.last_action = int(action)
            if terminated or truncated:
                st.session_state.sim_done = True
                st.success(f"Episode finished: return {st.session_state.sim_return:+.1f}" if terminated else f"Truncated at time limit: {st.session_state.sim_return:+.1f}")
                break
        time.sleep(0.03)
        st.rerun()

elif page == "Training Monitor":
    st.header("Training Monitor")
    training_log = Path(cfg.log_dir) / "training_log.csv"
    eval_log = Path(cfg.log_dir) / "eval_log.csv"
    if not training_log.exists():
        st.info("No logs yet — run: python train.py --config configs/default.yaml")
    else:
        import pandas as pd
        tl = pd.read_csv(training_log)
        st.line_chart(tl.set_index("total_steps")[["actor_loss", "value_loss"]])
        st.line_chart(tl.set_index("total_steps")[["entropy", "approx_kl"]])
        if eval_log.exists():
            ev = pd.read_csv(eval_log)
            st.line_chart(ev.set_index("total_steps")[["mean_return"]])
            st.dataframe(ev.tail(10))

elif page == "Scientific Results":
    st.header("Scientific Results")
    plots = sorted(Path(cfg.plot_dir).glob("*.png"))
    if not plots:
        st.info("No plots yet — run training first, then visualization.make_plots() runs automatically at the end.")
    for plot in plots:
        st.image(str(plot), caption=plot.name, use_container_width=True)

elif page == "Model Evaluation":
    st.header("Model Evaluation")
    if st.button("Run 20 deterministic episodes (seeds 10000+)"):
        with st.spinner("Evaluating..."):
            agent = get_agent(cfg)
            returns, successes, lengths = [], 0, []
            try:
                for i in range(20):
                    env = make_env(seed=10000 + i)
                    obs, _ = env.reset(seed=10000 + i)
                    total, length, terminated, truncated = 0.0, 0, False, False
                    while not (terminated or truncated):
                        action = agent.deterministic_action(obs)
                        obs, reward, terminated, truncated, _ = env.step(action)
                        total += reward
                        length += 1
                    env.close()
                    returns.append(total)
                    lengths.append(length)
                    successes += int(is_landing_success(terminated, truncated, total))
            finally:
                pass
            st.metric("Mean return", f"{np.mean(returns):+.1f} +/- {np.std(returns):.1f}")
            st.metric("Success rate", f"{successes}/20 ({successes/20:.0%})")
            st.line_chart(returns)

elif page == "Automated Tests":
    st.header("Automated Tests")
    if st.button("Run pytest -v"):
        import subprocess
        result = subprocess.run(["python", "-m", "pytest", "-v"], capture_output=True, text=True, timeout=600)
        st.code(result.stdout + result.stderr, language="text")
        st.success("All tests passed.") if result.returncode == 0 else st.error("Some tests failed — see output above.")
