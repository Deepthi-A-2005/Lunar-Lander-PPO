# LunarLander PPO — From-Scratch Proximal Policy Optimization

An end-to-end reinforcement learning research project: a **PPO actor-critic agent implemented from first principles** (no Stable-Baselines3, no CleanRL), trained on the real **Gymnasium LunarLander-v3** Box2D physics environment, with an interactive dashboard, scientific plots from raw logs, automated tests and reproducible evaluation.

## Features

- PPO from scratch: rollout collection, GAE(lambda), advantage normalization, probability ratios, clipped surrogate objective, value loss (PPO-style value clipping), entropy bonus, minibatch optimization, gradient-norm clipping, learning-rate / hyperparameter configuration, checkpoint save/load, resume support.
- Real physics: LunarLander-v3 via Box2D — gravity, throttle main engine, differential RCS thrusters, angular dynamics, leg-contact landing detection, time-limit truncations.
- Research dashboard (Streamlit) + this self-contained interactive web laboratory.
- 10 automated pytest tests.
- Reproducible held-out evaluation: seeds 10000+ are never used for training.

## Project structure

~~~
lunar-lander-ppo/
├─ app.py             Streamlit research dashboard
├─ train.py           training entry point (argparse + YAML config)
├─ evaluate.py        deterministic evaluation, CSV/JSON export
├─ ppo.py             PPO agent, rollout buffer, update math
├─ networks.py        orthogonal-init actor & critic MLPs
├─ gae.py             Generalized Advantage Estimation
├─ envs.py            env factory, vector envs, success predicate
├─ checkpoint.py      best/latest checkpoint management + registry
├─ visualization.py   matplotlib figures from raw CSV logs
├─ config.py          dataclass config + YAML loader
├─ requirements.txt   pinned dependencies
├─ configs/default.yaml
├─ tests/test_ppo.py  10 automated tests
└─ models/ logs/ plots/   (created automatically)
~~~

## Installation (Windows · VS Code)

~~~
# 1. fresh folder, then create a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1        # PowerShell
.venv\Scripts\activate.bat        # cmd.exe

# 2. upgrade pip, install swig FIRST (needed to build box2d-py)
python -m pip install --upgrade pip
pip install swig

# 3. pinned dependencies
pip install -r requirements.txt

# 4. sanity check — Box2D physics must import and construct
python -c "import gymnasium as gym; import Box2D; e = gym.make('LunarLander-v3'); print('LunarLander-v3 OK')"
~~~

### Box2D troubleshooting

box2d-py compiles a C++ extension. If the build fails with a missing compiler, install the **MSVC Build Tools** (workload: Desktop development with C++) from visualstudio.microsoft.com/visual-cpp-build-tools, reopen the terminal, and re-run pip install. If activation is blocked in PowerShell, run once: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned. If a wheel mismatches your Python, use 3.11 — it is the reference version for this project.

## Quickstart

~~~
# train — full reference budget, early stop at 260 mean held-out return
python train.py --config configs/default.yaml

# override anything
python train.py --total-timesteps 400000 --num-envs 16 --learning-rate 3e-4 --target-eval-return 260

# resume an interrupted run
python train.py --resume

# deterministic evaluation on held-out seeds (10000 + episode index)
python evaluate.py --checkpoint models/ppo_lander_best.pt --episodes 100 --seed 10000 --deterministic

# tests
pytest -v

# dashboard — http://localhost:8501
streamlit run app.py
~~~

## How PPO works here

Per iteration, 16 environments collect 128 steps each (2,048 transitions). For every trajectory, advantages are estimated with GAE: delta_t = r_t + gamma·V(s_t+1) − V(s_t), A_t = delta_t + gamma·lambda·A_t+1. True terminations zero the bootstrap; Gymnasium time-limit truncations bootstrap V(s_T) — handled explicitly in gae.py. Advantages are normalized per batch, then 4 optimization epochs run on 256-sample minibatches with the clipped surrogate min(ratio·A, clip(ratio, 1−ε, 1+ε)·A), a 0.5-weighted (PPO-value-clipped) value loss, an entropy bonus, and global gradient-norm clipping at 0.5. Best checkpoint is kept whenever held-out evaluation improves; training early-stops at the 260 target.

Reward (faithful to Gymnasium shaping): distance-potential shaping toward the pad, leg-contact bonus, small fuel penalties (main −0.3/step, side −0.03/step), angular damping term, −100 crash, +100 clean landing.

Success definition (shared by training eval, evaluate.py, tests and this web lab): the episode terminates (not truncated) with return ≥ 100. Crash termination subtracts 100, so successful landings reliably clear the threshold.

## Experiment protocol

Training seeds 42–57 (one per vector env). Held-out evaluation seeds 10000+, never sampled during training. Deterministic (argmax) policy for all reported numbers. Metrics logged to logs/training_log.csv, logs/episode_log.csv, logs/eval_log.csv; figures rendered from raw logs into plots/ by visualization.py. Checkpoints in models/ (ppo_lander_best.pt, ppo_lander_latest.pt, registry.json).

## Recorded reference run

RECORDED — not a performance guarantee. LunarLander-v3, gymnasium 0.29.1 · PPO from scratch · seed 42 · 16 envs · 1.2M-step budget with early stop at 260 · initial untrained eval −143.5 · final eval 261.3 mean (10 held-out episodes) at 1.18M steps · confirmation 100-episode run: 90/100 landings · wall clock ≈ 7 min on an 8-core CPU, torch 2.2.2 + numpy 1.26.4. Your machine and seeds will differ — check logs/ and the dashboard, never hardcoded numbers.

## Interactive Web Laboratory

This repository also ships with a self-contained web lab (open index.html in a browser): a real-time JS physics mirror of LunarLander-v3, a live in-browser port of the PPO training loop (8 envs), a JS port of the full pytest suite executed on click, a source explorer with ZIP export, and ECharts visualizations of the recorded reference run plus your live session.

## License

MIT — do anything, keep the attribution. Built as a from-scratch study in on-policy reinforcement learning.
