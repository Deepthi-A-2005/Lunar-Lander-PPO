# LunarLander PPO — From-Scratch Proximal Policy Optimization

An end-to-end reinforcement learning research project featuring a **PPO actor-critic agent implemented from first principles**, trained in the Gymnasium LunarLander-v3 Box2D physics environment, with an interactive dashboard, scientific plots, automated tests, and reproducible evaluation.

## Features

- **PPO from scratch:** rollout collection, GAE(λ), advantage normalization, probability ratios, clipped surrogate objective, value loss, entropy bonus, minibatch optimization, gradient-norm clipping, configurable hyperparameters, checkpoints, and resume support.
- **Real physics:** LunarLander-v3 using Box2D, including gravity, engine thrust, RCS thrusters, angular dynamics, landing detection, and time-limit truncations.
- **Research dashboard:** Streamlit interface and interactive web laboratory.
- **Automated testing:** pytest test suite.
- **Held-out evaluation:** evaluation seeds are kept separate from training seeds.
- **Reproducibility:** configurable experiments, CSV logs, evaluation metrics, and saved model checkpoints.

## Project Structure

```text
lunar-lander-ppo/
├── app.py
├── train.py
├── evaluate.py
├── ppo.py
├── networks.py
├── gae.py
├── envs.py
├── checkpoint.py
├── visualization.py
├── config.py
├── requirements.txt
├── configs/
│   └── default.yaml
├── tests/
│   └── test_ppo.py
└── models/ logs/ plots/
```

The `models/`, `logs/`, and `plots/` directories are generated or populated during experiments, depending on the project configuration.

## Installation — Windows and VS Code

Create and activate a virtual environment in the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install swig
pip install -r requirements.txt
```

Verify that Gymnasium and Box2D can initialize the environment:

```powershell
python -c "import gymnasium as gym; import Box2D; e = gym.make('LunarLander-v3'); print('LunarLander-v3 OK'); e.close()"
```

### Box2D Troubleshooting

If the Box2D build fails because a C++ compiler is missing, install the Visual Studio Build Tools with the **Desktop development with C++** workload. Reopen your terminal and retry the installation.

If PowerShell blocks virtual-environment activation, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Use a Python version supported by the pinned dependencies in `requirements.txt`.

## Quickstart

Train an agent using the default configuration:

```powershell
python train.py --config configs/default.yaml
```

Override training parameters if supported by the training script:

```powershell
python train.py --total-timesteps 400000 --num-envs 16 --learning-rate 3e-4 --target-eval-return 260
```

Resume a training run:

```powershell
python train.py --resume
```

Evaluate a saved checkpoint:

```powershell
python evaluate.py --checkpoint models/ppo_lander_best.pt --episodes 100 --seed 10000 --deterministic
```

Run tests:

```powershell
pytest -v
```

Launch the Streamlit dashboard:

```powershell
streamlit run app.py
```

## How PPO Works

Each training iteration collects experience from parallel environments. Generalized Advantage Estimation (GAE) estimates how much better an action performed than expected.

The PPO update uses normalized advantages and a clipped probability-ratio objective to limit excessively large policy updates. Additional components can include value-function loss, entropy regularization, minibatch optimization, and gradient-norm clipping.

The implementation handles episode termination and time-limit truncation separately when calculating value bootstrapping. Refer to `ppo.py`, `gae.py`, and `train.py` for the actual implementation details.

## Experiment Protocol

Training and evaluation should use separate random seeds to reduce the risk of reporting results on training episodes. Training logs, evaluation records, plots, and checkpoints depend on the configuration and the runs actually completed.

Inspect the generated files in `logs/`, `plots/`, and `models/` to verify experiment results.

## Recorded Reference Run

Any reported reward, landing rate, training duration, or benchmark should be treated as a **reference result, not a performance guarantee**. Verify the actual run using the corresponding logs and evaluation output before presenting metrics as reproduced results.

## Interactive Web Laboratory

If `index.html` is included in the repository, open it in a browser to explore the web laboratory. Its features and reported results should be verified against the implementation and the current session.

## License

MIT License. See the repository's license file for the applicable terms.
