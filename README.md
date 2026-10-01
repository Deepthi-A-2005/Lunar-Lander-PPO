# Lunar Lander Landing — Streamlit PPO Demo

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, and `README.md`.
3. Go to Streamlit Community Cloud.
4. Select the GitHub repository.
5. Set the main file to `app.py`.
6. Deploy.

## Important

This version is a PPO-style educational simulation. It includes an adaptive hand-built policy and demo PPO telemetry. It does not train a neural-network PPO model.

For a true PPO implementation, connect a PyTorch/Stable-Baselines3 PPO agent trained on Gymnasium LunarLander.
