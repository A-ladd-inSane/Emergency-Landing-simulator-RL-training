#!/usr/bin/env python3
"""
train_ppo.py — Train PPO on the JAX Ball Hover env via Stable-Baselines3.

This is the standard SB3 workflow, identical to training on MuJoCo envs
like Hopper-v4 or Ant-v4. The only difference is the physics backend.

Usage:
    python train_ppo.py                         # train from scratch
    python train_ppo.py --eval models/ppo_ball  # evaluate trained agent
    python train_ppo.py --timesteps 1_000_000   # custom training length

Requirements:
    pip install jax jaxlib gymnasium stable-baselines3[extra]
"""

import argparse
import os
import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import set_global_seed

from ball_env import BallHoverEnv

# ── Environment factory ──

def make_env(rank: int, seed: int = 0):
    """Return a thunk that creates a seeded env instance."""
    def _init():
        env = BallHoverEnv()
        env.reset(seed=seed + rank)
        return env
    return _init

# ── Training ──

def train(total_timesteps: int = 500_000, n_envs: int = 8, seed: int = 0):
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs",  exist_ok=True)

    # Parallel rollout workers (SubprocVecEnv = multiprocess)
    env = SubprocVecEnv([make_env(i, seed) for i in range(n_envs)])

    # Separate eval env (single process, deterministic)
    eval_env = BallHoverEnv()

    model = PPO(
        policy="MlpPolicy",
        env=env,
        # ── Hyperparameters (tuned for simple 2D continuous control) ──
        learning_rate  = 3e-4,
        n_steps        = 2048,     # rollout horizon per env
        batch_size     = 64,
        n_epochs       = 10,
        gamma          = 0.99,
        gae_lambda     = 0.95,
        clip_range     = 0.2,
        ent_coef       = 0.01,    # encourage exploration
        vf_coef        = 0.5,
        max_grad_norm  = 0.5,
        # ── Misc ──
        verbose        = 1,
        tensorboard_log= "./tb_logs/",
        seed           = seed,
    )

    # Callbacks
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="./models/",
        log_path="./logs/",
        eval_freq=10_000 // n_envs,  # evaluate every ~10k total steps
        deterministic=True,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=50_000 // n_envs,
        save_path="./models/checkpoints/",
    )

    print(f"Training PPO for {total_timesteps:,} steps with {n_envs} envs...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_cb, ckpt_cb],
        progress_bar=True,
    )
    model.save("models/ppo_ball_hover")
    print("Saved → models/ppo_ball_hover.zip")
    env.close()

# ── Evaluation ──

def evaluate(model_path: str, n_episodes: int = 5):
    env = BallHoverEnv(render_mode="human")
    model = PPO.load(model_path, env=env)

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        total_r = 0
        for step in range(env.MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(action)
            total_r += r
            env.render()
            if term or trunc:
                break
        x, y = float(env.state[0]), float(env.state[1])
        print(f"  Ep {ep+1}: reward={total_r:8.1f}  "
              f"steps={step+1:4d}  final_pos=({x:.1f}, {y:.1f})m")
    env.close()

# ── Entry ──

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO training on JAX Ball sim")
    parser.add_argument("--eval", type=str, default=None,
                        help="Path to saved model .zip for evaluation")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--n_envs", type=int, default=8)
    args = parser.parse_args()

    if args.eval:
        evaluate(args.eval)
    else:
        train(total_timesteps=args.timesteps, n_envs=args.n_envs)
