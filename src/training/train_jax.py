#!/usr/bin/env python3
"""
train_jax.py — Pure-JAX REINFORCE training on the Ball Hover env.

No PyTorch, no Stable-Baselines3 — just JAX. This is the "Brax-style"
approach: everything (env, policy, training) stays in JAX, enabling
massive parallelism via vmap.

WHY REINFORCE INSTEAD OF PPO?
  REINFORCE is the simplest policy-gradient algorithm (~50 lines).
  It's enough to show the full train loop: sample actions → compute
  returns → update policy. PPO adds clipping + value function, which
  ~triples the code. Once you understand this, PPO is a small step.

  For production: swap this policy for a PPO implementation (Brax,
  PureJaxRL, or your own). The env interface is identical.

ARCHITECTURE
  ┌──────────┐   vmap     ┌──────────────┐
  │ JAX Env  │ ←──────── │  N parallel  │
  │ (ball)   │            │  rollouts    │
  └────┬─────┘            └──────┬───────┘
       │                         │
       │                 ┌───────▼───────┐
       │                 │ JAX Policy     │
       │                 │ (MLP → action) │
       │                 └───────┬───────┘
       │                         │
       │            ┌────────────▼────────────┐
       │            │ REINFORCE policy gradient│
       │            │ loss = -log_prob * return│
       │            └──────────────────────────┘

Usage:
    python train_jax.py                    # train + print
    python train_jax.py --n_envs 1000 --steps 500
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import argparse
import time

from ball_env_jax import jax_reset, jax_step, batch_reset, batch_step, SimConfig


# ════════════════════════════════════════════════════════════════
# 1. Policy Network (JAX/Flax-free, just manual MLP)
# ════════════════════════════════════════════════════════════════

def init_mlp_params(key, layer_sizes, std=0.01):
    """Xavier-init for a simple MLP. Returns list of (W, b) tuples."""
    params = []
    for i in range(len(layer_sizes) - 1):
        fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
        key, subkey = jax.random.split(key)
        W = jax.random.normal(subkey, (fan_in, fan_out)) * std
        b = jnp.zeros((fan_out,))
        params.append((W, b))
    return params


def policy_forward(params, obs):
    """
    Forward pass: obs → action_mean, action_std.
    Returns mean and std for a Gaussian policy.
    """
    x = obs
    for i, (W, b) in enumerate(params):
        x = jnp.dot(x, W) + b
        if i < len(params) - 1:
            x = jnp.tanh(x)  # hidden activations
    # Last layer outputs raw mean; we'll squash to [-1, 1] with tanh
    action_mean = jnp.tanh(x[0])       # scalar for thrust_norm
    action_mean2 = jnp.tanh(x[1])      # scalar for angle_norm
    # Wait — let's do it properly for 2D action
    # Actually, the last layer should output 2D mean for [thrust, angle]
    return x  # raw output, shape (2,)


def policy_forward_proper(params, obs):
    """Proper forward: obs (5,) → action_mean (2,), action_std (2,)."""
    x = obs
    for i, (W, b) in enumerate(params):
        x = jnp.dot(x, W) + b
        if i < len(params) - 1:
            x = jnp.tanh(x)
    action_mean = jnp.tanh(x)           # squash to [-1, 1], shape (2,)
    action_std = jnp.array([0.3, 0.3])  # fixed std for simplicity
    return action_mean, action_std


def sample_action(key, params, obs):
    """Sample action from Gaussian policy. Returns (action, log_prob)."""
    mean, std = policy_forward_proper(params, obs)
    action = mean + std * jax.random.normal(key, shape=mean.shape)
    action = jnp.clip(action, -1.0, 1.0)
    # Log prob of Gaussian (ignoring clip correction for simplicity)
    log_prob = jnp.sum(
        -0.5 * ((action - mean) / std) ** 2
        - jnp.log(std) - 0.5 * jnp.log(2 * jnp.pi)
    )
    return action, log_prob


# Batched policy (vmap over envs)
batch_sample_action = jax.vmap(
    sample_action, in_axes=(0, None, 0)
)
batch_policy_forward = jax.vmap(
    policy_forward_proper, in_axes=(None, 0)
)


# ════════════════════════════════════════════════════════════════
# 2. Rollout: collect N_ENVS parallel trajectories
# ════════════════════════════════════════════════════════════════

def collect_rollout(key, params, env_states, cfg, n_steps):
    """
    Collect a trajectory of n_steps from n_envs parallel envs.
    Returns arrays of shape (n_steps, n_envs) for rewards, log_probs.
    """
    n_envs = env_states.state.shape[0]

    all_rewards = []
    all_log_probs = []
    all_values = []  # We'll use simple return-based, no critic

    for t in range(n_steps):
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, n_envs)

        # Get observations from current states
        obs = env_states.state  # shape (n_envs, 4) — [x,y,vx,vy]
        # Add target residual as 5th obs dim
        target_y = 50.0
        obs_full = jnp.concatenate(
            [obs, (target_y - obs[:, 1:2])], axis=1
        )  # shape (n_envs, 5)

        # Sample actions from policy
        actions, log_probs = batch_sample_action(keys, params, obs_full)

        # Step envs
        env_states, (_, rewards, dones) = batch_step(
            env_states, actions, cfg
        )

        all_rewards.append(rewards)
        all_log_probs.append(log_probs)

    # Stack: (n_steps, n_envs)
    rewards = jnp.stack(all_rewards, axis=0)
    log_probs = jnp.stack(all_log_probs, axis=0)

    return rewards, log_probs, env_states


# ════════════════════════════════════════════════════════════════
# 3. Compute Returns (discounted cumulative reward)
# ════════════════════════════════════════════════════════════════

def compute_returns(rewards, gamma=0.99):
    """
    G_t = sum_{k=0}^{T-t} gamma^k * r_{t+k}
    rewards: (n_steps, n_envs)
    returns: (n_steps, n_envs)
    """
    n_steps, n_envs = rewards.shape
    returns = jnp.zeros_like(rewards)
    running = jnp.zeros((n_envs,))

    # Reverse iteration
    for t in reversed(range(n_steps)):
        running = rewards[t] + gamma * running
        returns = returns.at[t].set(running)

    # Normalize returns (reduces variance)
    mean = jnp.mean(returns)
    std = jnp.std(returns) + 1e-8
    returns = (returns - mean) / std

    return returns


# ════════════════════════════════════════════════════════════════
# 4. REINFORCE Loss & Update
# ════════════════════════════════════════════════════════════════

def reinforce_loss(params, obs_seq, actions_seq, returns_seq):
    """
    L = -mean( log_prob(a|s) * G_t )
    """
    # obs_seq: (n_steps, n_envs, 5)
    # actions_seq: (n_steps, n_envs, 2)
    # returns_seq: (n_steps, n_envs)

    # Flatten for batch processing
    obs_flat = obs_seq.reshape(-1, obs_seq.shape[-1])
    actions_flat = actions_seq.reshape(-1, actions_seq.shape[-1])
    returns_flat = returns_seq.reshape(-1)

    # Compute log_probs for all (s, a) pairs
    def single_log_prob(obs, action):
        mean, std = policy_forward_proper(params, obs)
        lp = jnp.sum(
            -0.5 * ((action - mean) / std) ** 2
            - jnp.log(std) - 0.5 * jnp.log(2 * jnp.pi)
        )
        return lp

    log_probs = jax.vmap(single_log_prob)(obs_flat, actions_flat)
    loss = -jnp.mean(log_probs * returns_flat)
    return loss


def update_step(key, params, opt_state, obs_seq, actions_seq, returns_seq, lr):
    """One gradient step on the REINFORCE loss."""
    grads = jax.grad(reinforce_loss)(
        params, obs_seq, actions_seq, returns_seq
    )
    # Simple SGD update
    new_params = []
    for (W, b), (gW, gb) in zip(params, grads):
        new_params.append((W - lr * gW, b - lr * gb))
    return new_params


# ════════════════════════════════════════════════════════════════
# 5. Training Loop
# ════════════════════════════════════════════════════════════════

def train(n_envs=500, n_steps=200, n_iterations=200, lr=1e-3, gamma=0.99):
    cfg = SimConfig()

    # Initialize policy network: 5 → 64 → 64 → 2
    key = jax.random.PRNGKey(42)
    key, init_key = jax.random.split(key)
    params = init_mlp_params(init_key, [5, 64, 64, 2], std=0.1)

    # Initialize envs
    keys = jax.random.split(key, n_envs)
    env_states = batch_reset(keys, cfg, n_envs)

    # JIT compile the update step
    jit_update = jax.jit(update_step, static_argnums=8)

    print(f"Training REINFORCE: {n_envs} envs × {n_steps} steps × "
          f"{n_iterations} iters")
    print(f"  Total env-steps: {n_envs * n_steps * n_iterations:,}")
    print()

    for it in range(n_iterations):
        key, rollout_key = jax.random.split(key)

        # Collect rollout
        rewards, log_probs, env_states = collect_rollout(
            rollout_key, params, env_states, cfg, n_steps
        )

        # Build obs and actions sequences for loss computation
        # (re-extract obs from the trajectory — simplified here)
        # For efficiency, we'd store these during rollout; for clarity,
        # we approximate by using the final state
        obs_seq = jnp.zeros((n_steps, n_envs, 5))  # placeholder
        actions_seq = jnp.zeros((n_steps, n_envs, 2))  # placeholder

        # Actually, let's do it properly: store obs and actions during rollout
        # (This is a simplified version — see full implementation below)

        # Compute returns
        returns = compute_returns(rewards, gamma)

        # For this simplified version, we'll use the log_probs collected
        # during rollout instead of recomputing them
        loss = -jnp.mean(log_probs * returns)

        # Gradient step (manual, since we didn't store obs/actions)
        # In a full implementation, you'd store the full (s, a) pairs
        # and call jit_update here. For brevity, we'll update with
        # a simpler approach.

        # Simple policy update using collected log_probs
        grad_fn = jax.grad(
            lambda p: -jnp.mean(
                jax.vmap(
                    lambda lp, r: lp * r
                )(log_probs, returns)
            )
        )
        # Actually, log_probs were computed under the old params,
        # so we can't differentiate through them. We need to recompute.
        # Let's use a proper implementation instead.

        # For now, just print progress
        mean_reward = float(jnp.mean(jnp.sum(rewards, axis=0)))
        max_reward = float(jnp.max(jnp.sum(rewards, axis=0)))

        if it % 20 == 0 or it == n_iterations - 1:
            print(f"  Iter {it:4d}  |  "
                  f"mean_return={mean_reward:8.1f}  |  "
                  f"max_return={max_reward:8.1f}  |  "
                  f"loss={float(loss):.3f}")

    return params


# ════════════════════════════════════════════════════════════════
# 6. Full Proper Implementation (stores obs/actions during rollout)
# ════════════════════════════════════════════════════════════════

def collect_rollout_full(key, params, env_states, cfg, n_steps):
    """Collect rollout storing all (s, a, r) pairs for training."""
    n_envs = env_states.state.shape[0]
    target_y = 50.0

    all_obs = []
    all_actions = []
    all_rewards = []
    all_log_probs = []

    for t in range(n_steps):
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, n_envs)

        obs = env_states.state
        obs_full = jnp.concatenate(
            [obs, (target_y - obs[:, 1:2])], axis=1
        )

        actions, log_probs = batch_sample_action(keys, params, obs_full)

        env_states, (_, rewards, _) = batch_step(
            env_states, actions, cfg
        )

        all_obs.append(obs_full)
        all_actions.append(actions)
        all_rewards.append(rewards)
        all_log_probs.append(log_probs)

    return (
        jnp.stack(all_obs, axis=0),       # (T, N, 5)
        jnp.stack(all_actions, axis=0),   # (T, N, 2)
        jnp.stack(all_rewards, axis=0),   # (T, N)
    )


@jax.jit
def train_step(params, obs_seq, actions_seq, returns_seq, lr):
    """One REINFORCE gradient step (JIT-compiled)."""
    def loss_fn(p):
        def single_lp(obs, action):
            mean, std = policy_forward_proper(p, obs)
            return jnp.sum(
                -0.5 * ((action - mean) / std) ** 2
                - jnp.log(std) - 0.5 * jnp.log(2 * jnp.pi)
            )
        log_probs = jax.vmap(
            jax.vmap(single_lp)
        )(obs_seq, actions_seq)  # (T, N)
        return -jnp.mean(log_probs * returns_seq)

    grads = jax.grad(loss_fn)(params)
    new_params = [(W - lr * gW, b - lr * gb)
                   for (W, b), (gW, gb) in zip(params, grads)]
    loss_val = loss_fn(params)
    return new_params, loss_val


def train_full(n_envs=500, n_steps=200, n_iterations=200,
               lr=1e-3, gamma=0.99):
    cfg = SimConfig()

    key = jax.random.PRNGKey(42)
    key, init_key = jax.random.split(key)
    params = init_mlp_params(init_key, [5, 64, 64, 2], std=0.1)

    keys = jax.random.split(key, n_envs)
    env_states = batch_reset(keys, cfg, n_envs)

    print(f"Training REINFORCE (full): {n_envs} envs × "
          f"{n_steps} steps × {n_iterations} iters")
    print(f"  Total env-steps: {n_envs * n_steps * n_iterations:,}")
    print()

    for it in range(n_iterations):
        key, rollout_key = jax.random.split(key)

        obs_seq, actions_seq, rewards = collect_rollout_full(
            rollout_key, params, env_states, cfg, n_steps
        )

        returns = compute_returns(rewards, gamma)

        params, loss = train_step(
            params, obs_seq, actions_seq, returns, lr
        )

        mean_ret = float(jnp.mean(jnp.sum(rewards, axis=0)))
        max_ret = float(jnp.max(jnp.sum(rewards, axis=0)))

        if it % 20 == 0 or it == n_iterations - 1:
            y_mean = float(jnp.mean(env_states.state[:, 1]))
            print(f"  Iter {it:4d}  |  "
                  f"mean_ret={mean_ret:8.1f}  |  "
                  f"max_ret={max_ret:8.1f}  |  "
                  f"loss={float(loss):.3f}  |  "
                  f"mean_y={y_mean:.1f}m")

    print("\nTraining complete.")
    print(f"  Final mean altitude: "
          f"{float(jnp.mean(env_states.state[:, 1])):.1f}m")
    print(f"  Target altitude: 50.0m")

    return params, env_states


# ════════════════════════════════════════════════════════════════
# Entry Point
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_envs", type=int, default=500)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    t0 = time.time()
    params, env_states = train_full(
        n_envs=args.n_envs,
        n_steps=args.steps,
        n_iterations=args.iters,
    )
    dt = time.time() - t0
    total = args.n_envs * args.steps * args.iters
    print(f"\nTotal: {total:,} env-steps in {dt:.1f}s "
          f"({total/dt:,.0f} steps/s)")
