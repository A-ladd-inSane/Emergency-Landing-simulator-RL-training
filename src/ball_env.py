#!/usr/bin/env python3
"""
ball_env.py — Gymnasium Environment wrapping the JAX Ball Simulation.

This is the exact same pattern as MuJoCo:
    MuJoCo:  mj_step(state, action)  →  Gym Env  →  SB3/RLlib
    Ours:    physics_step(state, action, cfg)  →  Gym Env  →  SB3/RLlib

The only difference: physics_step is JAX-jitted instead of C-backed.
From SB3's perspective, the Env is indistinguishable from a MuJoCo env.

Task: HOVER at target altitude (y=50m) using thrust + angle control.
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ball_sim_jax import physics_step, SimConfig


class BallHoverEnv(gym.Env):
    """
    Observation (5D):  [x, y, vx, vy, target_y - y]
    Action (2D):       [thrust_norm, angle_norm]  both in [-1, 1]
                       thrust_norm=-1 → 0N,  thrust_norm=+1 → MAX_THRUST
                       angle_norm=-1  → -π,   angle_norm=+1  → +π

    Reward:
        -w_alt * |y - target_y|        altitude error
        -w_fuel * thrust               fuel cost
        -w_vel * |v|                   velocity penalty (encourage hover)
        +w_alive * 1                   per-step alive bonus
        +crash_penalty                 if y <= 0

    Termination:
        y <= 0          crash
        |x| > BOUNDARY  out of bounds
        step >= MAX_STEPS  truncation (timeout)
    """

    metadata = {"render_modes": ["human"]}

    # ── Physics constants ──
    MAX_THRUST = 50.0       # N
    BOUNDARY   = 100.0      # m
    MAX_STEPS  = 1000       # 10 s at dt=0.01

    # ── Task ──
    TARGET_Y = 50.0        # m

    # ── Reward weights ──
    W_ALTITUDE  = 1.0
    W_FUEL      = 0.01
    W_VELOCITY  = 0.05
    W_ALIVE     = 0.1
    W_CRASH     = -100.0
    W_BOUNDARY  = -50.0

    def __init__(self, render_mode=None):
        super().__init__()
        self.cfg = SimConfig()

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        obs_lo = np.array(
            [-self.BOUNDARY, 0, -50, -50, -100], dtype=np.float32)
        obs_hi = np.array(
            [self.BOUNDARY, 200, 50, 50, 100], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=obs_lo, high=obs_hi, dtype=np.float32)

        self.render_mode = render_mode
        self.state = None
        self.step_count = 0
        self._fig = None

    # ── Helpers ──

    def _denormalize(self, action):
        """Map [-1,1] action → physical [thrust (N), angle (rad)]."""
        thrust = (action[0] + 1.0) / 2.0 * self.MAX_THRUST   # [0, 50]
        angle  =  action[1] * np.pi                            # [-π, π]
        return jnp.array([thrust, angle])

    def _obs(self):
        x, y, vx, vy = [float(v) for v in self.state]
        return np.array([x, y, vx, vy, self.TARGET_Y - y], dtype=np.float32)

    def _reward(self, action):
        x, y, vx, vy = [float(v) for v in self.state]
        speed  = float(jnp.sqrt(vx**2 + vy**2))
        thrust = (action[0] + 1.0) / 2.0 * self.MAX_THRUST

        return (
            -abs(y - self.TARGET_Y) * self.W_ALTITUDE
            - thrust                 * self.W_FUEL
            - speed                  * self.W_VELOCITY
            + self.W_ALIVE
        )

    # ── Gym API ──

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        x0  = self.np_random.uniform(-10, 10)
        y0  = self.np_random.uniform(5, 20)
        vx0 = self.np_random.uniform(-2, 2)
        vy0 = self.np_random.uniform(0, 5)
        self.state = jnp.array([x0, y0, vx0, vy0])
        self.step_count = 0
        return self._obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        ctrl = self._denormalize(action)

        # ── THE KEY LINE: JAX physics replaces MuJoCo's mj_step ──
        self.state = physics_step(self.state, ctrl, self.cfg)
        # ────────────────────────────────────────────────────────

        self.step_count += 1
        obs    = self._obs()
        reward = self._reward(action)

        x, y = float(self.state[0]), float(self.state[1])
        terminated = False
        truncated  = False

        if y <= 0.0:
            terminated = True
            reward += self.W_CRASH
        if abs(x) > self.BOUNDARY:
            terminated = True
            reward += self.W_BOUNDARY
        if self.step_count >= self.MAX_STEPS:
            truncated = True

        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode != "human":
            return
        import matplotlib.pyplot as plt
        if self._fig is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(10, 6))
        self._ax.clear()
        self._ax.set_xlim(-self.BOUNDARY, self.BOUNDARY)
        self._ax.set_ylim(0, 100)
        self._ax.set_aspect('equal')
        self._ax.axhline(self.TARGET_Y, color='green', ls='--', alpha=0.3, label='target')
        self._ax.axhline(0, color='brown', lw=2)
        x, y, vx, vy = [float(v) for v in self.state]
        self._ax.plot(x, y, 'ro', ms=10)
        self._ax.quiver(x, y, vx, vy, color='green', scale=20)
        self._ax.set_title(f'Step {self.step_count}  y={y:.1f}m')
        plt.pause(0.01)


# ── Quick smoke test ──
if __name__ == "__main__":
    env = BallHoverEnv()
    obs, _ = env.reset(seed=42)
    print(f"Obs shape: {obs.shape}  →  {obs}")
    print(f"Action space: {env.action_space}")

    total_r = 0
    for i in range(100):
        a = env.action_space.sample()
        obs, r, term, trunc, _ = env.step(a)
        total_r += r
        if term or trunc:
            print(f"  ended at step {i+1}: term={term} trunc={trunc}")
            break
    print(f"Random policy reward over {i+1} steps: {total_r:.1f}")
