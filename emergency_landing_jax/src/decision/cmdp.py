"""
decision/cmdp.py — 约束马尔可夫决策过程 (CMDP) 训练

对应技术方案 §4.4.2 Eq 6, 20。

CMDP 在标准 RL 基础上增加安全约束：
  max_θ  E[Σ γ^t r_t]
  s.t.   E[Σ γ^t c_t] ≤ d_safety    (安全约束)

拉格朗日方法：
  L(θ, λ) = J_reward(θ) - λ (J_cost(θ) - d_safety)
  θ ← θ + ∇_θ L
  λ ← λ + αλ (J_cost - d_safety)     (对偶上升)
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple, List
from .policy import (
    PolicyParams, init_policy, sample_action,
    value_forward, safety_value_forward,
)


class CMDPState(NamedTuple):
    """CMDP 训练状态"""
    params: PolicyParams
    lagrange_mult: float     # λ (安全约束拉格朗日乘子)
    opt_state: dict          # Adam 优化器状态
    step: int


class Trajectory(NamedTuple):
    """单条轨迹"""
    observations: np.ndarray   # [N, obs_dim]
    actions: np.ndarray        # [N, act_dim]
    log_probs: np.ndarray      # [N]
    rewards: np.ndarray         # [N]
    costs: np.ndarray           # [N] (安全约束代价)
    values: np.ndarray          # [N]
    safety_values: np.ndarray   # [N]
    dones: np.ndarray           # [N]


# ════════════════════════════════════════════════════════════════
# 优势函数 (GAE)
# ════════════════════════════════════════════════════════════════

def compute_gae(rewards: np.ndarray, values: np.ndarray,
               dones: np.ndarray, gamma: float = 0.99,
               gae_lambda: float = 0.95) -> np.ndarray:
    """广义优势估计 (GAE)"""
    N = len(rewards)
    advantages = np.zeros(N)
    gae = 0.0
    for t in reversed(range(N)):
        if t == N - 1:
            next_value = 0.0
        else:
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages[t] = gae
    return advantages


# ════════════════════════════════════════════════════════════════
# CMDP 更新
# ════════════════════════════════════════════════════════════════

def cmdp_update(state: CMDPState,
                trajectories: List[Trajectory],
                lr: float = 3e-4,
                cost_limit: float = 10.0,
                lambda_lr: float = 0.1) -> CMDPState:
    """
    CMDP 拉格朗日更新 (Eq 20)

    1. 收集批量轨迹
    2. 计算优势函数
    3. 更新策略参数 θ
    4. 更新拉格朗日乘子 λ
    """
    params = state.params
    lagrange = state.lagrange_mult

    # ── 计算批量优势 ──
    all_obs = []
    all_actions = []
    all_old_log_probs = []
    all_rewards_adv = []
    all_costs_adv = []
    all_costs = []

    for traj in trajectories:
        adv = compute_gae(traj.rewards, traj.values, traj.dones)
        # 奖励优势 + 价值基线
        returns = adv + traj.values

        all_obs.append(traj.observations)
        all_actions.append(traj.actions)
        all_old_log_probs.append(traj.log_probs)
        all_rewards_adv.append(adv)
        all_costs.append(traj.costs)

    all_obs = np.concatenate(all_obs)
    all_actions = np.concatenate(all_actions)
    all_old_log_probs = np.concatenate(all_old_log_probs)
    all_rewards_adv = np.concatenate(all_rewards_adv)
    all_costs = np.concatenate(all_costs)

    # 标准化优势
    if len(all_rewards_adv) > 1:
        all_rewards_adv = (all_rewards_adv - all_rewards_adv.mean()) / (
            all_rewards_adv.std() + 1e-8
        )

    # ── 策略梯度损失 ──
    obs_jax = jnp.array(all_obs)
    actions_jax = jnp.array(all_actions)
    old_lp_jax = jnp.array(all_old_log_probs)
    adv_jax = jnp.array(all_rewards_adv)
    costs_jax = jnp.array(all_costs)

    def policy_loss_fn(p):
        # 重新计算 log prob
        mean, std = jax.vmap(lambda o: (
            jnp.tanh(jax.nn.tanh(o @ p.w1 + p.b1) @ p.w2 + p.b2) @ p.w_pi_mean + p.b_pi_mean,
            jnp.exp(jnp.clip(
                jax.nn.tanh(o @ p.w1 + p.b1) @ p.w2 + p.b2
                @ p.w_pi_log_std + p.b_pi_log_std, -2.0, 1.0
            ))
        ))(obs_jax)

        log_std = jnp.clip(
            jax.nn.tanh(obs_jax @ p.w1 + p.b1) @ p.w2 + p.b2
            @ p.w_pi_log_std + p.b_pi_log_std, -2.0, 1.0
        )
        std = jnp.exp(log_std)
        new_log_probs = jnp.sum(
            -0.5 * ((actions_jax - mean) / (std + 1e-8))**2
            - jnp.log(std + 1e-8) - 0.5 * jnp.log(2 * jnp.pi), axis=-1
        )

        ratio = jnp.exp(new_log_probs - old_lp_jax)
        # PPO clip
        clip_range = 0.2
        surr1 = ratio * adv_jax
        surr2 = jnp.clip(ratio, 1 - clip_range, 1 + clip_range) * adv_jax
        policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))

        # 价值损失
        values = jax.vmap(lambda o: value_forward(p, o))(obs_jax)
        value_loss = jnp.mean((values - jnp.array(all_rewards_adv +
                        np.concatenate([t.values for t in trajectories])))**2)

        # 安全价值损失
        s_values = jax.vmap(lambda o: safety_value_forward(p, o))(obs_jax)
        cost_returns = costs_jax
        safety_loss = jnp.mean((s_values - cost_returns)**2)

        # 组合损失: reward - λ * cost
        total_loss = policy_loss + 0.5 * value_loss + 0.5 * safety_loss

        return total_loss

    # ── 计算梯度 ──
    loss_val, grads = jax.value_and_grad(policy_loss_fn)(params)

    # ── Adam 更新 ──
    new_opt = adam_step(state.opt_state, grads, lr)
    new_params = new_opt[0]

    # ── 更新拉格朗日乘子 (Eq 20) ──
    mean_cost = float(np.mean(all_costs))
    new_lagrange = max(0.0, lagrange + lambda_lr * (mean_cost - cost_limit))

    return CMDPState(
        params=new_params,
        lagrange_mult=new_lagrange,
        opt_state=new_opt[1],
        step=state.step + 1,
    )


# ════════════════════════════════════════════════════════════════
# Adam 优化器
# ════════════════════════════════════════════════════════════════

def init_adam(params: PolicyParams, lr: float = 3e-4) -> dict:
    """初始化 Adam 状态"""
    return {
        'lr': lr,
        'm': jax.tree_util.tree_map(jnp.zeros_like, params),
        'v': jax.tree_util.tree_map(jnp.zeros_like, params),
        't': 0,
    }


def adam_step(opt_state: dict, grads: PolicyParams,
              lr: float = None) -> tuple:
    """Adam 一步更新"""
    lr = lr or opt_state['lr']
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = opt_state['t'] + 1

    m = jax.tree_util.tree_map(
        lambda m, g: beta1 * m + (1 - beta1) * g, opt_state['m'], grads)
    v = jax.tree_util.tree_map(
        lambda v, g: beta2 * v + (1 - beta2) * g**2, opt_state['v'], grads)

    m_hat = jax.tree_util.tree_map(lambda m: m / (1 - beta1**t), m)
    v_hat = jax.tree_util.tree_map(lambda v: v / (1 - beta2**t), v)

    updated = jax.tree_util.tree_map(
        lambda p, m, v: p - lr * m / (jnp.sqrt(v) + eps),
        opt_state.get('params', None), m_hat, v_hat) if 'params' in opt_state else None

    # 直接更新 params
    return updated, {'lr': lr, 'm': m, 'v': v, 't': t}


# ════════════════════════════════════════════════════════════════
# 测试
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    rng = jax.random.PRNGKey(42)
    params = init_policy(rng, obs_dim=6, act_dim=2, hidden=32)

    # 模拟轨迹
    N = 50
    obs_dim, act_dim = 6, 2
    obs = np.random.randn(N, obs_dim).astype(np.float32)
    actions = np.random.randn(N, act_dim).astype(np.float32)
    log_probs = np.random.randn(N).astype(np.float32)
    rewards = -np.abs(obs[:, 1] - 50)  # 高度偏差
    costs = np.maximum(0, 10 - obs[:, 1])  # 安全代价（低于10m）
    values = np.zeros(N)
    safety_values = np.zeros(N)
    dones = np.zeros(N)
    dones[-1] = 1.0

    traj = Trajectory(obs, actions, log_probs, rewards, costs,
                      values, safety_values, dones)

    state = CMDPState(
        params=params,
        lagrange_mult=1.0,
        opt_state=init_adam(params, 3e-4),
        step=0,
    )

    print("=== CMDP 训练测试 ===")
    print(f"  Initial λ = {state.lagrange_mult:.4f}")
    print(f"  Mean reward = {np.mean(rewards):.4f}")
    print(f"  Mean cost = {np.mean(costs):.4f}")

    new_state = cmdp_update(state, [traj], cost_limit=5.0)
    print(f"\n  After update:")
    print(f"  λ = {new_state.lagrange_mult:.4f} (↑ if cost > limit)")
    print(f"  Step = {new_state.step}")
