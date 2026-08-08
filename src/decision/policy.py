"""
decision/policy.py — 策略网络定义

对应技术方案 §4.4.1。

使用 JAX/Flax 实现策略网络：
  - 共享特征提取 → 策略头（动作均值+方差）
  - 价值头（状态价值估计）
  - 安全价值头（安全约束代价估计）

如果 Flax 不可用，回退到纯 JAX MLP。
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple, Tuple


# ════════════════════════════════════════════════════════════════
# 策略参数容器
# ════════════════════════════════════════════════════════════════

class PolicyParams(NamedTuple):
    """策略网络参数"""
    # 共享特征层
    w1: jnp.ndarray   # [obs_dim, hidden]
    b1: jnp.ndarray   # [hidden]
    w2: jnp.ndarray   # [hidden, hidden]
    b2: jnp.ndarray   # [hidden]

    # 策略头
    w_pi_mean: jnp.ndarray  # [hidden, act_dim]
    b_pi_mean: jnp.ndarray  # [act_dim]
    w_pi_log_std: jnp.ndarray  # [hidden, act_dim]
    b_pi_log_std: jnp.ndarray  # [act_dim]

    # 价值头
    w_v: jnp.ndarray  # [hidden, 1]
    b_v: jnp.ndarray  # [1]

    # 安全价值头 (CMDP 约束代价)
    w_s: jnp.ndarray  # [hidden, 1]
    b_s: jnp.ndarray  # [1]


def init_policy(rng,
               obs_dim: int = 6, act_dim: int = 2,
               hidden: int = 64) -> PolicyParams:
    """Xavier 初始化策略参数"""
    def xavier(key, shape, fan_in, fan_out):
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return jax.random.uniform(key, shape,
                                   minval=-limit, maxval=limit)

    keys = jax.random.split(rng, 10)

    w1 = xavier(keys[0], (obs_dim, hidden), obs_dim, hidden)
    b1 = jnp.zeros(hidden)
    w2 = xavier(keys[1], (hidden, hidden), hidden, hidden)
    b2 = jnp.zeros(hidden)

    w_pi_mean = xavier(keys[2], (hidden, act_dim), hidden, act_dim)
    b_pi_mean = jnp.zeros(act_dim)
    w_pi_log_std = xavier(keys[3], (hidden, act_dim), hidden, act_dim)
    b_pi_log_std = jnp.log(jnp.ones(act_dim) * 0.5)

    w_v = xavier(keys[4], (hidden, 1), hidden, 1)
    b_v = jnp.zeros(1)

    w_s = xavier(keys[5], (hidden, 1), hidden, 1)
    b_s = jnp.zeros(1)

    return PolicyParams(w1=w1, b1=b1, w2=w2, b2=b2,
                         w_pi_mean=w_pi_mean, b_pi_mean=b_pi_mean,
                         w_pi_log_std=w_pi_log_std, b_pi_log_std=b_pi_log_std,
                         w_v=w_v, b_v=b_v, w_s=w_s, b_s=b_s)


# ════════════════════════════════════════════════════════════════
# 前向传播
# ════════════════════════════════════════════════════════════════

def forward_features(params: PolicyParams, obs: jnp.ndarray) -> jnp.ndarray:
    """共享特征层"""
    h = jnp.tanh(obs @ params.w1 + params.b1)
    h = jnp.tanh(h @ params.w2 + params.b2)
    return h


def policy_forward(params: PolicyParams, obs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    策略前向：返回动作分布参数
    Returns: (action_mean, action_std)
    """
    h = forward_features(params, obs)
    mean = jnp.tanh(h @ params.w_pi_mean + params.b_pi_mean)  # [-1, 1]
    log_std = h @ params.w_pi_log_std + params.b_pi_log_std
    log_std = jnp.clip(log_std, -2.0, 1.0)  # 限制方差范围
    std = jnp.exp(log_std)
    return mean, std


def value_forward(params: PolicyParams, obs: jnp.ndarray) -> jnp.ndarray:
    """价值函数前向"""
    h = forward_features(params, obs)
    v = h @ params.w_v + params.b_v
    return v.squeeze()


def safety_value_forward(params: PolicyParams, obs: jnp.ndarray) -> jnp.ndarray:
    """安全约束代价函数前向"""
    h = forward_features(params, obs)
    s = h @ params.w_s + params.b_s
    return s.squeeze()


def sample_action(params: PolicyParams, obs: jnp.ndarray,
                  rng,
                  deterministic: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    从策略中采样动作
    Returns: (action, log_prob)
    """
    mean, std = policy_forward(params, obs)

    if deterministic:
        return mean, jnp.array(0.0)

    noise = jax.random.normal(rng, mean.shape)
    action = mean + std * noise
    action = jnp.clip(action, -1.0, 1.0)

    # 对数概率（高斯）
    log_prob = jnp.sum(
        -0.5 * ((action - mean) / (std + 1e-8))**2
        - jnp.log(std + 1e-8) - 0.5 * jnp.log(2 * jnp.pi)
    )
    return action, log_prob


# ════════════════════════════════════════════════════════════════
# 测试
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    rng = jax.random.PRNGKey(42)
    params = init_policy(rng, obs_dim=6, act_dim=2, hidden=64)

    obs = jnp.array([5.0, 40.0, 3.0, -2.0, 10.0, 0.5])
    action, log_prob = sample_action(params, obs, rng)
    value = value_forward(params, obs)
    safety = safety_value_forward(params, obs)

    print("=== 策略网络测试 ===")
    print(f"Observation: {obs}")
    print(f"Action:      {action}  (thrust_norm, angle_norm)")
    print(f"Log prob:    {log_prob:.4f}")
    print(f"Value:       {value:.4f}")
    print(f"Safety cost: {safety:.4f}")

    # vmap 批量推理
    batch_obs = jnp.tile(obs, (100, 1))
    vmap_sample = jax.vmap(sample_action, in_axes=(None, 0, None))
    batch_keys = jax.random.split(rng, 100)
    actions, _ = vmap_sample(params, batch_obs, batch_keys)
    print(f"\nBatch (100): mean={jnp.mean(actions, axis=0)}, "
          f"std={jnp.std(actions, axis=0)}")
