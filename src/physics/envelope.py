"""
physics/envelope.py — 可行运动包络 (Feasible Motion Envelope, FME)

对应技术方案 §4.3.3 Eq 19。

FME 通过前向仿真网格，计算在当前故障状态 B(t) 下
未来 N 步内所有可达状态的包络。用于：
  - 安全约束的保守性量化
  - 路径规划的空间可行性预筛
  - CBF 安全集的构造
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from typing import NamedTuple
from .dynamics import faulty_physics_step


class FME(NamedTuple):
    """可行运动包络结果"""
    reachable_x: jnp.ndarray   # [N_ctrl, N_steps] 可达 x
    reachable_y: jnp.ndarray   # [N_ctrl, N_steps] 可达 y
    min_x: float
    max_x: float
    min_y: float
    max_y: float


def compute_fme(state: jnp.ndarray, fault: jnp.ndarray, params: tuple,
                 n_steps: int = 50, n_controls: int = 8,
                 max_thrust: float = 50.0) -> FME:
    """
    计算可行运动包络 (Eq 19)

    在控制空间网格上展开，前向仿真 N 步，
    取所有可达状态的 min/max 包络。

    Args:
        state:     [x, y, vx, vy, energy]
        fault:     [φ, δ, asym]
        params:    物理参数
        n_steps:   前向仿真步数
        n_controls: 控制方向离散化数
        max_thrust: 最大推力

    Returns:
        FME 包络
    """
    # 离散化控制方向（极坐标网格）
    angles = jnp.linspace(0, 2 * jnp.pi, n_controls, endpoint=False)

    # 对每个控制方向前向仿真
    def simulate_one(angle):
        ctrl = jnp.array([max_thrust * jnp.cos(angle),
                          max_thrust * jnp.sin(angle)])

        def scan_fn(carry, _):
            s = carry
            s_new = faulty_physics_step(s, ctrl, fault, params)
            return s_new, s_new

        _, traj = jax.lax.scan(scan_fn, state, None, length=n_steps)
        return traj[:, 0], traj[:, 1]  # x, y 轨迹

    # vmap 跨所有控制方向
    all_x, all_y = jax.vmap(simulate_one)(angles)

    return FME(
        reachable_x=all_x,
        reachable_y=all_y,
        min_x=float(jnp.min(all_x)),
        max_x=float(jnp.max(all_x)),
        min_y=float(jnp.min(all_y)),
        max_y=float(jnp.max(all_y)),
    )


def is_landing_reachable(fme: FME, target_x: float, target_y: float,
                          margin: float = 5.0) -> bool:
    """
    判断降落点是否在 FME 包络内
    """
    return (fme.min_x - margin <= target_x <= fme.max_x + margin and
            fme.min_y - margin <= target_y <= fme.max_y + margin)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '..')
    from config import SystemConfig
    from dynamics import pack_params

    cfg = SystemConfig()
    params = pack_params(cfg)

    state = jnp.array([0.0, 40.0, 5.0, 0.0, 1000.0])
    fault = jnp.array([0.0, 0.0, 0.0])  # 无故障

    fme = compute_fme(state, fault, params, n_steps=50, n_controls=16)
    print(f"FME (no fault): x=[{fme.min_x:.1f}, {fme.max_x:.1f}], "
          f"y=[{fme.min_y:.1f}, {fme.max_y:.1f}]")

    # 50%动力降级
    fault_deg = jnp.array([0.0, 0.5, 0.0])
    fme_deg = compute_fme(state, fault_deg, params, n_steps=50, n_controls=16)
    print(f"FME (50% deg):  x=[{fme_deg.min_x:.1f}, {fme_deg.max_x:.1f}], "
          f"y=[{fme_deg.min_y:.1f}, {fme_deg.max_y:.1f}]")

    # 严重故障
    fault_sev = jnp.array([0.5, 0.7, 0.5])
    fme_sev = compute_fme(state, fault_sev, params, n_steps=50, n_controls=16)
    print(f"FME (severe):   x=[{fme_sev.min_x:.1f}, {fme_sev.max_x:.1f}], "
          f"y=[{fme_sev.min_y:.1f}, {fme_sev.max_y:.1f}]")
