"""
perception/endurance.py — 剩余可飞时间估计

对应技术方案 §4.3.2 Eq 5。

T_remain(s, π_θ) = E[Σ_t γ^t | s, π_θ]

基于当前状态和策略，蒙特卡洛估计剩余可飞时间。
简化版：用能量模型解析估计。
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class EnduranceEstimate:
    """续航估计结果"""
    t_remain: float       # 剩余可飞时间 s
    range_x: float       # 水平可达距离 m
    range_y: float       # 垂直可达距离 m
    energy: float        # 剩余能量 J
    confidence: float


def estimate_endurance(state: np.ndarray, fault: np.ndarray,
                       mass: float = 1.0, g: float = 9.81,
                       max_thrust: float = 50.0) -> EnduranceEstimate:
    """
    估计剩余可飞时间 (Eq 5)

    简化模型：
      能量 = 剩余电池能量 - 已消耗
      悬停功耗 P_hover = (mg) / (B_eff * η) * v_hover
      前飞功耗 ≈ P_hover * (1 + v²/v_hover²) / 2

    T_remain = Energy / P_current
    """
    x, y, vx, vy, energy = state
    phi, delta, asym = fault

    # 当前控制有效性
    B_eff = 1.0 - delta  # 动力降级

    # 悬停推力
    thrust_hover = mass * g

    # 当前速度
    speed = np.sqrt(vx**2 + vy**2)

    # 功率估计（简化）
    P_hover = thrust_hover / max(B_eff, 0.1)  # 退化后功率上升
    v_ref = 10.0  # 参考速度
    P_current = P_hover * (1 + (speed / v_ref)**2) / 2

    # 伞扰动额外功耗
    if phi > 0:
        # 伞失效导致阻力增加
        P_drag = 0.5 * phi * speed**2 * 0.05
        P_current += P_drag

    # 剩余时间
    t_remain = energy / max(P_current, 1.0)

    # 可达范围
    range_x = speed * t_remain * 0.7  # 0.7=转向损耗
    range_y = (thrust_hover / max(B_eff, 0.1) - mass * g) * t_remain**2 / (2 * mass)

    return EnduranceEstimate(
        t_remain=float(t_remain),
        range_x=float(abs(range_x)),
        range_y=float(range_y),
        energy=float(energy),
        confidence=0.8,
    )


if __name__ == "__main__":
    state = np.array([0.0, 40.0, 5.0, 0.0, 1000.0])

    # 正常
    e1 = estimate_endurance(state, np.array([0, 0, 0]))
    print(f"正常:  T_remain={e1.t_remain:.1f}s  "
          f"range_x={e1.range_x:.1f}m  energy={e1.energy:.0f}J")

    # 50%动力降级
    e2 = estimate_endurance(state, np.array([0, 0.5, 0]))
    print(f"50%δ: T_remain={e2.t_remain:.1f}s  "
          f"range_x={e2.range_x:.1f}m  energy={e2.energy:.0f}J")

    # 严重故障
    e3 = estimate_endurance(state, np.array([0.5, 0.7, 0.3]))
    print(f"严重:  T_remain={e3.t_remain:.1f}s  "
          f"range_x={e3.range_x:.1f}m  energy={e3.energy:.0f}J")
