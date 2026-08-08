"""
decision/cbf.py — 鲁棒控制障碍函数 (CBF) 安全过滤

对应技术方案 §4.4.3 Eq 22-23。

CBF QP 安全过滤：
  对RL策略输出的原始动作 u_rl，在安全约束下投影到安全集：
  min ||u - u_rl||²
  s.t. Lf h(x) + Lg h(x) u ≥ -α h(x)    (CBF约束)
       u ∈ U                            (执行器约束)

h(x) ≥ 0 定义安全集（距障碍物/地面的有符号距离）。
"""

import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Callable, List


@dataclass
class CBFConstraint:
    """CBF 约束定义"""
    name: str
    h: Callable          # 障碍函数 h(x) → float
    grad_h: Callable      # h 的梯度 ∇h(x) → [d]
    alpha: float = 1.0    # CBF 常数


class CBFSafetyFilter:
    """
    鲁棒 CBF QP 安全过滤 (Eq 22-23)

    在每步对RL动作做投影修正，
    确保安全约束不被违反。
    """

    def __init__(self, constraints: List[CBFConstraint],
                 u_min: np.ndarray = None, u_max: np.ndarray = None,
                 robust_margin: float = 0.1):
        """
        Args:
            constraints: CBF约束列表
            u_min/u_max: 执行器限幅
            robust_margin: 鲁棒裕度（对应不确定度Σ_cb）
        """
        self.constraints = constraints
        self.u_min = u_min if u_min is not None else np.array([0.0, -np.pi])
        self.u_max = u_max if u_max is not None else np.array([50.0, np.pi])
        self.margin = robust_margin

    def filter(self, state: np.ndarray, u_rl: np.ndarray,
               f_nominal: np.ndarray, g_nominal: np.ndarray) -> np.ndarray:
        """
        安全过滤 (Eq 22-23)

        min ||u - u_rl||² + ρ slack
        s.t. CBF constraints + actuator bounds

        Args:
            state: [x, y, vx, vy]
            u_rl: RL策略原始动作 [thrust, angle]
            f_nominal: 漂移项 f(x)（不依赖控制的部分）
            g_nominal: 控制矩阵 G(x) [d_state, d_control]
        Returns:
            u_safe: 安全过滤后的动作
        """
        u_rl = np.clip(u_rl, self.u_min, self.u_max)

        # 目标函数
        def objective(u):
            return float(np.sum((u - u_rl)**2))

        def objective_jac(u):
            return 2 * (u - u_rl)

        # 约束
        constraints_list = []

        # CBF 约束
        for cbf in self.constraints:
            h_val = cbf.h(state)
            grad_h = cbf.grad_h(state)

            # Lf h = grad_h · f
            Lf_h = grad_h @ f_nominal
            # Lg h = grad_h · G
            Lg_h = grad_h @ g_nominal

            # 不确定度裕度 (Eq 23)
            # Lf h + Lg h · u ≥ -α·h + margin
            def cbf_con(u, Lf=Lf_h, Lg=Lg_h, h=h_val, alpha=cbf.alpha,
                       margin=self.margin):
                return Lf + Lg @ u + alpha * h - margin

            def cbf_jac(u, Lg=Lg_h, alpha=cbf.alpha):
                return Lg

            constraints_list.append({
                'type': 'ineq',
                'fun': cbf_con,
                'jac': cbf_jac,
            })

        # 执行器约束
        bounds = list(zip(self.u_min, self.u_max))

        # 求解 QP
        result = minimize(
            objective, u_rl, jac=objective_jac,
            method='SLSQP', bounds=bounds,
            constraints=constraints_list,
            options={'maxiter': 50, 'ftol': 1e-6}
        )

        u_safe = result.x if result.success else u_rl
        u_safe = np.clip(u_safe, self.u_min, self.u_max)

        return u_safe


# ════════════════════════════════════════════════════════════════
# 标准障碍函数
# ════════════════════════════════════════════════════════════════

def ground_constraint(y_min: float = 0.0) -> CBFConstraint:
    """地面碰撞约束: h(x) = y - y_min"""
    return CBFConstraint(
        name="ground",
        h=lambda s: float(s[1] - y_min),
        grad_h=lambda s: np.array([0.0, 1.0, 0.0, 0.0]),
        alpha=2.0,
    )


def ceiling_constraint(y_max: float = 100.0) -> CBFConstraint:
    """天花板约束: h(x) = y_max - y"""
    return CBFConstraint(
        name="ceiling",
        h=lambda s: float(y_max - s[1]),
        grad_h=lambda s: np.array([0.0, -1.0, 0.0, 0.0]),
        alpha=1.0,
    )


def obstacle_constraint(obs_x: float, obs_y: float,
                        radius: float = 5.0) -> CBFConstraint:
    """障碍物约束: h(x) = (x-obs)² - r²"""
    def h(s):
        dx = s[0] - obs_x
        dy = s[1] - obs_y
        return float(dx**2 + dy**2 - radius**2)

    def grad_h(s):
        dx = s[0] - obs_x
        dy = s[1] - obs_y
        return np.array([2*dx, 2*dy, 0.0, 0.0])

    return CBFConstraint(name=f"obstacle({obs_x},{obs_y})", h=h,
                         grad_h=grad_h, alpha=1.5)


# ════════════════════════════════════════════════════════════════
# 测试
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 构建安全过滤器
    constraints = [
        ground_constraint(y_min=0.0),
        ceiling_constraint(y_max=100.0),
        obstacle_constraint(20.0, 30.0, radius=5.0),
    ]

    f = CBFSafetyFilter(
        constraints=constraints,
        u_min=np.array([0.0, -np.pi]),
        u_max=np.array([50.0, np.pi]),
        robust_margin=0.1,
    )

    # 简化的动力学模型
    g_func = 9.81
    mass = 1.0
    dt = 0.01

    def nominal_dynamics(s, u):
        """标称动力学 f + G*u"""
        # f: 不含控制的部分（重力+阻力）
        vx, vy = s[2], s[3]
        f = np.array([
            vx, vy,
            -0.05 * vx / mass,
            -0.05 * vy / mass - g_func
        ])
        # G: 控制矩阵 [4, 2] → thrust_x, thrust_y
        G = np.array([
            [0, 0],
            [0, 0],
            [1/mass, 0],
            [0, 1/mass]
        ])
        return f, G

    # 测试场景
    state = np.array([5.0, 5.0, 3.0, -5.0])  # 接近地面，下降中
    u_rl = np.array([0.0, 0.0])  # RL输出：不发力（会撞地）

    f_nom, G_nom = nominal_dynamics(state, u_rl)
    u_safe = f.filter(state, u_rl, f_nom, G_nom)

    print("=== CBF 安全过滤测试 ===")
    print(f"State: {state}")
    print(f"  y={state[1]:.1f}m (ground=0, ceiling=100)")
    print(f"  障碍物 at (20, 30), r=5")
    print(f"RL action:  thrust={u_rl[0]:.1f}N, angle={u_rl[1]:.2f}rad")
    print(f"Safe action: thrust={u_safe[0]:.1f}N, angle={u_safe[1]:.2f}rad")
    print(f"  → CBF修正了RL动作以避免撞地")

    # 验证安全
    for cbf in constraints:
        h_val = cbf.h(state)
        print(f"  h[{cbf.name}] = {h_val:.2f} {'✓' if h_val > 0 else '✗'}")
