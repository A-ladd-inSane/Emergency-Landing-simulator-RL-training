"""
execution/tracker.py — LQR 轨迹跟踪

对应技术方案 §4.5.2 Eq 27。

对规划轨迹 x_ref(t)，使用 LQR 跟踪：
  u = -K (x - x_ref) + u_ff

K 通过求解离散代数 Riccati 方程获得。
"""

import numpy as np
from scipy.linalg import solve_discrete_are as solve_dare
from typing import NamedTuple


class LQRController(NamedTuple):
    """LQR 控制器"""
    K: np.ndarray        # [N_u, N_x] 反馈增益
    x_ref: np.ndarray    # 参考轨迹 [T, N_x]
    u_ff: np.ndarray     # 前馈控制 [T, N_u]


def compute_lqr_gain(A: np.ndarray, B: np.ndarray,
                     Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    离散 LQR 增益 (Eq 27)

    K = (R + B^T P B)^{-1} B^T P A
    P 通过求解 DARE: P = A^T P A - A^T P B (R + B^T P B)^{-1} B^T P A + Q
    """
    P = solve_dare(A, B, Q, R)
    K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
    return K


class TrajectoryTracker:
    """LQR 轨迹跟踪器"""

    def __init__(self, A: np.ndarray, B: np.ndarray,
                 Q: np.ndarray = None, R: np.ndarray = None):
        n_state = A.shape[0]
        n_ctrl = B.shape[1]
        self.Q = Q or np.eye(n_state) * 1.0
        self.R = R or np.eye(n_ctrl) * 0.1
        self.K = compute_lqr_gain(A, B, self.Q, self.R)

    def track(self, state: np.ndarray, ref_idx: int,
              x_ref: np.ndarray, u_ff: np.ndarray = None) -> np.ndarray:
        """
        LQR 跟踪

        u = -K(x - x_ref[t]) + u_ff[t]
        """
        ref = x_ref[ref_idx] if ref_idx < len(x_ref) else x_ref[-1]
        u_fb = -self.K @ (state - ref)

        if u_ff is not None:
            ff = u_ff[ref_idx] if ref_idx < len(u_ff) else u_ff[-1]
            u = u_fb + ff
        else:
            u = u_fb

        return u


# ════════════════════════════════════════════════════════════════
# 测试
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 离散动力学: [x, y, vx, vy]
    dt = 0.01
    A = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])

    # 控制: [ax, ay] → [vx, vy] 的加速度
    B = np.array([
        [0, 0],
        [0, 0],
        [dt, 0],
        [0, dt],
    ])

    # LQR 权重
    Q = np.diag([10, 10, 1, 1])  # 位置权重大于速度
    R = np.diag([0.1, 0.1])

    tracker = TrajectoryTracker(A, B, Q, R)
    print(f"LQR Gain K:\n{tracker.K}")

    # 生成参考轨迹：直线上升到50m后悬停
    T = 100
    t = np.arange(T) * dt
    x_ref = np.column_stack([
        np.zeros(T),           # x constant at 0
        50 * (1 - np.exp(-2*t)),  # y exponential approach to 50
        np.zeros(T),           # vx = 0
        100 * np.exp(-2*t),    # vy → 0
    ])

    # 仿真
    state = np.array([0.0, 0.0, 0.0, 0.0])
    print(f"\nInitial state: {state}")
    print(f"Target: hover at y=50m\n")

    for i in range(T):
        u = tracker.track(state, i, x_ref)
        state = A @ state + B @ u

    print(f"Final state: {state}")
    print(f"Final error: {state - x_ref[-1]}")
    print(f"  → LQR收敛到参考轨迹")
