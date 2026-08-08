"""
execution/control_alloc.py — 控制分配

对应技术方案 §4.5.1 Eq 26。

在故障条件下，将期望广义力 [Fx, Fy] 分配到可用执行器。
使用加权伪逆方法，考虑故障掩码 M(φ,δ)。
"""

import numpy as np
from typing import NamedTuple


class ControlAllocation(NamedTuple):
    """控制分配结果"""
    u_command: np.ndarray      # [N_actuators] 各执行器指令
    u_achieved: np.ndarray     # [F_x, F_y] 实际广义力
    saturation: np.ndarray     # [N_actuators] 饱和度 [0,1]
    feasible: bool            # 是否可行


def weighted_pseudo_inverse(B: np.ndarray,
                            weights: np.ndarray = None) -> np.ndarray:
    """
    加权伪逆 B†_W = W⁻¹ B^T (B W⁻¹ B^T)⁻¹

    Args:
        B: [2, N] 控制有效性矩阵
        weights: [N] 各执行器权重（故障后降低）
    Returns:
        B_pinv: [N, 2]
    """
    n = B.shape[1]
    if weights is None:
        weights = np.ones(n)

    W_inv = np.diag(1.0 / (weights + 1e-10))
    BW_inv = B @ W_inv
    inner = BW_inv @ B.T
    if np.abs(np.linalg.det(inner)) < 1e-10:
        # 退化情况：使用最小范数解
        return W_inv @ B.T @ np.linalg.pinv(inner)

    B_pinv = W_inv @ B.T @ np.linalg.inv(inner)
    return B_pinv


def allocate_control(desired_force: np.ndarray,
                     B: np.ndarray,
                     u_min: np.ndarray = None,
                     u_max: np.ndarray = None,
                     weights: np.ndarray = None) -> ControlAllocation:
    """
    控制分配 (Eq 26)

    Args:
        desired_force: [Fx, Fy] 期望广义力
        B: [2, N] 控制有效性矩阵 (可能已退化)
        u_min/u_max: 各执行器限幅
        weights: 各执行器健康度权重
    """
    n = B.shape[1]
    if u_min is None:
        u_min = np.zeros(n)
    if u_max is None:
        u_max = np.ones(n) * 50.0

    # 加权伪逆
    B_pinv = weighted_pseudo_inverse(B, weights)

    # 分配
    u_cmd = B_pinv @ desired_force

    # 饱和
    u_cmd_clipped = np.clip(u_cmd, u_min, u_max)
    saturation = np.abs(u_cmd_clipped / (u_max + 1e-10))

    # 实际广义力
    u_achieved = B @ u_cmd_clipped

    # 可行性检查
    force_error = np.linalg.norm(desired_force - u_achieved)
    feasible = force_error < 0.1 * np.linalg.norm(desired_force + 1e-10)

    return ControlAllocation(
        u_command=u_cmd_clipped,
        u_achieved=u_achieved,
        saturation=saturation,
        feasible=feasible,
    )


# ════════════════════════════════════════════════════════════════
# 测试
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 标称控制矩阵：2个推力器（左右各一）
    # B[0] = [Fx], B[1] = [Fy] for each actuator
    B_nominal = np.array([
        [1.0, 1.0],   # Fx from actuator 0, 1
        [0.0, 1.0],   # Fy from actuator 0, 1
    ])

    # 标称情况
    desired = np.array([10.0, 15.0])  # 10N right, 15N up
    result = allocate_control(desired, B_nominal)
    print("=== 标称控制分配 ===")
    print(f"  Desired: {desired}")
    print(f"  Command: {result.u_command}")
    print(f"  Achieved: {result.u_achieved}")
    print(f"  Saturat.: {result.saturation}")
    print(f"  Feasible: {result.feasible}")

    # 故障：执行器1功率降级50%
    B_fault = np.array([
        [1.0, 0.5],
        [0.0, 0.5],
    ])
    weights = np.array([1.0, 0.5])  # 健康度权重
    result_fault = allocate_control(desired, B_fault, weights=weights)
    print("\n=== 故障控制分配 (act1 @ 50%) ===")
    print(f"  Desired: {desired}")
    print(f"  Command: {result_fault.u_command}")
    print(f"  Achieved: {result_fault.u_achieved}")
    print(f"  Saturat.: {result_fault.saturation}")
    print(f"  Feasible: {result_fault.feasible}")

    # 严重故障：执行器1失效
    B_severe = np.array([
        [1.0, 0.0],
        [0.0, 0.0],
    ])
    weights_severe = np.array([1.0, 0.01])
    result_severe = allocate_control(desired, B_severe, weights=weights_severe)
    print("\n=== 严重故障 (act1 failed) ===")
    print(f"  Desired: {desired}")
    print(f"  Command: {result_severe.u_command}")
    print(f"  Achieved: {result_severe.u_achieved}")
    print(f"  Feasible: {result_severe.feasible}")
