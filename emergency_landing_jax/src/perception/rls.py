"""
perception/rls.py — 双遗忘因子递推最小二乘 (RLS)

对应技术方案 §4.2.2 Eq 11-13。

双遗忘因子 RLS 同时跟踪：
  - λ_f (快变): 跟踪阶跃故障（如动力突变）
  - λ_s (慢变): 跟踪渐进退化（如电池老化）

输出：控制有效性矩阵的在线估计 B̂(t)，
供决策层使用 B(t) = B̂(t) 而非标称 B₀。
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class RLSState:
    """RLS 估计器内部状态"""
    # 快变估计
    theta_f: np.ndarray    # 参数向量 (n_params,)
    P_f: np.ndarray         # 协方差矩阵 (n_params, n_params)
    # 慢变估计
    theta_s: np.ndarray
    P_s: np.ndarray
    # 融合权重
    w_f: float = 0.5        # 快变权重
    step: int = 0


class DualForgettingRLS:
    """
    双遗忘因子 RLS (Eq 11-13)

    模型: y = Φ^T θ + noise
    更新:
      θ_{k+1} = θ_k + K_k (y_k - φ_k^T θ_k)
      K_k = P_k φ_k / (λ + φ_k^T P_k φ_k)
      P_{k+1} = (P_k - K_k φ_k^T P_k) / λ

    双通道:
      快变 λ_f=0.98 → 灵敏但噪声大
      慢变 λ_s=0.999 → 稳定但滞后
      融合 θ̂ = w_f θ_f + (1-w_f) θ_s
    """

    def __init__(self, n_params: int,
                 lambda_f: float = 0.98,
                 lambda_s: float = 0.999,
                 P0_scale: float = 100.0):
        """
        Args:
            n_params: 参数维度（2D: thrust_x, thrust_y → 2）
            lambda_f: 快变遗忘因子
            lambda_s: 慢变遗忘因子
            P0_scale: 初始协方差对角值
        """
        self.n = n_params
        self.lf = lambda_f
        self.ls = lambda_s

        # 初始状态
        theta0 = np.zeros(n_params)
        P0 = np.eye(n_params) * P0_scale

        self.state = RLSState(
            theta_f=theta0.copy(),
            P_f=P0.copy(),
            theta_s=theta0.copy(),
            P_s=P0.copy(),
        )

    def update(self, phi: np.ndarray, y: float) -> Tuple[float, float]:
        """
        单步 RLS 更新

        Args:
            phi: 回归向量 [n_params]（控制输入）
            y:  观测值（加速度变化）
        Returns:
            (theta_fused, convergence_indicator)
        """
        s = self.state

        # ── 快变通道 ──
        y_pred_f = phi @ s.theta_f
        e_f = y - y_pred_f
        K_f = s.P_f @ phi / (self.lf + phi @ s.P_f @ phi)
        s.theta_f = s.theta_f + K_f * e_f
        s.P_f = (s.P_f - np.outer(K_f, phi @ s.P_f)) / self.lf

        # ── 慢变通道 ──
        y_pred_s = phi @ s.theta_s
        e_s = y - y_pred_s
        K_s = s.P_s @ phi / (self.ls + phi @ s.P_s @ phi)
        s.theta_s = s.theta_s + K_s * e_s
        s.P_s = (s.P_s - np.outer(K_s, phi @ s.P_s @ phi)) / self.ls

        # ── 自适应融合权重 ──
        # 阶跃时快变残差远大于慢变 → 信任快变
        # 渐变时两者接近 → 信任慢变（更稳定）
        ratio = abs(e_f) / (abs(e_s) + 1e-10)
        s.w_f = 1.0 / (1.0 + np.exp(-(ratio - 2.0)))  # sigmoid

        # ── 融合估计 ──
        theta_fused = s.w_f * s.theta_f + (1 - s.w_f) * s.theta_s

        # 收敛指标：P 矩阵迹的倒数
        convergence = 1.0 / (np.trace(s.P_f) / self.n + 1e-10)

        s.step += 1
        return theta_fused, convergence

    def get_B_hat(self) -> np.ndarray:
        """
        返回当前控制有效性估计 B̂(t)

        2D简化：θ = [B00, B11]（对角阵）
        """
        s = self.state
        theta = s.w_f * s.theta_f + (1 - s.w_f) * s.theta_s
        return np.diag(theta)

    @property
    def step(self) -> int:
        return self.state.step


# ════════════════════════════════════════════════════════════════
# 测试
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)

    rls = DualForgettingRLS(n_params=2, lambda_f=0.98, lambda_s=0.999)

    # 标称 B = [1.0, 1.0]
    B_true = np.array([1.0, 1.0])

    print("=== Phase 1: 正常运行 (B=[1,1]) ===")
    for i in range(50):
        ctrl = np.random.randn(2) * 10  # 随机控制
        y = ctrl @ B_true + np.random.randn() * 0.5  # 加噪声
        theta, conv = rls.update(ctrl, y)

    B_hat = rls.get_B_hat()
    print(f"  B_hat = diag({B_hat[0,0]:.3f}, {B_hat[1,1]:.3f})  step={rls.step}")

    print("\n=== Phase 2: 阶跃故障 (B_y → 0.5) ===")
    B_true_fault = np.array([1.0, 0.5])
    for i in range(50):
        ctrl = np.random.randn(2) * 10
        y = ctrl @ B_true_fault + np.random.randn() * 0.5
        theta, conv = rls.update(ctrl, y)
        if i in [5, 10, 20, 49]:
            B_hat = rls.get_B_hat()
            w_f = rls.state.w_f
            print(f"  step {i+1:3d}: B_hat=diag({B_hat[0,0]:.3f}, {B_hat[1,1]:.3f})  "
                  f"w_f={w_f:.3f}")

    print("\n=== Phase 3: 渐进退化 (B → 0.3) ===")
    for i in range(100):
        degradation = i / 100 * 0.7
        B_deg = np.array([1.0, 1.0 - degradation])
        ctrl = np.random.randn(2) * 10
        y = ctrl @ B_deg + np.random.randn() * 0.5
        theta, conv = rls.update(ctrl, y)
        if i in [20, 50, 99]:
            B_hat = rls.get_B_hat()
            print(f"  step {i+1:3d}: B_hat=diag({B_hat[0,0]:.3f}, {B_hat[1,1]:.3f})  "
                  f"(true: {B_deg[1]:.3f})")
