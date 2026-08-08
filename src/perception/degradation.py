"""
perception/degradation.py — 组合退化度与 SVD 截断

对应技术方案 §4.2.2 Eq 4, 14。

Δμ_eff(t) = Σ_k w_k · μ_k(fault_k)

综合故障检测、RLS辨识、直接传感器三路信息，
给出当前飞行能力的标量退化度量 [0, 1]。
"""

import numpy as np
from dataclasses import dataclass
from typing import List
from .fdi import FDIResult
from .rls import DualForgettingRLS


@dataclass
class DegradationResult:
    """退化度评估结果"""
    mu_eff: float              # 组合退化度 [0, 1]
    mu_power: float            # 动力退化
    mu_parachute: float        # 伞退化
    mu_rls: float              # RLS辨识退化
    mu_combined: float         # 加权融合
    sv_rank: int               # SVD数值秩
    sv_singular_values: np.ndarray  # 奇异值
    confidence: float          # 置信度


def compute_degradation(fdi_result: FDIResult,
                         rls: DualForgettingRLS,
                         B0_nominal: np.ndarray = np.diag([1.0, 1.0])
                         ) -> DegradationResult:
    """
    计算组合退化度 Δμ_eff(t) (Eq 4)

    三路信息融合：
      μ_fdi: FDI 检测到的故障严重度
      μ_rls: RLS B̂(t) vs B₀ 的偏差
      μ_sensor: 直接传感器（此处简化为0）

    融合权重 w_k 由各路置信度自适应调节。
    """
    # ── μ_fdi: FDI 严重度 ──
    if fdi_result.detected:
        mu_fdi = fdi_result.severity
        conf_fdi = fdi_result.confidence
    else:
        mu_fdi = 0.0
        conf_fdi = 0.0

    # ── μ_rls: RLS 辨识退化 ──
    B_hat = rls.get_B_hat()
    # 退化度 = 1 - min(B_hat / B0)  (最差通道)
    ratio = np.minimum(B_hat / B0_nominal, 1.0 / np.maximum(B_hat, 1e-10))
    mu_rls = float(1.0 - np.min(np.diag(ratio)))
    mu_rls = np.clip(mu_rls, 0.0, 1.0)
    conf_rls = min(rls.step / 50.0, 1.0)  # 收敛后置信度高

    # ── μ_sensor: 直接传感器（简化） ──
    mu_sensor = 0.0
    conf_sensor = 0.5

    # ── 自适应权重 (Eq 4) ──
    total_conf = conf_fdi + conf_rls + conf_sensor + 1e-10
    w_fdi = conf_fdi / total_conf
    w_rls = conf_rls / total_conf
    w_sensor = conf_sensor / total_conf

    mu_power = mu_fdi if fdi_result.fault_type in ("power", "combined") else mu_rls
    mu_parachute = mu_fdi if fdi_result.fault_type in ("parachute", "combined") else 0.0

    mu_combined = w_fdi * mu_fdi + w_rls * mu_rls + w_sensor * mu_sensor
    mu_combined = float(np.clip(mu_combined, 0.0, 1.0))

    # ── Eq 14: SVD 截断 ──
    # 对 B̂(t) 做 SVD，截断小于阈值的奇异值
    U, S, Vt = np.linalg.svd(B_hat)
    sv_threshold = 0.1  # 奇异值阈值
    rank = int(np.sum(S > sv_threshold))
    S_truncated = S.copy()
    S_truncated[S_truncated < sv_threshold] = 0

    confidence = float(max(conf_fdi, conf_rls))

    return DegradationResult(
        mu_eff=mu_combined,
        mu_power=float(mu_power),
        mu_parachute=float(mu_parachute),
        mu_rls=float(mu_rls),
        mu_combined=mu_combined,
        sv_rank=rank,
        sv_singular_values=S,
        confidence=confidence,
    )


def degradation_to_fault(degradation: DegradationResult) -> np.ndarray:
    """
    将退化度转换为故障向量 [φ, δ, asym]
    """
    phi = degradation.mu_parachute
    delta = degradation.mu_power
    asym = 0.0  # 需要额外的不对称检测
    return np.array([phi, delta, asym])


if __name__ == "__main__":
    from fdi import GLRDetector, FDIResult

    # 正常状态
    rls = DualForgettingRLS(n_params=2)
    detector = GLRDetector(window_size=20, state_dim=2)

    # 模拟正常运行
    B_true = np.array([1.0, 1.0])
    for i in range(30):
        ctrl = np.random.randn(2) * 10
        y = ctrl @ B_true + np.random.randn() * 0.5
        rls.update(ctrl, y)

    fdi_result = FDIResult(
        detected=False, fault_type="none", severity=0.0,
        confidence=0.0, glr_statistic=0.0, timestamp=0.0
    )

    deg = compute_degradation(fdi_result, rls)
    print("=== 正常状态 ===")
    print(f"  μ_eff = {deg.mu_eff:.4f}")
    print(f"  SVD rank = {deg.sv_rank}, σ = {deg.sv_singular_values}")
    print(f"  confidence = {deg.confidence:.3f}")

    # 故障状态
    B_fault = np.array([1.0, 0.3])
    for i in range(50):
        ctrl = np.random.randn(2) * 10
        y = ctrl @ B_fault + np.random.randn() * 0.5
        rls.update(ctrl, y)

    fdi_fault = FDIResult(
        detected=True, fault_type="power", severity=0.7,
        confidence=0.9, glr_statistic=100.0, timestamp=0.3
    )

    deg2 = compute_degradation(fdi_fault, rls)
    print("\n=== 故障状态 ===")
    print(f"  μ_eff = {deg2.mu_eff:.4f}")
    print(f"  μ_power = {deg2.mu_power:.4f}")
    print(f"  μ_rls = {deg2.mu_rls:.4f}")
    print(f"  SVD rank = {deg2.sv_rank}, σ = {deg2.sv_singular_values}")
