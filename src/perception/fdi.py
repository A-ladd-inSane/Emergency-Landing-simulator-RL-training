"""
perception/fdi.py — 故障检测与隔离 (FDI)

对应技术方案 §4.2.1 Eq 7-10。

三层 FDI 架构：
  1. GLR 检测：广义似然比检测突变故障 (Eq 7-8)
  2. 结构化残差隔离：基于残差方向向量匹配故障类型 (Eq 9)
  3. 严重度估计：故障幅度估计

2D简化版：监测 [vx, vy] 的预测残差，
检测动力降级 δ 和伞失效 φ。
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.stats import chi2
from dataclasses import dataclass
from typing import NamedTuple


# ════════════════════════════════════════════════════════════════
# 故障检测结果
# ════════════════════════════════════════════════════════════════

@dataclass
class FDIResult:
    """FDI 输出"""
    detected: bool           # 是否检测到故障
    fault_type: str          # "none" / "power" / "parachute" / "combined"
    severity: float          # 严重度 [0, 1]
    confidence: float       # 置信度 [0, 1]
    glr_statistic: float     # GLR 检验统计量
    timestamp: float         # 检测时刻


# ════════════════════════════════════════════════════════════════
# GLR 检测器 (Eq 7-8)
# ════════════════════════════════════════════════════════════════

class GLRDetector:
    """
    广义似然比 (GLR) 突变故障检测器

    原理：在滑动窗口内计算对数似然比，
    若超过卡方阈值则判定故障发生。

    H0: 残差 ~ N(0, Σ)      (正常)
    H1: 残差 ~ N(μ_fault, Σ)  (故障，均值偏移)

    GLR = max_μ [Σ r_i^T Σ⁻¹ μ - n/2 μ^T Σ⁻¹ μ]
    ~ χ²(d)  under H0

    (Eq 7-8)
    """

    def __init__(self, window_size: int = 20, alpha: float = 1e-6,
                 state_dim: int = 2):
        """
        Args:
            window_size: 滑动窗口 N
            alpha: 虚警概率上界
            state_dim: 状态维度 d
        """
        self.window = window_size
        self.alpha = alpha
        self.d = state_dim

        # 卡方阈值 (Eq 8)
        self.threshold = float(chi2.ppf(1 - alpha, df=state_dim))

        # 残差历史
        self._residuals = np.zeros((window_size, state_dim))
        self._idx = 0
        self._full = False

        # 标称协方差（简化：单位阵）
        self.Sigma_inv = np.eye(state_dim)

    def update(self, residual: np.ndarray) -> float:
        """
        推入新残差，返回 GLR 统计量

        Args:
            residual: [d] 当前步预测残差
        Returns:
            glr_stat: GLR 检验统计量
        """
        self._residuals[self._idx] = residual
        self._idx = (self._idx + 1) % self.window
        if self._idx == 0:
            self._full = True

        if not self._full:
            return 0.0

        # GLR 统计量 (Eq 7)
        # 简化版：窗口内残差二范数平方
        r_window = self._residuals.copy()
        r_mean = r_window.mean(axis=0)
        glr_stat = float(r_mean @ self.Sigma_inv @ r_mean * self.window)

        return glr_stat

    def detect(self, residual: np.ndarray, t: float = 0.0) -> FDIResult:
        """
        完整检测流程

        Args:
            residual: [d] 预测残差
            t: 当前时间
        Returns:
            FDIResult
        """
        glr_stat = self.update(residual)
        detected = glr_stat > self.threshold

        if not detected:
            return FDIResult(
                detected=False, fault_type="none", severity=0.0,
                confidence=0.0, glr_statistic=glr_stat, timestamp=t
            )

        # ── Eq 9: 结构化残差隔离 ──
        fault_type, severity, confidence = self._isolate(residual)

        return FDIResult(
            detected=True, fault_type=fault_type, severity=severity,
            confidence=confidence, glr_statistic=glr_stat, timestamp=t
        )

    def _isolate(self, residual: np.ndarray):
        """
        结构化残差隔离 (Eq 9)

        不同故障在残差空间中具有不同方向特征：
          - 动力降级 δ: 残差主要在控制方向（y轴，垂直）
          - 伞失效 φ:  残差主要在速度方向（x轴，水平，减速异常）
          - 复合:       两方向都有显著残差
        """
        r = residual
        r_norm = np.linalg.norm(r) + 1e-10

        # 残差方向向量
        dir_x = abs(r[0]) / r_norm  # 水平方向占比
        dir_y = abs(r[1]) / r_norm  # 垂直方向占比

        # 方向阈值
        theta = 0.65

        if dir_y > theta and dir_x < (1 - theta):
            fault_type = "power"
        elif dir_x > theta and dir_y < (1 - theta):
            fault_type = "parachute"
        else:
            fault_type = "combined"

        # 严重度估计：残差幅度 → [0, 1]
        severity = min(r_norm / 10.0, 1.0)

        # 置信度：GLR 超过阈值的倍数
        confidence = min(glr_stat := self._last_glr / self.threshold, 1.0) \
            if hasattr(self, '_last_glr') else min(glr_stat / self.threshold, 1.0)

        return fault_type, severity, confidence

    @property
    def _last_glr(self):
        """最近一次 GLR 统计量"""
        return getattr(self, '__last_glr', 0.0)


# ════════════════════════════════════════════════════════════════
# 残差计算
# ════════════════════════════════════════════════════════════════

def compute_residual(state_prev: np.ndarray, state_curr: np.ndarray,
                     control: np.ndarray, nominal_model) -> np.ndarray:
    """
    计算预测残差 r = y - ŷ

    使用标称模型预测下一步状态，
    与实际状态之差作为残差。

    Args:
        state_prev: 上一步状态 [x, y, vx, vy]
        state_curr: 当前实际状态 [x, y, vx, vy]
        control:   当前控制 [thrust_x, thrust_y]
        nominal_model: 标称预测函数 (state, ctrl) -> predicted_state
    Returns:
        residual: [vx_res, vy_res] 速度残差
    """
    predicted = nominal_model(state_prev, control)
    residual = state_curr[2:4] - predicted[2:4]  # 速度残差
    return residual


# ════════════════════════════════════════════════════════════════
# 测试
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)

    detector = GLRDetector(window_size=20, alpha=1e-6, state_dim=2)
    print(f"GLR threshold (χ², α=1e-6, d=2): {detector.threshold:.4f}")

    # 模拟正常残差（高斯噪声）
    print("\n--- 正常运行 ---")
    for i in range(30):
        r = np.random.randn(2) * 0.1
        result = detector.detect(r, t=i * 0.01)
        if result.detected:
            print(f"  t={i*0.01:.2f}: FALSE ALARM glr={result.glr_statistic:.4f}")

    # 注入突变故障
    print("\n--- 故障注入 ---")
    for i in range(30):
        if i < 10:
            r = np.random.randn(2) * 0.1
        else:
            # 动力降级：垂直方向残差增大
            r = np.random.randn(2) * 0.1 + np.array([0, 3.0])

        result = detector.detect(r, t=(30+i) * 0.01)
        if result.detected:
            print(f"  t={result.timestamp:.2f}: DETECTED "
                  f"type={result.fault_type} sev={result.severity:.2f} "
                  f"glr={result.glr_statistic:.4f}")
