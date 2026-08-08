"""
evaluation/metrics.py — 适航安全指标计算

对应技术方案 §5。

核心指标：
  1. p_safe: 安全着陆概率（Clopper-Pearson 置信区间）
  2. p_casualty: 地面伤亡概率（FMEA）
  3. 鲁棒性: 对抗风场/多重故障下的表现

Clopper-Pearson 精确二项分布置信区间：
  [Beta(α/2; k, n-k+1), Beta(1-α/2; k+1, n-k)]
"""

import numpy as np
from scipy.stats import beta, binom
from dataclasses import dataclass
from typing import List


@dataclass
class SafetyMetrics:
    """安全指标"""
    n_trials: int
    n_success: int
    n_failure: int
    p_point: float             # 点估计
    p_lower: float             # Clopper-Pearson 下界
    p_upper: float             # Clopper-Pearson 上界
    confidence_level: float
    p_casualty: float          # 地面伤亡概率
    passed: bool               # 是否通过适航标准
    margin: float              # 安全裕度


def clopper_pearson_ci(n_success: int, n_trials: int,
                        confidence: float = 0.95) -> tuple:
    """
    Clopper-Pearson 精确二项分布置信区间 (§5.6.1)

    对于 n_trials 次试验中 n_success 次成功，
    给出成功率 p 的 (1-α) 置信区间。

    公式:
      下界 = Beta(α/2; k, n-k+1)
      上界 = Beta(1-α/2; k+1, n-k)

    特殊情况:
      k=0: 下界=0
      k=n: 上界=1
    """
    alpha = 1 - confidence
    k = n_success
    n = n_trials

    if k == 0:
        lower = 0.0
    else:
        lower = beta.ppf(alpha / 2, k, n - k + 1)

    if k == n:
        upper = 1.0
    else:
        upper = beta.ppf(1 - alpha / 2, k + 1, n - k)

    return lower, upper


def compute_safety_metrics(trial_results: List[bool],
                             casualty_events: List[bool] = None,
                             target_p_safe: float = 0.95,
                             target_p_casualty: float = 1e-4,
                             confidence: float = 0.95) -> SafetyMetrics:
    """
    计算适航安全指标 (§5)

    Args:
        trial_results: 每次试验是否安全着陆
        casualty_events: 每次试验是否发生地面伤亡
        target_p_safe: 安全着陆概率目标
        target_p_casualty: 地面伤亡概率目标
        confidence: 置信水平
    """
    n_trials = len(trial_results)
    n_success = sum(trial_results)
    n_failure = n_trials - n_success

    p_point = n_success / n_trials if n_trials > 0 else 0.0
    p_lower, p_upper = clopper_pearson_ci(n_success, n_trials, confidence)

    p_casualty = 0.0
    if casualty_events:
        n_casualty = sum(casualty_events)
        p_casualty = n_casualty / len(casualty_events)

    passed = (p_lower >= target_p_safe) and (p_casualty <= target_p_casualty)
    margin = p_lower - target_p_safe

    return SafetyMetrics(
        n_trials=n_trials,
        n_success=n_success,
        n_failure=n_failure,
        p_point=p_point,
        p_lower=p_lower,
        p_upper=p_upper,
        confidence_level=confidence,
        p_casualty=p_casualty,
        passed=passed,
        margin=margin,
    )


def required_trials_for_target(target_p: float = 0.95,
                                confidence: float = 0.95,
                                allowed_failures: int = 5) -> int:
    """
    计算达到目标所需的蒙特卡洛试验次数 (§5.6.1)

    使用 Clopper-Pearson 下界 ≥ target_p，
    在允许 k 次失败时，求最小 n。
    """
    alpha = 1 - confidence
    n = allowed_failures + 1
    while True:
        n_success = n - allowed_failures
        lower, _ = clopper_pearson_ci(n_success, n, confidence)
        if lower >= target_p:
            return n
        n += 1
        if n > 10000:
            return -1  # 不可达


# ════════════════════════════════════════════════════════════════
# FMEA 辅助
# ════════════════════════════════════════════════════════════════

@dataclass
class FMEAItem:
    """FMEA (故障模式与影响分析) 条目"""
    component: str
    failure_mode: str
    effect: str
    severity: int        # 1-10
    occurrence: int     # 1-10
    detectability: int  # 1-10
    rpn: int = 0        # Risk Priority Number = S × O × D

    def __post_init__(self):
        self.rpn = self.severity * self.occurrence * self.detectability


def compute_fmea_rpn(items: List[FMEAItem]) -> dict:
    """计算 FMEA 风险优先数"""
    return {
        'total_items': len(items),
        'total_rpn': sum(i.rpn for i in items),
        'max_rpn': max(i.rpn for i in items) if items else 0,
        'high_risk': sum(1 for i in items if i.rpn > 100),
        'items': items,
    }


# ════════════════════════════════════════════════════════════════
# 测试
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Clopper-Pearson 测试
    print("=== Clopper-Pearson 置信区间 ===")
    for k, n in [(150, 150), (145, 150), (140, 150), (135, 150), (0, 10)]:
        lo, hi = clopper_pearson_ci(k, n, 0.95)
        print(f"  {k}/{n} successes: [{lo:.4f}, {hi:.4f}]")

    # 所需试验次数
    print("\n=== 所需试验次数 ===")
    for allowed in [0, 3, 5, 8]:
        n = required_trials_for_target(0.95, 0.95, allowed)
        print(f"  allowed_failures={allowed}: n={n}")

    # 安全指标计算
    print("\n=== 安全指标 ===")
    # 模拟150次试验
    np.random.seed(42)
    results = np.random.random(150) > 0.02  # 98% 成功率
    casualties = np.random.random(150) < 0.001  # 0.1% 伤亡率

    metrics = compute_safety_metrics(
        list(results), list(casualties),
        target_p_safe=0.95, target_p_casualty=1e-4
    )

    print(f"  Trials: {metrics.n_trials}")
    print(f"  Success: {metrics.n_success}/{metrics.n_trials}")
    print(f"  p_point: {metrics.p_point:.4f}")
    print(f"  p_lower: {metrics.p_lower:.4f} (target ≥ 0.95)")
    print(f"  p_upper: {metrics.p_upper:.4f}")
    print(f"  p_casualty: {metrics.p_casualty:.6f} (target ≤ 1e-4)")
    print(f"  PASSED: {metrics.passed}")
    print(f"  Margin: {metrics.margin:+.4f}")

    # FMEA
    print("\n=== FMEA ===")
    items = [
        FMEAItem("伞系统", "伞未展开", "无减速", 10, 3, 4),
        FMEAItem("动力系统", "旋翼失效", "推力下降", 8, 4, 3),
        FMEAItem("传感器", "IMU漂移", "状态估计偏差", 6, 5, 6),
    ]
    fmea = compute_fmea_rpn(items)
    print(f"  Total RPN: {fmea['total_rpn']}")
    print(f"  Max RPN: {fmea['max_rpn']}")
    print(f"  High risk items: {fmea['high_risk']}")
