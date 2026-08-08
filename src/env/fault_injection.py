"""
env/fault_injection.py — 故障注入系统

对应技术方案 §3.2-3.3。

故障类型：
  1. 伞系统失效 (φ): 完全失效、部分展开、缠线
  2. 动力降级 (δ): 部分功率损失、完全失效
  3. 不对称降级 (asym): 一侧旋翼失效

故障注入策略（训练时）：
  - Domain Randomization: 随机故障参数
  - Curriculum: 从简单到复杂
  - Domain Adaptation: sim2real gap
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict
from enum import Enum


class FaultType(Enum):
    """故障类型枚举"""
    NONE = "none"
    CHUTE_FAIL = "chute_fail"          # 伞完全失效
    CHUTE_PARTIAL = "chute_partial"    # 伞部分展开
    POWER_LOSS = "power_loss"          # 动力降级
    POWER_TOTAL = "power_total"        # 动力完全失效
    ASYMMETRIC = "asymmetric"          # 不对称
    COMBINED = "combined"              # 复合故障


@dataclass
class FaultScenario:
    """故障场景定义"""
    fault_type: FaultType = FaultType.NONE
    phi: float = 0.0       # 伞失效 [0,1]
    delta: float = 0.0     # 动力降级 [0,1]
    asym: float = 0.0      # 不对称 [0,1]
    onset_time: float = 0.0  # 故障发生时间 s
    onset_mode: str = "step"  # "step" or "ramp"
    ramp_duration: float = 0.0  # 渐变持续时间
    description: str = ""


class FaultInjector:
    """
    故障注入器

    支持模式：
      - fixed: 固定故障
      - random: 域随机化
      - curriculum: 课程学习（逐步增加难度）
    """

    # 预定义故障场景
    SCENARIOS: Dict[str, FaultScenario] = {
        "normal": FaultScenario(
            FaultType.NONE, description="正常运行"),
        "chute_fail": FaultScenario(
            FaultType.CHUTE_FAIL, phi=1.0,
            description="伞完全失效"),
        "chute_partial": FaultScenario(
            FaultType.CHUTE_PARTIAL, phi=0.5,
            description="伞部分展开50%"),
        "power_50": FaultScenario(
            FaultType.POWER_LOSS, delta=0.5,
            description="动力降级50%"),
        "power_80": FaultScenario(
            FaultType.POWER_LOSS, delta=0.8,
            description="动力降级80%"),
        "power_total": FaultScenario(
            FaultType.POWER_TOTAL, delta=1.0,
            description="动力完全失效"),
        "asym_50": FaultScenario(
            FaultType.ASYMMETRIC, asym=0.5,
            description="不对称降级50%"),
        "combined_severe": FaultScenario(
            FaultType.COMBINED, phi=0.5, delta=0.7, asym=0.3,
            description="复合严重故障"),
    }

    def __init__(self, mode: str = "fixed",
                 seed: int = 42):
        self.mode = mode
        self.rng = np.random.RandomState(seed)
        self.current_scenario = FaultScenario()

    def sample_fault(self, difficulty: float = 0.0) -> FaultScenario:
        """
        采样故障场景

        Args:
            difficulty: 课程难度 [0, 1]
                       0=仅轻微故障, 1=严重故障
        """
        if self.mode == "fixed":
            return self.current_scenario

        elif self.mode == "random":
            # 域随机化
            phi = self.rng.beta(2, 5) * difficulty  # 多数轻微
            delta = self.rng.beta(2, 5) * difficulty
            asym = self.rng.beta(2, 5) * difficulty * 0.5

            return FaultScenario(
                fault_type=FaultType.COMBINED,
                phi=float(phi), delta=float(delta), asym=float(asym),
                onset_time=float(self.rng.uniform(0, 2.0)),
                onset_mode="step",
                description=f"random(φ={phi:.2f},δ={delta:.2f},asym={asym:.2f})",
            )

        elif self.mode == "curriculum":
            # 课程学习：难度随训练进度增加
            if difficulty < 0.2:
                # 初期：无故障或轻微
                if self.rng.random() < 0.7:
                    return FaultScenario()
                else:
                    return FaultScenario(
                        FaultType.POWER_LOSS,
                        delta=float(self.rng.uniform(0, 0.3)),
                        description="curriculum easy",
                    )
            elif difficulty < 0.5:
                # 中期：中等故障
                return FaultScenario(
                    FaultType.COMBINED,
                    delta=float(self.rng.uniform(0.2, 0.5)),
                    phi=float(self.rng.uniform(0, 0.3)),
                    description="curriculum medium",
                )
            else:
                # 后期：严重故障
                return FaultScenario(
                    FaultType.COMBINED,
                    delta=float(self.rng.uniform(0.5, 0.9)),
                    phi=float(self.rng.uniform(0.3, 1.0)),
                    asym=float(self.rng.uniform(0, 0.5)),
                    description="curriculum hard",
                )

        return FaultScenario()

    def get_fault_vector(self, scenario: FaultScenario,
                         t: float) -> np.ndarray:
        """
        获取当前时刻的故障向量 [φ, δ, asym]

        支持阶跃和渐变故障。
        """
        if t < scenario.onset_time:
            return np.array([0.0, 0.0, 0.0])

        if scenario.onset_mode == "ramp":
            elapsed = t - scenario.onset_time
            if scenario.ramp_duration > 0:
                factor = min(elapsed / scenario.ramp_duration, 1.0)
            else:
                factor = 1.0
        else:
            factor = 1.0

        return np.array([
            scenario.phi * factor,
            scenario.delta * factor,
            scenario.asym * factor,
        ])


# ════════════════════════════════════════════════════════════════
# 测试
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 固定场景
    injector = FaultInjector(mode="fixed")
    injector.current_scenario = FaultInjector.SCENARIOS["power_50"]

    print("=== 固定故障场景 ===")
    for t in [0.0, 0.5, 1.0]:
        fv = injector.get_fault_vector(injector.current_scenario, t)
        print(f"  t={t:.1f}: fault={fv}")

    # 域随机化
    print("\n=== 域随机化 (difficulty=0.5) ===")
    inj_rand = FaultInjector(mode="random", seed=42)
    for i in range(5):
        sc = inj_rand.sample_fault(difficulty=0.5)
        print(f"  #{i}: {sc.description}")

    # 课程学习
    print("\n=== 课程学习 ===")
    inj_curr = FaultInjector(mode="curriculum", seed=42)
    for diff in [0.1, 0.3, 0.5, 0.8, 1.0]:
        sc = inj_curr.sample_fault(difficulty=diff)
        print(f"  diff={diff:.1f}: {sc.description}")

    # 渐变故障
    print("\n=== 渐变故障 ===")
    ramp_scenario = FaultScenario(
        FaultType.POWER_LOSS, delta=0.8,
        onset_time=1.0, onset_mode="ramp",
        ramp_duration=2.0,
        description="ramp 2s"
    )
    inj_ramp = FaultInjector(mode="fixed")
    for t in [0.0, 1.0, 2.0, 3.0, 4.0]:
        fv = inj_ramp.get_fault_vector(ramp_scenario, t)
        print(f"  t={t:.1f}: fault={fv}")
