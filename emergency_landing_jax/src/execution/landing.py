"""
execution/landing.py — 着陆程序

对应技术方案 §4.5.3 Eq 28。

分三段着陆：
  Phase 1: Approach (进场) — 下降到目标点上方
  Phase 2: Flare (拉平)     — 减速，浅角度接地
  Phase 3: Touchdown (接地) — 垂直速度趋零

每段有不同的速度/角度约束。
"""

import numpy as np
from dataclasses import dataclass, field
from typing import NamedTuple
from enum import Enum


class LandingPhase(Enum):
    """着陆阶段"""
    APPROACH = 1      # 进场段
    FLARE = 2          # 拉平段
    TOUCHDOWN = 3      # 接地段
    LANDED = 4         # 已着陆


@dataclass
class LandingState:
    """着陆状态"""
    phase: LandingPhase = LandingPhase.APPROACH
    target_x: float = 0.0        # 目标着陆点 x
    target_y: float = 0.0        # 目标着陆点 y
    flare_height: float = 5.0    # 拉平高度 m
    descent_rate: float = 2.0     # 进场下降率 m/s
    flare_rate: float = 0.5       # 拉平下降率 m/s
    touch_speed_max: float = 1.0  # 最大接地速度 m/s
    phase_time: float = 0.0       # 当前阶段时间
    total_time: float = 0.0       # 总着陆时间


class LandingController:
    """
    着陆控制器 (Eq 28)

    根据当前状态和目标着陆点，
    输出期望推力和角度。
    """

    def __init__(self, config: LandingState = None):
        self.state = config or LandingState()

    def update_phase(self, pos: np.ndarray, vel: np.ndarray) -> LandingPhase:
        """更新着陆阶段"""
        x, y = pos[0], pos[1]
        vx, vy = vel[0], vel[1]
        speed = np.sqrt(vx**2 + vy**2)

        if self.state.phase == LandingPhase.APPROACH:
            # 到达拉平高度
            if y <= self.state.flare_height:
                self.state.phase = LandingPhase.FLARE

        elif self.state.phase == LandingPhase.FLARE:
            # 接近地面
            if y <= 1.0:
                self.state.phase = LandingPhase.TOUCHDOWN

        elif self.state.phase == LandingPhase.TOUCHDOWN:
            if y <= 0.05 and speed < self.state.touch_speed_max:
                self.state.phase = LandingPhase.LANDED

        return self.state.phase

    def compute_control(self, pos: np.ndarray, vel: np.ndarray,
                        dt: float = 0.01) -> tuple:
        """
        计算着陆控制指令

        Returns: (thrust, angle, phase)
        """
        phase = self.update_phase(pos, vel)
        x, y = pos[0], pos[1]
        vx, vy = vel[0], vel[1]

        mass = 1.0
        g = 9.81

        if phase == LandingPhase.APPROACH:
            # 朝目标点下降
            dx = self.state.target_x - x
            dy = self.state.target_y + self.state.flare_height - y

            # 期望速度：水平指向目标，垂直为下降率
            desired_vx = np.clip(dx * 0.5, -5.0, 5.0)
            desired_vy = -self.state.descent_rate

            # PID式推力
            thrust_x = mass * (desired_vx - vx) * 2.0
            thrust_y = mass * (desired_vy - vy) * 2.0 + mass * g

            thrust = np.sqrt(thrust_x**2 + thrust_y**2)
            angle = np.arctan2(thrust_y, thrust_x)

        elif phase == LandingPhase.FLARE:
            # 减速，浅角度
            desired_vx = np.clip(
                (self.state.target_x - x) * 0.3, -2.0, 2.0)
            desired_vy = -self.state.flare_rate

            thrust_x = mass * (desired_vx - vx) * 3.0
            thrust_y = mass * (desired_vy - vy) * 3.0 + mass * g

            thrust = np.sqrt(thrust_x**2 + thrust_y**2)
            angle = np.arctan2(thrust_y, thrust_x)

        elif phase == LandingPhase.TOUCHDOWN:
            # 垂直接地，速度趋零
            desired_vx = 0.0
            desired_vy = -0.3

            thrust_x = mass * (desired_vx - vx) * 5.0
            thrust_y = mass * (desired_vy - vy) * 5.0 + mass * g

            thrust = np.sqrt(thrust_x**2 + thrust_y**2)
            angle = np.arctan2(thrust_y, thrust_x)

        else:  # LANDED
            thrust = mass * g  # 仅维持重力
            angle = np.pi / 2

        self.state.phase_time += dt
        self.state.total_time += dt

        return float(thrust), float(angle), phase


# ════════════════════════════════════════════════════════════════
# 测试
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    controller = LandingController(LandingState(
        target_x=0.0, target_y=0.0,
        flare_height=5.0,
        descent_rate=2.0,
        flare_rate=0.5,
    ))

    # 初始状态：高度40m，有水平速度
    pos = np.array([10.0, 40.0])
    vel = np.array([0.0, 0.0])

    dt = 0.01
    mass = 1.0
    g = 9.81

    print("=== 着陆仿真 ===")
    print(f"  Start: pos={pos}, vel={vel}")
    print(f"  Target: ({controller.state.target_x}, {controller.state.target_y})")
    print()

    for i in range(5000):
        thrust, angle, phase = controller.compute_control(pos, vel, dt)

        # 简单动力学
        ax = thrust * np.cos(angle) / mass
        ay = thrust * np.sin(angle) / mass - g

        vel = vel + np.array([ax, ay]) * dt
        pos = pos + vel * dt

        # 地面
        if pos[1] < 0:
            pos[1] = 0
            vel[1] = -vel[1] * 0.3  # 弹跳
            if abs(vel[1]) < 0.5:
                vel = np.zeros(2)
                phase = LandingPhase.LANDED
                break

        if i % 500 == 0 or phase != LandingPhase.APPROACH:
            speed = np.sqrt(vel[0]**2 + vel[1]**2)
            print(f"  t={i*dt:5.2f}s  pos=({pos[0]:6.1f}, {pos[1]:5.1f}) "
                  f"vel=({vel[0]:5.2f}, {vel[1]:5.2f}) "
                  f"|v|={speed:.2f} phase={phase.name}")

    print(f"\n  Final: pos=({pos[0]:.2f}, {pos[1]:.2f}), vel={vel}")
    print(f"  Total time: {controller.state.total_time:.2f}s")
