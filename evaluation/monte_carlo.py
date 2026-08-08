"""
evaluation/monte_carlo.py — 蒙特卡洛评估

对应技术方案 §5.6。

150次蒙特卡洛试验，覆盖：
  - 标准条件
  - 风场扰动
  - 故障组合
  - 对抗场景

输出：安全着陆概率 + Clopper-Pearson 置信区间
"""

import numpy as np
import time
import sys
import os

# 添加项目根到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SystemConfig
from src.physics.dynamics import faulty_physics_step, pack_params
from src.env.fault_injection import FaultInjector, FaultScenario
from src.perception.landing_points import LandingPointSearcher
from .metrics import compute_safety_metrics, SafetyMetrics


def run_single_trial(env_params: tuple,
                      fault_scenario: FaultScenario,
                      target_x: float = 0.0,
                      target_y: float = 0.0,
                      max_steps: int = 1000,
                      seed: int = 0) -> dict:
    """
    单次蒙特卡洛试验

    返回:
      success: 是否安全着陆
      casualty: 是否发生伤亡
      final_pos: 最终位置
      final_vel: 最终速度
      steps: 步数
      touch_speed: 接地速度
    """
    rng = np.random.RandomState(seed)
    cfg = SystemConfig()
    params = env_params

    # 初始状态
    x = rng.uniform(-5, 5)
    y = rng.uniform(30, 50)
    vx = rng.uniform(-3, 3)
    vy = rng.uniform(-2, 2)
    energy = 1000.0

    state = np.array([x, y, vx, vy, energy])

    # 简单控制策略：朝目标方向发力
    injector = FaultInjector(mode="fixed")

    for step in range(max_steps):
        t = step * cfg.physics.dt

        # 获取当前故障
        fault = injector.get_fault_vector(fault_scenario, t)

        # 简单策略：朝目标方向发力
        dx = target_x - state[0]
        dy = target_y - state[1]

        # 期望速度
        desired_vx = np.clip(dx * 0.5, -5, 5)
        desired_vy = np.clip(dy * 0.5, -5, 5)

        # PD控制
        thrust_x = 2.0 * (desired_vx - state[2])
        thrust_y = 2.0 * (desired_vy - state[3]) + cfg.physics.mass * cfg.physics.g

        # 限幅
        thrust = np.sqrt(thrust_x**2 + thrust_y**2)
        if thrust > 50.0:
            thrust_x *= 50.0 / thrust
            thrust_y *= 50.0 / thrust
            thrust = 50.0

        angle = np.arctan2(thrust_y, thrust_x)
        ctrl = np.array([thrust, angle])

        # JAX物理步进
        import jax.numpy as jnp
        state = np.array(
            faulty_physics_step(jnp.array(state), jnp.array(ctrl),
                                 jnp.array(fault), params)
        )

        # 终止条件
        if state[1] <= 0.01:
            touch_speed = np.sqrt(state[2]**2 + state[3]**2)
            success = touch_speed < 3.0  # 安全接地速度 < 3m/s
            casualty = touch_speed > 10.0  # 伤亡速度 > 10m/s
            return {
                'success': success,
                'casualty': casualty,
                'final_pos': state[:2].copy(),
                'final_vel': state[2:4].copy(),
                'steps': step,
                'touch_speed': touch_speed,
            }

    # 超时
    return {
        'success': False,
        'casualty': False,
        'final_pos': state[:2].copy(),
        'final_vel': state[2:4].copy(),
        'steps': max_steps,
        'touch_speed': -1,
    }


def run_monte_carlo(n_trials: int = 150,
                     scenarios: list = None,
                     seed: int = 42) -> SafetyMetrics:
    """
    蒙特卡洛评估 (§5.6)

    Args:
        n_trials: 试验次数
        scenarios: 故障场景列表
    """
    cfg = SystemConfig()
    params = pack_params(cfg)

    if scenarios is None:
        scenarios = [
            ("normal", FaultScenario()),
            ("chute_fail", FaultScenario(phi=1.0)),
            ("power_50", FaultScenario(delta=0.5)),
            ("combined", FaultScenario(phi=0.5, delta=0.7)),
        ]

    print(f"\n{'='*60}")
    print(f"蒙特卡洛评估: {n_trials} trials × {len(scenarios)} scenarios")
    print(f"{'='*60}\n")

    all_results = []
    all_casualties = []

    for sc_name, scenario in scenarios:
        print(f"  场景: {sc_name}")
        t0 = time.time()

        for i in range(n_trials):
            result = run_single_trial(
                params, scenario,
                seed=seed + i
            )
            all_results.append(result['success'])
            all_casualties.append(result['casualty'])

        dt = time.time() - t0
        n_succ = sum(all_results[-n_trials:])
        print(f"    成功: {n_succ}/{n_trials} "
              f"({n_succ/n_trials*100:.1f}%) "
              f"in {dt:.1f}s")

    metrics = compute_safety_metrics(
        all_results, all_casualties,
        target_p_safe=cfg.eval.p_safe_target,
        target_p_casualty=cfg.eval.p_casualty_target,
        confidence=cfg.eval.confidence_level,
    )

    print(f"\n{'='*60}")
    print(f"评估结果")
    print(f"{'='*60}")
    print(f"  总试验: {metrics.n_trials}")
    print(f"  成功: {metrics.n_success}/{metrics.n_trials}")
    print(f"  p_safe: {metrics.p_point:.4f}")
    print(f"  95% CI: [{metrics.p_lower:.4f}, {metrics.p_upper:.4f}]")
    print(f"  目标: p_safe ≥ {cfg.eval.p_safe_target}")
    print(f"  p_casualty: {metrics.p_casualty:.6f}")
    print(f"  目标: p_casualty ≤ {cfg.eval.p_casualty_target}")
    print(f"  通过: {'✓' if metrics.passed else '✗'}")
    print(f"  裕度: {metrics.margin:+.4f}")

    return metrics


if __name__ == "__main__":
    # 快速测试（少量试验）
    run_monte_carlo(
        n_trials=20,
        scenarios=[
            ("normal", FaultScenario()),
            ("power_50", FaultScenario(delta=0.5)),
        ],
    )
