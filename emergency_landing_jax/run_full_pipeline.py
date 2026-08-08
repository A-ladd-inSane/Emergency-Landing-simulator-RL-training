#!/usr/bin/env python3
"""
run_full_pipeline.py — 端到端流水线演示

依次执行：
  1. 物理引擎验证（正常+故障）
  2. 感知层：FDI + RLS + 退化度
  3. 可行运动包络计算
  4. 降落点检索
  5. CBF安全过滤
  6. 蒙特卡洛安全评估
"""

import sys
import os
import time

# 路径设置
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SystemConfig
from src.physics.dynamics import faulty_physics_step, pack_params
from src.physics.envelope import compute_fme
from src.perception.fdi import GLRDetector
from src.perception.rls import DualForgettingRLS
from src.perception.degradation import compute_degradation
from src.perception.endurance import estimate_endurance
from src.perception.landing_points import LandingPointSearcher
from src.decision.cbf import CBFSafetyFilter, CBFConstraint
from src.decision.policy import init_policy, sample_action
from src.execution.control_alloc import allocate_control
from src.execution.tracker import TrajectoryTracker
from src.execution.landing import LandingController, LandingPhase
from src.env.fault_injection import FaultInjector, FaultScenario
from evaluation.monte_carlo import run_single_trial
from evaluation.metrics import compute_safety_metrics

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np


def banner(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    cfg = SystemConfig()
    params = pack_params(cfg)

    # ════════════════════════════════════════════════════════════
    # 1. 物理引擎
    # ════════════════════════════════════════════════════════════
    banner("1. 物理引擎验证")
    state = jnp.array([0.0, 40.0, 5.0, 0.0, 1000.0])
    ctrl = jnp.array([10.0, np.pi/2])
    fault_none = jnp.array([0.0, 0.0, 0.0])
    fault_sev = jnp.array([0.5, 0.7, 0.3])

    s1 = faulty_physics_step(state, ctrl, fault_none, params)
    s2 = faulty_physics_step(state, ctrl, fault_sev, params)
    print(f"  初始:  {state.tolist()}")
    print(f"  正常:  {s1.tolist()}")
    print(f"  故障:  {s2.tolist()}")

    # 性能
    import time; t0 = time.time()
    for _ in range(10000):
        state = faulty_physics_step(state, ctrl, fault_none, params)
    state.block_until_ready()
    print(f"  10k steps: {10000/(time.time()-t0):,.0f} steps/s")

    # ════════════════════════════════════════════════════════════
    # 2. 感知层
    # ════════════════════════════════════════════════════════════
    banner("2. 感知层")

    # FDI
    detector = GLRDetector(window_size=20, alpha=1e-6, state_dim=2)
    print(f"  FDI: GLR阈值={detector.threshold:.4f}")

    # RLS
    rls = DualForgettingRLS(n_params=2, lambda_f=0.98, lambda_s=0.999)
    B_fault = np.array([1.0, 0.5])
    for i in range(50):
        ctrl_np = np.random.randn(2) * 10
        y = ctrl_np @ B_fault + np.random.randn() * 0.5
        rls.update(ctrl_np, y)
    B_hat = rls.get_B_hat()
    print(f"  RLS: B̂=diag({B_hat[0,0]:.3f}, {B_hat[1,1]:.3f}) (真值=1.0, 0.5)")

    # 退化度
    from src.perception.fdi import FDIResult
    fdi_result = FDIResult(
        detected=True, fault_type="power", severity=0.7,
        confidence=0.9, glr_statistic=100.0, timestamp=0.3
    )
    deg = compute_degradation(fdi_result, rls)
    print(f"  退化度: μ_eff={deg.mu_eff:.4f} (rank={deg.sv_rank})")

    # 续航
    endurance = estimate_endurance(
        np.array([0, 40, 5, 0, 1000]),
        np.array([0.5, 0.7, 0.3])
    )
    print(f"  续航: T_remain={endurance.t_remain:.1f}s, range={endurance.range_x:.1f}m")

    # ════════════════════════════════════════════════════════════
    # 3. 可行运动包络
    # ════════════════════════════════════════════════════════════
    banner("3. 可行运动包络 (FME)")
    state_fme = jnp.array([0.0, 40.0, 5.0, 0.0, 1000.0])
    fme = compute_fme(state_fme, fault_none, params, n_steps=50, n_controls=16)
    print(f"  正常: x=[{fme.min_x:.1f}, {fme.max_x:.1f}], y=[{fme.min_y:.1f}, {fme.max_y:.1f}]")
    fme_f = compute_fme(state_fme, fault_sev, params, n_steps=50, n_controls=16)
    print(f"  故障: x=[{fme_f.min_x:.1f}, {fme_f.max_x:.1f}], y=[{fme_f.min_y:.1f}, {fme_f.max_y:.1f}]")

    # ════════════════════════════════════════════════════════════
    # 4. 降落点检索
    # ════════════════════════════════════════════════════════════
    banner("4. 降落点检索")
    np.random.seed(42)
    gx, gy = np.meshgrid(np.linspace(-100, 100, 21), np.linspace(0, 80, 9))
    candidates = np.column_stack([gx.ravel(), gy.ravel()])
    slopes = np.random.rand(len(candidates)) * 0.3
    obstacles = np.array([[20, 30], [-30, 40], [50, 20]])

    searcher = LandingPointSearcher(candidates, slopes, obstacles)
    results = searcher.search(
        np.array([10.0, 40.0]),
        np.array([-5.0, -2.0]),
        (-80, 80, 0, 60),
        top_k=3,
    )
    for i, p in enumerate(results):
        print(f"  #{i+1}: ({p.x:+.1f}, {p.y:.1f}) score={p.score:.3f}")

    # ════════════════════════════════════════════════════════════
    # 5. 安全过滤
    # ════════════════════════════════════════════════════════════
    banner("5. CBF 安全过滤")
    constraints = [
        CBFConstraint("ground", lambda s: s[1], lambda s: np.array([0, 1, 0, 0])),
        CBFConstraint("ceiling", lambda s: 100 - s[1], lambda s: np.array([0, -1, 0, 0])),
    ]
    filter = CBFSafetyFilter(constraints,
                              u_max=np.array([50.0, np.pi]))
    state_cbf = np.array([5.0, 3.0, 0.0, -5.0])
    u_rl = np.array([0.0, 0.0])
    f_dyn = np.array([0, 0, 0, -9.81])
    G = np.array([[0,0],[0,0],[1,0],[0,1]])
    u_safe = filter.filter(state_cbf, u_rl, f_dyn, G)
    print(f"  RL动作: {u_rl}")
    print(f"  安全动作: {u_safe}")

    # ════════════════════════════════════════════════════════════
    # 6. 控制分配
    # ════════════════════════════════════════════════════════════
    banner("6. 控制分配")
    B = np.array([[1, 0], [0, 1.0]])
    result = allocate_control(np.array([5.0, 15.0]), B)
    print(f"  正常: cmd={result.u_command}, sat={result.saturation}")

    B_deg = np.array([[1, 0], [0, 0.3]])
    result_deg = allocate_control(np.array([5.0, 15.0]), B_deg, weights=np.array([1.0, 0.3]))
    print(f"  故障: cmd={result_deg.u_command}, sat={result_deg.saturation}")

    # ════════════════════════════════════════════════════════════
    # 7. 蒙特卡洛评估（快速版）
    # ════════════════════════════════════════════════════════════
    banner("7. 蒙特卡洛安全评估 (快速版)")
    all_results = []
    all_casualties = []

    scenarios = [
        ("normal", FaultScenario()),
        ("power_50", FaultScenario(delta=0.5)),
        ("chute_fail", FaultScenario(phi=1.0)),
    ]

    for sc_name, scenario in scenarios:
        for i in range(10):
            r = run_single_trial(params, scenario, seed=i)
            all_results.append(r['success'])
            all_casualties.append(r['casualty'])
        n_succ = sum(all_results[-10:])
        print(f"  {sc_name}: {n_succ}/10")

    metrics = compute_safety_metrics(all_results, all_casualties)
    print(f"\n  总计: {metrics.n_success}/{metrics.n_trials}")
    print(f"  p_safe: {metrics.p_point:.4f}")
    print(f"  95% CI: [{metrics.p_lower:.4f}, {metrics.p_upper:.4f}]")
    print(f"  通过: {'✓' if metrics.passed else '✗'}")

    print(f"\n{'='*60}")
    print("  流水线完成")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
