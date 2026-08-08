"""
physics/dynamics.py — 扩展动力学模型

对应技术方案 §3.4 数学形式化 (Eq 1-3)。

核心扩展：
  - 故障掩码矩阵 M(φ,δ,t) 对控制有效性 B₀ 的乘性退化 (Eq 2)
  - 伞系统加性扰动 d_chute(v,φ) (Eq 3)
  - 组合动力学：ṡ = f(s) + B(t)·u + d_chute(v,φ) (Eq 1)

在原始 ball_sim_jax.py 的 physics_step 基础上，
增加故障掩码和伞扰动两项。
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from dataclasses import dataclass
from typing import NamedTuple

# ════════════════════════════════════════════════════════════════
# 状态与故障定义
# ════════════════════════════════════════════════════════════════

class FlightState(NamedTuple):
    """飞行汽车状态（2D简化版）
    完整版：6DOF位置+速度+姿态+角速度+电池 = 15-18维
    简化版：2D位置+速度+电池 = 5维
    """
    x: float = 0.0       # 水平位置 m
    y: float = 40.0     # 垂直位置 m
    vx: float = 5.0     # 水平速度 m/s
    vy: float = 0.0     # 垂直速度 m/s
    energy: float = 1000.0  # 剩余能量 J


class FaultMode(NamedTuple):
    """故障状态
    φ: 伞失效模式 (0=完好, 0.5=部分展开, 1.0=完全失效)
    δ: 动力降级模式 (0=无降级, 1.0=完全失效)
    δ_asym: 不对称降级 (0=对称, 1=一侧完全失效)
    """
    parachute_failure: float = 0.0    # φ ∈ [0, 1]
    power_degradation: float = 0.0   # δ ∈ [0, 1]
    asymmetry: float = 0.0            # 不对称因子 ∈ [0, 1]


# ════════════════════════════════════════════════════════════════
# Eq 2: 故障掩码矩阵 M(φ,δ,t)
# ════════════════════════════════════════════════════════════════

def control_effectiveness_mask(fault: FaultMode) -> jnp.ndarray:
    """
    M(φ,δ,t) ∈ [0,1]^{2×2} — 故障掩码矩阵 (Eq 2)

    B(t) = B₀ ⊙ M(φ,δ,t)

    2D简化：推力x通道和推力y通道可独立降级。
    完整版为 m×n 矩阵，此处为对角2×2。
    """
    # 动力降级 δ → 控制效率衰减 (1-δ)
    power_factor = 1.0 - fault.power_degradation

    # 不对称降级 → x通道额外衰减
    asym_x = 1.0 - fault.asymmetry * 0.5

    M = jnp.array([
        [power_factor * asym_x, 0.0],
        [0.0, power_factor]
    ])
    return M


def effective_control_matrix(B0: jnp.ndarray, fault: FaultMode) -> jnp.ndarray:
    """
    B(t) = B₀ ⊙ M(φ,δ,t)  (Eq 2)

    标称控制矩阵经故障掩码后的实际控制有效性。
    """
    M = control_effectiveness_mask(fault)
    return B0 * M  # Hadamard积（2D简化为逐元素）


# ════════════════════════════════════════════════════════════════
# Eq 3: 伞系统加性扰动 d_chute(v, φ)
# ════════════════════════════════════════════════════════════════

def parachute_disturbance(state: FlightState, fault: FaultMode,
                           CD: float, A: float, l_arm: float,
                           rho: float = 1.225) -> jnp.ndarray:
    """
    d_chute(v, φ) = ½ ρ ‖v‖² C_D(φ) A(φ) · ℓ_arm  (Eq 3)

    伞系统产生的加性扰动力矩（2D简化为力）。
    依赖飞行速度 v 和伞失效模式 φ。

    φ=0: 伞完好 → 正常阻力（减速作用）
    φ=0.5: 部分展开 → 不对称阻力扰动
    φ=1: 完全失效 → 无伞阻力
    """
    speed = jnp.sqrt(state.vx**2 + state.vy**2)

    # 伞阻力系数和面积随失效模式变化
    # φ=0: C_D=CD_nominal, A=A_nominal
    # φ=1: C_D→0, A→0（完全失效无伞）
    CD_eff = CD * (1.0 - fault.parachute_failure)
    A_eff = A * (1.0 - fault.parachute_failure)

    # 阻力方向与速度反向
    speed_safe = jnp.maximum(speed, 1e-6)
    drag_dir_x = -state.vx / speed_safe
    drag_dir_y = -state.vy / speed_safe

    # 不对称扰动：部分展开时产生侧向力
    asym_force = fault.parachute_failure * 0.3 * speed**2 * l_arm

    # 总扰动力 [Fx, Fy]
    drag_mag = 0.5 * rho * speed**2 * CD_eff * A_eff
    F_parachute = jnp.array([
        drag_mag * drag_dir_x + asym_force,
        drag_mag * drag_dir_y
    ])

    return F_parachute


# ════════════════════════════════════════════════════════════════
# Eq 1: 完整动力学 ṡ = f(s) + B(t)·u + d_chute
# ════════════════════════════════════════════════════════════════

# 标称控制矩阵 B₀（2D: 推力→加速度）
B0_DEFAULT = jnp.array([
    [1.0, 0.0],   # thrust_x → ax
    [0.0, 1.0]    # thrust_y → ay
])


@jax.jit
def faulty_physics_step(state_arr, control_arr, fault_arr, params):
    """
    扩展物理步进 — 含故障掩码和伞扰动 (Eq 1-3)

    Args:
        state_arr:  [x, y, vx, vy, energy]  (5D)
        control_arr: [thrust_x, thrust_y]   (2D, 物理单位 N)
        fault_arr:  [φ, δ, asym]            (3D)
        params:     (g, mass, drag_k, dt, B0_flat, rho, CD, A, l_arm, E_min, P_hover, k_power)

    Returns:
        new_state: [x, y, vx, vy, energy]  (5D)
    """
    x, y, vx, vy, energy = state_arr
    thrust_x, thrust_y = control_arr
    phi, delta, asym = fault_arr

    (g, mass, drag_k, dt,
     b00, b01, b10, b11,
     rho, CD, A, l_arm,
     E_min, P_hover, k_power) = params

    # ── Eq 2: 故障掩码 ──
    power_factor = 1.0 - delta
    asym_x = 1.0 - asym * 0.5
    M00 = power_factor * asym_x
    M11 = power_factor

    # B(t) = B₀ ⊙ M
    B00_eff = b00 * M00
    B11_eff = b11 * M11

    # ── Eq 3: 伞扰动 ──
    speed = jnp.sqrt(vx**2 + vy**2)
    speed_safe = jnp.maximum(speed, 1e-6)
    CD_eff = CD * (1.0 - phi)
    A_eff = A * (1.0 - phi)
    drag_mag = 0.5 * rho * speed**2 * CD_eff * A_eff
    drag_fx = drag_mag * (-vx / speed_safe) + phi * 0.3 * speed**2 * l_arm
    drag_fy = drag_mag * (-vy / speed_safe)

    # ── Eq 1: 总力 ──
    # f(s): 重力 + 线性阻力
    F_gravity = jnp.array([0.0, -mass * g])
    F_drag_linear = jnp.array([-drag_k * vx, -drag_k * vy])

    # B(t)·u: 控制力（经故障掩码）
    F_control = jnp.array([B00_eff * thrust_x, B11_eff * thrust_y])

    # d_chute: 伞扰动
    F_parachute = jnp.array([drag_fx, drag_fy])

    F_total = F_gravity + F_drag_linear + F_control + F_parachute

    # ── 积分：半隐式 Euler ──
    ax, ay = F_total / mass
    new_vx = vx + ax * dt
    new_vy = vy + ay * dt
    new_x = x + new_vx * dt
    new_y = y + new_vy * dt

    # 地面碰撞
    hit = new_y <= 0.0
    new_y = jnp.where(hit, 0.0, new_y)
    new_vy = jnp.where(hit, -new_vy * 0.3, new_vy)

    # ── Eq 5: 能量消耗 ──
    P_parasite = 0.5 * rho * speed**3 * CD_eff * A_eff
    P_total = P_hover + k_power * (thrust_x**2 + thrust_y**2) + P_parasite
    new_energy = jnp.maximum(energy - P_total * dt, E_min)

    return jnp.array([new_x, new_y, new_vx, new_vy, new_energy])


# ════════════════════════════════════════════════════════════════
# 批量仿真（jax.lax.scan）
# ════════════════════════════════════════════════════════════════

@jax.jit
def batch_faulty_simulate(state0, controls, faults, params):
    """
    批量含故障仿真

    Args:
        state0:  [5] 初始状态
        controls: [N, 2] 控制序列
        faults:  [N, 3] 故障序列（可随时间变化）
        params:  物理参数 tuple

    Returns:
        trajectory: [N, 5]
    """
    def scan_fn(carry, inputs):
        s = carry
        ctrl, flt = inputs
        s_new = faulty_physics_step(s, ctrl, flt, params)
        return s_new, s_new

    _, traj = jax.lax.scan(scan_fn, state0, (controls, faults))
    return traj


# ════════════════════════════════════════════════════════════════
# 参数打包工具
# ════════════════════════════════════════════════════════════════

def pack_params(cfg):
    """将 config 对象打包为 JAX-compatible tuple"""
    p = cfg.physics
    perc = cfg.perception
    B0 = B0_DEFAULT
    return (
        p.g, p.mass, p.drag_k, p.dt,
        float(B0[0, 0]), float(B0[0, 1]),
        float(B0[1, 0]), float(B0[1, 1]),
        p.rho_air, p.CD_nominal, p.A_nominal, p.l_arm,
        perc.E_min, perc.P_hover, perc.power_k,
    )


# ════════════════════════════════════════════════════════════════
# 测试
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '..')
    from config import SystemConfig

    cfg = SystemConfig()
    params = pack_params(cfg)

    # 正常状态
    state = jnp.array([0.0, 40.0, 5.0, 0.0, 1000.0])
    ctrl = jnp.array([10.0, 15.0])  # 10N x, 15N y
    fault = jnp.array([0.0, 0.0, 0.0])  # 无故障

    s1 = faulty_physics_step(state, ctrl, fault, params)
    print(f"Normal:     {state.tolist()} → {s1.tolist()}")

    # 伞完全失效
    fault_chute = jnp.array([1.0, 0.0, 0.0])
    s2 = faulty_physics_step(state, ctrl, fault_chute, params)
    print(f"No chute:   {state.tolist()} → {s2.tolist()}")

    # 50%动力降级
    fault_power = jnp.array([0.0, 0.5, 0.0])
    s3 = faulty_physics_step(state, ctrl, fault_power, params)
    print(f"50% power:  {state.tolist()} → {s3.tolist()}")

    # 复合故障
    fault_combo = jnp.array([0.5, 0.5, 0.3])
    s4 = faulty_physics_step(state, ctrl, fault_combo, params)
    print(f"Combo:      {state.tolist()} → {s4.tolist()}")

    # 性能测试
    import time
    t0 = time.time()
    for _ in range(10000):
        state = faulty_physics_step(state, ctrl, fault, params)
    state.block_until_ready()
    print(f"\n10k steps: {time.time()-t0:.3f}s → {10000/(time.time()-t0):,.0f} steps/s")
