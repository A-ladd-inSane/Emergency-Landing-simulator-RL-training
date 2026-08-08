"""
config.py — 全局参数配置

对应技术方案中各公式的物理参数、安全阈值、训练超参。
所有参数集中管理，便于适航审查时的可追溯性。
"""

from dataclasses import dataclass, field
from typing import NamedTuple
import numpy as np


# ════════════════════════════════════════════════════════════════
# 1. 物理参数 (对应 §3.4 数学形式化)
# ════════════════════════════════════════════════════════════════

@dataclass
class PhysicsConfig:
    """飞行汽车（小球简化）物理参数"""
    # 基本物理
    g: float = 9.81              # 重力加速度 m/s²
    mass: float = 1.0            # 质量 kg
    drag_k: float = 0.05         # 线性阻力系数 N·s/m
    dt: float = 0.01             # 时间步长 s（100Hz）

    # 标称控制矩阵 B₀（2D简化：[thrust_x, thrust_y]）
    # 完整版为 m×n 矩阵，此处简化为对角阵
    B0_nominal: tuple = (1.0, 1.0)  # 标称控制有效性

    # 伞系统参数 (Eq 3)
    rho_air: float = 1.225        # 空气密度 kg/m³
    CD_nominal: float = 1.0       # 标称阻力系数
    A_nominal: float = 0.5        # 标称展开面积 m²
    l_arm: float = 0.3             # 伞挂点力臂 m

    # 故障不确定度
    CD_uncertainty: float = 0.5   # C_D ±50% (§4.2.3)
    A_uncertainty: float = 0.5    # A ±50%
    mass_uncertainty: float = 0.1 # 质量 ±10%
    battery_uncertainty: float = 0.3 # 内阻 ±30%

    # 地面碰撞
    bounce_damping: float = 0.3   # 弹跳能量保留率

    # 边界
    x_boundary: float = 200.0     # 水平边界 m
    y_ground: float = 0.0         # 地面高度 m


# ════════════════════════════════════════════════════════════════
# 2. 感知层参数 (对应 §4.2)
# ════════════════════════════════════════════════════════════════

@dataclass
class PerceptionConfig:
    """感知层参数"""
    # GLR 故障检测 (Eq 7-9)
    glr_window: int = 20           # 滑动窗口 N
    glr_threshold_alpha: float = 1e-6  # 虚警概率上界
    glr_chi2_dof: int = 4          # 卡方自由度（状态维度）

    # RLS 在线辨识 (Eq 11-13)
    rls_lambda_f: float = 0.98    # 快变遗忘因子（跟踪阶跃）
    rls_lambda_s: float = 0.999   # 慢变遗忘因子（跟踪退化）
    rls_P0_scale: float = 100.0   # 协方差重置值 σ²
    rls_convergence_steps: int = 50  # 收敛步数估计

    # SVD 截断 (Eq 14)
    svd_kappa_machine: float = 2.2e-16  # 机器精度
    svd_delta_sensor: float = 1e-4     # 传感器噪声等效奇异值下界

    # 退化度 (Eq 4)
    degradation_alpha: float = 0.7  # 加权系数 α
    degradation_epsilon: float = 1e-3  # 截断阈值

    # 续航评估 (Eq 5)
    E_min: float = 50.0            # 最低安全能量 J
    P_hover: float = 20.0         # 悬停平均功率 W
    power_k: float = 0.01         # 控制力矩平方功率系数
    cos_bias: float = 1.0         # 伞非对称效应简化

    # 降落点检索 (Eq 16-17)
    landing_d0: float = 100.0      # 距离特征尺度
    landing_rho0: float = 10.0    # 人口密度特征尺度
    landing_sigma0: float = 1.0    # 粗糙度特征尺度
    landing_w1: float = 1.0        # 距离权重
    landing_w2: float = 0.5        # 人口权重
    landing_w3: float = 0.3        # 粗糙度权重

    # 两阶段决策时间线 (§4.3.6)
    T0_budget: float = 0.1        # T0 阶段预算 100ms
    T1_budget: float = 2.0        # T1 阶段预算 2s
    RLS_convergence_time: float = 0.7  # RLS收敛约700ms


# ════════════════════════════════════════════════════════════════
# 3. 决策层参数 (对应 §4.3)
# ════════════════════════════════════════════════════════════════

@dataclass
class DecisionConfig:
    """决策层参数"""
    # PPO 超参
    policy_lr: float = 3e-4       # 策略学习率
    value_lr: float = 1e-3        # 价值函数学习率
    gamma: float = 0.99           # 折扣因子
    gae_lambda: float = 0.95     # GAE 参数
    clip_range: float = 0.2       # PPO clip
    ent_coef: float = 0.01       # 熵系数
    n_steps: int = 2048           # 每轮采样步数
    batch_size: int = 64
    n_epochs: int = 10

    # CMDP Lagrangian (Eq 20)
    lambda_lag_init: float = 0.0  # 初始 Lagrangian 乘子
    lambda_lag_lr: float = 0.05  # λ 自适应学习率
    d_safe: float = 10.0          # 安全预算上界

    # CBF/CLF QP (Eq 22-23)
    cbf_beta: float = 1.0         # CBF 衰减率
    clf_alpha: float = 1.0        # CLF 衰减率
    qp_rho1: float = 1e3          # CLF 松弛惩罚权重
    qp_rho2: float = 1.0         # CBF 松弛惩罚权重
    cbf_dtcbf_Ts: float = 0.01   # 离散时间周期

    # FME (Eq 19)
    fme_N_steps: int = 50         # 前向仿真步数
    fme_n_controls: int = 10      # 控制离散化粒度

    # 可微梯度注入 (Eq 21)
    grad_inject_init: float = 0.1  # 初始注入系数
    grad_inject_decay: float = 0.99  # 退火衰减率

    # OOD 检测
    ood_threshold: float = 2.0   # 能量分数阈值

    # 策略网络
    policy_hidden: tuple = (64, 64)  # MLP 隐藏层
    policy_log_std_init: float = 0.0  # 初始 log std


# ════════════════════════════════════════════════════════════════
# 4. 执行层参数 (对应 §4.4)
# ════════════════════════════════════════════════════════════════

@dataclass
class ExecutionConfig:
    """执行层参数"""
    # 控制分配 (Eq 26)
    max_thrust: float = 50.0      # 最大推力 N
    kappa_warn: float = 10.0      # 条件数警告阈值
    kappa_degrade: float = 100.0  # 条件数降级阈值

    # LQR (Eq 27)
    lqr_Q: tuple = (10.0, 10.0, 1.0, 1.0)  # 状态权重
    lqr_R: tuple = (0.1, 0.1)             # 控制权重

    # 着陆 (Eq 28)
    h_hover: float = 5.0          # 悬停高度阈值 m
    v_descent: float = 1.5        # 标称下降速率 m/s

    # 轨迹跟踪
    tracker_dt: float = 0.01      # 跟踪控制周期


# ════════════════════════════════════════════════════════════════
# 5. 评估参数 (对应 §5.6)
# ════════════════════════════════════════════════════════════════

@dataclass
class EvalConfig:
    """评估参数"""
    # 评估指标 (§5.6.1)
    d_impact_crit: float = 5.0    # 着陆冲击距离阈值 m
    v_touch_max: float = 3.0     # 触地速度上限 m/s

    # 统计验证 (§5.6.2)
    p_safe_target: float = 0.95   # 着陆成功率目标
    p_casualty_target: float = 1e-4  # 地面伤亡率目标
    confidence_level: float = 0.95    # 置信水平
    mc_n_trials: int = 150       # 蒙特卡洛试验次数
    mc_allowed_failures: int = 5  # 允许失败次数

    # 对抗测试 (§5.6.5)
    adversarial_wind_max: float = 15.0  # 极端风速 m/s
    adversarial_multi_fault: bool = True  # 多重故障叠加


# ════════════════════════════════════════════════════════════════
# 6. 全局配置聚合
# ════════════════════════════════════════════════════════════════

@dataclass
class SystemConfig:
    """全局系统配置"""
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # 随机种子
    seed: int = 42

    # 日志
    log_dir: str = "./logs"
    model_dir: str = "./models"
    log_arinc653: bool = True     # ARINC 653 格式结构化日志


# 全局单例
CONFIG = SystemConfig()
