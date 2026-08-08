"""
perception/landing_points.py — 降落点检索

对应技术方案 §4.3.1 Eq 16-17。

KD-Tree + 多目标评分：
  - 地形坡度（平坦优先）
  - 风场适配（逆风降落）
  - 可达性（FME 包络内）
  - 障碍距离（远离建筑）

输出 Top-K 候选降落点，按综合得分排序。
"""

import numpy as np
from scipy.spatial import KDTree
from dataclasses import dataclass, field
from typing import List


@dataclass
class LandingPoint:
    """降落点候选"""
    x: float
    y: float
    slope: float         # 地形坡度
    wind_alignment: float  # 风场适配 [0,1]
    reachable: bool      # 是否在FME包络内
    obstacle_dist: float # 距最近障碍距离 m
    score: float = 0.0   # 综合得分


@dataclass
class LandingPointSearcher:
    """
    KD-Tree 降落点检索 (Eq 16-17)

    维护一个地形图数据库，支持：
      - 空间近邻检索（KD-Tree）
      - 多目标评分（Eq 16）
      - Top-K 排序（Eq 17）
    """
    candidates: np.ndarray = None  # [N, 2] 候选点坐标
    slopes: np.ndarray = None      # [N] 地形坡度
    obstacles: np.ndarray = None   # [M, 2] 障碍物坐标
    _kdtree: KDTree = field(default=None, repr=False)
    _obs_tree: KDTree = field(default=None, repr=False)

    def __init__(self, candidates: np.ndarray, slopes: np.ndarray,
                 obstacles: np.ndarray = None):
        self.candidates = np.asarray(candidates, dtype=float)
        self.slopes = np.asarray(slopes, dtype=float)
        self._kdtree = KDTree(self.candidates)

        if obstacles is not None:
            self.obstacles = np.asarray(obstacles, dtype=float)
            self._obs_tree = KDTree(self.obstacles)

    def search(self, current_pos: np.ndarray, wind_vec: np.ndarray,
               fme_bounds: tuple = None, top_k: int = 5) -> List[LandingPoint]:
        """
        检索最优降落点

        Args:
            current_pos: [x, y] 当前位置
            wind_vec: [wx, wy] 风向量
            fme_bounds: (min_x, max_x, min_y, max_y) FME包络
            top_k: 返回前K个

        Returns:
            排序后的降落点列表
        """
        # 空间近邻检索 (Eq 16)
        # 在当前位置附近 200m 范围内搜索
        radius = 200.0
        idx = self._kdtree.query_ball_point(current_pos, r=radius)

        if len(idx) == 0:
            return []

        results = []
        for i in idx:
            pos = self.candidates[i]
            slope = self.slopes[i]

            # 风场适配：逆风降落得分高
            to_point = pos - current_pos
            norm = np.linalg.norm(to_point) + 1e-10
            dir_to_point = to_point / norm
            wind_norm = np.linalg.norm(wind_vec) + 1e-10
            wind_dir = wind_vec / wind_norm
            # 逆风降落: 方向与风向相反
            wind_alignment = max(0, -dir_to_point @ wind_dir)

            # FME 可达性
            reachable = True
            if fme_bounds is not None:
                min_x, max_x, min_y, max_y = fme_bounds
                reachable = (min_x <= pos[0] <= max_x and
                             min_y <= pos[1] <= max_y)

            # 障碍距离
            if self._obs_tree is not None:
                obs_dist, _ = self._obs_tree.query(pos)
                obs_dist = float(obs_dist)
            else:
                obs_dist = 100.0

            # ── Eq 17: 多目标综合评分 ──
            # w1*slope + w2*wind + w3*distance + w4*obstacle
            dist = np.linalg.norm(pos - current_pos)
            score = (0.4 * (1 - min(slope, 1.0))  # 坡度
                     + 0.2 * wind_alignment        # 风场
                     + 0.2 * (1 - min(dist / radius, 1.0))  # 距离
                     + 0.2 * min(obs_dist / 50.0, 1.0))  # 障碍
            if not reachable:
                score *= 0.1  # 不可达大幅降分

            results.append(LandingPoint(
                x=float(pos[0]), y=float(pos[1]),
                slope=float(slope),
                wind_alignment=float(wind_alignment),
                reachable=reachable,
                obstacle_dist=float(obs_dist),
                score=float(score)
            ))

        # 排序
        results.sort(key=lambda p: p.score, reverse=True)
        return results[:top_k]


# ════════════════════════════════════════════════════════════════
# 测试
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)

    # 生成候选降落点网格
    grid_x, grid_y = np.meshgrid(
        np.linspace(-100, 100, 21),
        np.linspace(0, 80, 9)
    )
    candidates = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    slopes = np.random.rand(len(candidates)) * 0.3  # 坡度 0-0.3
    obstacles = np.array([[20, 30], [-30, 40], [50, 20]])  # 障碍物

    searcher = LandingPointSearcher(candidates, slopes, obstacles)

    # 搜索
    current_pos = np.array([10.0, 40.0])
    wind = np.array([-5.0, -2.0])  # 西风
    fme_bounds = (-80, 80, 0, 60)

    results = searcher.search(current_pos, wind, fme_bounds, top_k=5)

    print("=== 降落点检索结果 ===")
    print(f"当前位置: ({current_pos[0]}, {current_pos[1]})")
    print(f"风场: ({wind[0]}, {wind[1]}) m/s")
    print()
    for i, p in enumerate(results):
        print(f"  #{i+1}: pos=({p.x:+6.1f}, {p.y:5.1f}) "
              f"slope={p.slope:.2f} wind={p.wind_alignment:.2f} "
              f"obs={p.obstacle_dist:.1f}m "
              f"reach={'Y' if p.reachable else 'N'} "
              f"score={p.score:.3f}")
