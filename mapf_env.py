"""
MAPF Environment — 网格地图，支持障碍物、智能体起点/终点。
"""

from typing import List, Tuple, Set

# 四方向移动 + 原地等待
MOVES = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]


class MAPFEnv:
    """基于二维网格的 Multi-Agent Path Finding 环境。

    Attributes:
        width:  地图宽度（列数）
        height: 地图高度（行数）
        obstacles: 障碍物坐标集合
        starts:  各智能体起点列表
        goals:   各智能体终点列表
    """

    def __init__(
        self,
        width: int,
        height: int,
        obstacles: List[Tuple[int, int]],
        starts: List[Tuple[int, int]],
        goals: List[Tuple[int, int]],
    ):
        assert len(starts) == len(goals), "起点和终点数量必须一致"
        self.width = width
        self.height = height
        self.obstacles: Set[Tuple[int, int]] = set(obstacles)
        self.starts = starts
        self.goals = goals
        self.num_agents = len(starts)

        # 校验合法性
        for s in starts:
            assert self._in_bounds(s) and s not in self.obstacles, f"起点 {s} 不合法"
        for g in goals:
            assert self._in_bounds(g) and g not in self.obstacles, f"终点 {g} 不合法"

    # ------ 查询接口 ------

    def _in_bounds(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        return 0 <= r < self.height and 0 <= c < self.width

    def is_free(self, pos: Tuple[int, int]) -> bool:
        """位置是否可通行（在边界内且非障碍）。"""
        return self._in_bounds(pos) and pos not in self.obstacles

    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """返回 pos 的所有合法后继位置（含原地等待）。"""
        result = []
        for dr, dc in MOVES:
            nxt = (pos[0] + dr, pos[1] + dc)
            if self.is_free(nxt):
                result.append(nxt)
        return result

    # ------ 可视化 ------

    def render(self, paths: List[List[Tuple[int, int]]] = None, timestep: int = 0):
        """打印地图。若提供 paths，则显示 timestep 时刻各智能体位置。"""
        agent_positions = {}
        if paths:
            for i, path in enumerate(paths):
                t = min(timestep, len(path) - 1)
                agent_positions[path[t]] = i

        for r in range(self.height):
            row_str = ""
            for c in range(self.width):
                pos = (r, c)
                if pos in agent_positions:
                    row_str += f" {agent_positions[pos]} "
                elif pos in self.obstacles:
                    row_str += " @ "
                elif pos in self.starts:
                    row_str += " S "
                elif pos in self.goals:
                    row_str += " G "
                else:
                    row_str += " . "
            print(row_str)
        print()
