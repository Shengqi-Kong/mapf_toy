"""
ECBS (Enhanced Conflict-Based Search) — 次优 MAPF 求解算法。

参考: Barer et al., "Suboptimal Variants of the Conflict-Based Search Algorithm
      for the Multi-Agent Pathfinding Problem", SoCS 2014.

次优界 w: 返回的 SoC <= w * 最优 SoC。w=1.0 退化为标准 CBS。
"""

import heapq
import time
from typing import Dict, List, Optional, Set, Tuple

from mapf_env import MAPFEnv

Constraint = tuple


# ---------- 数据结构 ----------

class ECBSNode:
    """Constraint-Tree 节点，额外记录冲突数用于 focal 排序。"""

    __slots__ = ("constraints", "solution", "cost", "num_conflicts")

    def __init__(self):
        self.constraints: List[Constraint] = []
        self.solution: List[List[Tuple[int, int]]] = []
        self.cost: int = 0
        self.num_conflicts: int = 0

    def __lt__(self, other: "ECBSNode"):
        # OPEN 堆按 cost 排序
        return self.cost < other.cost


# ---------- 冲突检测 ----------

def _get_pos(path: List[Tuple[int, int]], t: int) -> Tuple[int, int]:
    return path[min(t, len(path) - 1)]


def _detect_first_conflict(solution: List[List[Tuple[int, int]]]) -> Optional[dict]:
    n = len(solution)
    max_t = max(len(p) for p in solution)
    for t in range(max_t):
        for i in range(n):
            pi = _get_pos(solution[i], t)
            for j in range(i + 1, n):
                pj = _get_pos(solution[j], t)
                if pi == pj:
                    return {"type": "vertex", "agents": (i, j), "pos": pi, "timestep": t}
        if t > 0:
            for i in range(n):
                prev_i = _get_pos(solution[i], t - 1)
                curr_i = _get_pos(solution[i], t)
                for j in range(i + 1, n):
                    prev_j = _get_pos(solution[j], t - 1)
                    curr_j = _get_pos(solution[j], t)
                    if prev_i == curr_j and prev_j == curr_i:
                        return {"type": "edge", "agents": (i, j),
                                "pos": (prev_i, curr_i), "timestep": t}
    return None


def _count_conflicts(solution: List[List[Tuple[int, int]]]) -> int:
    """统计 solution 中所有冲突数（用于 focal 启发）。"""
    n = len(solution)
    max_t = max(len(p) for p in solution)
    count = 0
    for t in range(max_t):
        positions = [_get_pos(solution[i], t) for i in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if positions[i] == positions[j]:
                    count += 1
        if t > 0:
            for i in range(n):
                for j in range(i + 1, n):
                    if (_get_pos(solution[i], t - 1) == _get_pos(solution[j], t) and
                            _get_pos(solution[j], t - 1) == _get_pos(solution[i], t)):
                        count += 1
    return count


# ---------- 低层 focal A* ----------

def _build_constraint_table(constraints: List[Constraint], agent: int):
    vertex_table: Dict[int, Set[Tuple[int, int]]] = {}
    edge_table: Dict[int, Set] = {}
    for c in constraints:
        if c[0] != agent:
            continue
        if len(c) == 3:
            _, pos, t = c
            vertex_table.setdefault(t, set()).add(pos)
        elif len(c) == 4:
            _, pf, pt, t = c
            edge_table.setdefault(t, set()).add((pf, pt))
    return vertex_table, edge_table


def _future_vertex_conflict(vertex_table, goal, current_t) -> bool:
    for t, positions in vertex_table.items():
        if t > current_t and goal in positions:
            return True
    return False


def _conflict_count_at(
    pos: Tuple[int, int],
    t: int,
    other_paths: List[List[Tuple[int, int]]],
    prev_pos: Optional[Tuple[int, int]] = None,
) -> int:
    """计算 (pos, t) 与其他智能体路径的冲突数，用作 focal 启发值。"""
    count = 0
    for path in other_paths:
        other_pos = _get_pos(path, t)
        if other_pos == pos:
            count += 1
        # 边冲突
        if prev_pos is not None and t > 0:
            other_prev = _get_pos(path, t - 1)
            if other_prev == pos and other_pos == prev_pos:
                count += 1
    return count


def _focal_low_level(
    env: MAPFEnv,
    agent: int,
    constraints: List[Constraint],
    other_paths: List[List[Tuple[int, int]]],
    w: float,
    deadline: float = None,
) -> Optional[List[Tuple[int, int]]]:
    """带 focal 启发的时空 A*。"""
    start = env.starts[agent]
    goal = env.goals[agent]
    vertex_table, edge_table = _build_constraint_table(constraints, agent)
    max_t = min(env.width * env.height, 4096)

    def h(pos):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    # OPEN: (f, g, conflict_count, pos, t, prev_pos)
    start_state = (h(start), 0, 0, start, 0, None)
    open_list = [start_state]
    # 记录每个 (pos, t) 的最优 g 值
    best_g: Dict[Tuple, int] = {(start, 0): 0}
    came_from: Dict[Tuple, Tuple] = {}

    # focal bound 基于 open_list 中最小 f
    f_min = h(start)
    iters = 0

    while open_list:
        iters += 1
        if iters % 2000 == 0 and deadline is not None and time.time() > deadline:
            return None  # 低层超时

        # 从 OPEN 中找 focal 候选 (f <= w * f_min) 中 conflict_count 最小的
        focal_bound = w * f_min
        best_idx = None
        best_fc = None
        for idx, item in enumerate(open_list):
            f_val, g_val, fc, pos, t, prev = item
            if f_val <= focal_bound:
                if best_fc is None or fc < best_fc or (fc == best_fc and f_val < open_list[best_idx][0]):
                    best_fc = fc
                    best_idx = idx

        if best_idx is None:
            # focal 为空，退回到普通 A* 最优节点
            best_idx = 0
            for idx in range(1, len(open_list)):
                if open_list[idx][0] < open_list[best_idx][0]:
                    best_idx = idx

        f_val, g_val, fc, pos, t, prev = open_list.pop(best_idx)

        # 更新 f_min
        if open_list:
            f_min = min(item[0] for item in open_list)

        state_key = (pos, t)
        if best_g.get(state_key, float('inf')) < g_val:
            continue

        if pos == goal and not _future_vertex_conflict(vertex_table, goal, t):
            return _reconstruct(came_from, state_key)

        if t >= max_t:
            continue

        for nxt in env.get_neighbors(pos):
            nt = t + 1
            if nt in vertex_table and nxt in vertex_table[nt]:
                continue
            if nt in edge_table and (pos, nxt) in edge_table[nt]:
                continue

            new_g = g_val + 1
            nxt_key = (nxt, nt)
            if best_g.get(nxt_key, float('inf')) <= new_g:
                continue

            best_g[nxt_key] = new_g
            came_from[nxt_key] = state_key
            new_f = new_g + h(nxt)
            new_fc = fc + _conflict_count_at(nxt, nt, other_paths, pos)
            heapq.heappush(open_list, (new_f, new_g, new_fc, nxt, nt, pos))

    return None


def _reconstruct(came_from, state):
    path = [state[0]]
    while state in came_from:
        state = came_from[state]
        path.append(state[0])
    path.reverse()
    return path


# ---------- ECBS 高层搜索 ----------

def ecbs_search(
    env: MAPFEnv,
    w: float = 1.5,
    timeout: float = None,
) -> Optional[List[List[Tuple[int, int]]]]:
    """ECBS 主函数。返回各智能体路径或 None（无解/超时）。"""
    deadline = time.time() + timeout if timeout is not None else None

    root = ECBSNode()
    root.constraints = []

    # 初始无约束规划（用空 other_paths）
    for i in range(env.num_agents):
        other = [p for j, p in enumerate(root.solution) if j != i]
        path = _focal_low_level(env, i, [], other, w, deadline)
        if path is None:
            return None
        root.solution.append(path)

    root.cost = sum(len(p) - 1 for p in root.solution)
    root.num_conflicts = _count_conflicts(root.solution)

    open_list: List[ECBSNode] = [root]

    while open_list:
        if deadline is not None and time.time() > deadline:
            return None  # 超时

        f_min = open_list[0].cost
        focal_bound = w * f_min

        # 从 focal 集合中选冲突数最少的节点
        best_node = None
        best_idx = None
        for idx, node in enumerate(open_list):
            if node.cost <= focal_bound:
                if best_node is None or node.num_conflicts < best_node.num_conflicts:
                    best_node = node
                    best_idx = idx

        if best_node is None:
            best_idx = 0
            best_node = open_list[0]

        open_list.pop(best_idx)
        # 维护堆性质
        heapq.heapify(open_list)

        if best_node.num_conflicts == 0:
            return best_node.solution

        conflict = _detect_first_conflict(best_node.solution)
        if conflict is None:
            return best_node.solution

        i, j = conflict["agents"]
        for agent in (i, j):
            child = ECBSNode()
            child.constraints = list(best_node.constraints)

            if conflict["type"] == "vertex":
                child.constraints.append((agent, conflict["pos"], conflict["timestep"]))
            else:
                pos_from, pos_to = conflict["pos"]
                if agent == i:
                    child.constraints.append((agent, pos_from, pos_to, conflict["timestep"]))
                else:
                    child.constraints.append((agent, pos_to, pos_from, conflict["timestep"]))

            child.solution = [list(p) for p in best_node.solution]
            other_paths = [child.solution[k] for k in range(env.num_agents) if k != agent]
            new_path = _focal_low_level(env, agent, child.constraints, other_paths, w, deadline)
            if new_path is None:
                continue

            child.solution[agent] = new_path
            child.cost = sum(len(p) - 1 for p in child.solution)
            child.num_conflicts = _count_conflicts(child.solution)
            heapq.heappush(open_list, child)

    return None
