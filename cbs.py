"""
CBS (Conflict-Based Search) — MAPF 的经典最优求解算法。

参考: Sharon et al., "Conflict-Based Search for Optimal Multi-Agent Pathfinding", AIJ 2015.
"""

import heapq
from typing import List, Tuple, Dict, Optional, Set
from mapf_env import MAPFEnv

# ---------- 数据结构 ----------

# 约束: (agent, position, timestep)  —— 顶点约束
# 约束: (agent, pos_from, pos_to, timestep) —— 边约束
Constraint = tuple


class CTNode:
    """Constraint-Tree 节点。"""

    def __init__(self):
        self.constraints: List[Constraint] = []
        self.solution: List[List[Tuple[int, int]]] = []  # 每个智能体的路径
        self.cost: int = 0  # SIC (Sum of Individual Costs)

    def __lt__(self, other: "CTNode"):
        return self.cost < other.cost


# ---------- 冲突检测 ----------

def _detect_first_conflict(
    solution: List[List[Tuple[int, int]]],
) -> Optional[dict]:
    """检测 solution 中最早的一个冲突，返回冲突描述 dict 或 None。"""
    num_agents = len(solution)
    max_t = max(len(p) for p in solution)

    for t in range(max_t):
        # 顶点冲突
        for i in range(num_agents):
            pos_i = _get_pos(solution[i], t)
            for j in range(i + 1, num_agents):
                pos_j = _get_pos(solution[j], t)
                if pos_i == pos_j:
                    return {
                        "type": "vertex",
                        "agents": (i, j),
                        "pos": pos_i,
                        "timestep": t,
                    }

        # 边冲突（交换位置）
        if t > 0:
            for i in range(num_agents):
                prev_i = _get_pos(solution[i], t - 1)
                curr_i = _get_pos(solution[i], t)
                for j in range(i + 1, num_agents):
                    prev_j = _get_pos(solution[j], t - 1)
                    curr_j = _get_pos(solution[j], t)
                    if prev_i == curr_j and prev_j == curr_i:
                        return {
                            "type": "edge",
                            "agents": (i, j),
                            "pos": (prev_i, curr_i),
                            "timestep": t,
                        }
    return None


def _get_pos(path: List[Tuple[int, int]], t: int) -> Tuple[int, int]:
    """到达终点后原地等待。"""
    return path[min(t, len(path) - 1)]


# ---------- 低层 A* (带时间维度 + 约束) ----------

def _build_constraint_table(
    constraints: List[Constraint], agent: int
) -> Dict[int, set]:
    """将约束按 timestep 分组，方便查询。"""
    vertex_table: Dict[int, Set[Tuple[int, int]]] = {}
    edge_table: Dict[int, Set[Tuple[Tuple[int, int], Tuple[int, int]]]] = {}

    for c in constraints:
        if c[0] != agent:
            continue
        if len(c) == 3:  # 顶点约束 (agent, pos, t)
            _, pos, t = c
            vertex_table.setdefault(t, set()).add(pos)
        elif len(c) == 4:  # 边约束 (agent, pos_from, pos_to, t)
            _, pf, pt, t = c
            edge_table.setdefault(t, set()).add((pf, pt))

    return vertex_table, edge_table


def _low_level_search(
    env: MAPFEnv,
    agent: int,
    constraints: List[Constraint],
) -> Optional[List[Tuple[int, int]]]:
    """带约束的时空 A* 搜索，为单个智能体规划路径。"""
    start = env.starts[agent]
    goal = env.goals[agent]
    vertex_table, edge_table = _build_constraint_table(constraints, agent)

    # 搜索上界：地图大小 * 智能体数（足够宽松）
    max_timestep = env.width * env.height * env.num_agents

    def h(pos):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    # (f, g, pos, timestep)
    open_list = [(h(start), 0, start, 0)]
    closed_set: Set[Tuple[Tuple[int, int], int]] = set()

    came_from: Dict[Tuple[Tuple[int, int], int], Tuple[Tuple[int, int], int]] = {}

    while open_list:
        f, g, pos, t = heapq.heappop(open_list)

        if (pos, t) in closed_set:
            continue
        closed_set.add((pos, t))

        # 到达目标 —— 还需确认后续时间步不存在约束
        if pos == goal and not _future_vertex_conflict(vertex_table, goal, t):
            return _reconstruct(came_from, (pos, t))

        if t >= max_timestep:
            continue

        for nxt in env.get_neighbors(pos):
            nt = t + 1
            # 顶点约束检查
            if nt in vertex_table and nxt in vertex_table[nt]:
                continue
            # 边约束检查
            if nt in edge_table and (pos, nxt) in edge_table[nt]:
                continue
            if (nxt, nt) in closed_set:
                continue

            new_g = g + 1
            new_f = new_g + h(nxt)
            came_from[(nxt, nt)] = (pos, t)
            heapq.heappush(open_list, (new_f, new_g, nxt, nt))

    return None  # 无解


def _future_vertex_conflict(
    vertex_table: Dict[int, set], goal: Tuple[int, int], current_t: int
) -> bool:
    """检查目标位置在 current_t 之后是否还有顶点约束。"""
    for t, positions in vertex_table.items():
        if t > current_t and goal in positions:
            return True
    return False


def _reconstruct(
    came_from: dict,
    state: Tuple[Tuple[int, int], int],
) -> List[Tuple[int, int]]:
    path = [state[0]]
    while state in came_from:
        state = came_from[state]
        path.append(state[0])
    path.reverse()
    return path


# ---------- CBS 高层搜索 ----------

def cbs_search(env: MAPFEnv) -> Optional[List[List[Tuple[int, int]]]]:
    """CBS 主函数。返回各智能体的路径列表，或 None（无解）。"""
    root = CTNode()
    root.constraints = []

    # 为每个智能体做无约束 A*
    for i in range(env.num_agents):
        path = _low_level_search(env, i, [])
        if path is None:
            return None
        root.solution.append(path)
    root.cost = sum(len(p) - 1 for p in root.solution)

    open_list: List[CTNode] = [root]

    while open_list:
        node = heapq.heappop(open_list)

        conflict = _detect_first_conflict(node.solution)
        if conflict is None:
            return node.solution  # 无冲突，找到最优解

        i, j = conflict["agents"]

        # 对冲突的两个智能体分别添加约束，生成两个子节点
        for agent in (i, j):
            child = CTNode()
            child.constraints = list(node.constraints)

            if conflict["type"] == "vertex":
                child.constraints.append(
                    (agent, conflict["pos"], conflict["timestep"])
                )
            else:  # edge
                pos_from, pos_to = conflict["pos"]
                if agent == i:
                    child.constraints.append(
                        (agent, pos_from, pos_to, conflict["timestep"])
                    )
                else:
                    child.constraints.append(
                        (agent, pos_to, pos_from, conflict["timestep"])
                    )

            # 只需重新规划受约束的智能体
            child.solution = [list(p) for p in node.solution]
            new_path = _low_level_search(env, agent, child.constraints)
            if new_path is None:
                continue  # 该分支无解，剪枝
            child.solution[agent] = new_path
            child.cost = sum(len(p) - 1 for p in child.solution)
            heapq.heappush(open_list, child)

    return None  # 无解
