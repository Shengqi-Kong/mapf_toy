"""
数据生成 — 随机 MAPF 实例生成 + CBS 专家数据采集。

用法:
    python data_gen.py --num_instances 500 --num_agents 4 --map_size 8 --output expert_data.pt
"""

import argparse
import random
from collections import deque
from typing import List, Tuple, Optional

import torch
from torch_geometric.data import Data

from mapf_env import MAPFEnv, MOVES
from cbs import cbs_search


# ---------- 观测构建 ----------

def build_graph_observation(
    env: MAPFEnv,
    positions: List[Tuple[int, int]],
    goals: List[Tuple[int, int]],
    fov_radius: int = 2,
    comm_radius: int = 5,
) -> Data:
    """为当前时间步构建所有 agent 的图观测。

    节点特征:
      spatial (3, fov_size, fov_size):
        - channel 0: 障碍物/越界
        - channel 1: 其他 agent
        - channel 2: 其他目标
      scalar (4,):
        - 归一化位置 (2)
        - 相对目标方向 (2)

    边: 曼哈顿距离 <= comm_radius 的 agent 对之间建立无向边。
    """
    num_agents = len(positions)
    fov_size = 2 * fov_radius + 1

    pos_set = set(positions)
    goal_set = set(goals)

    spatial_list = []
    scalar_list = []

    for i in range(num_agents):
        r, c = positions[i]
        gr, gc = goals[i]

        # 标量特征
        scalar = [
            r / max(env.height - 1, 1),
            c / max(env.width - 1, 1),
            (gr - r) / max(env.height, 1),
            (gc - c) / max(env.width, 1),
        ]
        scalar_list.append(scalar)

        # 空间特征: 3 个 channel, 每个 fov_size x fov_size
        channels = [[[0.0] * fov_size for _ in range(fov_size)] for _ in range(3)]
        for dr_idx, dr in enumerate(range(-fov_radius, fov_radius + 1)):
            for dc_idx, dc in enumerate(range(-fov_radius, fov_radius + 1)):
                nr, nc = r + dr, c + dc
                # channel 0: 障碍物/越界
                if not env._in_bounds((nr, nc)) or (nr, nc) in env.obstacles:
                    channels[0][dr_idx][dc_idx] = 1.0
                # channel 1: 其他 agent
                if (nr, nc) in pos_set and (nr, nc) != (r, c):
                    channels[1][dr_idx][dc_idx] = 1.0
                # channel 2: 其他目标
                if (nr, nc) in goal_set and (nr, nc) != goals[i]:
                    channels[2][dr_idx][dc_idx] = 1.0

        spatial_list.append(channels)

    spatial = torch.tensor(spatial_list, dtype=torch.float)  # [N, 3, fov, fov]
    scalar = torch.tensor(scalar_list, dtype=torch.float)    # [N, 4]

    # 通信图
    src, dst = [], []
    for i in range(num_agents):
        for j in range(num_agents):
            dist = abs(positions[i][0] - positions[j][0]) + abs(positions[i][1] - positions[j][1])
            if dist <= comm_radius:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    return Data(spatial=spatial, scalar=scalar, edge_index=edge_index, num_nodes=num_agents)


# ---------- 随机实例生成 ----------

def _bfs_reachable(env: MAPFEnv, start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
    """BFS 检查 start 能否到达 goal。"""
    if start == goal:
        return True
    visited = {start}
    queue = deque([start])
    while queue:
        pos = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nxt = (pos[0] + dr, pos[1] + dc)
            if nxt not in visited and env.is_free(nxt):
                if nxt == goal:
                    return True
                visited.add(nxt)
                queue.append(nxt)
    return False


def generate_random_instance(
    width: int = 8,
    height: int = 8,
    num_agents: int = 4,
    obstacle_density: float = 0.1,
    rng: random.Random = None,
) -> Optional[MAPFEnv]:
    """随机生成一个合法的 MAPF 实例。返回 None 表示生成失败。"""
    if rng is None:
        rng = random.Random()

    all_cells = [(r, c) for r in range(height) for c in range(width)]
    num_obstacles = int(len(all_cells) * obstacle_density)

    for _ in range(50):  # 最多重试 50 次
        rng.shuffle(all_cells)
        obstacles = all_cells[:num_obstacles]
        obstacle_set = set(obstacles)
        free_cells = [cell for cell in all_cells if cell not in obstacle_set]

        if len(free_cells) < 2 * num_agents:
            continue

        rng.shuffle(free_cells)
        starts = free_cells[:num_agents]
        goals = free_cells[num_agents: 2 * num_agents]

        env = MAPFEnv(width, height, obstacles, starts, goals)

        # 检查每个 agent 的起点能到达终点
        if all(_bfs_reachable(env, s, g) for s, g in zip(starts, goals)):
            return env

    return None


# ---------- 专家数据采集 ----------

def collect_expert_data(
    num_instances: int = 500,
    width: int = 8,
    height: int = 8,
    num_agents: int = 4,
    obstacle_density: float = 0.1,
    fov_radius: int = 2,
    comm_radius: int = 5,
    seed: int = 42,
    mixed_agents: str = None,
) -> List[Data]:
    """生成随机实例，用 CBS 求解，采集 (观测, 专家动作) 对。

    Args:
        mixed_agents: 混合 agent 数量范围，格式 "min-max"，如 "2-5"。
                      设置后 num_agents 参数被忽略，每个实例随机选择 agent 数量。
    """
    rng = random.Random(seed)
    dataset = []
    solved = 0

    agent_range = None
    if mixed_agents:
        lo, hi = map(int, mixed_agents.split("-"))
        agent_range = (lo, hi)
        print(f"混合 agent 模式: {lo}-{hi} agents")

    for idx in range(num_instances):
        na = rng.randint(agent_range[0], agent_range[1]) if agent_range else num_agents
        env = generate_random_instance(width, height, na, obstacle_density, rng)
        if env is None:
            continue

        solution = cbs_search(env)
        if solution is None:
            continue

        solved += 1
        max_t = max(len(p) for p in solution)

        for t in range(max_t - 1):
            positions = [p[min(t, len(p) - 1)] for p in solution]
            goals = list(env.goals)

            data = build_graph_observation(env, positions, goals, fov_radius, comm_radius)

            # 提取专家动作
            actions = []
            for i, path in enumerate(solution):
                curr = path[min(t, len(path) - 1)]
                nxt = path[min(t + 1, len(path) - 1)]
                move = (nxt[0] - curr[0], nxt[1] - curr[1])
                actions.append(MOVES.index(move))
            data.y = torch.tensor(actions, dtype=torch.long)
            data.idx = len(dataset)

            dataset.append(data)

        if (idx + 1) % 50 == 0:
            print(f"  进度: {idx + 1}/{num_instances} 实例, 已求解 {solved}, 数据量 {len(dataset)}")

    print(f"完成: {solved}/{num_instances} 实例成功求解, 共 {len(dataset)} 条数据")
    return dataset


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(description="生成 MAPF 模仿学习专家数据")
    parser.add_argument("--num_instances", type=int, default=500)
    parser.add_argument("--num_agents", type=int, default=4)
    parser.add_argument("--mixed_agents", type=str, default=None,
                        help="混合 agent 数量范围，如 '2-5'，设置后忽略 --num_agents")
    parser.add_argument("--map_size", type=int, default=8)
    parser.add_argument("--obstacle_density", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="expert_data.pt")
    args = parser.parse_args()

    print(f"生成专家数据: {args.num_instances} 实例, {args.num_agents} agents, "
          f"{args.map_size}x{args.map_size} 地图, 障碍密度 {args.obstacle_density}")

    dataset = collect_expert_data(
        num_instances=args.num_instances,
        width=args.map_size,
        height=args.map_size,
        num_agents=args.num_agents,
        obstacle_density=args.obstacle_density,
        seed=args.seed,
        mixed_agents=args.mixed_agents,
    )

    torch.save(dataset, args.output)
    print(f"数据已保存至 {args.output}")


if __name__ == "__main__":
    main()
