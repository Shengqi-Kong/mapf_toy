"""
map_loader.py — 解析 MovingAI .map 文件并生成 MAPFEnv 实例。

支持格式:
    type octile
    height H
    width W
    map
    ....@@@....
"""

import random
from collections import deque
from typing import List, Optional, Tuple

from mapf_env import MAPFEnv

# 可通行地块
FREE_TILES = {'.', 'S', 'G'}
# 障碍地块
BLOCK_TILES = {'@', 'T', 'W', 'O'}


def parse_map(filepath: str) -> Tuple[int, int, List[Tuple[int, int]]]:
    """解析 .map 文件，返回 (height, width, obstacles)。坐标为 (row, col)。"""
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    height = width = 0
    map_start = 0
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("height"):
            height = int(line.split()[1])
        elif line.startswith("width"):
            width = int(line.split()[1])
        elif line == "map":
            map_start = i + 1
            break

    obstacles = []
    for r, line in enumerate(lines[map_start: map_start + height]):
        for c, ch in enumerate(line.rstrip("\n")):
            if ch in BLOCK_TILES or (ch not in FREE_TILES and ch != '\n'):
                obstacles.append((r, c))

    return height, width, obstacles


def _bfs_reachable(free_set: set, start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
    """BFS 检查 start 能否到达 goal。"""
    if start == goal:
        return True
    visited = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nb = (r + dr, c + dc)
            if nb not in visited and nb in free_set:
                if nb == goal:
                    return True
                visited.add(nb)
                queue.append(nb)
    return False


def _largest_component(free_cells: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """返回 free_cells 中最大连通分量的格子列表。"""
    free_set = set(free_cells)
    visited: set = set()
    best: List[Tuple[int, int]] = []

    for start in free_cells:
        if start in visited:
            continue
        # BFS
        component = []
        queue = deque([start])
        visited.add(start)
        while queue:
            r, c = queue.popleft()
            component.append((r, c))
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nb = (r + dr, c + dc)
                if nb not in visited and nb in free_set:
                    visited.add(nb)
                    queue.append(nb)
        if len(component) > len(best):
            best = component

    return best


def place_agents_random(
    height: int,
    width: int,
    obstacles: List[Tuple[int, int]],
    num_agents: int,
    seed: int = 42,
    max_retries: int = 300,
) -> Optional[Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]]:
    """在地图上随机放置 num_agents 个起点和终点，保证可达性。"""
    obs_set = set(obstacles)
    all_free = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if (r, c) not in obs_set
    ]
    # 只在最大连通分量内采样，避免孤立格子导致可达性失败
    free_cells = _largest_component(all_free)

    if len(free_cells) < num_agents * 2:
        return None

    rng = random.Random(seed)

    for _ in range(max_retries):
        chosen = rng.sample(free_cells, num_agents * 2)
        starts = chosen[:num_agents]
        goals = chosen[num_agents:]

        # 检查起点互不重叠、终点互不重叠
        if len(set(starts)) < num_agents or len(set(goals)) < num_agents:
            continue

        free_set = set(free_cells)
        ok = all(_bfs_reachable(free_set, starts[i], goals[i]) for i in range(num_agents))
        if ok:
            return starts, goals

    return None


def load_mapf_env(
    filepath: str,
    num_agents: int,
    seed: int = 42,
) -> Optional[MAPFEnv]:
    """加载 .map 文件并随机放置智能体，返回 MAPFEnv 或 None（放置失败）。"""
    height, width, obstacles = parse_map(filepath)
    result = place_agents_random(height, width, obstacles, num_agents, seed)
    if result is None:
        return None
    starts, goals = result
    return MAPFEnv(width, height, obstacles, starts, goals)
