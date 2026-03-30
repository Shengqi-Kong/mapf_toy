"""
test.py — 验证 CBS 算法在不同场景下的正确性。
"""

from mapf_env import MAPFEnv
from cbs import cbs_search


def validate_solution(env: MAPFEnv, solution):
    """校验解的合法性：无碰撞、起终点正确、移动合法。"""
    if solution is None:
        print("  无解")
        return False

    max_t = max(len(p) for p in solution)

    for i, path in enumerate(solution):
        # 起点/终点
        assert path[0] == env.starts[i], f"智能体 {i} 起点不匹配"
        assert path[-1] == env.goals[i], f"智能体 {i} 终点不匹配"
        # 每步移动合法
        for t in range(len(path) - 1):
            assert path[t + 1] in env.get_neighbors(path[t]), (
                f"智能体 {i} 在 t={t} 移动不合法: {path[t]} -> {path[t+1]}"
            )

    # 冲突检测
    for t in range(max_t):
        positions = []
        for i, path in enumerate(solution):
            pos = path[min(t, len(path) - 1)]
            positions.append(pos)
        # 顶点冲突
        assert len(positions) == len(set(positions)), f"t={t} 存在顶点冲突: {positions}"
        # 边冲突
        if t > 0:
            for i in range(len(solution)):
                for j in range(i + 1, len(solution)):
                    pi_prev = solution[i][min(t - 1, len(solution[i]) - 1)]
                    pi_curr = solution[i][min(t, len(solution[i]) - 1)]
                    pj_prev = solution[j][min(t - 1, len(solution[j]) - 1)]
                    pj_curr = solution[j][min(t, len(solution[j]) - 1)]
                    assert not (pi_prev == pj_curr and pj_prev == pi_curr), (
                        f"t={t} 智能体 {i} 和 {j} 存在边冲突"
                    )

    print("  验证通过")
    return True


def print_solution(solution):
    if solution is None:
        print("  无解")
        return
    cost = sum(len(p) - 1 for p in solution)
    print(f"  总代价 (SIC): {cost}")
    for i, path in enumerate(solution):
        print(f"  智能体 {i}: {path}")


# ============ 测试场景 ============

def test_simple():
    """场景 1: 两个智能体对向行走，必须绕行或等待。"""
    print("=== 场景 1: 对向冲突 ===")
    #  . . . . .
    #  S0. . .G0
    #  . . . . .
    #  S1. . .G1
    #  . . . . .
    env = MAPFEnv(
        width=5, height=5,
        obstacles=[],
        starts=[(1, 0), (3, 0)],
        goals=[(1, 4), (3, 4)],
    )
    solution = cbs_search(env)
    print_solution(solution)
    validate_solution(env, solution)
    env.render(solution, timestep=0)


def test_swap():
    """场景 2: 两个智能体需要交换位置。"""
    print("=== 场景 2: 交换位置 ===")
    env = MAPFEnv(
        width=5, height=3,
        obstacles=[],
        starts=[(1, 0), (1, 4)],
        goals=[(1, 4), (1, 0)],
    )
    solution = cbs_search(env)
    print_solution(solution)
    validate_solution(env, solution)


def test_with_obstacles():
    """场景 3: 带障碍物的地图。"""
    print("=== 场景 3: 带障碍物 ===")
    # 8x8 地图，中间有一堵墙，留一个缺口
    obstacles = [(r, 4) for r in range(8) if r != 3]
    env = MAPFEnv(
        width=8, height=8,
        obstacles=obstacles,
        starts=[(1, 1), (6, 1)],
        goals=[(1, 6), (6, 6)],
    )
    solution = cbs_search(env)
    print_solution(solution)
    validate_solution(env, solution)
    env.render(solution, timestep=0)


def test_bottleneck():
    """场景 4: 三个智能体争抢同一个瓶颈通道。"""
    print("=== 场景 4: 瓶颈通道 ===")
    # 7x7 地图，中间只有 (3,3) 可通行
    obstacles = []
    for r in range(7):
        for c in range(7):
            if r == 3 and c == 3:
                continue
            if (r == 2 or r == 3 or r == 4) and c == 3:
                continue
            if r == 3 and (c == 2 or c == 3 or c == 4):
                continue
            if c == 3 and 2 <= r <= 4:
                continue
            if r == 3 and 2 <= c <= 4:
                continue
    # 简化：用一排墙隔开左右，只留中间通道
    obstacles = [(r, 3) for r in range(7) if r != 3]
    env = MAPFEnv(
        width=7, height=7,
        obstacles=obstacles,
        starts=[(1, 1), (3, 1), (5, 1)],
        goals=[(1, 5), (3, 5), (5, 5)],
    )
    solution = cbs_search(env)
    print_solution(solution)
    validate_solution(env, solution)
    env.render(solution, timestep=0)


if __name__ == "__main__":
    test_simple()
    print()
    test_swap()
    print()
    test_with_obstacles()
    print()
    test_bottleneck()
    print("\n所有测试完成！")
