"""
评估 — 用训练好的 GNN 策略进行分散式推理，计算各项指标。

用法:
    python evaluate.py --model gnn_policy.pt --num_instances 100 --num_agents 4 --map_size 8
"""

import argparse
import json
import os
import random
from datetime import datetime
from typing import List, Tuple

import torch

from mapf_env import MAPFEnv, MOVES
from cbs import cbs_search
from models.gnn_policy import GNNPolicy
from data_gen import build_graph_observation, generate_random_instance


# ---------- 碰撞处理 ----------

def resolve_collisions(
    env: MAPFEnv,
    current: List[Tuple[int, int]],
    intended: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """简单的优先级碰撞处理: 低编号 agent 优先。

    规则:
      1. 目标位置不可通行 → 原地不动
      2. Vertex 冲突 (多个 agent 想去同一格) → 低编号优先，其余不动
      3. Edge 冲突 (两个 agent 交换位置) → 双方不动
    """
    n = len(current)
    result = list(intended)

    # 不可通行检查
    for i in range(n):
        if not env.is_free(result[i]):
            result[i] = current[i]

    # Edge 冲突: 交换位置 → 双方不动
    for i in range(n):
        for j in range(i + 1, n):
            if result[i] == current[j] and result[j] == current[i]:
                result[i] = current[i]
                result[j] = current[j]

    # Vertex 冲突: 低编号优先
    occupied = {}
    for i in range(n):
        if result[i] in occupied:
            result[i] = current[i]  # 高编号让步
        else:
            occupied[result[i]] = i

    return result


# ---------- 分散式 Rollout ----------

def rollout(
    env: MAPFEnv,
    model: GNNPolicy,
    device: torch.device,
    max_steps: int = 50,
    fov_radius: int = 2,
    comm_radius: int = 5,
) -> Tuple[List[List[Tuple[int, int]]], bool]:
    """用 GNN 策略进行分散式推理。返回 (paths, success)。"""
    model.eval()
    positions = list(env.starts)
    paths = [[s] for s in env.starts]

    with torch.no_grad():
        for _ in range(max_steps):
            # 检查是否全部到达目标
            if all(positions[i] == env.goals[i] for i in range(env.num_agents)):
                return paths, True

            data = build_graph_observation(env, positions, env.goals, fov_radius, comm_radius)
            data = data.to(device)
            logits = model(data)
            actions = logits.argmax(dim=1).cpu().tolist()

            intended = []
            for i, a in enumerate(actions):
                dr, dc = MOVES[a]
                intended.append((positions[i][0] + dr, positions[i][1] + dc))

            positions = resolve_collisions(env, positions, intended)
            for i, p in enumerate(positions):
                paths[i].append(p)

    # 最后再检查一次
    success = all(positions[i] == env.goals[i] for i in range(env.num_agents))
    return paths, success


# ---------- 评估主流程 ----------

def evaluate(
    model: GNNPolicy,
    device: torch.device,
    num_instances: int = 100,
    width: int = 8,
    height: int = 8,
    num_agents: int = 4,
    obstacle_density: float = 0.1,
    max_steps: int = 50,
    seed: int = 123,
):
    rng = random.Random(seed)

    successes = 0
    total_valid = 0
    soc_gnn = []
    soc_cbs = []
    partial_success_rates = []

    for idx in range(num_instances):
        env = generate_random_instance(width, height, num_agents, obstacle_density, rng)
        if env is None:
            continue

        # CBS 最优解 (作为基准)
        cbs_solution = cbs_search(env)
        if cbs_solution is None:
            continue

        total_valid += 1

        # GNN rollout
        paths, success = rollout(env, model, device, max_steps)

        # 部分成功率: 本回合中到达目标的 agent 比例
        agents_at_goal = sum(1 for i in range(env.num_agents) if paths[i][-1] == env.goals[i])
        partial_success_rates.append(agents_at_goal / env.num_agents)

        if success:
            successes += 1
            soc_gnn.append(sum(len(p) - 1 for p in paths))
            soc_cbs.append(sum(len(p) - 1 for p in cbs_solution))

        if (idx + 1) % 20 == 0:
            rate = successes / total_valid if total_valid > 0 else 0
            print(f"  进度: {idx + 1}/{num_instances}, 成功率: {rate:.2%} ({successes}/{total_valid})")

    # 汇总
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "num_instances": num_instances,
            "num_valid": total_valid,
            "map_size": f"{width}x{height}",
            "num_agents": num_agents,
            "obstacle_density": obstacle_density,
            "max_steps": max_steps,
            "seed": seed,
        },
        "success_rate": successes / total_valid if total_valid > 0 else 0,
        "partial_success_rate": round(sum(partial_success_rates) / len(partial_success_rates), 4) if partial_success_rates else 0,
        "successes": successes,
        "total_valid": total_valid,
    }

    print("\n" + "=" * 50)
    print(f"评估结果 ({total_valid} 个有效实例)")
    print("=" * 50)

    if total_valid == 0:
        print("没有有效实例")
        return results

    print(f"总体成功率: {successes}/{total_valid} = {successes / total_valid:.2%}")
    print(f"部分成功率: {sum(partial_success_rates) / len(partial_success_rates):.2%}")

    if successes > 0:
        avg_soc_gnn = sum(soc_gnn) / len(soc_gnn)
        avg_soc_cbs = sum(soc_cbs) / len(soc_cbs)

        results["avg_soc_gnn"] = round(avg_soc_gnn, 2)
        results["avg_soc_cbs"] = round(avg_soc_cbs, 2)
        results["soc_ratio"] = round(avg_soc_gnn / avg_soc_cbs, 2)

        print(f"平均 Sum of Costs:  GNN={avg_soc_gnn:.2f}  CBS={avg_soc_cbs:.2f}  "
              f"比值={avg_soc_gnn / avg_soc_cbs:.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="评估 GNN 分散式策略")
    parser.add_argument("--model", type=str, default="gnn_policy.pt")
    parser.add_argument("--num_instances", type=int, default=100)
    parser.add_argument("--num_agents", type=int, default=4)
    parser.add_argument("--map_size", type=int, default=8)
    parser.add_argument("--obstacle_density", type=float, default=0.1)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    model = GNNPolicy()
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=False))
    model = model.to(device)
    print(f"模型已加载: {args.model}")

    results = evaluate(
        model, device,
        num_instances=args.num_instances,
        width=args.map_size,
        height=args.map_size,
        num_agents=args.num_agents,
        obstacle_density=args.obstacle_density,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    # 保存结果到 JSON 文件
    results["model"] = args.model
    log_dir = "eval_logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"eval_{timestamp}.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存至 {log_path}")


if __name__ == "__main__":
    main()
