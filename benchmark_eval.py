"""
benchmark_eval.py — 在 5 个 MovingAI benchmark 场景上评估 CBS / ECBS / GNN。

用法:
    python benchmark_eval.py
    python benchmark_eval.py --num_instances 20 --timeout 60 --model checkpoints/latest.pt
"""

import argparse
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import torch

from cbs import cbs_search
from ecbs import ecbs_search
from map_loader import load_mapf_env

# ---------- 配置 ----------

MAP_DIR = "maps"

MAP_FILES = {
    "empty":     "empty-32-32.map",
    "maze":      "maze-32-32-2.map",
    "random":    "random-32-32-10.map",
    "den520d":   "den520d.map",
    "warehouse": "warehouse-10-20-10-2-1.map",
}

MAP_URLS = {}

AGENT_COUNTS = [8, 16, 32]

# 按智能体数量设置超时（秒）
CBS_TIMEOUTS  = {8: 5,   16: 15,  32: 30,  64: 60}
ECBS_TIMEOUTS = {8: 3,   16: 8,   32: 15,  64: 30}


# ---------- 地图检验 ----------

def download_maps(map_dir: str = MAP_DIR) -> None:
    """检查地图文件是否存在，不存在则提示用户手动放置。"""
    os.makedirs(map_dir, exist_ok=True)
    for filename in MAP_FILES.values():
        dest = os.path.join(map_dir, filename)
        if os.path.exists(dest):
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ 缺少地图文件: {dest}，请手动下载并放入 {map_dir}/ 目录")


# ---------- GNN rollout (可选) ----------

def _gnn_rollout(env, model, device, max_steps: int = 200):
    """从 evaluate.py 复用 rollout 逻辑，避免循环导入。"""
    from evaluate import rollout
    return rollout(env, model, device, max_steps)


# ---------- 单实例评估 ----------

def run_single_instance(
    env,
    num_agents: int,
    cbs_timeout: float,
    ecbs_timeout: float,
    model=None,
    device=None,
    max_steps: int = 200,
) -> Dict:
    record = {}

    # CBS
    t0 = time.time()
    cbs_sol = cbs_search(env, timeout=cbs_timeout)
    cbs_time = time.time() - t0
    if cbs_sol is not None:
        record["cbs"] = {
            "solved": True,
            "soc": sum(len(p) - 1 for p in cbs_sol),
            "elapsed": round(cbs_time, 4),
        }
    else:
        record["cbs"] = {"solved": False, "soc": None, "elapsed": round(cbs_time, 4)}

    # ECBS
    t0 = time.time()
    ecbs_sol = ecbs_search(env, w=1.5, timeout=ecbs_timeout)
    ecbs_time = time.time() - t0
    if ecbs_sol is not None:
        record["ecbs"] = {
            "solved": True,
            "soc": sum(len(p) - 1 for p in ecbs_sol),
            "elapsed": round(ecbs_time, 4),
        }
    else:
        record["ecbs"] = {"solved": False, "soc": None, "elapsed": round(ecbs_time, 4)}

    # GNN (可选)
    if model is not None:
        paths, success = _gnn_rollout(env, model, device, max_steps)
        agents_at_goal = sum(
            1 for i in range(env.num_agents) if paths[i][-1] == env.goals[i]
        )
        record["gnn"] = {
            "solved": success,
            "partial_rate": round(agents_at_goal / env.num_agents, 4),
            "soc": sum(len(p) - 1 for p in paths) if success else None,
        }

    return record


# ---------- 单场景评估 ----------

def run_scenario(
    map_name: str,
    map_path: str,
    num_agents: int,
    num_instances: int = 20,
    cbs_timeout: float = 60.0,
    ecbs_timeout: float = 30.0,
    seed: int = 42,
    model=None,
    device=None,
) -> Dict:
    records = []
    for inst_idx in range(num_instances):
        env = load_mapf_env(map_path, num_agents, seed=seed + inst_idx)
        if env is None:
            print(f"    [{map_name} n={num_agents}] 实例 {inst_idx} 放置失败，跳过")
            continue
        rec = run_single_instance(env, num_agents, cbs_timeout, ecbs_timeout, model, device)
        records.append(rec)

    def agg(algo: str) -> Dict:
        solved = [r[algo] for r in records if r.get(algo, {}).get("solved")]
        total = len(records)
        sr = len(solved) / total if total > 0 else 0
        avg_soc = sum(s["soc"] for s in solved) / len(solved) if solved else None
        return {
            "success_rate": round(sr, 4),
            "avg_soc": round(avg_soc, 2) if avg_soc is not None else None,
            "solved_count": len(solved),
            "total": total,
        }

    result = {"cbs": agg("cbs"), "ecbs": agg("ecbs")}
    if model is not None:
        gnn_recs = [r["gnn"] for r in records if "gnn" in r]
        if gnn_recs:
            solved_gnn = [r for r in gnn_recs if r["solved"]]
            total = len(gnn_recs)
            result["gnn"] = {
                "success_rate": round(len(solved_gnn) / total, 4) if total > 0 else 0,
                "partial_success_rate": round(
                    sum(r["partial_rate"] for r in gnn_recs) / total, 4
                ) if total > 0 else 0,
                "avg_soc": round(
                    sum(r["soc"] for r in solved_gnn) / len(solved_gnn), 2
                ) if solved_gnn else None,
                "solved_count": len(solved_gnn),
                "total": total,
            }

    return result


# ---------- 保存结果 ----------

def save_results(results: Dict, output_path: str) -> None:
    tmp = output_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    os.replace(tmp, output_path)


# ---------- 主流程 ----------

def run_benchmark(
    agent_counts: List[int] = None,
    num_instances: int = 20,
    timeout: float = None,
    model_path: Optional[str] = None,
    output_dir: str = "benchmark_results",
    seed: int = 42,
    ecbs_w: float = 1.5,
) -> str:
    if agent_counts is None:
        agent_counts = AGENT_COUNTS

    os.makedirs(output_dir, exist_ok=True)
    download_maps()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    if model_path and os.path.exists(model_path):
        from models.gnn_policy import GNNPolicy
        model = GNNPolicy()
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        model = model.to(device)
        model.eval()
        print(f"GNN 模型已加载: {model_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"benchmark_{timestamp}"
    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    output_path = os.path.join(run_dir, f"{run_name}.json")

    results = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "agent_counts": agent_counts,
            "maps": list(MAP_FILES.keys()),
            "num_instances": num_instances,
            "ecbs_w": ecbs_w,
            "model": model_path,
        },
        "results": {},
    }

    for map_name, map_file in MAP_FILES.items():
        map_path = os.path.join(MAP_DIR, map_file)
        if not os.path.exists(map_path):
            print(f"地图文件不存在，跳过: {map_path}")
            continue

        results["results"][map_name] = {}
        print(f"\n=== 场景: {map_name} ===")

        for n_agents in agent_counts:
            cbs_to = timeout if timeout else CBS_TIMEOUTS.get(n_agents, 60)
            ecbs_to = timeout if timeout else ECBS_TIMEOUTS.get(n_agents, 30)

            print(f"  智能体数: {n_agents}  (CBS超时={cbs_to}s, ECBS超时={ecbs_to}s)")
            scenario_result = run_scenario(
                map_name, map_path, n_agents,
                num_instances=num_instances,
                cbs_timeout=cbs_to,
                ecbs_timeout=ecbs_to,
                seed=seed,
                model=model,
                device=device,
            )
            results["results"][map_name][str(n_agents)] = scenario_result

            cbs_sr = scenario_result["cbs"]["success_rate"]
            ecbs_sr = scenario_result["ecbs"]["success_rate"]
            print(f"    CBS  成功率={cbs_sr:.0%}  avg_soc={scenario_result['cbs']['avg_soc']}")
            print(f"    ECBS 成功率={ecbs_sr:.0%}  avg_soc={scenario_result['ecbs']['avg_soc']}")
            if "gnn" in scenario_result:
                gnn_sr = scenario_result["gnn"]["success_rate"]
                print(f"    GNN  成功率={gnn_sr:.0%}  avg_soc={scenario_result['gnn']['avg_soc']}")

            # 每完成一个 (map, n_agents) 就保存一次，防止中途崩溃丢失数据
            save_results(results, output_path)

    print(f"\n结果已保存至 {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="MAPF Benchmark 评估")
    parser.add_argument("--num_instances", type=int, default=20)
    parser.add_argument("--agent_counts", type=int, nargs="+", default=AGENT_COUNTS)
    parser.add_argument("--timeout", type=float, default=None,
                        help="统一超时秒数（覆盖默认按智能体数的超时）")
    parser.add_argument("--model", type=str, default=None, help="GNN 模型路径（可选）")
    parser.add_argument("--output_dir", type=str, default="benchmark_results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ecbs_w", type=float, default=1.5)
    args = parser.parse_args()

    run_benchmark(
        agent_counts=args.agent_counts,
        num_instances=args.num_instances,
        timeout=args.timeout,
        model_path=args.model,
        output_dir=args.output_dir,
        seed=args.seed,
        ecbs_w=args.ecbs_w,
    )


if __name__ == "__main__":
    main()
