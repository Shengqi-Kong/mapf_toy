"""
plot_results.py — 从 benchmark_eval.py 生成的 JSON 结果文件绘制对比图。

用法:
    python plot_results.py benchmark_results/benchmark_YYYYMMDD_HHMMSS.json
    python plot_results.py benchmark_results/benchmark_YYYYMMDD_HHMMSS.json --output_dir benchmark_results
"""

import argparse
import json
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.font_manager as fm
import numpy as np

# ---------- 中文字体配置 ----------

def _setup_chinese_font():
    """尝试设置支持中文的字体，找不到则回退到英文标签。"""
    candidates = ["Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC", "WenQuanYi Micro Hei"]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams["font.family"] = font
            return True
    return False

_HAS_CHINESE = _setup_chinese_font()

def _cn(text: str, fallback: str) -> str:
    """有中文字体时返回中文，否则返回英文 fallback。"""
    return text if _HAS_CHINESE else fallback

# ---------- 样式 ----------

ALGO_STYLE = {
    "cbs":  {"label": "CBS",  "color": "#2196F3", "linestyle": "-",  "marker": "o"},
    "ecbs": {"label": "ECBS (w=1.5)", "color": "#FF9800", "linestyle": "--", "marker": "s"},
    "gnn":  {"label": "GNN",  "color": "#4CAF50", "linestyle": ":",  "marker": "^"},
}

MAP_ORDER = ["empty", "maze", "random", "den520d", "warehouse"]

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})


# ---------- 数据加载 ----------

def load_results(json_path: str) -> Dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_agent_counts(results: Dict) -> List[int]:
    return sorted(results["metadata"]["agent_counts"])


def _get_maps(results: Dict) -> List[str]:
    available = list(results["results"].keys())
    return [m for m in MAP_ORDER if m in available] + [m for m in available if m not in MAP_ORDER]


def _extract_series(
    results: Dict,
    map_name: str,
    algo: str,
    metric: str,
    agent_counts: List[int],
) -> List[Optional[float]]:
    """提取某算法在某地图上的指标序列，缺失值用 None。"""
    series = []
    for n in agent_counts:
        entry = results["results"].get(map_name, {}).get(str(n), {}).get(algo, {})
        val = entry.get(metric)
        series.append(val)
    return series


# ---------- 绘图工具 ----------

def _make_grid(n_maps: int):
    """返回 (fig, axes) — 2行3列，最后一格为汇总。"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)
    return fig, axes.flatten()


def _plot_one_ax(
    ax,
    results: Dict,
    map_name: str,
    agent_counts: List[int],
    metric: str,
    algos: List[str],
    ylabel: str,
    title: str,
):
    x = np.arange(len(agent_counts))
    has_data = False
    for algo in algos:
        style = ALGO_STYLE[algo]
        y = _extract_series(results, map_name, algo, metric, agent_counts)
        # 只绘制非 None 的点
        valid_x = [x[i] for i, v in enumerate(y) if v is not None]
        valid_y = [v for v in y if v is not None]
        if valid_y:
            ax.plot(valid_x, valid_y,
                    label=style["label"],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    linewidth=1.8,
                    markersize=6)
            has_data = True

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in agent_counts])
    ax.set_xlabel(_cn("智能体数量", "Num Agents"))
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if not has_data:
        ax.text(0.5, 0.5, _cn("无数据", "No Data"), ha="center", va="center", transform=ax.transAxes)


def _plot_aggregate_ax(
    ax,
    results: Dict,
    agent_counts: List[int],
    maps: List[str],
    metric: str,
    algos: List[str],
    ylabel: str,
):
    """汇总子图：对所有地图取均值。"""
    x = np.arange(len(agent_counts))
    for algo in algos:
        style = ALGO_STYLE[algo]
        agg_y = []
        for n in agent_counts:
            vals = []
            for m in maps:
                v = results["results"].get(m, {}).get(str(n), {}).get(algo, {}).get(metric)
                if v is not None:
                    vals.append(v)
            agg_y.append(np.mean(vals) if vals else None)

        valid_x = [x[i] for i, v in enumerate(agg_y) if v is not None]
        valid_y = [v for v in agg_y if v is not None]
        if valid_y:
            ax.plot(valid_x, valid_y,
                    label=style["label"],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    linewidth=2.2,
                    markersize=7)

    ax.set_title(_cn("汇总 (所有地图均值)", "Aggregate (All Maps)"))
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in agent_counts])
    ax.set_xlabel(_cn("智能体数量", "Num Agents"))
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


# ---------- 成功率图 ----------

def plot_success_rate(
    results: Dict,
    output_path: str,
    include_gnn: bool = True,
) -> None:
    agent_counts = _get_agent_counts(results)
    maps = _get_maps(results)
    algos = ["cbs", "ecbs"] + (["gnn"] if include_gnn else [])

    fig, axes = _make_grid(len(maps))

    for idx, map_name in enumerate(maps[:5]):
        _plot_one_ax(
            axes[idx], results, map_name, agent_counts,
            metric="success_rate",
            algos=algos,
            ylabel=_cn("成功率", "Success Rate"),
            title=map_name,
        )
        axes[idx].set_ylim(-0.05, 1.05)
        axes[idx].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    # 汇总
    _plot_aggregate_ax(axes[5], results, agent_counts, maps, "success_rate", algos, _cn("成功率", "Success Rate"))
    axes[5].set_ylim(-0.05, 1.05)
    axes[5].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    # 隐藏多余子图
    for idx in range(len(maps), 5):
        axes[idx].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(algos), bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(_cn("成功率对比 (CBS / ECBS / GNN)", "Success Rate (CBS / ECBS / GNN)"), fontsize=14, y=1.01)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"成功率图已保存: {output_path}")


# ---------- Sum of Costs 图 ----------

def plot_sum_of_costs(
    results: Dict,
    output_path: str,
    include_gnn: bool = True,
) -> None:
    agent_counts = _get_agent_counts(results)
    maps = _get_maps(results)
    algos = ["cbs", "ecbs"] + (["gnn"] if include_gnn else [])

    fig, axes = _make_grid(len(maps))

    for idx, map_name in enumerate(maps[:5]):
        _plot_one_ax(
            axes[idx], results, map_name, agent_counts,
            metric="avg_soc",
            algos=algos,
            ylabel=_cn("平均 Sum of Costs", "Avg Sum of Costs"),
            title=map_name,
        )

    _plot_aggregate_ax(axes[5], results, agent_counts, maps, "avg_soc", algos, _cn("平均 Sum of Costs", "Avg Sum of Costs"))

    for idx in range(len(maps), 5):
        axes[idx].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(algos), bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(_cn("平均 Sum of Costs 对比 (仅成功实例)", "Avg Sum of Costs (Solved Instances Only)"), fontsize=14, y=1.01)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Sum of Costs 图已保存: {output_path}")


# ---------- 主函数 ----------

def main():
    parser = argparse.ArgumentParser(description="绘制 MAPF benchmark 对比图")
    parser.add_argument("results_json", help="benchmark_eval.py 输出的 JSON 文件路径")
    parser.add_argument("--output_dir", default=None,
                        help="图片输出目录（默认与 JSON 同目录）")
    parser.add_argument("--no_gnn", action="store_true", help="不绘制 GNN 曲线")
    args = parser.parse_args()

    results = load_results(args.results_json)
    include_gnn = not args.no_gnn

    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.results_json))
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(args.results_json))[0]
    plot_success_rate(results,
                      os.path.join(out_dir, f"{base}_success_rate.png"),
                      include_gnn=include_gnn)
    plot_sum_of_costs(results,
                      os.path.join(out_dir, f"{base}_sum_of_costs.png"),
                      include_gnn=include_gnn)


if __name__ == "__main__":
    main()
