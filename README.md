
这是我用claude code来测试mapf工作的一个小demo。

## 项目结构

- `mapf_env.py` — 网格环境，定义障碍物、智能体起点/终点及邻居查询
- `cbs.py` — CBS（Conflict-Based Search）最优多智能体路径规划算法
- `ecbs.py` — ECBS（Enhanced CBS）次优 MAPF 算法，次优界 w=1.5
- `models/gnn_policy.py` — CNN + GNN 分散式策略网络（CNN 提取空间特征 + GNN 通信）
- `data_gen.py` — 随机 MAPF 实例生成 + CBS 专家数据采集
- `train.py` — 行为克隆训练
- `evaluate.py` — 分散式推理评估，指标：总体成功率、部分成功率、Sum of Costs
- `map_loader.py` — 解析 MovingAI .map 文件，随机放置智能体生成 MAPFEnv
- `benchmark_eval.py` — 多场景 / 多智能体数量 benchmark 主脚本
- `plot_results.py` — 从 JSON 结果生成成功率和 Sum of Costs 对比图
- `simulation.py` — 基于 tkinter 的可视化仿真 GUI
- `test.py` — CBS 算法正确性测试
- `maps/` — MovingAI .map 文件目录（需手动下载）
- `benchmark_results/` — benchmark 结果 JSON 和图片输出目录

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 生成专家数据 (CBS 求解 500 个随机实例)
python data_gen.py --num_instances 500 --num_agents 4 --map_size 8 --output expert_data.pt

# 也支持混合 agent 数量
python data_gen.py --num_instances 500 --mixed_agents 2-5 --map_size 8 --output expert_data.pt

# 3. 训练 CNN+GNN 策略网络
python train.py --data expert_data.pt --epochs 50 --lr 1e-3

# 4. 评估 (输出总体成功率、部分成功率、Sum of Costs 对比)
python evaluate.py --model checkpoints/latest.pt --num_instances 100 --num_agents 4 --map_size 8
```

## Benchmark 评估

在 5 个 MovingAI 标准场景上对比 CBS / ECBS / GNN，智能体数量分别为 8、16、32。

### 准备地图文件

请手动从 [movingai.com/benchmarks/mapf/](https://movingai.com/benchmarks/mapf/) 下载以下文件，放入 `maps/` 目录：

- `empty-32-32.map`
- `maze-32-32-2.map`
- `random-32-32-10.map`
- `den520d.map`
- `warehouse-10-20-10-2-1.map`

```bash
# 运行 benchmark（需提前手动下载地图，见上方"准备地图文件"说明，仅 CBS + ECBS 对比）
python benchmark_eval.py

# 加入 GNN 对比
python benchmark_eval.py --model checkpoints/latest.pt

# 自定义参数
python benchmark_eval.py --num_instances 20 --agent_counts 8 16 32 64 --timeout 60

# 生成对比图（成功率 + Sum of Costs）
python plot_results.py benchmark_results/benchmark_YYYYMMDD_HHMMSS.json

# 不绘制 GNN 曲线
python plot_results.py benchmark_results/benchmark_YYYYMMDD_HHMMSS.json --no_gnn
```

### benchmark_eval.py 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_instances` | 20 | 每个场景每组智能体数的测试实例数 |
| `--agent_counts` | 8 16 32 | 智能体数量组 |
| `--timeout` | 自动 | 统一超时秒数（覆盖默认按智能体数的超时） |
| `--model` | 无 | GNN 模型路径（可选） |
| `--output_dir` | benchmark_results | 结果保存目录 |
| `--ecbs_w` | 1.5 | ECBS 次优界 |


## simulation.py 使用说明

运行：

```bash
python simulation.py
```

功能：

- 顶部切换编辑模式：障碍物 / 起点 / 终点
- 左键点击网格放置，右键点击取消
- 起点与终点按放置顺序一一配对（S0↔G0, S1↔G1, ...）
- 点击「▶ 运行」调用 CBS 求解并自动播放路径动画
- 支持通过 Spinbox 调整网格行列数，点击「重置网格」生效
- 最多支持 8 种颜色区分不同智能体
