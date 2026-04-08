
这是我用claude code来测试mapf工作的一个小demo。

## 项目结构

- `mapf_env.py` — 网格环境，定义障碍物、智能体起点/终点及邻居查询
- `cbs.py` — CBS（Conflict-Based Search）最优多智能体路径规划算法
- `models/gnn_policy.py` — CNN + GNN 分散式策略网络（CNN 提取空间特征 + GNN 通信）
- `data_gen.py` — 随机 MAPF 实例生成 + CBS 专家数据采集
- `train.py` — 行为克隆训练
- `evaluate.py` — 分散式推理评估，对比 GNN 与 CBS 指标
- `simulation.py` — 基于 tkinter 的可视化仿真 GUI
- `test.py` — CBS 算法正确性测试

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

# 4. 评估
python evaluate.py --model checkpoints/latest.pt --num_instances 100 --num_agents 4 --map_size 8
```

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
