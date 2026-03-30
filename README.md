
这是我用claude code来测试mapf工作的一个小demo。

## 项目结构

- `mapf_env.py` — 网格环境，定义障碍物、智能体起点/终点及邻居查询
- `cbs.py` — CBS（Conflict-Based Search）最优多智能体路径规划算法
- `simulation.py` — 基于 tkinter 的可视化仿真 GUI
- `test.py` — 单元测试

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

