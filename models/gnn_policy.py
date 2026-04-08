"""
CNN + GNN 分散式策略网络 — 用于 MAPF 的模仿学习。

架构: CNN(空间特征) + MLP(标量特征) -> 3x SAGEConv -> Linear
输入: 每个 agent 的局部观测 (3×fov×fov 空间特征 + 4维标量 + 通信图)
输出: 5 个动作的 logits (对应 MOVES: wait, right, left, down, up)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GNNPolicy(nn.Module):

    def __init__(self, scalar_dim=4, hidden_dim=128, output_dim=5):
        super().__init__()

        # CNN 提取空间特征: (3, fov_size, fov_size) -> embedding
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # (64, 1, 1)
            nn.Flatten(),             # (64,)
        )

        # 融合 CNN 输出 + 标量特征
        self.fusion = nn.Linear(64 + scalar_dim, hidden_dim)

        # GNN 通信层
        self.conv1 = SAGEConv(hidden_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim // 2)

        # 动作输出
        self.action_head = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, data):
        spatial = data.spatial       # [N, 3, fov, fov]
        scalar = data.scalar         # [N, 4]
        edge_index = data.edge_index

        # CNN 提取空间特征
        cnn_out = self.cnn(spatial)  # [N, 64]

        # 融合
        x = torch.cat([cnn_out, scalar], dim=1)  # [N, 68]
        x = F.relu(self.fusion(x))                # [N, hidden_dim]

        # GNN 通信
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        return self.action_head(x)  # [N, 5]
