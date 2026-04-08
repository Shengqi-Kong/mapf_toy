"""
行为克隆训练 — 用 CBS 专家数据训练 GNN 策略网络。

用法:
    python train.py --data expert_data.pt --epochs 50 --lr 1e-3 --output gnn_policy.pt
"""

import argparse
import os
import shutil
import random
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import DataLoader

from models.gnn_policy import GNNPolicy


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_nodes = 0

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)  # [N_total, 5]
        loss = F.cross_entropy(logits, batch.y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.y.size(0)
        total_correct += (logits.argmax(dim=1) == batch.y).sum().item()
        total_nodes += batch.y.size(0)

    return total_loss / total_nodes, total_correct / total_nodes


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_nodes = 0

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        loss = F.cross_entropy(logits, batch.y)

        total_loss += loss.item() * batch.y.size(0)
        total_correct += (logits.argmax(dim=1) == batch.y).sum().item()
        total_nodes += batch.y.size(0)

    return total_loss / total_nodes, total_correct / total_nodes


def main():
    parser = argparse.ArgumentParser(description="GNN 策略网络行为克隆训练")
    parser.add_argument("--data", type=str, default="expert_data.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="gnn_policy.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    print(f"设备: {device}")

    # 加载数据
    dataset = torch.load(args.data, weights_only=False)
    random.seed(args.seed)
    random.shuffle(dataset)

    split = int(len(dataset) * (1 - args.val_ratio))
    train_set = dataset[:split]
    val_set = dataset[split:]
    print(f"数据集: {len(train_set)} 训练 / {len(val_set)} 验证")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    # 模型
    model = GNNPolicy().to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0

    # 输出目录
    os.makedirs("checkpoints", exist_ok=True)
    temp_path = os.path.join("checkpoints", "_best_temp.pt")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, device)

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), temp_path)

    # 用时间戳 + 最佳准确率命名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    acc_str = f"{best_val_acc:.4f}".replace(".", "p")
    save_name = f"gnn_policy_{timestamp}_acc{acc_str}.pt"
    save_path = os.path.join("checkpoints", save_name)

    if os.path.exists(temp_path):
        os.rename(temp_path, save_path)
    latest_path = os.path.join("checkpoints", "latest.pt")
    shutil.copy2(save_path, latest_path)
    print(f"训练完成, 最佳验证准确率: {best_val_acc:.4f}, 模型已保存至 {save_path}")


if __name__ == "__main__":
    main()
