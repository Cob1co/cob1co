"""Transformer 权重控制器训练脚本

功能：
- 从 DataCollector 生成的 transformer_training_data.pkl 中读取训练样本；
- 使用 TransformerBackbone 网络进行监督学习，拟合 (alpha_soc, alpha_grid, alpha_cost)；
- 训练完成后，将最优模型权重保存到 phase3_config.yaml 中配置的路径。

使用方法（在项目根目录）：

    python -m LMPC.training.train_transformer

要求：
- 先运行 tests/run_data_collection_march.py 生成训练数据；
- phase3_config.yaml 中的优化器和训练配置有效。
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import yaml

from LMPC.core.transformer_controller import TransformerBackbone


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TransformerDataset(Dataset):
    """简单的 (state_sequence, optimal_weights) 数据集封装。"""

    def __init__(self, states: np.ndarray, targets: np.ndarray) -> None:
        # states: (N, 24, 12), targets: (N, 3)
        self.states = torch.as_tensor(states, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return self.states.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.states[idx], self.targets[idx]


def load_config() -> Dict[str, Any]:
    """加载 Phase 3 配置。"""
    cfg_path = PROJECT_ROOT / "LMPC" / "phase3_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"找不到配置文件: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_training_data(config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """从 transformer_training_data.pkl 加载训练数据。

    返回：
        X: (N, 24, 12)
        y: (N, 3)
    """
    data_cfg = config.get("data", {})
    rel_path = data_cfg.get("transformer_training_data", "data/transformer_training_data.pkl")
    path = PROJECT_ROOT / rel_path
    if not path.exists():
        raise FileNotFoundError(f"找不到训练数据文件: {path}")

    with open(path, "rb") as f:
        dataset: List[Dict[str, Any]] = pickle.load(f)

    if not isinstance(dataset, list) or not dataset:
        raise ValueError("transformer_training_data.pkl 内容为空或格式不正确")

    states_list: List[np.ndarray] = []
    targets_list: List[np.ndarray] = []
    for item in dataset:
        state_seq = np.asarray(item["state_sequence"], dtype=np.float32)
        opt_w = np.asarray(item["optimal_weights"], dtype=np.float32)
        if state_seq.shape != (24, 12):
            raise ValueError(f"state_sequence 形状应为 (24, 12)，当前为 {state_seq.shape}")
        if opt_w.shape != (3,):
            raise ValueError(f"optimal_weights 形状应为 (3,)，当前为 {opt_w.shape}")
        states_list.append(state_seq)
        targets_list.append(opt_w)

    X = np.stack(states_list, axis=0)  # (N, 24, 12)
    y = np.stack(targets_list, axis=0)  # (N, 3)
    return X, y


def build_model_and_optim(config: Dict[str, Any], device: torch.device) -> Tuple[TransformerBackbone, torch.optim.Optimizer, Any]:
    """根据配置构建模型、优化器和学习率调度器。"""
    model = TransformerBackbone(config).to(device)

    t_cfg = config.get("transformer", {}).get("training", {})
    opt_cfg = t_cfg.get("optimizer", {})
    sch_cfg = t_cfg.get("scheduler", {})

    lr = float(opt_cfg.get("lr", 1e-4))
    weight_decay = float(opt_cfg.get("weight_decay", 1e-5))

    optimizer_name = str(opt_cfg.get("name", "AdamW")).lower()
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"不支持的优化器: {opt_cfg.get('name')}")

    scheduler = None
    scheduler_name = str(sch_cfg.get("name", "")).lower()
    if scheduler_name == "cosineannealinglr":
        T_max = int(sch_cfg.get("T_max", 100))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

    return model, optimizer, scheduler


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: nn.Module,
) -> float:
    """单轮训练。"""
    model.train()
    total_loss = 0.0
    total_batches = 0

    for states, targets in dataloader:
        states = states.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(states)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1

    return total_loss / max(total_batches, 1)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
) -> float:
    """在验证集上评估。"""
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for states, targets in dataloader:
            states = states.to(device)
            targets = targets.to(device)

            outputs = model(states)
            loss = loss_fn(outputs, targets)

            total_loss += float(loss.item())
            total_batches += 1

    return total_loss / max(total_batches, 1)


def save_best_model(model: nn.Module, config: Dict[str, Any]) -> Path:
    """保存最优模型权重到配置文件指定位置。"""
    models_cfg = config.get("models", {})
    rel_path = models_cfg.get("transformer_weights", "LMPC/models/transformer_weights.pth")
    path = PROJECT_ROOT / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)

    # 只保存 state_dict，便于 TransformerController 直接加载
    torch.save(model.state_dict(), path)
    print(f"\n✅ 最优模型已保存到: {path}")
    return path


def main() -> None:
    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载配置和数据
    config = load_config()
    X, y = load_training_data(config)
    print(f"加载训练样本数: {X.shape[0]}")

    # 构建数据集和划分训练/验证集（简单按 8:2 划分）
    full_dataset = TransformerDataset(X, y)
    n_total = len(full_dataset)
    n_train = max(int(n_total * 0.8), 1)
    n_val = max(n_total - n_train, 1)
    if n_train + n_val > n_total:
        n_train = n_total - n_val
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])

    t_cfg = config.get("transformer", {}).get("training", {})
    opt_cfg = t_cfg.get("optimizer", {})
    batch_size = int(opt_cfg.get("batch_size", 64))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 构建模型与优化器
    model, optimizer, scheduler = build_model_and_optim(config, device)
    loss_fn = nn.MSELoss()

    max_epochs = int(opt_cfg.get("max_epochs", 100))
    patience = int(opt_cfg.get("patience", 10))

    best_val_loss = float("inf")
    best_state_dict: Dict[str, Any] | None = None
    epochs_no_improve = 0

    print("\n开始训练 Transformer 权重控制器...")
    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
        val_loss = evaluate(model, val_loader, device, loss_fn)

        if scheduler is not None:
            scheduler.step()

        print(
            f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )

        # 早停逻辑
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state_dict = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"验证集 loss {patience} 个 epoch 未提升，提前停止训练。"
                )
                break

    # 恢复最优权重并保存
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    save_best_model(model, config)

    print("\n训练结束。")


if __name__ == "__main__":
    main()
