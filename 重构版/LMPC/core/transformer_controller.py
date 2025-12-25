"""Transformer 权重控制器

功能：
- 输入：24 步 × 12 维特征序列 (24, 12)
- 输出：3 个动态权重 α_soc, α_grid, α_cost，范围 [alpha_min, alpha_max]
- 推理阶段可选低通滤波，平滑权重变化。
"""

from __future__ import annotations

from typing import Any, Dict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class PositionalEncoding(nn.Module):
    """标准正弦/余弦位置编码。"""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TransformerBackbone(nn.Module):
    """Transformer 主干网络：(B, 24, 12) -> (B, 3)"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        model_cfg = config.get("transformer", {}).get("model", {})

        self.state_dim = int(model_cfg.get("state_dim", 12))
        self.d_model = int(model_cfg.get("d_model", 128))
        self.nhead = int(model_cfg.get("nhead", 8))
        self.num_layers = int(model_cfg.get("num_layers", 4))
        self.dim_feedforward = int(model_cfg.get("dim_feedforward", 512))
        self.dropout = float(model_cfg.get("dropout", 0.1))

        # α 范围从 mpc.alpha_range 读取
        mpc_cfg = config.get("mpc", {})
        alpha_cfg = mpc_cfg.get("alpha_range", {})
        self.alpha_min = float(alpha_cfg.get("min", 0.5))
        self.alpha_max = float(alpha_cfg.get("max", 2.0))

        # 输入嵌入
        self.embedding = nn.Linear(self.state_dim, self.d_model)

        # 位置编码
        self.pos_encoding = PositionalEncoding(self.d_model, max_len=int(model_cfg.get("seq_len", 24)))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # 输出头
        self.output_head = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # 三个权重
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向计算

        参数：
            x: (batch, 24, 12)
        返回：
            weights: (batch, 3)，已经映射到 [alpha_min, alpha_max]
        """
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.encoder(x)  # (B, 24, d_model)
        x = x[:, -1, :]      # 取最后一个时间步
        x = self.output_head(x)  # (B, 3)

        # Sigmoid 映射到 [alpha_min, alpha_max]
        x = torch.sigmoid(x)
        scale = self.alpha_max - self.alpha_min
        x = x * scale + self.alpha_min
        return x


class TransformerController:
    """Transformer 权重控制器封装

    - 负责加载模型权重；
    - 提供 predict_weights 接口；
    - 内部可选低通滤波平滑。
    """

    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = TransformerBackbone(config).to(self.device)

        # 加载权重
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"找不到 Transformer 模型权重文件: {model_file}")

        # 安全加载权重：仅反序列化 state_dict，避免 FutureWarning
        state = torch.load(model_file, map_location=self.device, weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state_dict = state["state_dict"]
        else:
            state_dict = state
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # 平滑配置
        smooth_cfg = config.get("transformer", {}).get("smoothing", {})
        self.use_smoothing = bool(smooth_cfg.get("enable", True))
        self.smooth_alpha = float(smooth_cfg.get("alpha", 0.3))
        self.prev_weights: np.ndarray | None = None

    def reset(self) -> None:
        """重置内部平滑状态。"""
        self.prev_weights = None

    def predict_weights(self, state_seq: np.ndarray, apply_filter: bool = True) -> Dict[str, float]:
        """预测动态权重。

        参数：
            state_seq: np.array(24, 12)
            apply_filter: 是否应用低通滤波
        返回：
            {'alpha_soc', 'alpha_grid', 'alpha_cost'}
        """
        if state_seq.ndim != 2:
            raise ValueError(f"state_seq 形状应为 (24, 12)，当前为 {state_seq.shape}")

        x = torch.as_tensor(state_seq, dtype=torch.float32, device=self.device).unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            raw = self.model(x)[0].cpu().numpy()  # (3,)

        if apply_filter and self.use_smoothing:
            if self.prev_weights is None:
                filtered = raw
            else:
                a = self.smooth_alpha
                filtered = a * raw + (1.0 - a) * self.prev_weights
        else:
            filtered = raw

        self.prev_weights = filtered.copy()

        return {
            "alpha_soc": float(filtered[0]),
            "alpha_grid": float(filtered[1]),
            "alpha_cost": float(filtered[2]),
        }
