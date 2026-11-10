"""
Multi-Scale Feature Fusion Module.
Implements gated fusion, concatenation projection, and cross-attention fusion
between global and local visual representations while preserving spatial token sequence lengths.
"""

from typing import Literal, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleFeatureFusion(nn.Module):
    """
    Fuses aligned global visual features and local visual features.
    Guarantees output shape matches the global visual sequence length (e.g. 64 tokens).
    """

    def __init__(
        self,
        hidden_dim: int,
        fusion_method: Literal["gated", "concat", "attention"] = "gated",
        dropout: float = 0.05,
    ):
        """
        Args:
            hidden_dim: Feature dimension of visual tokens (e.g. 2048 or 2560).
            fusion_method: Strategy for combining global and local features.
            dropout: Dropout probability.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fusion_method = fusion_method

        if fusion_method == "gated":
            # Gated fusion: G = sigmoid(W [F_g; F_l] + b)
            self.gate_proj = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            # Initialize second layer bias to positive value (~1.0) so gate starts near sigmoid(1.0) ~ 0.73
            # favoring global features at initialization for stable training.
            nn.init.constant_(self.gate_proj[-1].bias, 1.0)
            self.out_norm = nn.LayerNorm(hidden_dim, eps=1e-6)

        elif fusion_method == "concat":
            # Linear projection of concatenated features: F_fused = W [F_g; F_l] + F_g
            self.proj = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.out_norm = nn.LayerNorm(hidden_dim, eps=1e-6)

        elif fusion_method == "attention":
            # Multihead attention: Query = F_g, Key/Value = F_l
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True,
            )
            self.out_norm = nn.LayerNorm(hidden_dim, eps=1e-6)

        else:
            raise ValueError(f"Unknown fusion method '{fusion_method}'. Options: gated, concat, attention")

    def forward(self, f_global: torch.Tensor, f_local: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_global: Global visual features [B, T, D]
            f_local: Aligned local visual features [B, T, D]

        Returns:
            f_fused: Fused visual features [B, T, D]
        """
        assert f_global.shape == f_local.shape, (
            f"Shape mismatch between global {f_global.shape} and local {f_local.shape}"
        )

        if self.fusion_method == "gated":
            cat = torch.cat([f_global, f_local], dim=-1)
            gate = torch.sigmoid(self.gate_proj(cat))
            fused = gate * f_global + (1.0 - gate) * f_local
            return self.out_norm(fused + f_global)

        elif self.fusion_method == "concat":
            cat = torch.cat([f_global, f_local], dim=-1)
            projected = self.proj(cat)
            return self.out_norm(projected + f_global)

        elif self.fusion_method == "attention":
            attn_out, _ = self.cross_attn(query=f_global, key=f_local, value=f_local)
            return self.out_norm(attn_out + f_global)
