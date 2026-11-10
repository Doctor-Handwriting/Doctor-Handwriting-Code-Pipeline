"""
Multi-Scale Visual Extractor Module.
Processes global and local visual representations and reduces/aligns local tokens
into the global spatial grid before calling the feature fusion module.
"""

import math
from typing import List, Literal, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_fusion import MultiScaleFeatureFusion


class MultiScaleVisualExtractor(nn.Module):
    """
    Combines global visual tokens with local high-resolution crop visual tokens.
    Guarantees output shape [B, T_g, D] preserving Qwen3-VL token counts and positional layout.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: Optional[int] = None,
        reduction_method: Literal["attention", "spatial_pool"] = "attention",
        fusion_method: Literal["gated", "concat", "attention"] = "gated",
        grid_size: int = 2,
        num_heads: int = 8,
        dropout: float = 0.05,
    ):
        """
        Args:
            in_dim: Input feature dimension from vision encoder (e.g. 2048 or 2560).
            hidden_dim: Common hidden dimension for fusion (defaults to in_dim).
            reduction_method: How local crop features are aggregated ('attention' or 'spatial_pool').
            fusion_method: How global and local features are fused ('gated', 'concat', 'attention').
            grid_size: Spatial grid division (e.g. 2 for 2x2 = 4 crops).
            num_heads: Attention heads for cross-attention.
            dropout: Dropout probability.
        """
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim or in_dim
        self.reduction_method = reduction_method
        self.grid_size = grid_size

        # Projections to common hidden dimension
        self.proj_global = (
            nn.Linear(in_dim, self.hidden_dim) if in_dim != self.hidden_dim else nn.Identity()
        )
        self.proj_local = (
            nn.Linear(in_dim, self.hidden_dim) if in_dim != self.hidden_dim else nn.Identity()
        )

        # Local feature reduction mechanism
        if self.reduction_method == "attention":
            self.local_to_global_attn = nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.local_norm = nn.LayerNorm(self.hidden_dim, eps=1e-6)

        elif self.reduction_method == "spatial_pool":
            self.pool_norm = nn.LayerNorm(self.hidden_dim, eps=1e-6)

        # Feature fusion module
        self.fusion = MultiScaleFeatureFusion(
            hidden_dim=self.hidden_dim,
            fusion_method=fusion_method,
            dropout=dropout,
        )

    def _align_via_attention(
        self,
        f_global: torch.Tensor,
        f_local_crops: Union[torch.Tensor, List[torch.Tensor]],
    ) -> torch.Tensor:
        """
        Cross-attend from global token positions to all fine-grained local crop tokens.
        Query: Global tokens [B, T_g, D]
        Key/Value: Flattened local crop tokens [B, M * T_c, D]
        """
        if isinstance(f_local_crops, list):
            f_local_flat = torch.cat(f_local_crops, dim=1)  # [B, M*T_c, D]
        elif f_local_crops.dim() == 4:
            b, m, tc, d = f_local_crops.shape
            f_local_flat = f_local_crops.view(b, m * tc, d)
        else:
            f_local_flat = f_local_crops

        aligned, _ = self.local_to_global_attn(
            query=f_global,
            key=f_local_flat,
            value=f_local_flat,
        )
        return self.local_norm(aligned)

    def _align_via_spatial_pooling(
        self,
        f_global: torch.Tensor,
        f_local_crops: Union[torch.Tensor, List[torch.Tensor]],
    ) -> torch.Tensor:
        """
        Pool each crop's feature map into its corresponding quadrant in the global spatial grid.
        Assumes T_g = H_g * W_g (e.g. 64 = 8x8).
        """
        b, t_g, d = f_global.shape
        grid_dim = int(math.isqrt(t_g))
        assert grid_dim * grid_dim == t_g, f"Global tokens {t_g} cannot form square grid"

        # List of M crops
        if isinstance(f_local_crops, list):
            crop_list = f_local_crops
        elif f_local_crops.dim() == 4:
            crop_list = [f_local_crops[:, i] for i in range(f_local_crops.shape[1])]
        else:
            raise ValueError(f"Unexpected local crops tensor shape: {f_local_crops.shape}")

        quad_dim = grid_dim // self.grid_size
        canvas = torch.zeros(b, grid_dim, grid_dim, d, device=f_global.device, dtype=f_global.dtype)

        # Place the first grid_size * grid_size crops into quadrants
        num_grid_crops = self.grid_size * self.grid_size
        for idx in range(min(num_grid_crops, len(crop_list))):
            row = idx // self.grid_size
            col = idx % self.grid_size

            c_feat = crop_list[idx]  # [B, T_c, D]
            tc = c_feat.shape[1]
            c_dim = int(math.isqrt(tc))
            c_grid = c_feat.view(b, c_dim, c_dim, d).permute(0, 3, 1, 2)  # [B, D, H_c, W_c]
            pooled = F.adaptive_avg_pool2d(c_grid, (quad_dim, quad_dim)).permute(0, 2, 3, 1)  # [B, quad_dim, quad_dim, D]

            r_start = row * quad_dim
            r_end = (row + 1) * quad_dim
            c_start = col * quad_dim
            c_end = (col + 1) * quad_dim
            canvas[:, r_start:r_end, c_start:c_end, :] = pooled

        # If a 5th crop (center crop) is provided, blend it into the center
        if len(crop_list) > num_grid_crops:
            center_crop = crop_list[num_grid_crops]
            tc = center_crop.shape[1]
            c_dim = int(math.isqrt(tc))
            c_grid = center_crop.view(b, c_dim, c_dim, d).permute(0, 3, 1, 2)
            pooled_c = F.adaptive_avg_pool2d(c_grid, (quad_dim, quad_dim)).permute(0, 2, 3, 1)

            c_r_start = grid_dim // 4
            c_r_end = c_r_start + quad_dim
            c_c_start = grid_dim // 4
            c_c_end = c_c_start + quad_dim
            canvas[:, c_r_start:c_r_end, c_c_start:c_c_end, :] = (
                0.5 * canvas[:, c_r_start:c_r_end, c_c_start:c_c_end, :] + 0.5 * pooled_c
            )

        aligned = canvas.view(b, t_g, d)
        return self.pool_norm(aligned)

    def forward(
        self,
        f_global: torch.Tensor,
        f_local_crops: Union[torch.Tensor, List[torch.Tensor]],
    ) -> torch.Tensor:
        """
        Forward pass:
        1. Projects global and local features to hidden_dim.
        2. Aligns local features with global token grid.
        3. Fuses representations via configured fusion mechanism.

        Args:
            f_global: Global visual tokens [B, T_g, in_dim]
            f_local_crops: List of crop tokens or [B, M, T_c, in_dim]

        Returns:
            f_fused: Fused visual tokens [B, T_g, hidden_dim]
        """
        # 1. Project
        h_global = self.proj_global(f_global)

        if isinstance(f_local_crops, list):
            h_local_crops = [self.proj_local(crop) for crop in f_local_crops]
        else:
            h_local_crops = self.proj_local(f_local_crops)

        # 2. Align local features to global shape [B, T_g, hidden_dim]
        if self.reduction_method == "attention":
            h_local_aligned = self._align_via_attention(h_global, h_local_crops)
        elif self.reduction_method == "spatial_pool":
            h_local_aligned = self._align_via_spatial_pooling(h_global, h_local_crops)
        else:
            raise ValueError(f"Unknown reduction method: {self.reduction_method}")

        # 3. Fuse
        f_fused = self.fusion(h_global, h_local_aligned)
        return f_fused
