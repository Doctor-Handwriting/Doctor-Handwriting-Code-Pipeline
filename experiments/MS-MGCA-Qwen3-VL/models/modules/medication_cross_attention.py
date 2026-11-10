"""
Medication-Guided Cross-Attention Module (Component B).
Introduces K task-specific learnable queries to extract medication-relevant
representations from visual features and integrates them via a visual adapter.
"""

from typing import Literal, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class MedicationCrossAttention(nn.Module):
    """
    Task-specific Medication-Guided Cross-Attention.
    Uses K learnable medication queries Q_m to attend to visual features,
    and integrates the refined representations back into visual tokens via a visual adapter.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_queries: int = 4,
        num_heads: int = 8,
        dropout: float = 0.05,
        integration_mode: Literal["adapter", "prefix"] = "adapter",
    ):
        """
        Args:
            hidden_dim: Dimension of visual tokens (e.g. 2048 or 2560).
            num_queries: Number of learnable medication queries K (e.g. 1, 4, 8).
            num_heads: Attention heads for multi-head cross-attention.
            dropout: Dropout probability.
            integration_mode: 'adapter' (residual modulation preserving token count) or 'prefix'.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.integration_mode = integration_mode

        # 1. Learnable medication queries Q_m in R^(K x d)
        self.med_queries = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)

        # 2. Query -> Visual Cross-Attention: H_m = MHA(Q_m, F_visual, F_visual)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-6)

        # Feed-forward refinement for queries
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-6)

        # 3. Integration Mechanism
        if integration_mode == "adapter":
            # Reverse cross-attention: Visual tokens attend to medication query summaries
            self.adapter_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.adapter_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.adapter_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
            # Learnable gating/scale factor initialized to 0.1 for stable initial warmup
            self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        f_visual: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            f_visual: Visual features [B, T_v, D] (e.g. [B, 64, 2048])

        Returns:
            f_out: Refined visual features [B, T_v, D] (if integration_mode == 'adapter')
                   or (f_out, h_med) if integration_mode == 'prefix'.
        """
        b = f_visual.shape[0]

        # Expand learnable queries to batch dimension: [B, K, D]
        q_m = self.med_queries.unsqueeze(0).expand(b, -1, -1)

        # 1. Cross-Attention: Query = Medication Queries, Key/Value = Visual Features
        h_m, _ = self.cross_attn(
            query=q_m,
            key=f_visual,
            value=f_visual,
        )
        h_prime = self.norm1(q_m + h_m)
        h_prime = self.norm2(h_prime + self.mlp(h_prime))  # [B, K, D]

        # 2. Integration into visual pipeline
        if self.integration_mode == "adapter":
            # Query = Visual tokens, Key/Value = Medication representation
            mod_out, _ = self.adapter_attn(
                query=f_visual,
                key=h_prime,
                value=h_prime,
            )
            f_refined = f_visual + self.gamma * self.adapter_mlp(mod_out)
            return self.adapter_norm(f_refined)

        elif self.integration_mode == "prefix":
            return f_visual, h_prime
        else:
            raise ValueError(f"Unknown integration mode '{self.integration_mode}'")
