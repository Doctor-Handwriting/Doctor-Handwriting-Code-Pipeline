"""
Confidence Head Module (Component D).
Predicts the correctness probability of generated medication transcriptions
from decoder hidden representations using Binary Cross-Entropy loss.
"""

from typing import Literal, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfidenceHead(nn.Module):
    """
    MLP or Linear projection predicting transcription correctness probability c in [0, 1].
    c = sigmoid(W_c h + b_c)
    """

    def __init__(
        self,
        hidden_dim: int,
        head_type: Literal["linear", "mlp"] = "mlp",
        dropout: float = 0.1,
    ):
        """
        Args:
            hidden_dim: Language decoder hidden dimension (e.g. 2048 or 2560).
            head_type: Architecture type ('linear' or 'mlp').
            dropout: Dropout probability.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_type = head_type

        if head_type == "mlp":
            mid_dim = hidden_dim // 2
            self.net = nn.Sequential(
                nn.Linear(hidden_dim, mid_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mid_dim, 1),
            )
        elif head_type == "linear":
            self.net = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            raise ValueError(f"Unknown head type '{head_type}'. Options: linear, mlp")

        # Initialize final layer bias to 0.0 (starts at 50% prior probability)
        if isinstance(self.net[-1], nn.Linear):
            nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: Pooled decoder representation [B, D]

        Returns:
            logits: Raw confidence logits [B, 1]
            probs: Predicted correctness probability c in [0, 1], shape [B, 1]
        """
        logits = self.net(h)
        probs = torch.sigmoid(logits)
        return logits, probs

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Binary Cross-Entropy loss with logits.

        Args:
            logits: Predicted logits [B, 1] or [B]
            targets: Binary correctness labels z in {0, 1}, shape [B, 1] or [B]

        Returns:
            Scalar BCE loss.
        """
        logits_flat = logits.view(-1).float()
        targets_flat = targets.view(-1).float()
        return F.binary_cross_entropy_with_logits(logits_flat, targets_flat)
