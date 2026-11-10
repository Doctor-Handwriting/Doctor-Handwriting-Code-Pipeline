"""
Uncertainty Decoder Module (Component D).
Performs token pooling over generated decoder hidden states, estimates confidence,
and executes the clinical thresholding decision rule (Accepted vs. Requires Review).
"""

from typing import Dict, List, Literal, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .confidence_head import ConfidenceHead


class UncertaintyDecoder(nn.Module):
    """
    Decodes confidence estimates from language decoder hidden states.
    Supports mean pooling, last token representation, and attention pooling.
    """

    def __init__(
        self,
        hidden_dim: int,
        head_type: Literal["linear", "mlp"] = "mlp",
        pooling_method: Literal["mean", "last", "attention"] = "mean",
        confidence_threshold: float = 0.70,
        dropout: float = 0.1,
    ):
        """
        Args:
            hidden_dim: Language model hidden dimension (e.g. 2048 or 2560).
            head_type: 'linear' or 'mlp'.
            pooling_method: 'mean', 'last', or 'attention'.
            confidence_threshold: Tau threshold for clinical review flagging.
            dropout: Dropout probability.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pooling_method = pooling_method
        self.confidence_threshold = confidence_threshold

        self.confidence_head = ConfidenceHead(
            hidden_dim=hidden_dim,
            head_type=head_type,
            dropout=dropout,
        )

        if pooling_method == "attention":
            self.query_pool = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
            self.pool_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True,
            )

    def pool_hidden_states(
        self,
        hidden_states: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pools token sequence hidden states into a single vector h in R^(B x D).

        Args:
            hidden_states: Sequence hidden states [B, L, D]
            token_mask: Boolean mask [B, L] indicating valid generated or target tokens.

        Returns:
            h: Pooled representation [B, D]
        """
        b, l, d = hidden_states.shape

        if token_mask is None:
            token_mask = torch.ones(b, l, dtype=torch.bool, device=hidden_states.device)

        if self.pooling_method == "mean":
            mask_expanded = token_mask.unsqueeze(-1).to(hidden_states.dtype)  # [B, L, 1]
            sum_h = (hidden_states * mask_expanded).sum(dim=1)
            count = mask_expanded.sum(dim=1).clamp(min=1.0)
            return sum_h / count

        elif self.pooling_method == "last":
            # Select the last unmasked token per row
            h_last = []
            for row in range(b):
                unmasked_indices = token_mask[row].nonzero(as_tuple=True)[0]
                if len(unmasked_indices) > 0:
                    last_idx = unmasked_indices[-1]
                    h_last.append(hidden_states[row, last_idx])
                else:
                    h_last.append(hidden_states[row, -1])
            return torch.stack(h_last, dim=0)

        elif self.pooling_method == "attention":
            q = self.query_pool.expand(b, -1, -1)  # [B, 1, D]
            key_padding_mask = ~token_mask  # True where padding
            attn_out, _ = self.pool_attn(
                query=q,
                key=hidden_states,
                value=hidden_states,
                key_padding_mask=key_padding_mask,
            )
            return attn_out.squeeze(1)  # [B, D]

        else:
            raise ValueError(f"Unknown pooling method '{self.pooling_method}'")

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: pool tokens and predict correctness probability.

        Args:
            hidden_states: Sequence hidden states [B, L, D]
            token_mask: Boolean mask [B, L]

        Returns:
            logits: Confidence logits [B, 1]
            probs: Predicted probabilities c in [0, 1], shape [B, 1]
        """
        h = self.pool_hidden_states(hidden_states, token_mask)
        return self.confidence_head(h)

    def evaluate_decision(
        self,
        confidence: float,
        threshold: Optional[float] = None,
    ) -> Dict[str, Union[float, bool]]:
        """
        Execute thresholding decision: c < tau => Requires Review.
        """
        tau = threshold if threshold is not None else self.confidence_threshold
        requires_review = bool(confidence < tau)
        return {
            "confidence": float(round(confidence, 4)),
            "threshold": float(tau),
            "requires_review": requires_review,
            "accepted": not requires_review,
        }
