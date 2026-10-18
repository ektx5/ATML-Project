"""
attention_pooling.py — Hand-implemented multi-head self-attention pooling.

Maps to ATML Unit 4: Attention mechanisms.

This is a COURSE REQUIREMENT: multi-head attention is implemented
manually in this module (not delegated to nn.MultiheadAttention) to demonstrate
first-principles understanding of query-key-value transformations, scaled dot-product
attention, and statistics aggregation.

Architecture
------------
Given a sequence of frame embeddings H ∈ R^{B × T × D}:

1. Compute query, key, value projections for each head h:
     Q_h = H W_Q^h,   K_h = H W_K^h,   V_h = H W_V^h
2. Compute scaled dot-product attention weights:
     A_h = softmax( Q_h K_h^T / sqrt(d_k) ) ∈ R^{B × T × T}
3. Attention-weighted value context:
     Context_h = A_h V_h
4. Concatenate all heads + linear projection + layer norm.
5. Statistics attention pooling:
     Compute frame-level importance weights by averaging attention across heads/queries,
     then compute attention-weighted mean and attention-weighted standard deviation.
     Output = concat([weighted_mean, weighted_std]) ∈ R^{B × 2D}.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Single-head scaled dot-product attention.
    Equation: Attention(Q, K, V) = softmax( (Q K^T) / sqrt(d_k) ) V
    """

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,  # [B, T_q, d_k]
        key: torch.Tensor,    # [B, T_k, d_k]
        value: torch.Tensor,  # [B, T_k, d_v]
        mask: Optional[torch.Tensor] = None,  # [B, T_q, T_k] bool mask (True = ignore)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d_k = query.size(-1)
        scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        context = torch.bmm(weights, value)

        return context, weights


class MultiHeadAttentionPooling(nn.Module):
    """
    Multi-head attention pooling layer with attentive statistics aggregation.

    Input: [B, T, embed_dim]
    Output: [B, 2 * embed_dim] (mean + std concatenated)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads

        # Projections for Q, K, V
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

        # Output projection and normalization
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.layer_norm = nn.LayerNorm(embed_dim)

        self.attn = ScaledDotProductAttention(dropout=dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        [B, T, D] -> [B * num_heads, T, d_k]
        """
        B, T, _ = x.size()
        x = x.view(B, T, self.num_heads, self.d_k)
        x = x.permute(0, 2, 1, 3).contiguous()  # [B, num_heads, T, d_k]
        return x.view(B * self.num_heads, T, self.d_k)

    def _merge_heads(self, x: torch.Tensor, B: int) -> torch.Tensor:
        """
        [B * num_heads, T, d_k] -> [B, T, D]
        """
        T = x.size(1)
        x = x.view(B, self.num_heads, T, self.d_k)
        x = x.permute(0, 2, 1, 3).contiguous()  # [B, T, num_heads, d_k]
        return x.view(B, T, self.embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor [B, T, embed_dim]
        mask : Optional[torch.Tensor]

        Returns
        -------
        pooled : torch.Tensor [B, 2 * embed_dim]
        """
        B, T, _ = x.size()

        # 1. Project Q, K, V and split into heads
        q = self._split_heads(self.W_q(x))  # [B*num_heads, T, d_k]
        k = self._split_heads(self.W_k(x))
        v = self._split_heads(self.W_v(x))

        expanded_mask = None
        if mask is not None:
            # Expand mask for all heads
            expanded_mask = mask.repeat_interleave(self.num_heads, dim=0)

        # 2. Scaled dot-product attention per head
        context, attn_weights = self.attn(q, k, v, mask=expanded_mask)

        # 3. Merge heads back & project
        context = self._merge_heads(context, B)  # [B, T, embed_dim]
        context = self.layer_norm(self.out_proj(context))

        # 4. Attention Statistics Pooling (mean + std)
        # Average attention across query positions and heads to get scalar weight per frame
        # attn_weights: [B*num_heads, T, T] -> view [B, num_heads, T, T]
        attn_4d = attn_weights.view(B, self.num_heads, T, T)
        frame_weights = attn_4d.mean(dim=(1, 2))  # [B, T]
        frame_weights = F.softmax(frame_weights, dim=-1).unsqueeze(-1)  # [B, T, 1]

        # Weighted Mean
        weighted_mean = torch.sum(context * frame_weights, dim=1)  # [B, embed_dim]

        # Weighted Standard Deviation
        weighted_var = torch.sum(frame_weights * (context ** 2), dim=1) - (weighted_mean ** 2)
        weighted_std = torch.sqrt(torch.clamp(weighted_var, min=1e-9))  # [B, embed_dim]

        # Fixed-length concatenated embedding: [B, 2 * embed_dim]
        pooled = torch.cat([weighted_mean, weighted_std], dim=-1)
        return pooled
