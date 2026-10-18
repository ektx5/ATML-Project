"""
contrastive_loss.py — Contrastive loss for speaker verification.

Maps to ATML Unit 1: Siamese networks.

Reference: "Dimensionality Reduction by Learning an Invariant Mapping"
           (Hadsell, Chopra, LeCun, 2006).

Loss:
  L = is_same * 0.5 * d^2 + (1 - is_same) * 0.5 * max(0, margin - d)^2

where d = pairwise_distance(emb_a, emb_b), is_same = 1 for same speaker, 0 for different.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for Siamese speaker verification.

    Parameters
    ----------
    margin : float
        Minimum distance expected between embeddings of different speakers.
        Default 1.0 (for L2-normalized embeddings, distance ranges from 0 to 2).
    reduction : str
        'mean' | 'sum' | 'none'
    """

    def __init__(self, margin: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        is_same: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        emb_a, emb_b : torch.Tensor [B, D] L2-normalized embeddings
        is_same : torch.Tensor [B] float or long (1.0 = same speaker, 0.0 = different)

        Returns
        -------
        loss : torch.Tensor scalar (or [B] if reduction='none')
        """
        if is_same.dtype != torch.float32 and is_same.dtype != torch.float64:
            is_same = is_same.float()

        # Euclidean distance
        d = F.pairwise_distance(emb_a, emb_b, p=2)

        same_loss = 0.5 * torch.pow(d, 2)
        diff_loss = 0.5 * torch.pow(F.relu(self.margin - d), 2)

        loss_per_sample = is_same * same_loss + (1.0 - is_same) * diff_loss

        if self.reduction == "mean":
            return loss_per_sample.mean()
        elif self.reduction == "sum":
            return loss_per_sample.sum()
        elif self.reduction == "none":
            return loss_per_sample
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")
