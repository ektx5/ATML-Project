"""
triplet_loss.py — Triplet loss with semi-hard negative mining.

Maps to ATML Unit 1: Siamese networks + metric learning.

Reference: "FaceNet: A Unified Embedding for Face Recognition and Clustering"
           (Schroff, Kalenichenko, Philbin, 2015).

Standard triplet loss:
  L = max(0, d(a,p) - d(a,n) + margin)

Semi-hard negative mining:
  For each anchor-positive pair, choose negatives that are:
    - HARDER than the positive (d(a,n) > d(a,p))
    - But NOT the hardest (d(a,n) < d(a,p) + margin)
  If no semi-hard negative exists in the batch, fall back to the hardest negative
  with d(a,n) > d(a,p) or hardest negative overall.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Triplet loss with online negative mining strategy.

    Parameters
    ----------
    margin : float
        Triplet margin (default 0.2 for cosine-space / normalized embeddings).
    mining : str
        'semihard' | 'hardest' | 'all'
    distance : str
        'cosine' | 'euclidean'
    reduction : str
        'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        margin: float = 0.2,
        mining: str = "semihard",
        distance: str = "cosine",
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.margin = margin
        self.mining = mining
        self.distance = distance
        self.reduction = reduction

    def _pairwise_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise distance matrix [B, B].
        """
        if self.distance == "cosine":
            # For L2-normalized embeddings, cosine distance = 1 - cos_sim
            norm_emb = F.normalize(embeddings, p=2, dim=-1)
            sim = torch.mm(norm_emb, norm_emb.t())
            # Clamp to [-1, 1] to avoid numerical precision errors
            sim = torch.clamp(sim, -1.0, 1.0)
            return torch.clamp(1.0 - sim, min=0.0, max=2.0)
        else:
            # Euclidean distance
            diff = embeddings.unsqueeze(1) - embeddings.unsqueeze(0)
            return torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-12)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute triplet loss over a batch with labels.

        Parameters
        ----------
        embeddings : torch.Tensor [B, D]
        labels : torch.Tensor [B] int64 speaker IDs
        """
        B = embeddings.size(0)
        dist_matrix = self._pairwise_distances(embeddings)  # [B, B]

        # Masks for positive and negative pairs
        labels = labels.view(-1, 1)
        matches = torch.eq(labels, labels.t())  # [B, B] boolean
        diff_speakers = ~matches

        # Eliminate diagonal (self-match)
        eye = torch.eye(B, dtype=torch.bool, device=embeddings.device)
        pos_mask = matches & (~eye)

        if self.mining == "all":
            # Form all valid (a, p, n) triplets
            d_ap = dist_matrix.unsqueeze(2)  # [B, B, 1]
            d_an = dist_matrix.unsqueeze(1)  # [B, 1, B]
            triplet_loss = F.relu(d_ap - d_an + self.margin)

            triplet_mask = pos_mask.unsqueeze(2) & diff_speakers.unsqueeze(1)
            valid_losses = triplet_loss[triplet_mask]

            if valid_losses.numel() == 0:
                return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

            if self.reduction == "mean":
                # Average over active (loss > 0) or all valid triplets
                active_losses = valid_losses[valid_losses > 0.0]
                return active_losses.mean() if active_losses.numel() > 0 else valid_losses.mean()
            elif self.reduction == "sum":
                return valid_losses.sum()
            else:
                return valid_losses

        # Online mining: iterate over each anchor-positive pair
        losses = []
        for a_idx in range(B):
            pos_indices = torch.nonzero(pos_mask[a_idx], as_tuple=False).squeeze(-1)
            neg_indices = torch.nonzero(diff_speakers[a_idx], as_tuple=False).squeeze(-1)

            if pos_indices.numel() == 0 or neg_indices.numel() == 0:
                continue

            for p_idx in pos_indices:
                p_idx = p_idx.item()
                d_ap = dist_matrix[a_idx, p_idx]
                d_an_candidates = dist_matrix[a_idx, neg_indices]

                selected_d_an = None

                if self.mining == "semihard":
                    # Semi-hard: d_ap < d_an < d_ap + margin
                    semihard_mask = (d_an_candidates > d_ap) & (d_an_candidates < d_ap + self.margin)
                    semihard_candidates = d_an_candidates[semihard_mask]

                    if semihard_candidates.numel() > 0:
                        # Choose the hardest among semi-hard (closest to d_ap)
                        selected_d_an = torch.min(semihard_candidates)
                    else:
                        # Fallback to hardest negative overall (smallest d_an)
                        selected_d_an = torch.min(d_an_candidates)
                elif self.mining == "hardest":
                    # Smallest d_an
                    selected_d_an = torch.min(d_an_candidates)

                if selected_d_an is not None:
                    loss_val = F.relu(d_ap - selected_d_an + self.margin)
                    losses.append(loss_val)

        if len(losses) == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        stacked_losses = torch.stack(losses)
        if self.reduction == "mean":
            return stacked_losses.mean()
        elif self.reduction == "sum":
            return stacked_losses.sum()
        else:
            return stacked_losses

    def verify_loss_properties(
        self,
        anchor_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        negative_emb: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Direct calculation for a single anchor, positive, negative triplet.
        """
        batch = torch.cat([anchor_emb, positive_emb, negative_emb], dim=0)
        labels = torch.tensor([0, 0, 1], dtype=torch.long, device=batch.device)

        d_matrix = self._pairwise_distances(batch)
        d_ap = d_matrix[0, 1].item()
        d_an = d_matrix[0, 2].item()
        loss = max(0.0, d_ap - d_an + self.margin)

        return {
            "loss": loss,
            "d_ap": d_ap,
            "d_an": d_an,
        }
