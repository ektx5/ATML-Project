"""
test_triplet_loss.py — Unit tests for the triplet loss with semi-hard mining.

Run with: pytest tests/test_triplet_loss.py -v
"""

import pytest
import torch
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.losses.triplet_loss import TripletLoss


EMBEDDING_DIM = 64
MARGIN = 0.2


def make_embeddings(n_speakers: int, n_per_speaker: int, dim: int = EMBEDDING_DIM):
    embeddings = []
    labels = []
    for spk in range(n_speakers):
        spk_emb = F.normalize(torch.randn(1, dim), p=2, dim=-1)
        # Small perturbations for same speaker so they are close but not strictly identical
        for _ in range(n_per_speaker):
            noise = torch.randn(1, dim) * 0.01
            perturbed = F.normalize(spk_emb + noise, p=2, dim=-1)
            embeddings.append(perturbed)
            labels.append(spk)
    return torch.cat(embeddings, dim=0), torch.tensor(labels, dtype=torch.long)


class TestTripletLossBasic:
    @pytest.fixture
    def loss_fn(self):
        return TripletLoss(margin=MARGIN, mining="all", distance="cosine")

    def test_zero_loss_for_perfect_triplet(self, loss_fn):
        """
        When d_an > d_ap + margin, loss must be exactly 0.
        """
        # Form orthogocal / opposite vectors
        dim = EMBEDDING_DIM
        anchor = torch.zeros(1, dim)
        anchor[0, 0] = 1.0  # [1, 0, ...]
        positive = torch.zeros(1, dim)
        positive[0, 0] = 1.0  # identical to anchor, d_ap = 0
        negative = torch.zeros(1, dim)
        negative[0, 0] = -1.0  # opposite, cos_sim = -1, d_an = 2.0

        res = loss_fn.verify_loss_properties(anchor, positive, negative)
        assert res["loss"] == 0.0
        assert res["d_ap"] == 0.0
        assert res["d_an"] == 2.0

    def test_positive_loss_for_violated_margin(self, loss_fn):
        """
        When d_an < d_ap + margin, loss must be positive.
        """
        dim = EMBEDDING_DIM
        anchor = torch.zeros(1, dim)
        anchor[0, 0] = 1.0
        # Positive is somewhat far: d_ap ~ 0.5
        positive = F.normalize(torch.tensor([[0.5, 0.866] + [0.0] * (dim - 2)]), p=2, dim=-1)
        # Negative is closer: d_an ~ 0.1
        negative = F.normalize(torch.tensor([[0.95, 0.31] + [0.0] * (dim - 2)]), p=2, dim=-1)

        res = loss_fn.verify_loss_properties(anchor, positive, negative)
        assert res["loss"] > 0.0
        assert res["d_ap"] > res["d_an"]  # severely violated


class TestTripletLossMining:
    def test_semihard_mining_does_not_crash(self):
        loss_fn = TripletLoss(margin=MARGIN, mining="semihard")
        embeddings, labels = make_embeddings(n_speakers=4, n_per_speaker=3)
        loss = loss_fn(embeddings, labels)
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0

    def test_hardest_mining_does_not_crash(self):
        loss_fn = TripletLoss(margin=MARGIN, mining="hardest")
        embeddings, labels = make_embeddings(n_speakers=4, n_per_speaker=3)
        loss = loss_fn(embeddings, labels)
        assert torch.isfinite(loss)

    def test_all_mining_does_not_crash(self):
        loss_fn = TripletLoss(margin=MARGIN, mining="all")
        embeddings, labels = make_embeddings(n_speakers=4, n_per_speaker=3)
        loss = loss_fn(embeddings, labels)
        assert torch.isfinite(loss)
