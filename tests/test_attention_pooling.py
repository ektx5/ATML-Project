"""
test_attention_pooling.py — Unit tests for the hand-implemented multi-head
attention pooling module.

Run with: pytest tests/test_attention_pooling.py -v
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.attention_pooling import MultiHeadAttentionPooling, ScaledDotProductAttention


class TestScaledDotProductAttention:
    def test_output_shape(self):
        B, T, d_k = 4, 50, 64
        attn = ScaledDotProductAttention()
        Q = torch.randn(B, T, d_k)
        K = torch.randn(B, T, d_k)
        V = torch.randn(B, T, d_k)
        context, weights = attn(Q, K, V)
        assert context.shape == (B, T, d_k)
        assert weights.shape == (B, T, T)

    def test_attention_weights_sum_to_one(self):
        B, T, d_k = 2, 20, 32
        attn = ScaledDotProductAttention()
        Q = torch.randn(B, T, d_k)
        K = torch.randn(B, T, d_k)
        V = torch.randn(B, T, d_k)
        _, weights = attn(Q, K, V)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_mask_ignores_positions(self):
        B, T, d_k = 2, 10, 16
        attn = ScaledDotProductAttention()
        Q = torch.randn(B, T, d_k)
        K = torch.randn(B, T, d_k)
        V = torch.randn(B, T, d_k)
        # Mask out everything except position 0
        mask = torch.zeros(B, T, T, dtype=torch.bool)
        mask[:, :, 1:] = True
        _, weights = attn(Q, K, V, mask=mask)
        # Position 0 should receive ~1.0 weight
        assert torch.allclose(weights[:, :, 0], torch.ones(B, T), atol=1e-4)
        assert torch.all(weights[:, :, 1:] < 1e-4)


class TestMultiHeadAttentionPooling:
    @pytest.fixture
    def pooling(self):
        return MultiHeadAttentionPooling(embed_dim=128, num_heads=4, dropout=0.0)

    def test_output_shape(self, pooling):
        B, T, D = 8, 100, 128
        x = torch.randn(B, T, D)
        out = pooling(x)
        assert out.shape == (B, 2 * D)

    def test_gradient_flow(self, pooling):
        B, T, D = 4, 50, 128
        x = torch.randn(B, T, D, requires_grad=True)
        out = pooling(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum().item() > 0.0

    def test_different_sequence_lengths(self, pooling):
        D = 128
        for T in [10, 50, 150]:
            x = torch.randn(2, T, D)
            out = pooling(x)
            assert out.shape == (2, 2 * D)

    def test_embed_dim_not_divisible_raises(self):
        with pytest.raises(AssertionError):
            MultiHeadAttentionPooling(embed_dim=100, num_heads=3)

    def test_deterministic_in_eval_mode(self, pooling):
        pooling.eval()
        x = torch.randn(2, 30, 128)
        with torch.no_grad():
            out1 = pooling(x)
            out2 = pooling(x)
        assert torch.allclose(out1, out2)
