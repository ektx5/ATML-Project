"""
branch_b_siamese.py — Speaker Verification via Siamese Network.

Maps to ATML Unit 1: Siamese networks + few-shot / meta-learning.

Architecture
------------
Twin encoder (shared weights):
  ECAPA-TDNN (Desplanques et al., 2020)
  OR ResNet-1D (from scratch fallback)

Both encoders produce a fixed-length speaker embedding [B, embedding_dim].

Verification:
  cosine_similarity(embedding_a, embedding_b) vs. a tuned threshold.
  Threshold is found by minimising EER on the validation set — not hardcoded.

Few-shot enrollment:
  A new speaker is enrolled from N=3-5 clips by averaging their embeddings.
  No retraining occurs.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Squeeze-and-Excitation Block (used in ECAPA-TDNN)
# ---------------------------------------------------------------------------

class SE_Block(nn.Module):
    """
    Channel-wise Squeeze-and-Excitation block (Hu et al., 2018).
    Adaptively rescales channel responses using global temporal context.
    """

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        bottleneck = max(16, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, bottleneck, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, T]
        returns: [B, C, T]
        """
        b, c, _ = x.size()
        scale = self.avg_pool(x).view(b, c)
        scale = self.fc(scale).view(b, c, 1)
        return x * scale


# ---------------------------------------------------------------------------
# TDNN Block (Time-Delay Neural Network layer)
# ---------------------------------------------------------------------------

class TDNNBlock(nn.Module):
    """
    1-D dilated convolution + BatchNorm + activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        padding = ((kernel_size - 1) * dilation) // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(x)))


class SERes2NetBlock(nn.Module):
    """
    Dilated TDNN block with Squeeze-and-Excitation and residual connection.
    """

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 2) -> None:
        super().__init__()
        self.tdnn = TDNNBlock(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.se = SE_Block(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.tdnn(x)
        out = self.se(out)
        return out + residual


# ---------------------------------------------------------------------------
# ECAPA-TDNN Encoder
# ---------------------------------------------------------------------------

class ECAPA_TDNN(nn.Module):
    """
    ECAPA-TDNN speaker encoder (Desplanques et al., 2020).
    Input:  log-Mel spectrogram [B, n_mels, T_frames]
    Output: speaker embedding    [B, embedding_dim]
    """

    def __init__(
        self,
        n_mels: int = 80,
        channels: int = 512,
        embedding_dim: int = 192,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.embedding_dim = embedding_dim

        # Initial stem
        self.layer1 = TDNNBlock(n_mels, channels, kernel_size=5, dilation=1)

        # 3 Res2Net-style blocks with increasing dilation
        self.layer2 = SERes2NetBlock(channels, kernel_size=3, dilation=2)
        self.layer3 = SERes2NetBlock(channels, kernel_size=3, dilation=3)
        self.layer4 = SERes2NetBlock(channels, kernel_size=3, dilation=4)

        # Multi-scale feature aggregation (MFA): concatenate layers 2, 3, 4
        self.mfa_conv = nn.Conv1d(channels * 3, channels * 3, kernel_size=1)
        self.mfa_norm = nn.BatchNorm1d(channels * 3)
        self.mfa_relu = nn.ReLU(inplace=True)

        # Attentive Statistics Pooling
        self.attention = nn.Sequential(
            nn.Conv1d(channels * 3, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Tanh(),
            nn.Conv1d(128, channels * 3, kernel_size=1),
        )

        # Projection head to target embedding dim
        # Pooled output is (mean + std) -> dimension is 2 * (channels * 3)
        self.fc = nn.Linear((channels * 3) * 2, embedding_dim)
        self.bn_out = nn.BatchNorm1d(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor [B, n_mels, T_frames]

        Returns
        -------
        embedding : torch.Tensor [B, embedding_dim] L2-normalized
        """
        # Intermediate representations for MFA
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # Multi-layer aggregation
        cat_features = torch.cat([x2, x3, x4], dim=1)  # [B, 3*channels, T]
        mfa = self.mfa_relu(self.mfa_norm(self.mfa_conv(cat_features)))

        # Attentive statistics pooling
        attn_weights = F.softmax(self.attention(mfa), dim=-1)  # [B, 3*channels, T]

        # Weighted mean
        mean = torch.sum(mfa * attn_weights, dim=-1)  # [B, 3*channels]

        # Weighted std
        var = torch.sum(attn_weights * (mfa ** 2), dim=-1) - (mean ** 2)
        std = torch.sqrt(torch.clamp(var, min=1e-9))

        pooled = torch.cat([mean, std], dim=-1)  # [B, 6*channels]

        # Projection & normalization
        emb = self.bn_out(self.fc(pooled))
        return F.normalize(emb, p=2, dim=-1)


# ---------------------------------------------------------------------------
# ResNet-1D Encoder (fallback architecture)
# ---------------------------------------------------------------------------

class ResBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + self.shortcut(x))
        return out


class ResNet1D(nn.Module):
    def __init__(
        self,
        n_mels: int = 80,
        embedding_dim: int = 192,
        layers: List[int] = [3, 4, 6, 3],
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_mels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )

        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.stage1 = self._make_stage(base_channels, channels[0], layers[0], stride=1)
        self.stage2 = self._make_stage(channels[0], channels[1], layers[1], stride=2)
        self.stage3 = self._make_stage(channels[1], channels[2], layers[2], stride=2)
        self.stage4 = self._make_stage(channels[2], channels[3], layers[3], stride=2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[3], embedding_dim)

    def _make_stage(self, in_c: int, out_c: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [ResBlock1D(in_c, out_c, stride=stride)]
        for _ in range(1, blocks):
            layers.append(ResBlock1D(out_c, out_c, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.pool(out).squeeze(-1)
        emb = self.fc(out)
        return F.normalize(emb, p=2, dim=-1)


# ---------------------------------------------------------------------------
# Siamese Speaker Verifier (Twin Encoder with Shared Weights)
# ---------------------------------------------------------------------------

class SiameseSpeakerVerifier(nn.Module):
    """
    Siamese speaker verification model.
    The shared-weight twin encoder processes both enrollment and query audio.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        enc_cfg = cfg["encoder"]
        audio_cfg = cfg["audio"]

        enc_type = enc_cfg.get("type", "ecapa_tdnn")
        self.embedding_dim = enc_cfg.get("embedding_dim", 192)

        if enc_type == "ecapa_tdnn":
            self.encoder = ECAPA_TDNN(
                n_mels=audio_cfg.get("n_mels", 80),
                channels=enc_cfg.get("channels", 512),
                embedding_dim=self.embedding_dim,
            )
        elif enc_type == "resnet1d":
            self.encoder = ResNet1D(
                n_mels=audio_cfg.get("n_mels", 80),
                embedding_dim=self.embedding_dim,
                layers=enc_cfg.get("resnet_layers", [3, 4, 6, 3]),
                base_channels=enc_cfg.get("resnet_base_channels", 64),
            )
        else:
            raise ValueError(f"Unknown encoder type: {enc_type}")

    def encode(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Encode mel spectrogram tensor [B, n_mels, T_frames] -> L2-normalized embedding [B, D].
        """
        return self.encoder(mel_spec)

    def forward(
        self,
        anchor: torch.Tensor,
        positive: Optional[torch.Tensor] = None,
        negative: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Siamese forward pass for training.
        """
        anchor_emb = self.encode(anchor)
        positive_emb = self.encode(positive) if positive is not None else None
        negative_emb = self.encode(negative) if negative is not None else None

        return {
            "anchor_emb": anchor_emb,
            "positive_emb": positive_emb,
            "negative_emb": negative_emb,
        }

    def compute_similarity(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cosine similarity between embedding vectors in [-1.0, 1.0].
        """
        return F.cosine_similarity(emb_a, emb_b, dim=-1)


# ---------------------------------------------------------------------------
# Few-shot Enrollment & Verification
# ---------------------------------------------------------------------------

def enroll_speaker(
    model: SiameseSpeakerVerifier,
    mel_specs: List[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """
    Enroll a new speaker from N=3-5 reference clips by averaging their embeddings.
    Zero retraining required (meta-learning / few-shot).
    """
    model.eval()
    embeddings = []
    with torch.no_grad():
        for mel in mel_specs:
            if mel.ndim == 2:
                mel = mel.unsqueeze(0)
            emb = model.encode(mel.to(device))
            embeddings.append(emb)

    avg_embedding = torch.mean(torch.stack(embeddings, dim=0), dim=0)
    return F.normalize(avg_embedding, p=2, dim=-1)


def verify_speaker(
    model: SiameseSpeakerVerifier,
    query_mel: torch.Tensor,
    enrollment_embedding: torch.Tensor,
    threshold: float,
    device: torch.device,
) -> Tuple[bool, float]:
    """
    Verify whether query clip matches the enrolled speaker identity.
    """
    model.eval()
    if query_mel.ndim == 2:
        query_mel = query_mel.unsqueeze(0)
    if enrollment_embedding.ndim == 1:
        enrollment_embedding = enrollment_embedding.unsqueeze(0)

    with torch.no_grad():
        query_emb = model.encode(query_mel.to(device))
        sim = model.compute_similarity(query_emb, enrollment_embedding.to(device))
        similarity_val = sim.item() if sim.numel() == 1 else sim[0].item()

    is_match = similarity_val >= threshold
    return is_match, float(similarity_val)
