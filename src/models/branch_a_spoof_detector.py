"""
branch_a_spoof_detector.py — Spoof / Deepfake Detection model.

Maps to ATML Unit 3 (GANs — synthetic speech artifacts) and
         ATML Unit 4 (Attention — multi-head attention pooling).

Architecture
------------
Input: raw waveform tensor [B, T_samples] or Mel spectrogram [B, n_mels, T_frames]
  |
  v
Backbone: wav2vec2-base (freeze first 6 transformer layers)
  OR ConvBackbone (CNN stack fallback for compute-constrained / offline settings)
  |  frame-level embeddings: [B, T_frames, 768 or 256]
  v
MultiHeadAttentionPooling (manual multi-head attention pooling)
  |  fixed-length utterance embedding: [B, 2 * embed_dim]
  v
Head: Dense(2*embed_dim -> 256) -> BatchNorm1d -> LeakyReLU(0.2) -> Dropout(0.3) -> Dense(256 -> 2)
  |
  v
Softmax / CrossEntropyLoss (bonafide=0 vs. spoof=1)
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.attention_pooling import MultiHeadAttentionPooling

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CNN Backbone (compute-constrained or offline fallback)
# ---------------------------------------------------------------------------

class ConvBackbone(nn.Module):
    """
    Lightweight 2-D CNN backbone over log-Mel spectrograms.
    Maps [B, n_mels, T_frames] -> [B, T_frames, output_dim].
    """

    def __init__(self, n_mels: int = 80, output_dim: int = 256) -> None:
        super().__init__()
        self.output_dim = output_dim

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),  # Halve frequency dimension only
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )

        remaining_mels = n_mels // 8
        self.proj = nn.Linear(128 * remaining_mels, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        mel_spec: [B, n_mels, T]
        returns:  [B, T, output_dim]
        """
        if mel_spec.ndim == 2:
            mel_spec = mel_spec.unsqueeze(0)

        x = mel_spec.unsqueeze(1)  # [B, 1, n_mels, T]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)  # [B, 128, n_mels//8, T]

        b, c, m, t = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * m)  # [B, T, C*M]
        x = self.norm(self.proj(x))
        return x


# ---------------------------------------------------------------------------
# Main Branch A Model: SpoofDetector
# ---------------------------------------------------------------------------

class SpoofDetector(nn.Module):
    """
    Branch A model: backbone (wav2vec2 or CNN) + attention pooling + classifier head.
    """

    def __init__(
        self,
        cfg: dict,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        backbone_cfg = cfg["backbone"]
        attn_cfg = cfg["attention_pooling"]
        head_cfg = cfg["head"]

        self.backbone_name = backbone_cfg.get("name", "wav2vec2")
        self.embed_dim = backbone_cfg.get("output_dim", 768)

        # 1. Backbone Initialization
        if self.backbone_name == "wav2vec2":
            try:
                from transformers import Wav2Vec2Model
                pretrained_name = backbone_cfg.get("pretrained_model", "facebook/wav2vec2-base")
                self.backbone = Wav2Vec2Model.from_pretrained(pretrained_name)
                self.embed_dim = self.backbone.config.hidden_size

                # Freeze first N transformer layers
                freeze_n = backbone_cfg.get("freeze_layers", 6)
                if hasattr(self.backbone, "feature_extractor"):
                    for p in self.backbone.feature_extractor.parameters():
                        p.requires_grad = False
                if hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layers"):
                    for layer in self.backbone.encoder.layers[:freeze_n]:
                        for p in layer.parameters():
                            p.requires_grad = False
                logger.info("Loaded wav2vec2-base with first %d layers frozen.", freeze_n)
            except Exception as e:
                logger.warning("Could not load HuggingFace wav2vec2 (%s). Falling back to ConvBackbone.", e)
                self.backbone_name = "cnn_attention"
                self.embed_dim = 256
                self.backbone = ConvBackbone(
                    n_mels=cfg.get("audio", {}).get("n_mels", 80),
                    output_dim=self.embed_dim,
                )
        elif self.backbone_name == "cnn_attention":
            self.embed_dim = backbone_cfg.get("output_dim", 256)
            self.backbone = ConvBackbone(
                n_mels=cfg.get("audio", {}).get("n_mels", 80),
                output_dim=self.embed_dim,
            )
        else:
            raise ValueError(f"Unknown backbone: {self.backbone_name}")

        # 2. Hand-implemented Multi-Head Attention Pooling
        self.attention_pooling = MultiHeadAttentionPooling(
            embed_dim=self.embed_dim,
            num_heads=attn_cfg.get("num_heads", 8),
            dropout=attn_cfg.get("dropout", 0.1),
        )

        # 3. Classifier Head (input is 2 * embed_dim due to mean + std statistics pooling)
        hidden_dim = head_cfg.get("hidden_dim", 256)
        num_classes = head_cfg.get("num_classes", 2)
        dropout_p = head_cfg.get("dropout", 0.3)

        self.head = nn.Sequential(
            nn.Linear(2 * self.embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, num_classes),
        )

        # 4. Class-weighted Cross-Entropy Loss
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract fixed-length utterance embedding [B, 2 * embed_dim].
        x: raw waveform [B, T] for wav2vec2 OR Mel spectrogram [B, n_mels, T] for ConvBackbone.
        """
        if self.backbone_name == "wav2vec2":
            if x.ndim == 1:
                x = x.unsqueeze(0)
            outputs = self.backbone(x)
            frame_embs = outputs.last_hidden_state  # [B, T_frames, hidden_size]
        else:
            frame_embs = self.backbone(x)  # [B, T_frames, output_dim]

        pooled = self.attention_pooling(frame_embs)  # [B, 2 * embed_dim]
        return pooled

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        Returns:
            'logits': [B, 2]
            'probs': [B, 2]
            'embedding': [B, 2 * embed_dim]
            'loss': scalar (optional if labels provided)
        """
        embedding = self.extract_embedding(x)
        logits = self.head(embedding)
        probs = F.softmax(logits, dim=-1)

        res = {
            "logits": logits,
            "probs": probs,
            "embedding": embedding,
        }

        if labels is not None:
            res["loss"] = self.criterion(logits, labels)

        return res


def save_checkpoint(
    model: SpoofDetector,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_eer: float,
    path: str,
) -> None:
    torch.save({
        "epoch": epoch,
        "val_eer": val_eer,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": model.cfg,
    }, path)


def load_checkpoint(
    model: SpoofDetector,
    path: str,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return {"epoch": ckpt.get("epoch", 0), "val_eer": ckpt.get("val_eer", 1.0)}
