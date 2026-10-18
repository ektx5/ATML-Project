"""
fusion.py — Risk Score Fusion Layer.

Combines three independent signals into a single 0-100 risk score:
  1. spoof_probability     (Branch A output, in [0, 1])
  2. ae_error_normalised   (Autoencoder reconstruction error, z-score normalized)
  3. speaker_similarity    (Branch B cosine similarity, in [-1, 1])

Outputs:
  - risk_score : float in [0, 100]
  - verdict    : 'LOW RISK' | 'MEDIUM RISK' | 'HIGH RISK — flag for human review'
  - explanation: dict with percentage / weight breakdown of signals
  - disclaimer : explicit notice that this is a risk signal for review, not legal proof

Maps to explainability requirement and multi-branch biometric fusion.
"""

import json
import logging
import os
import pickle
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DISCLAIMER_TEXT = (
    "This score is a probabilistic risk signal for human-in-the-loop review, "
    "not legal-grade mathematical proof of fraud or identity."
)


def build_result(
    risk_score: float,
    spoof_prob: float,
    ae_error_norm: float,
    speaker_similarity: float,
    w_spoof: float,
    w_ae: float,
    w_mismatch: float,
    verdict_thresholds: Optional[Dict] = None,
) -> Dict:
    """
    Standardized payload builder.
    """
    # Enforce bounds
    risk_score = float(np.clip(risk_score, 0.0, 100.0))

    if verdict_thresholds is None:
        verdict_thresholds = {
            "low_risk_below": 30.0,
            "medium_risk_below": 60.0,
            "high_risk_above": 60.0,
        }

    low_th = verdict_thresholds.get("low_risk_below", 30.0)
    med_th = verdict_thresholds.get("medium_risk_below", 60.0)

    if risk_score < low_th:
        verdict = "LOW RISK"
    elif risk_score < med_th:
        verdict = "MEDIUM RISK"
    else:
        verdict = "HIGH RISK — flag for human review"

    # Compute relative signal contributions for explainability
    # Normalize inputs to [0, 1] range:
    ae_clamped = float(np.clip(max(0.0, ae_error_norm) / 10.0, 0.0, 1.0))
    # Cosine similarity in [-1, 1] -> mismatch penalty in [0, 1]
    # For high match (sim >= 0.70), mismatch penalty is near zero
    if speaker_similarity >= 0.70:
        spk_mismatch = float(max(0.0, (1.0 - speaker_similarity) * 0.2))
    else:
        spk_mismatch = float(np.clip((1.0 - speaker_similarity) / 2.0, 0.0, 1.0))

    spoof_contrib = float(w_spoof * spoof_prob)
    ae_contrib = float(w_ae * ae_clamped)
    spk_contrib = float(w_mismatch * spk_mismatch)

    total_contrib = spoof_contrib + ae_contrib + spk_contrib + 1e-9

    explanation = {
        "spoof_contribution": spoof_contrib,
        "ae_contribution": ae_contrib,
        "speaker_contribution": spk_contrib,
        "spoof_percentage": round((spoof_contrib / total_contrib) * 100.0, 1),
        "ae_percentage": round((ae_contrib / total_contrib) * 100.0, 1),
        "speaker_percentage": round((spk_contrib / total_contrib) * 100.0, 1),
    }

    return {
        "risk_score": round(risk_score, 1),
        "verdict": verdict,
        "spoof_probability": float(spoof_prob),
        "ae_error_normalised": float(ae_error_norm),
        "speaker_similarity": float(speaker_similarity),
        "explanation": explanation,
        "disclaimer": DISCLAIMER_TEXT,
    }


class WeightedSumFusion:
    """
    Explainable weighted combination of Branch A, Autoencoder, and Branch B.
    """

    def __init__(
        self,
        w_spoof: float = 0.5,
        w_ae_error: float = 0.2,
        w_speaker_mismatch: float = 0.3,
        verdict_thresholds: Optional[Dict] = None,
    ) -> None:
        total = w_spoof + w_ae_error + w_speaker_mismatch
        if total <= 0:
            w_spoof, w_ae_error, w_speaker_mismatch = 0.5, 0.2, 0.3
            total = 1.0

        self.w_spoof = w_spoof / total
        self.w_ae = w_ae_error / total
        self.w_mismatch = w_speaker_mismatch / total
        self.verdict_thresholds = verdict_thresholds

    def predict(
        self,
        spoof_prob: float,
        ae_error_norm: float,
        speaker_similarity: float,
    ) -> Dict:
        ae_clamped = np.clip(max(0.0, ae_error_norm) / 10.0, 0.0, 1.0)
        if speaker_similarity >= 0.70:
            spk_mismatch = max(0.0, (1.0 - speaker_similarity) * 0.2)
        else:
            spk_mismatch = np.clip((1.0 - speaker_similarity) / 2.0, 0.0, 1.0)

        risk_raw = (
            self.w_spoof * spoof_prob
            + self.w_ae * ae_clamped
            + self.w_mismatch * spk_mismatch
        )
        risk_score = np.clip(risk_raw, 0.0, 1.0) * 100.0

        return build_result(
            risk_score=risk_score,
            spoof_prob=spoof_prob,
            ae_error_norm=ae_error_norm,
            speaker_similarity=speaker_similarity,
            w_spoof=self.w_spoof,
            w_ae=self.w_ae,
            w_mismatch=self.w_mismatch,
            verdict_thresholds=self.verdict_thresholds,
        )


class LogisticFusion:
    """
    Logistic regression model trained on calibration triples.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        verdict_thresholds: Optional[Dict] = None,
    ) -> None:
        self.verdict_thresholds = verdict_thresholds
        self._model = None
        self._fallback = WeightedSumFusion(verdict_thresholds=verdict_thresholds)

        if model_path and os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    self._model = pickle.load(f)
                logger.info("Loaded calibrated LogisticRegression fusion model from %s", model_path)
            except Exception as e:
                logger.warning("Could not load logistic fusion model: %s", e)

    def fit(self, calibration_csv: str, model_save_path: str) -> None:
        import pandas as pd
        from sklearn.linear_model import LogisticRegression

        df = pd.read_csv(calibration_csv)
        # Features: [spoof_prob, ae_error_norm, spk_mismatch]
        df["spk_mismatch"] = (1.0 - df["speaker_sim"]) / 2.0
        X = df[["spoof_prob", "ae_error_norm", "spk_mismatch"]].values
        y = df["label"].values

        clf = LogisticRegression(class_weight="balanced", max_iter=1000)
        clf.fit(X, y)
        self._model = clf

        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        with open(model_save_path, "wb") as f:
            pickle.dump(clf, f)
        logger.info("Calibrated and saved LogisticRegression fusion model to %s", model_save_path)

    def predict(
        self,
        spoof_prob: float,
        ae_error_norm: float,
        speaker_similarity: float,
    ) -> Dict:
        if self._model is None:
            return self._fallback.predict(spoof_prob, ae_error_norm, speaker_similarity)

        spk_mismatch = (1.0 - speaker_similarity) / 2.0
        features = np.array([[spoof_prob, ae_error_norm, spk_mismatch]], dtype=np.float32)

        probs = self._model.predict_proba(features)[0]
        # P(threat/attack)
        risk_score = probs[1] * 100.0

        coefs = self._model.coef_[0]
        w_spoof, w_ae, w_mismatch = abs(coefs[0]), abs(coefs[1]), abs(coefs[2])

        return build_result(
            risk_score=risk_score,
            spoof_prob=spoof_prob,
            ae_error_norm=ae_error_norm,
            speaker_similarity=speaker_similarity,
            w_spoof=w_spoof,
            w_ae=w_ae,
            w_mismatch=w_mismatch,
            verdict_thresholds=self.verdict_thresholds,
        )


class FusionEngine:
    """
    Top-level FusionEngine handling config-driven dispatch.
    """

    def __init__(self, fusion_cfg: dict) -> None:
        f_cfg = fusion_cfg.get("fusion", {})
        method = f_cfg.get("method", "weighted_sum")
        verdict_cfg = f_cfg.get("verdict", {})

        if method == "logistic":
            lr_path = f_cfg.get("logistic", {}).get("model_path")
            self._fuser = LogisticFusion(model_path=lr_path, verdict_thresholds=verdict_cfg)
        else:
            ws_cfg = f_cfg.get("weighted_sum", {})
            self._fuser = WeightedSumFusion(
                w_spoof=ws_cfg.get("w_spoof", 0.5),
                w_ae_error=ws_cfg.get("w_ae_error", 0.2),
                w_speaker_mismatch=ws_cfg.get("w_speaker_mismatch", 0.3),
                verdict_thresholds=verdict_cfg,
            )

    @classmethod
    def from_config(cls, cfg: dict) -> "FusionEngine":
        return cls(cfg)

    def score(
        self,
        spoof_prob: float,
        ae_error_norm: float,
        speaker_similarity: float,
    ) -> Dict:
        return self._fuser.predict(
            spoof_prob=spoof_prob,
            ae_error_norm=ae_error_norm,
            speaker_similarity=speaker_similarity,
        )
