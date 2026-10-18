"""
test_fusion.py — Unit tests for the Fusion layer.

Run with: pytest tests/test_fusion.py -v
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.fusion import WeightedSumFusion, FusionEngine


MINIMAL_CFG = {
    "fusion": {
        "method": "weighted_sum",
        "weighted_sum": {
            "w_spoof": 0.5,
            "w_ae_error": 0.2,
            "w_speaker_mismatch": 0.3,
        },
        "verdict": {
            "low_risk_below": 30,
            "medium_risk_below": 60,
            "high_risk_above": 60,
        },
    }
}


class TestWeightedSumFusion:
    @pytest.fixture
    def fuser(self):
        return WeightedSumFusion(w_spoof=0.5, w_ae_error=0.2, w_speaker_mismatch=0.3)

    def test_score_range_all_zeros(self, fuser):
        result = fuser.predict(spoof_prob=0.0, ae_error_norm=-99.0, speaker_similarity=1.0)
        assert 0.0 <= result["risk_score"] <= 100.0

    def test_score_range_all_ones(self, fuser):
        result = fuser.predict(spoof_prob=1.0, ae_error_norm=99.0, speaker_similarity=-1.0)
        assert 0.0 <= result["risk_score"] <= 100.0

    def test_high_spoof_gives_high_score(self, fuser):
        result = fuser.predict(spoof_prob=1.0, ae_error_norm=3.0, speaker_similarity=-0.8)
        assert result["risk_score"] > 50.0

    def test_low_risk_all_signals_clean(self, fuser):
        result = fuser.predict(spoof_prob=0.05, ae_error_norm=-2.0, speaker_similarity=0.95)
        assert result["risk_score"] < 30.0

    def test_verdict_string_present(self, fuser):
        result = fuser.predict(spoof_prob=0.5, ae_error_norm=0.0, speaker_similarity=0.5)
        assert "verdict" in result
        assert result["verdict"] in ("LOW RISK", "MEDIUM RISK", "HIGH RISK — flag for human review")

    def test_disclaimer_present(self, fuser):
        result = fuser.predict(spoof_prob=0.5, ae_error_norm=0.0, speaker_similarity=0.5)
        assert "disclaimer" in result
        assert len(result["disclaimer"]) > 10

    def test_explanation_dict_present(self, fuser):
        result = fuser.predict(spoof_prob=0.5, ae_error_norm=0.0, speaker_similarity=0.5)
        assert "explanation" in result
        assert isinstance(result["explanation"], dict)


class TestFusionEngine:
    @pytest.fixture
    def engine(self):
        return FusionEngine.from_config(MINIMAL_CFG)

    def test_edge_case_extreme_inputs(self, engine):
        for spoof_prob in [0.0, 0.5, 1.0]:
            for ae_error in [-10.0, 0.0, 10.0]:
                for sim in [-1.0, 0.0, 1.0]:
                    result = engine.score(spoof_prob, ae_error, sim)
                    assert 0.0 <= result["risk_score"] <= 100.0

    def test_monotone_in_spoof_prob(self, engine):
        low = engine.score(0.1, 0.0, 0.5)["risk_score"]
        high = engine.score(0.9, 0.0, 0.5)["risk_score"]
        assert high >= low
