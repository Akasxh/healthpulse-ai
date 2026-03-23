"""Tests for src/models.py — risk scoring, ensemble model, recommendations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import (
    MetricRisk,
    RiskAssessment,
    _compute_metric_risk,
    _generate_recommendations,
    assess_health_risk,
    build_ensemble_model,
)


class TestComputeMetricRisk:
    def test_in_range_gives_zero_risk(self) -> None:
        values = pd.Series([72, 75, 68, 80, 74])
        mr = _compute_metric_risk(values, "heart_rate_bpm")
        assert mr.risk_score == 0.0
        assert mr.status == "Normal"

    def test_high_values_give_elevated_risk(self) -> None:
        values = pd.Series([110, 115, 120, 112, 108])
        mr = _compute_metric_risk(values, "heart_rate_bpm")
        assert mr.risk_score > 0
        assert mr.status in ("Warning", "High Risk")

    def test_empty_series_returns_no_data(self) -> None:
        values = pd.Series([], dtype=float)
        mr = _compute_metric_risk(values, "heart_rate_bpm")
        assert mr.status == "No data"
        assert mr.risk_score == 50

    def test_trend_with_enough_data(self) -> None:
        # 14 days in range -> trend should not be "Insufficient data"
        values = pd.Series([72] * 7 + [70] * 7)
        mr = _compute_metric_risk(values, "heart_rate_bpm")
        assert mr.trend in ("Improving", "Stable", "Worsening")


class TestBuildEnsembleModel:
    def test_model_returns_correct_tuple(self) -> None:
        model, scaler, features = build_ensemble_model()
        assert hasattr(model, "predict_proba")
        assert hasattr(scaler, "transform")
        assert len(features) == 10
        assert "heart_rate_bpm" in features

    def test_model_predicts_three_classes(self) -> None:
        model, scaler, features = build_ensemble_model()
        sample = np.array([[72, 120, 75, 7.5, 8000, 45, 4, 98, 75, 8]])
        scaled = scaler.transform(sample)
        proba = model.predict_proba(scaled)
        assert proba.shape == (1, 3)
        assert abs(proba.sum() - 1.0) < 1e-6


class TestAssessHealthRisk:
    def test_returns_risk_assessment(self, small_df: pd.DataFrame) -> None:
        result = assess_health_risk(small_df)
        assert isinstance(result, RiskAssessment)
        assert 0 <= result.overall_risk_score <= 100
        assert result.risk_level in ("Low", "Moderate", "High", "Critical")
        assert len(result.metric_risks) > 0
        assert len(result.recommendations) > 0
        assert len(result.feature_importances) == 10

    def test_empty_df_returns_low_risk(self) -> None:
        result = assess_health_risk(pd.DataFrame())
        assert result.overall_risk_score == 0.0
        assert result.risk_level == "Low"
        assert result.recommendations == ["No data available for analysis."]


class TestGenerateRecommendations:
    def test_all_normal_gets_positive_message(self) -> None:
        risks = {
            "heart_rate_bpm": MetricRisk(
                name="heart_rate_bpm", label="HR", current_value=72,
                healthy_range=(60, 100), risk_score=0, status="Normal", trend="Stable",
            ),
        }
        recs = _generate_recommendations(risks, overall_risk=10)
        assert len(recs) == 1
        assert "within normal" in recs[0].lower()

    def test_high_risk_inserts_checkup_warning(self) -> None:
        risks = {
            "heart_rate_bpm": MetricRisk(
                name="heart_rate_bpm", label="HR", current_value=110,
                healthy_range=(60, 100), risk_score=60, status="High Risk", trend="Worsening",
            ),
        }
        recs = _generate_recommendations(risks, overall_risk=65)
        assert any("check-up" in r.lower() or "elevated" in r.lower() for r in recs)
