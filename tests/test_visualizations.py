"""Tests for src/visualizations.py — chart generation."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import pytest

from src.models import RiskAssessment
from src.visualizations import (
    correlation_heatmap,
    distribution_chart,
    feature_importance_chart,
    metric_trend_chart,
    multi_metric_chart,
    risk_gauge,
    risk_heatmap,
    weekly_comparison_chart,
)


class TestMetricTrendChart:
    def test_returns_figure(self, small_df: pd.DataFrame) -> None:
        fig = metric_trend_chart(small_df, "heart_rate_bpm")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # rolling avg + raw markers


class TestMultiMetricChart:
    def test_returns_figure_with_subplots(self, small_df: pd.DataFrame) -> None:
        fig = multi_metric_chart(small_df, ["heart_rate_bpm", "steps"])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2

    def test_empty_columns_returns_empty_figure(self, small_df: pd.DataFrame) -> None:
        fig = multi_metric_chart(small_df, [])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0


class TestCorrelationHeatmap:
    def test_returns_figure(self, small_df: pd.DataFrame) -> None:
        fig = correlation_heatmap(small_df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1  # single heatmap trace


class TestRiskGauge:
    @pytest.mark.parametrize("score", [0, 15, 30, 60, 95])
    def test_returns_figure_for_various_scores(self, score: float) -> None:
        fig = risk_gauge(score)
        assert isinstance(fig, go.Figure)


class TestRiskHeatmap:
    def test_returns_figure(self, sample_assessment: RiskAssessment) -> None:
        fig = risk_heatmap(sample_assessment)
        assert isinstance(fig, go.Figure)


class TestFeatureImportanceChart:
    def test_returns_figure(self, sample_assessment: RiskAssessment) -> None:
        fig = feature_importance_chart(sample_assessment.feature_importances)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1  # single bar trace


class TestDistributionChart:
    def test_returns_figure(self, small_df: pd.DataFrame) -> None:
        fig = distribution_chart(small_df, "steps")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # box + histogram


class TestWeeklyComparisonChart:
    def test_returns_figure(self, sample_df: pd.DataFrame) -> None:
        fig = weekly_comparison_chart(sample_df, "heart_rate_bpm")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
