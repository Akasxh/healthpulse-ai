"""Shared fixtures for the health agent test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_utils import DataSummary, compute_summary
from src.models import MetricRisk, RiskAssessment


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """90-day sample DataFrame with all known metric columns."""
    from src.sample_data import generate_sample_data

    return generate_sample_data(n_days=90, seed=42)


@pytest.fixture
def small_df() -> pd.DataFrame:
    """Minimal 5-row DataFrame for fast tests."""
    dates = pd.date_range("2025-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "heart_rate_bpm": [72, 75, 68, 80, 74],
            "bp_systolic": [118, 122, 115, 130, 120],
            "bp_diastolic": [78, 80, 72, 85, 76],
            "sleep_hours": [7.5, 6.0, 8.0, 5.5, 7.0],
            "steps": [8000, 6500, 10000, 4000, 9000],
            "stress_score": [3.0, 5.0, 2.0, 7.0, 4.0],
            "spo2_percent": [98.0, 97.5, 99.0, 96.0, 97.0],
            "water_intake_glasses": [8, 6, 10, 4, 7],
            "active_minutes": [45, 20, 60, 10, 35],
            "weight_kg": [75.0, 75.2, 74.8, 75.5, 74.9],
            "calories_burned": [2200, 1900, 2500, 1700, 2100],
        }
    )


@pytest.fixture
def empty_df() -> pd.DataFrame:
    """Empty DataFrame with correct columns."""
    return pd.DataFrame(
        columns=[
            "date",
            "heart_rate_bpm",
            "bp_systolic",
            "sleep_hours",
            "steps",
        ]
    )


@pytest.fixture
def sample_summary(small_df: pd.DataFrame) -> DataSummary:
    """DataSummary built from small_df."""
    return compute_summary(small_df)


@pytest.fixture
def sample_assessment(small_df: pd.DataFrame) -> RiskAssessment:
    """RiskAssessment built from small_df."""
    from src.models import assess_health_risk

    return assess_health_risk(small_df)


@pytest.fixture
def sample_metric_risk() -> MetricRisk:
    """A single MetricRisk instance for testing."""
    return MetricRisk(
        name="heart_rate_bpm",
        label="Heart Rate (bpm)",
        current_value=85.0,
        healthy_range=(60, 100),
        risk_score=10.0,
        status="Normal",
        trend="Stable",
    )
