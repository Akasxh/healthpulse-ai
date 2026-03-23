"""Tests for src/data_utils.py — data loading, BMI, calories, sleep analysis."""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd
import pytest

from src.data_utils import (
    KNOWN_COLUMNS,
    analyze_sleep_quality,
    calculate_bmi,
    compute_summary,
    estimate_daily_calories,
    get_metric_columns,
    load_csv,
)


class TestLoadCSV:
    def test_loads_valid_csv(self, small_df: pd.DataFrame) -> None:
        buf = BytesIO(small_df.to_csv(index=False).encode())
        result = load_csv(buf)
        assert "date" in result.columns
        assert len(result) == 5
        # Must have at least one known metric
        assert len(get_metric_columns(result)) > 0

    def test_raises_on_no_metric_columns(self) -> None:
        buf = BytesIO(b"name,age\nalice,30\nbob,25\n")
        with pytest.raises(ValueError, match="No recognized health metric"):
            load_csv(buf)

    def test_synthesizes_date_when_missing(self) -> None:
        buf = BytesIO(b"heart_rate_bpm\n72\n75\n68\n")
        result = load_csv(buf)
        assert "date" in result.columns
        assert len(result) == 3
        assert result["date"].notna().all()

    def test_normalizes_column_names(self) -> None:
        csv = "Date , Heart Rate Bpm , Steps\n2025-01-01,72,8000\n"
        buf = BytesIO(csv.encode())
        result = load_csv(buf)
        assert "heart_rate_bpm" in result.columns
        assert "steps" in result.columns


class TestGetMetricColumns:
    def test_returns_known_columns_only(self, small_df: pd.DataFrame) -> None:
        cols = get_metric_columns(small_df)
        assert "heart_rate_bpm" in cols
        assert "date" not in cols

    def test_empty_for_no_metrics(self) -> None:
        df = pd.DataFrame({"foo": [1], "bar": [2]})
        assert get_metric_columns(df) == []


class TestComputeSummary:
    def test_summary_values(self, small_df: pd.DataFrame) -> None:
        s = compute_summary(small_df)
        assert s.n_rows == 5
        assert s.n_days == 5
        assert len(s.available_metrics) >= 5
        assert isinstance(s.stats, pd.DataFrame)
        assert not s.stats.empty

    def test_empty_df_returns_zero_summary(self, empty_df: pd.DataFrame) -> None:
        s = compute_summary(empty_df)
        assert s.n_rows == 0
        assert s.n_days == 0
        assert s.available_metrics == []


class TestCalculateBMI:
    @pytest.mark.parametrize(
        "weight,height,expected_bmi,expected_cat",
        [
            (50, 175, 16.3, "Underweight"),
            (70, 175, 22.9, "Normal weight"),
            (85, 175, 27.8, "Overweight"),
            (110, 175, 35.9, "Obese"),
        ],
    )
    def test_bmi_categories(
        self, weight: float, height: float, expected_bmi: float, expected_cat: str
    ) -> None:
        bmi, cat = calculate_bmi(weight, height)
        assert bmi == expected_bmi
        assert cat == expected_cat

    def test_raises_on_non_positive(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            calculate_bmi(70, 0)
        with pytest.raises(ValueError, match="positive"):
            calculate_bmi(-5, 175)


class TestEstimateDailyCalories:
    def test_male_moderate(self) -> None:
        cals = estimate_daily_calories(80, 180, 30, "male", "moderate")
        # Mifflin-St Jeor: 10*80 + 6.25*180 - 5*30 + 5 = 1780; * 1.55 = 2759
        assert cals == 2759

    def test_female_sedentary(self) -> None:
        cals = estimate_daily_calories(60, 165, 25, "female", "sedentary")
        # 10*60 + 6.25*165 - 5*25 - 161 = 1345.25; * 1.2 = 1614.3 -> 1614
        assert cals == 1614

    def test_unknown_activity_defaults_moderate(self) -> None:
        default = estimate_daily_calories(70, 175, 30, "male", "moderate")
        unknown = estimate_daily_calories(70, 175, 30, "male", "UNKNOWN")
        assert default == unknown


class TestAnalyzeSleepQuality:
    def test_good_sleep(self) -> None:
        series = pd.Series([7.5, 8.0, 7.0, 8.5, 7.5, 8.0, 7.0])
        result = analyze_sleep_quality(series)
        assert result["label"] in ("Excellent", "Good")
        assert result["score"] >= 60
        assert result["details"]["average_hours"] == pytest.approx(7.6, abs=0.1)

    def test_poor_sleep(self) -> None:
        series = pd.Series([4.0, 3.5, 5.0, 4.5, 3.0, 5.5, 4.0])
        result = analyze_sleep_quality(series)
        assert result["score"] < 60
        assert len(result["recommendations"]) > 0

    def test_empty_sleep_series(self) -> None:
        series = pd.Series([], dtype=float)
        result = analyze_sleep_quality(series)
        assert result["score"] == 0
        assert result["label"] == "No data"
