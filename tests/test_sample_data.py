"""Tests for src/sample_data.py — synthetic data generation."""

from __future__ import annotations

import pandas as pd
import pytest

from src.sample_data import generate_sample_data, get_sample_csv_bytes


class TestGenerateSampleData:
    def test_correct_shape_and_columns(self) -> None:
        df = generate_sample_data(n_days=30, seed=0)
        assert len(df) == 30
        assert "date" in df.columns
        assert "heart_rate_bpm" in df.columns
        assert "sleep_hours" in df.columns
        assert df.shape[1] == 12  # date + 11 metrics

    def test_reproducible_with_seed(self) -> None:
        df1 = generate_sample_data(n_days=10, seed=99)
        df2 = generate_sample_data(n_days=10, seed=99)
        pd.testing.assert_frame_equal(df1, df2)

    def test_values_within_realistic_bounds(self) -> None:
        df = generate_sample_data(n_days=200, seed=7)
        assert df["heart_rate_bpm"].between(50, 120).all()
        assert df["spo2_percent"].between(90, 101).all()
        assert df["sleep_hours"].between(2, 12).all()


class TestGetSampleCSVBytes:
    def test_returns_valid_csv_bytes(self) -> None:
        data = get_sample_csv_bytes()
        assert isinstance(data, bytes)
        lines = data.decode().strip().split("\n")
        assert len(lines) == 91  # header + 90 rows
        assert "heart_rate_bpm" in lines[0]
