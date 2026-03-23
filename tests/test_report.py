"""Tests for src/report.py — PDF report generation."""

from __future__ import annotations

import pytest

from src.data_utils import DataSummary, compute_summary
from src.models import RiskAssessment, assess_health_risk
from src.report import generate_report


class TestGenerateReport:
    def test_returns_valid_pdf_bytes(
        self, small_df, sample_summary, sample_assessment
    ) -> None:
        pdf_bytes = generate_report(sample_summary, sample_assessment)
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 500
        # PDF magic bytes
        assert pdf_bytes[:5] == b"%PDF-"

    def test_includes_sleep_analysis(
        self, small_df, sample_summary, sample_assessment
    ) -> None:
        sleep = {
            "score": 75,
            "label": "Good",
            "details": {
                "average_hours": 7.3,
                "consistency_score": 80.0,
                "pct_in_range": 65.0,
                "pct_under_6h": 10.0,
            },
            "recommendations": [],
        }
        pdf_bytes = generate_report(sample_summary, sample_assessment, sleep_analysis=sleep)
        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes[:5] == b"%PDF-"
        # With sleep section the PDF should be larger
        pdf_no_sleep = generate_report(sample_summary, sample_assessment)
        assert len(pdf_bytes) > len(pdf_no_sleep)
