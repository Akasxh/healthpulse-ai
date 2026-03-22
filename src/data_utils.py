"""Data loading, validation, and preprocessing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd


# Columns we know how to analyze
KNOWN_COLUMNS: dict[str, dict[str, Any]] = {
    "heart_rate_bpm": {"label": "Heart Rate (bpm)", "healthy_range": (60, 100), "unit": "bpm"},
    "bp_systolic": {"label": "Systolic BP", "healthy_range": (90, 120), "unit": "mmHg"},
    "bp_diastolic": {"label": "Diastolic BP", "healthy_range": (60, 80), "unit": "mmHg"},
    "sleep_hours": {"label": "Sleep Duration", "healthy_range": (7, 9), "unit": "hours"},
    "steps": {"label": "Daily Steps", "healthy_range": (7000, 15000), "unit": "steps"},
    "calories_burned": {"label": "Calories Burned", "healthy_range": (1800, 2800), "unit": "kcal"},
    "active_minutes": {"label": "Active Minutes", "healthy_range": (30, 120), "unit": "min"},
    "weight_kg": {"label": "Weight", "healthy_range": (50, 100), "unit": "kg"},
    "spo2_percent": {"label": "Blood Oxygen (SpO2)", "healthy_range": (95, 100), "unit": "%"},
    "stress_score": {"label": "Stress Score", "healthy_range": (1, 4), "unit": "/10"},
    "water_intake_glasses": {"label": "Water Intake", "healthy_range": (8, 12), "unit": "glasses"},
}


@dataclass
class DataSummary:
    """Summary statistics and metadata for a health dataset."""
    n_rows: int
    n_days: int
    date_range: tuple[str, str]
    available_metrics: list[str]
    missing_pct: dict[str, float]
    stats: pd.DataFrame


def load_csv(uploaded_file: BytesIO) -> pd.DataFrame:
    """Load and validate a health data CSV.

    Args:
        uploaded_file: File-like object from Streamlit uploader.

    Returns:
        Cleaned DataFrame with parsed dates.

    Raises:
        ValueError: If the CSV has no recognizable health columns.
    """
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Try to find and parse a date column
    date_candidates = [c for c in df.columns if "date" in c or "time" in c]
    if date_candidates:
        date_col = date_candidates[0]
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
        if date_col != "date":
            df = df.drop(columns=[date_col])
    else:
        df["date"] = pd.date_range(end=pd.Timestamp.now().normalize(), periods=len(df), freq="D")

    df = df.sort_values("date").reset_index(drop=True)

    # Check we have at least one metric column
    metric_cols = [c for c in df.columns if c in KNOWN_COLUMNS]
    if not metric_cols:
        raise ValueError(
            f"No recognized health metric columns found. "
            f"Expected some of: {list(KNOWN_COLUMNS.keys())}"
        )

    return df


def get_metric_columns(df: pd.DataFrame) -> list[str]:
    """Return list of recognized health metric columns present in the DataFrame."""
    return [c for c in df.columns if c in KNOWN_COLUMNS]


def compute_summary(df: pd.DataFrame) -> DataSummary:
    """Compute summary statistics for health data."""
    metric_cols = get_metric_columns(df)
    numeric_df = df[metric_cols]

    missing_pct = {col: float(df[col].isna().mean() * 100) for col in metric_cols}

    stats = numeric_df.describe().T
    stats["healthy_low"] = [KNOWN_COLUMNS[c]["healthy_range"][0] for c in metric_cols]
    stats["healthy_high"] = [KNOWN_COLUMNS[c]["healthy_range"][1] for c in metric_cols]

    date_min = df["date"].min()
    date_max = df["date"].max()

    return DataSummary(
        n_rows=len(df),
        n_days=(date_max - date_min).days + 1,
        date_range=(str(date_min.date()), str(date_max.date())),
        available_metrics=metric_cols,
        missing_pct=missing_pct,
        stats=stats,
    )


def calculate_bmi(weight_kg: float, height_cm: float) -> tuple[float, str]:
    """Calculate BMI and return (value, category).

    Args:
        weight_kg: Weight in kilograms.
        height_cm: Height in centimeters.

    Returns:
        Tuple of (BMI value, category string).
    """
    if height_cm <= 0 or weight_kg <= 0:
        raise ValueError("Weight and height must be positive.")

    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)

    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal weight"
    elif bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"

    return round(bmi, 1), category


def estimate_daily_calories(
    weight_kg: float,
    height_cm: float,
    age: int,
    sex: str,
    activity_level: str,
) -> int:
    """Estimate daily calorie needs using Mifflin-St Jeor equation.

    Args:
        weight_kg: Weight in kilograms.
        height_cm: Height in centimeters.
        age: Age in years.
        sex: 'male' or 'female'.
        activity_level: One of 'sedentary', 'light', 'moderate', 'active', 'very_active'.

    Returns:
        Estimated daily calorie needs.
    """
    if sex.lower() == "male":
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161

    multipliers = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "very_active": 1.9,
    }
    factor = multipliers.get(activity_level, 1.55)
    return round(bmr * factor)


def analyze_sleep_quality(sleep_hours: pd.Series) -> dict[str, Any]:
    """Analyze sleep patterns and return quality assessment.

    Args:
        sleep_hours: Series of nightly sleep durations.

    Returns:
        Dict with quality metrics and recommendations.
    """
    clean = sleep_hours.dropna()
    if len(clean) == 0:
        return {"score": 0, "label": "No data", "details": {}, "recommendations": []}

    avg = float(clean.mean())
    std = float(clean.std()) if len(clean) > 1 else 0.0
    pct_good = float((clean.between(7, 9)).mean() * 100)
    pct_short = float((clean < 6).mean() * 100)

    # Consistency score (lower std = more consistent = better)
    consistency = max(0, 100 - std * 30)

    # Duration score
    if 7 <= avg <= 9:
        duration_score = 100
    elif 6 <= avg < 7 or 9 < avg <= 10:
        duration_score = 70
    else:
        duration_score = 40

    overall_score = round(0.6 * duration_score + 0.4 * consistency)

    if overall_score >= 80:
        label = "Excellent"
    elif overall_score >= 60:
        label = "Good"
    elif overall_score >= 40:
        label = "Fair"
    else:
        label = "Poor"

    recommendations: list[str] = []
    if avg < 7:
        recommendations.append("Aim for 7-9 hours of sleep per night.")
    if avg > 9:
        recommendations.append("Consistently sleeping >9 hours may indicate underlying issues. Consider consulting a doctor.")
    if std > 1.5:
        recommendations.append("Your sleep schedule is inconsistent. Try going to bed and waking up at the same time daily.")
    if pct_short > 30:
        recommendations.append(f"{pct_short:.0f}% of nights are under 6 hours. Prioritize sleep hygiene.")

    return {
        "score": overall_score,
        "label": label,
        "details": {
            "average_hours": round(avg, 1),
            "std_hours": round(std, 1),
            "pct_in_range": round(pct_good, 1),
            "pct_under_6h": round(pct_short, 1),
            "consistency_score": round(consistency, 1),
            "duration_score": duration_score,
        },
        "recommendations": recommendations,
    }
