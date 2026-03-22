"""Synthetic health data generator for demo mode."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_sample_data(n_days: int = 90, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic health data for demonstration.

    Args:
        n_days: Number of days of data to generate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with daily health metrics.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.now().normalize(), periods=n_days, freq="D")

    # Base patterns with weekly seasonality
    day_of_week = np.array([d.dayofweek for d in dates])
    is_weekend = (day_of_week >= 5).astype(float)

    # Gradual trend (simulates improving health over time)
    trend = np.linspace(0, 1, n_days)

    # Heart rate: 60-100 bpm, lower on rest days, improving trend
    hr_base = 75 - trend * 5
    hr_noise = rng.normal(0, 4, n_days)
    hr_weekend = -3 * is_weekend
    heart_rate = np.clip(hr_base + hr_noise + hr_weekend, 55, 105).round(0).astype(int)

    # Blood pressure systolic: 110-140
    bp_sys_base = 125 - trend * 8
    bp_sys_noise = rng.normal(0, 6, n_days)
    bp_systolic = np.clip(bp_sys_base + bp_sys_noise, 100, 155).round(0).astype(int)

    # Blood pressure diastolic: 70-90
    bp_dia_base = 82 - trend * 5
    bp_dia_noise = rng.normal(0, 4, n_days)
    bp_diastolic = np.clip(bp_dia_base + bp_dia_noise, 60, 100).round(0).astype(int)

    # Sleep hours: 5-9, more on weekends
    sleep_base = 6.8 + trend * 0.5
    sleep_noise = rng.normal(0, 0.6, n_days)
    sleep_weekend = 0.8 * is_weekend
    sleep_hours = np.clip(sleep_base + sleep_noise + sleep_weekend, 3.5, 10.5).round(1)

    # Steps: 3000-15000, fewer on weekends for some variety
    steps_base = 7500 + trend * 2000
    steps_noise = rng.normal(0, 1500, n_days)
    steps_weekend = -1500 * is_weekend + rng.normal(0, 500, n_days)
    steps = np.clip(steps_base + steps_noise + steps_weekend, 1000, 20000).round(0).astype(int)

    # Calories burned: correlated with steps
    calories = (steps * rng.uniform(0.035, 0.055, n_days) + rng.normal(1800, 150, n_days)).round(0).astype(int)

    # Active minutes: correlated with steps
    active_minutes = np.clip(
        (steps / 150 + rng.normal(0, 8, n_days)).round(0), 0, 180
    ).astype(int)

    # Weight (kg): slight downward trend
    weight_base = 78 - trend * 2.5
    weight_noise = rng.normal(0, 0.3, n_days)
    weight = (weight_base + weight_noise).round(1)

    # SpO2: 94-100%
    spo2 = np.clip(rng.normal(97.5, 0.8, n_days), 93, 100).round(1)

    # Stress score: 1-10, higher midweek
    stress_base = 5 - trend * 1.5
    stress_midweek = np.where((day_of_week >= 1) & (day_of_week <= 3), 1.2, 0)
    stress_noise = rng.normal(0, 1.2, n_days)
    stress_score = np.clip(stress_base + stress_midweek + stress_noise, 1, 10).round(1)

    # Water intake (glasses): 4-12
    water_intake = np.clip(rng.normal(7 + trend * 1, 1.5, n_days), 2, 14).round(0).astype(int)

    return pd.DataFrame({
        "date": dates,
        "heart_rate_bpm": heart_rate,
        "bp_systolic": bp_systolic,
        "bp_diastolic": bp_diastolic,
        "sleep_hours": sleep_hours,
        "steps": steps,
        "calories_burned": calories,
        "active_minutes": active_minutes,
        "weight_kg": weight,
        "spo2_percent": spo2,
        "stress_score": stress_score,
        "water_intake_glasses": water_intake,
    })


def get_sample_csv_bytes() -> bytes:
    """Return sample data as CSV bytes for download."""
    df = generate_sample_data()
    return df.to_csv(index=False).encode("utf-8")
