"""ML models for health risk scoring and analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .data_utils import KNOWN_COLUMNS, get_metric_columns


@dataclass
class RiskAssessment:
    """Result of health risk analysis."""
    overall_risk_score: float  # 0-100
    risk_level: str  # Low / Moderate / High / Critical
    metric_risks: dict[str, MetricRisk]
    recommendations: list[str]
    feature_importances: dict[str, float]


@dataclass
class MetricRisk:
    """Risk assessment for a single metric."""
    name: str
    label: str
    current_value: float
    healthy_range: tuple[float, float]
    risk_score: float  # 0-100
    status: str  # Normal / Warning / High Risk
    trend: str  # Improving / Stable / Worsening


def _compute_metric_risk(values: pd.Series, col: str) -> MetricRisk:
    """Compute risk for a single health metric.

    Args:
        values: Series of metric values (time-ordered).
        col: Column name matching KNOWN_COLUMNS key.

    Returns:
        MetricRisk assessment.
    """
    meta = KNOWN_COLUMNS[col]
    lo, hi = meta["healthy_range"]
    clean = values.dropna()

    if len(clean) == 0:
        return MetricRisk(
            name=col, label=meta["label"], current_value=0,
            healthy_range=(lo, hi), risk_score=50,
            status="No data", trend="Unknown",
        )

    current = float(clean.iloc[-1])
    recent_mean = float(clean.tail(7).mean())

    # Risk score: how far outside healthy range
    if lo <= recent_mean <= hi:
        risk = 0.0
    else:
        range_width = hi - lo
        if range_width == 0:
            range_width = 1.0
        if recent_mean < lo:
            deviation = (lo - recent_mean) / range_width
        else:
            deviation = (recent_mean - hi) / range_width
        risk = min(100.0, deviation * 80)

    # For stress_score, being in range means low stress = good
    # Invert semantics aren't needed since range already captures this

    if risk < 20:
        status = "Normal"
    elif risk < 50:
        status = "Warning"
    else:
        status = "High Risk"

    # Trend: compare last 7 days vs prior 7 days
    if len(clean) >= 14:
        recent = clean.tail(7).mean()
        prior = clean.iloc[-14:-7].mean()
        diff = recent - prior

        # For metrics where lower is better (HR, BP, stress, weight)
        lower_is_better = col in {"heart_rate_bpm", "bp_systolic", "bp_diastolic", "stress_score", "weight_kg"}
        # For metrics where higher is better (steps, sleep, spo2, water, active_minutes)
        threshold = 0.02 * (hi - lo) if (hi - lo) > 0 else 0.5

        if abs(diff) < threshold:
            trend = "Stable"
        elif lower_is_better:
            trend = "Improving" if diff < 0 else "Worsening"
        else:
            trend = "Improving" if diff > 0 else "Worsening"
    else:
        trend = "Insufficient data"

    return MetricRisk(
        name=col,
        label=meta["label"],
        current_value=round(current, 1),
        healthy_range=(lo, hi),
        risk_score=round(risk, 1),
        status=status,
        trend=trend,
    )


def _generate_synthetic_training_data(n_samples: int = 2000, seed: int = 123) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate synthetic training data for the ensemble model.

    Creates realistic health data with known risk labels for training.

    Returns:
        Tuple of (features, labels, feature_names).
    """
    rng = np.random.default_rng(seed)

    feature_specs: list[tuple[str, float, float]] = [
        ("heart_rate_bpm", 72, 12),
        ("bp_systolic", 122, 15),
        ("bp_diastolic", 78, 10),
        ("sleep_hours", 7.0, 1.2),
        ("steps", 8000, 3000),
        ("active_minutes", 45, 25),
        ("stress_score", 5, 2),
        ("spo2_percent", 97, 1.5),
        ("weight_kg", 75, 15),
        ("water_intake_glasses", 7, 2.5),
    ]

    feature_names = [f[0] for f in feature_specs]
    X = np.column_stack([
        rng.normal(mu, sigma, n_samples) for _, mu, sigma in feature_specs
    ])

    # Generate labels based on health rules
    labels = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        risk_count = 0
        hr, sys, dia, sleep, steps, active, stress, spo2, weight, water = X[i]
        if hr > 100 or hr < 55:
            risk_count += 1
        if sys > 140 or dia > 90:
            risk_count += 1.5
        if sleep < 5.5 or sleep > 10:
            risk_count += 1
        if steps < 4000:
            risk_count += 0.5
        if stress > 7:
            risk_count += 1
        if spo2 < 94:
            risk_count += 2
        if active < 15:
            risk_count += 0.5
        if water < 4:
            risk_count += 0.3

        # Add noise
        risk_count += rng.normal(0, 0.3)

        if risk_count >= 3:
            labels[i] = 2  # High risk
        elif risk_count >= 1.5:
            labels[i] = 1  # Moderate
        else:
            labels[i] = 0  # Low

    return X, labels, feature_names


def build_ensemble_model() -> tuple[VotingClassifier, StandardScaler, list[str]]:
    """Build and train the ensemble health risk model.

    Returns:
        Tuple of (trained model, fitted scaler, feature_names).
    """
    X, y, feature_names = _generate_synthetic_training_data()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    lr = LogisticRegression(max_iter=1000, random_state=42)

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("lr", lr)],
        voting="soft",
        weights=[0.6, 0.4],
    )
    ensemble.fit(X_scaled, y)

    return ensemble, scaler, feature_names


def assess_health_risk(df: pd.DataFrame) -> RiskAssessment:
    """Run full health risk assessment on the dataset.

    Args:
        df: Health data DataFrame.

    Returns:
        Comprehensive RiskAssessment.
    """
    if df.empty:
        return RiskAssessment(
            overall_risk_score=0.0, risk_level="Low",
            metric_risks={}, recommendations=["No data available for analysis."],
            feature_importances={},
        )

    metric_cols = get_metric_columns(df)

    # Per-metric risk
    metric_risks: dict[str, MetricRisk] = {}
    for col in metric_cols:
        metric_risks[col] = _compute_metric_risk(df[col], col)

    # Ensemble ML risk scoring
    model, scaler, model_features = build_ensemble_model()

    # Prepare input features from the most recent data
    recent = df.tail(7)
    feature_vector = []
    available_model_features = []
    for feat in model_features:
        if feat in df.columns and recent[feat].notna().any():
            feature_vector.append(float(recent[feat].mean()))
            available_model_features.append(feat)
        else:
            # Use population mean as fallback
            idx = model_features.index(feat)
            specs = [
                ("heart_rate_bpm", 72), ("bp_systolic", 122), ("bp_diastolic", 78),
                ("sleep_hours", 7.0), ("steps", 8000), ("active_minutes", 45),
                ("stress_score", 5), ("spo2_percent", 97), ("weight_kg", 75),
                ("water_intake_glasses", 7),
            ]
            feature_vector.append(specs[idx][1])
            available_model_features.append(feat)

    X_input = np.array(feature_vector).reshape(1, -1)
    X_scaled = scaler.transform(X_input)

    probas = model.predict_proba(X_scaled)[0]
    # Weighted risk score: P(moderate)*40 + P(high)*100
    ml_risk = float(probas[1] * 40 + probas[2] * 100) if len(probas) == 3 else 50.0

    # Combine metric-based risk with ML risk
    metric_risk_scores = [mr.risk_score for mr in metric_risks.values()]
    avg_metric_risk = float(np.mean(metric_risk_scores)) if metric_risk_scores else 50.0

    overall = 0.4 * avg_metric_risk + 0.6 * ml_risk
    overall = round(min(100.0, max(0.0, overall)), 1)

    if overall < 20:
        risk_level = "Low"
    elif overall < 45:
        risk_level = "Moderate"
    elif overall < 70:
        risk_level = "High"
    else:
        risk_level = "Critical"

    # Feature importances from RandomForest
    rf_model = model.named_estimators_["rf"]
    importances = dict(zip(model_features, rf_model.feature_importances_))

    # Generate recommendations
    recommendations = _generate_recommendations(metric_risks, overall)

    return RiskAssessment(
        overall_risk_score=overall,
        risk_level=risk_level,
        metric_risks=metric_risks,
        recommendations=recommendations,
        feature_importances=importances,
    )


def _generate_recommendations(
    metric_risks: dict[str, MetricRisk],
    overall_risk: float,
) -> list[str]:
    """Generate personalized health recommendations based on risk assessment."""
    recs: list[str] = []

    for col, mr in metric_risks.items():
        if mr.status == "No data":
            continue

        lo, hi = mr.healthy_range
        val = mr.current_value

        if col == "heart_rate_bpm" and val > hi:
            recs.append(
                f"Your resting heart rate ({val} bpm) is elevated. "
                "Regular cardiovascular exercise, stress management, and adequate hydration can help lower it."
            )
        elif col == "heart_rate_bpm" and val < lo:
            recs.append(
                f"Your resting heart rate ({val} bpm) is below normal. "
                "While this can indicate fitness, consult a doctor if you experience dizziness or fatigue."
            )

        if col == "bp_systolic" and val > 130:
            recs.append(
                f"Your systolic blood pressure ({val} mmHg) is elevated. "
                "Reduce sodium intake, maintain a healthy weight, exercise regularly, and limit alcohol."
            )

        if col == "bp_diastolic" and val > 85:
            recs.append(
                f"Your diastolic blood pressure ({val} mmHg) is above optimal. "
                "Consider the DASH diet, regular physical activity, and stress reduction techniques."
            )

        if col == "sleep_hours" and val < 6.5:
            recs.append(
                f"You're averaging {val} hours of sleep. "
                "Establish a consistent sleep schedule, limit screen time before bed, and create a dark, cool sleeping environment."
            )

        if col == "steps" and val < lo:
            recs.append(
                f"Your daily step count ({int(val)}) is below recommended levels. "
                "Try walking meetings, parking farther away, or taking short walks after meals."
            )

        if col == "stress_score" and val > 6:
            recs.append(
                f"Your stress score ({val}/10) is elevated. "
                "Practice mindfulness meditation, deep breathing exercises, or consider journaling to manage stress."
            )

        if col == "spo2_percent" and val < 95:
            recs.append(
                f"Your blood oxygen ({val}%) is below normal. "
                "This could indicate respiratory issues. Please consult a healthcare provider promptly."
            )

        if col == "water_intake_glasses" and val < 6:
            recs.append(
                f"Your water intake ({int(val)} glasses/day) is low. "
                "Aim for 8+ glasses daily. Keep a water bottle nearby and set hydration reminders."
            )

        if col == "active_minutes" and val < 20:
            recs.append(
                f"Your active minutes ({int(val)} min/day) are low. "
                "The WHO recommends 150+ minutes of moderate activity per week. Start with 10-minute walks."
            )

    if not recs:
        recs.append(
            "Your health metrics are within normal ranges. "
            "Keep up the great work! Continue maintaining a balanced diet, regular exercise, and good sleep habits."
        )

    if overall_risk > 60:
        recs.insert(0,
            "Your overall health risk score is elevated. "
            "Consider scheduling a check-up with your healthcare provider to discuss these findings."
        )

    return recs
