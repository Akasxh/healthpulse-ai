"""Interactive Plotly visualizations for health data."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .data_utils import KNOWN_COLUMNS, get_metric_columns
from .models import RiskAssessment


COLOR_GOOD = "#10b981"
COLOR_WARNING = "#f59e0b"
COLOR_DANGER = "#ef4444"
COLOR_PRIMARY = "#667eea"
COLOR_SECONDARY = "#764ba2"
COLOR_BG = "rgba(0,0,0,0)"

LAYOUT_DEFAULTS: dict[str, Any] = {
    "template": "plotly_white",
    "paper_bgcolor": COLOR_BG,
    "plot_bgcolor": COLOR_BG,
    "font": {"family": "Inter, -apple-system, sans-serif", "size": 12},
    "margin": {"l": 48, "r": 24, "t": 56, "b": 44},
    "title_font": {"size": 16, "color": "#1e293b"},
}


def metric_trend_chart(df: pd.DataFrame, column: str) -> go.Figure:
    """Create a time-series trend chart for a single metric.

    Args:
        df: Health data DataFrame with 'date' column.
        column: Name of the metric column to plot.

    Returns:
        Plotly Figure with trend line and healthy range band.
    """
    meta = KNOWN_COLUMNS.get(column, {})
    label = meta.get("label", column)
    unit = meta.get("unit", "")
    lo, hi = meta.get("healthy_range", (None, None))

    fig = go.Figure()

    # Healthy range band
    if lo is not None and hi is not None:
        fig.add_hrect(
            y0=lo, y1=hi,
            fillcolor=COLOR_GOOD, opacity=0.1,
            line_width=0,
            annotation_text="Healthy Range",
            annotation_position="top left",
        )

    # 7-day rolling average
    rolling = df[column].rolling(7, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=df["date"], y=rolling,
        mode="lines",
        name="7-day avg",
        line={"color": COLOR_PRIMARY, "width": 3},
    ))

    # Raw values
    fig.add_trace(go.Scatter(
        x=df["date"], y=df[column],
        mode="markers",
        name="Daily",
        marker={"color": COLOR_PRIMARY, "size": 4, "opacity": 0.4},
    ))

    fig.update_layout(
        title=f"{label} Over Time",
        xaxis_title="Date",
        yaxis_title=f"{label} ({unit})" if unit else label,
        hovermode="x unified",
        height=350,
        **LAYOUT_DEFAULTS,
    )

    return fig


def multi_metric_chart(df: pd.DataFrame, columns: list[str]) -> go.Figure:
    """Create a multi-axis chart showing multiple metrics."""
    n = len(columns)
    if n == 0:
        return go.Figure()

    colors = px.colors.qualitative.Set2[:n]

    fig = make_subplots(
        rows=n, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=[KNOWN_COLUMNS.get(c, {}).get("label", c) for c in columns],
    )

    for i, col in enumerate(columns, 1):
        rolling = df[col].rolling(7, min_periods=1).mean()
        fig.add_trace(
            go.Scatter(
                x=df["date"], y=rolling,
                mode="lines",
                name=KNOWN_COLUMNS.get(col, {}).get("label", col),
                line={"color": colors[i - 1], "width": 2},
            ),
            row=i, col=1,
        )

        meta = KNOWN_COLUMNS.get(col, {})
        lo, hi = meta.get("healthy_range", (None, None))
        if lo is not None:
            fig.add_hrect(
                y0=lo, y1=hi,
                fillcolor=colors[i - 1], opacity=0.08,
                line_width=0,
                row=i, col=1,
            )

    fig.update_layout(
        height=200 * n + 50,
        showlegend=False,
        hovermode="x unified",
        **LAYOUT_DEFAULTS,
    )

    return fig


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create a correlation heatmap of all available health metrics."""
    metric_cols = get_metric_columns(df)
    if len(metric_cols) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Need at least 2 metrics for correlation analysis", showarrow=False)
        return fig

    labels = [KNOWN_COLUMNS.get(c, {}).get("label", c) for c in metric_cols]
    corr = df[metric_cols].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=labels,
        y=labels,
        colorscale="RdBu_r",
        zmid=0,
        zmin=-1,
        zmax=1,
        text=corr.values.round(2),
        texttemplate="%{text}",
        textfont={"size": 10},
    ))

    fig.update_layout(
        title="Health Metrics Correlation Matrix",
        height=500,
        width=600,
        xaxis={"tickangle": 45},
        **LAYOUT_DEFAULTS,
    )

    return fig


def risk_gauge(score: float, title: str = "Overall Health Risk") -> go.Figure:
    """Create a gauge chart for risk score."""
    if score < 20:
        color = COLOR_GOOD
    elif score < 45:
        color = COLOR_WARNING
    else:
        color = COLOR_DANGER

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": title, "font": {"size": 16}},
        number={"suffix": "/100", "font": {"size": 28}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.7},
            "steps": [
                {"range": [0, 20], "color": "#e8f8f0"},
                {"range": [20, 45], "color": "#fef9e7"},
                {"range": [45, 70], "color": "#fdedec"},
                {"range": [70, 100], "color": "#f9ebea"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 2},
                "thickness": 0.75,
                "value": score,
            },
        },
    ))

    fig.update_layout(height=250, **LAYOUT_DEFAULTS)
    return fig


def risk_heatmap(assessment: RiskAssessment) -> go.Figure:
    """Create a heatmap showing risk levels per metric."""
    metrics = list(assessment.metric_risks.values())
    if not metrics:
        fig = go.Figure()
        fig.add_annotation(text="No metrics available", showarrow=False)
        return fig

    labels = [m.label for m in metrics]
    scores = [m.risk_score for m in metrics]
    statuses = [m.status for m in metrics]
    trends = [m.trend for m in metrics]

    # Custom text
    text = [f"{s}<br>{t}" for s, t in zip(statuses, trends)]

    fig = go.Figure(data=go.Heatmap(
        z=[scores],
        x=labels,
        y=["Risk Score"],
        colorscale=[
            [0, COLOR_GOOD],
            [0.3, "#f1c40f"],
            [0.6, COLOR_WARNING],
            [1.0, COLOR_DANGER],
        ],
        zmin=0,
        zmax=100,
        text=[text],
        texttemplate="%{text}",
        textfont={"size": 11},
    ))

    fig.update_layout(
        title="Health Risk Heatmap by Metric",
        height=200,
        xaxis={"tickangle": 45},
        yaxis={"showticklabels": False},
        **LAYOUT_DEFAULTS,
    )

    return fig


def feature_importance_chart(importances: dict[str, float]) -> go.Figure:
    """Bar chart of ML feature importances."""
    labels = [KNOWN_COLUMNS.get(k, {}).get("label", k) for k in importances]
    values = list(importances.values())

    # Sort by importance
    sorted_pairs = sorted(zip(labels, values), key=lambda x: x[1], reverse=True)
    labels, values = zip(*sorted_pairs) if sorted_pairs else ([], [])

    fig = go.Figure(go.Bar(
        x=list(values),
        y=list(labels),
        orientation="h",
        marker={"color": COLOR_PRIMARY, "opacity": 0.8},
    ))

    fig.update_layout(
        title="Feature Importance in Risk Model",
        xaxis_title="Importance",
        height=max(250, len(labels) * 35),
        **LAYOUT_DEFAULTS,
    )

    return fig


def distribution_chart(df: pd.DataFrame, column: str) -> go.Figure:
    """Create a histogram with box plot for a metric's distribution."""
    meta = KNOWN_COLUMNS.get(column, {})
    label = meta.get("label", column)
    lo, hi = meta.get("healthy_range", (None, None))

    fig = make_subplots(rows=2, cols=1, row_heights=[0.3, 0.7], shared_xaxes=True, vertical_spacing=0.05)

    # Box plot on top
    fig.add_trace(
        go.Box(x=df[column].dropna(), name="", marker_color=COLOR_PRIMARY, boxmean=True),
        row=1, col=1,
    )

    # Histogram below
    fig.add_trace(
        go.Histogram(x=df[column].dropna(), nbinsx=30, marker_color=COLOR_PRIMARY, opacity=0.7, name=label),
        row=2, col=1,
    )

    # Healthy range lines
    if lo is not None:
        for val, txt in [(lo, "Low"), (hi, "High")]:
            fig.add_vline(x=val, line_dash="dash", line_color=COLOR_GOOD, annotation_text=txt, row=2, col=1)

    fig.update_layout(
        title=f"{label} Distribution",
        showlegend=False,
        height=350,
        **LAYOUT_DEFAULTS,
    )

    return fig


def weekly_comparison_chart(df: pd.DataFrame, column: str) -> go.Figure:
    """Create a weekly average comparison chart."""
    meta = KNOWN_COLUMNS.get(column, {})
    label = meta.get("label", column)

    weekly = df.set_index("date")[column].resample("W").mean().dropna()

    colors = [COLOR_PRIMARY] * len(weekly)
    if len(weekly) >= 2:
        if weekly.iloc[-1] > weekly.iloc[-2]:
            colors[-1] = COLOR_GOOD if column in {"steps", "sleep_hours", "spo2_percent", "active_minutes", "water_intake_glasses"} else COLOR_WARNING
        else:
            colors[-1] = COLOR_GOOD if column in {"heart_rate_bpm", "bp_systolic", "bp_diastolic", "stress_score"} else COLOR_WARNING

    fig = go.Figure(go.Bar(
        x=weekly.index.strftime("%b %d"),
        y=weekly.values,
        marker_color=colors,
    ))

    fig.update_layout(
        title=f"Weekly Average: {label}",
        xaxis_title="Week",
        yaxis_title=label,
        height=300,
        **LAYOUT_DEFAULTS,
    )

    return fig
