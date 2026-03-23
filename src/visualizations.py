"""Interactive Plotly visualizations for health data."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_utils import KNOWN_COLUMNS, get_metric_columns
from src.models import RiskAssessment


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


def radar_comparison_chart(df: pd.DataFrame) -> go.Figure:
    """Create a radar/spider chart comparing user metrics to healthy ranges.

    Normalizes each metric to 0-1 scale where 1.0 = perfectly centered
    in the healthy range, and overlays the healthy reference polygon.

    Args:
        df: Health data DataFrame.

    Returns:
        Plotly Figure with radar chart.
    """
    metric_cols = get_metric_columns(df)
    if len(metric_cols) < 3:
        fig = go.Figure()
        fig.add_annotation(text="Need at least 3 metrics for radar chart", showarrow=False)
        return fig

    labels: list[str] = []
    user_scores: list[float] = []
    healthy_scores: list[float] = []

    for col in metric_cols:
        meta = KNOWN_COLUMNS[col]
        lo, hi = meta["healthy_range"]
        label = meta["label"]
        recent_mean = float(df[col].tail(7).mean())

        # Normalize: 1.0 = center of healthy range, 0.0 = far outside
        center = (lo + hi) / 2
        range_width = (hi - lo) / 2 if hi != lo else 1.0
        distance = abs(recent_mean - center) / range_width
        score = max(0.0, min(1.0, 1.0 - distance * 0.5))

        labels.append(label)
        user_scores.append(round(score, 2))
        healthy_scores.append(1.0)

    # Close the polygon
    labels_closed = labels + [labels[0]]
    user_closed = user_scores + [user_scores[0]]
    healthy_closed = healthy_scores + [healthy_scores[0]]

    fig = go.Figure()

    # Healthy reference polygon
    fig.add_trace(go.Scatterpolar(
        r=healthy_closed,
        theta=labels_closed,
        fill="toself",
        fillcolor="rgba(16,185,129,0.08)",
        line={"color": COLOR_GOOD, "width": 2, "dash": "dash"},
        name="Ideal Range",
    ))

    # User polygon
    fig.add_trace(go.Scatterpolar(
        r=user_closed,
        theta=labels_closed,
        fill="toself",
        fillcolor="rgba(102,126,234,0.15)",
        line={"color": COLOR_PRIMARY, "width": 2.5},
        name="Your Metrics",
        marker={"size": 6, "color": COLOR_PRIMARY},
    ))

    fig.update_layout(
        title="Health Metrics Radar: You vs Ideal",
        polar={
            "radialaxis": {
                "visible": True,
                "range": [0, 1.1],
                "tickvals": [0.25, 0.5, 0.75, 1.0],
                "ticktext": ["25%", "50%", "75%", "100%"],
                "gridcolor": "#e2e8f0",
            },
            "angularaxis": {"gridcolor": "#e2e8f0"},
            "bgcolor": "rgba(0,0,0,0)",
        },
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.15, "xanchor": "center", "x": 0.5},
        height=450,
        **LAYOUT_DEFAULTS,
    )

    return fig


def anomaly_timeline_chart(df: pd.DataFrame, metric_column: str | None = None) -> go.Figure:
    """Create a timeline showing anomalous days highlighted in red.

    Args:
        df: DataFrame with anomaly detection columns (is_anomaly, anomaly_score, anomaly_explanation).
        metric_column: Optional specific metric to plot. If None, shows overall anomaly scores.

    Returns:
        Plotly Figure with anomaly timeline.
    """
    if "is_anomaly" not in df.columns:
        return go.Figure()

    fig = go.Figure()

    # If a specific metric is selected, plot that metric
    if metric_column and metric_column in df.columns:
        meta = KNOWN_COLUMNS.get(metric_column, {})
        label = meta.get("label", metric_column)

        # Normal points
        normal = df[~df["is_anomaly"]]
        fig.add_trace(go.Scatter(
            x=normal["date"], y=normal[metric_column],
            mode="markers+lines",
            name="Normal",
            line={"color": COLOR_PRIMARY, "width": 1.5},
            marker={"size": 4, "opacity": 0.6},
        ))

        # Anomaly points (larger, red)
        anomalies = df[df["is_anomaly"]]
        fig.add_trace(go.Scatter(
            x=anomalies["date"], y=anomalies[metric_column],
            mode="markers",
            name="Anomaly",
            marker={"color": COLOR_DANGER, "size": 12, "symbol": "diamond", "line": {"width": 2, "color": "white"}},
            text=anomalies.get("anomaly_explanation", ""),
            hovertemplate="%{x}<br>%{y}<br>%{text}<extra>Anomaly</extra>",
        ))

        # Healthy range
        lo, hi = meta.get("healthy_range", (None, None))
        if lo is not None:
            fig.add_hrect(y0=lo, y1=hi, fillcolor=COLOR_GOOD, opacity=0.1, line_width=0)

        fig.update_layout(title=f"{label} — Anomaly Detection", yaxis_title=label)
    else:
        # Plot anomaly score timeline
        fig.add_trace(go.Bar(
            x=df["date"],
            y=df["anomaly_score"],
            marker_color=[COLOR_DANGER if a else COLOR_PRIMARY for a in df["is_anomaly"]],
            opacity=0.7,
            text=df["anomaly_explanation"],
            hovertemplate="%{x}<br>Score: %{y:.2f}<br>%{text}<extra></extra>",
        ))
        fig.update_layout(title="Anomaly Score Timeline", yaxis_title="Anomaly Score")

    fig.update_layout(
        height=350,
        hovermode="x unified",
        **LAYOUT_DEFAULTS,
    )
    return fig


def sparkline_figure(values: pd.Series, healthy_range: tuple[float, float]) -> go.Figure:
    """Create a tiny sparkline chart for inline metric display.

    Args:
        values: Last N data points to plot.
        healthy_range: (low, high) bounds for color coding.

    Returns:
        Minimal Plotly Figure suitable for small container.
    """
    s = pd.Series(values) if not isinstance(values, pd.Series) else values
    clean = s.dropna().tail(7)
    if len(clean) == 0:
        return go.Figure()

    last_val = float(clean.iloc[-1])
    lo, hi = healthy_range
    if lo <= last_val <= hi:
        line_color = COLOR_GOOD
    elif abs(last_val - lo) < (hi - lo) * 0.3 or abs(last_val - hi) < (hi - lo) * 0.3:
        line_color = COLOR_WARNING
    else:
        line_color = COLOR_DANGER

    # Convert hex colors to rgba for fill transparency
    _hex_to_rgba = {
        COLOR_GOOD: "rgba(16,185,129,0.1)",
        COLOR_WARNING: "rgba(245,158,11,0.1)",
        COLOR_DANGER: "rgba(239,68,68,0.1)",
    }
    fill_color = _hex_to_rgba.get(line_color, "rgba(102,126,234,0.1)")

    fig = go.Figure(go.Scatter(
        y=clean.values,
        mode="lines",
        line={"color": line_color, "width": 2},
        fill="tozeroy",
        fillcolor=fill_color,
    ))

    fig.update_layout(
        height=60,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"visible": False},
        yaxis={"visible": False},
        showlegend=False,
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
