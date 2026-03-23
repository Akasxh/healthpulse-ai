"""AI Health Insight Agent - Main Streamlit Application."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so `from src.xxx` imports work
# when Streamlit runs this file directly as __main__.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import streamlit as st
import pandas as pd

from src.data_utils import (
    analyze_sleep_quality,
    calculate_bmi,
    compute_summary,
    estimate_daily_calories,
    get_metric_columns,
    load_csv,
    KNOWN_COLUMNS,
)
from src.models import assess_health_risk, build_ensemble_model, detect_anomalies
from src.report import generate_report
from src.sample_data import generate_sample_data, get_sample_csv_bytes
from src.visualizations import (
    correlation_heatmap,
    distribution_chart,
    feature_importance_chart,
    metric_trend_chart,
    multi_metric_chart,
    radar_comparison_chart,
    risk_gauge,
    risk_heatmap,
    sparkline_figure,
    anomaly_timeline_chart,
    weekly_comparison_chart,
)


def configure_page() -> None:
    """Set Streamlit page configuration."""
    st.set_page_config(
        page_title="AI Health Insight Agent",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown('''<style>
    /* Hide Streamlit defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ===== Animations ===== */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 2px 8px rgba(102,126,234,0.1); }
        50% { box-shadow: 0 4px 20px rgba(102,126,234,0.25); }
    }
    @keyframes skeletonPulse {
        0% { background-position: -200px 0; }
        100% { background-position: calc(200px + 100%) 0; }
    }
    @keyframes countUp {
        from { opacity: 0; transform: scale(0.5); }
        to { opacity: 1; transform: scale(1); }
    }
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }

    /* Smooth page transitions */
    .main .block-container {
        animation: fadeInUp 0.5s ease-out;
    }

    /* Modern card styling with hover */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #667eea11, #764ba211);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(102,126,234,0.18);
        border-color: #667eea44;
    }

    /* Tab styling with active indicator */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f8fafc;
        padding: 4px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: #667eea15;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b, #0f172a);
    }
    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        border: none;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102,126,234,0.4);
    }

    /* Animated gradient hero text */
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2, #667eea, #10b981);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 6s ease infinite;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #7f8c8d;
        margin-top: -10px;
        margin-bottom: 20px;
        line-height: 1.6;
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #f1f5f9; border-radius: 4px; }
    ::-webkit-scrollbar-thumb { background: linear-gradient(135deg, #667eea, #764ba2); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #5a6fd6; }

    /* Table styling */
    .stDataFrame table { border-collapse: separate; border-spacing: 0; }
    .stDataFrame thead th {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        font-weight: 600;
        padding: 10px 12px;
    }
    .stDataFrame tbody tr:nth-child(even) { background: #f8fafc; }
    .stDataFrame tbody tr:hover { background: #667eea10 !important; transition: background 0.2s; }
    .stDataFrame td { padding: 8px 12px; }

    /* Toast notification styling */
    .stAlert {
        border-radius: 10px !important;
        border: none !important;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        animation: slideInLeft 0.4s ease-out;
    }

    /* Health score card */
    .health-score-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 16px;
        padding: 28px 32px;
        color: white;
        text-align: center;
        animation: fadeInUp 0.6s ease-out, pulseGlow 3s ease-in-out infinite;
        position: relative;
        overflow: hidden;
    }
    .health-score-card::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 60%);
        animation: gradientShift 8s linear infinite;
    }
    .health-score-number {
        font-size: 3.5rem;
        font-weight: 900;
        line-height: 1;
        animation: countUp 0.8s ease-out;
    }
    .health-score-label {
        font-size: 0.95rem;
        opacity: 0.9;
        margin-top: 4px;
    }

    /* Risk badge */
    .risk-badge {
        display: inline-block;
        padding: 4px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
        margin-top: 8px;
    }
    .risk-badge-low { background: rgba(16,185,129,0.2); color: #059669; }
    .risk-badge-moderate { background: rgba(245,158,11,0.2); color: #d97706; }
    .risk-badge-high { background: rgba(239,68,68,0.2); color: #dc2626; }
    .risk-badge-critical { background: rgba(192,57,43,0.3); color: #c0392b; }

    /* Data quality widget */
    .data-quality-item {
        display: flex;
        justify-content: space-between;
        padding: 6px 0;
        border-bottom: 1px solid #334155;
        font-size: 0.85rem;
    }
    .data-quality-item:last-child { border-bottom: none; }

    /* Recommendation cards */
    .rec-card {
        border-radius: 10px;
        padding: 16px 20px;
        margin: 10px 0;
        animation: fadeInUp 0.4s ease-out;
        transition: transform 0.2s;
    }
    .rec-card:hover { transform: translateX(4px); }
    .rec-cardiovascular { border-left: 4px solid #ef4444; background: #fef2f2; }
    .rec-sleep { border-left: 4px solid #8b5cf6; background: #f5f3ff; }
    .rec-activity { border-left: 4px solid #10b981; background: #f0fdf4; }
    .rec-nutrition { border-left: 4px solid #f59e0b; background: #fffbeb; }
    .rec-general { border-left: 4px solid #3b82f6; background: #eff6ff; }
    .quick-win-badge {
        display: inline-block;
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        font-size: 0.7rem;
        padding: 2px 8px;
        border-radius: 10px;
        font-weight: 600;
        margin-left: 8px;
        vertical-align: middle;
    }
    .severity-high { border-left-width: 6px !important; }
    .severity-normal { border-left-width: 3px; }

    /* Status dots */
    .status-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .status-green { background: #10b981; box-shadow: 0 0 6px #10b98155; }
    .status-yellow { background: #f59e0b; box-shadow: 0 0 6px #f59e0b55; }
    .status-red { background: #ef4444; box-shadow: 0 0 6px #ef444455; }

    /* Skeleton loading */
    .skeleton {
        background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
        background-size: 200px 100%;
        animation: skeletonPulse 1.5s ease-in-out infinite;
        border-radius: 8px;
    }

    /* Footer */
    .app-footer {
        margin-top: 60px;
        padding: 24px 0 16px 0;
        border-top: 1px solid #e2e8f0;
        text-align: center;
        color: #94a3b8;
        font-size: 0.82rem;
        line-height: 1.8;
        animation: fadeInUp 0.6s ease-out;
    }
    .app-footer a { color: #667eea; text-decoration: none; }
    .app-footer a:hover { text-decoration: underline; }

    /* Responsive tweaks */
    @media (max-width: 768px) {
        .main-header { font-size: 1.6rem !important; }
        .sub-header { font-size: 0.85rem !important; }
        .health-score-number { font-size: 2.5rem !important; }
        .health-score-card { padding: 20px 16px; }
        div[data-testid="stMetric"] { padding: 10px; }
        .stTabs [data-baseweb="tab"] { padding: 6px 12px; font-size: 0.85rem; }
    }
    @media (max-width: 480px) {
        .main-header { font-size: 1.3rem !important; }
        .health-score-number { font-size: 2rem !important; }
    }
</style>''', unsafe_allow_html=True)


def render_sidebar() -> pd.DataFrame | None:
    """Render sidebar with data loading options. Returns loaded DataFrame or None."""
    with st.sidebar:
        st.markdown("### Data Source")

        source = st.radio(
            "Choose data source:",
            ["Demo Dataset", "Upload CSV"],
            index=0,
            help="Use the demo dataset to explore features, or upload your own health data CSV.",
        )

        df: pd.DataFrame | None = None

        if source == "Demo Dataset":
            n_days = st.slider("Days of data", 30, 365, 90, step=30)
            df = generate_sample_data(n_days=n_days)
            st.success(f"Loaded {len(df)} days of synthetic data")

            st.download_button(
                label="Download Sample CSV",
                data=get_sample_csv_bytes(),
                file_name="sample_health_data.csv",
                mime="text/csv",
            )

        else:
            uploaded = st.file_uploader(
                "Upload health data CSV",
                type=["csv"],
                help="CSV with columns like heart_rate_bpm, bp_systolic, sleep_hours, steps, etc.",
            )
            if uploaded is not None:
                try:
                    df = load_csv(uploaded)
                    st.success(f"Loaded {len(df)} records")
                except ValueError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Error reading file: {e}")

        # Data quality indicator
        if df is not None:
            st.markdown("---")
            st.markdown("### Data Quality")
            _render_data_quality_widget(df)

        st.markdown("---")
        st.markdown("### About")
        st.markdown(
            "AI Health Insight Agent uses **machine learning** (Random Forest + Logistic Regression ensemble) "
            "to analyze health metrics, assess risks, and provide personalized recommendations. "
            "All processing happens locally - no data leaves your machine."
        )
        st.markdown(
            "<small>Built with Streamlit, scikit-learn, Plotly</small>",
            unsafe_allow_html=True,
        )

    return df


def _render_data_quality_widget(df: pd.DataFrame) -> None:
    """Render data quality metrics in the sidebar.

    Shows completeness, date range, metric count, and missing data warnings.
    """
    metric_cols = get_metric_columns(df)
    total_cells = len(df) * len(metric_cols) if metric_cols else 1
    missing_cells = sum(int(df[col].isna().sum()) for col in metric_cols) if metric_cols else 0
    completeness = ((total_cells - missing_cells) / total_cells * 100) if total_cells > 0 else 0

    date_min = df["date"].min().strftime("%b %d, %Y") if "date" in df.columns else "N/A"
    date_max = df["date"].max().strftime("%b %d, %Y") if "date" in df.columns else "N/A"

    bar_color = "#10b981" if completeness >= 90 else "#f59e0b" if completeness >= 70 else "#ef4444"

    st.markdown(f'''
    <div style="padding:4px 0;">
        <div class="data-quality-item">
            <span>Completeness</span>
            <span style="font-weight:600; color:{bar_color};">{completeness:.1f}%</span>
        </div>
        <div style="background:#334155; border-radius:4px; height:6px; margin:4px 0 8px 0;">
            <div style="background:{bar_color}; border-radius:4px; height:6px; width:{min(completeness, 100):.0f}%;"></div>
        </div>
        <div class="data-quality-item">
            <span>Date Range</span>
            <span style="font-weight:600; font-size:0.8rem;">{date_min} - {date_max}</span>
        </div>
        <div class="data-quality-item">
            <span>Metrics Available</span>
            <span style="font-weight:600;">{len(metric_cols)} / {len(KNOWN_COLUMNS)}</span>
        </div>
        <div class="data-quality-item">
            <span>Total Records</span>
            <span style="font-weight:600;">{len(df)}</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # Missing data warnings
    for col in metric_cols:
        missing_pct = df[col].isna().mean() * 100
        if missing_pct > 10:
            label = KNOWN_COLUMNS.get(col, {}).get("label", col)
            st.warning(f"{label}: {missing_pct:.0f}% missing")


def render_overview_tab(df: pd.DataFrame, summary: "DataSummary") -> None:  # noqa: F821
    """Render the overview dashboard tab."""
    st.markdown("### Dashboard Overview")

    # Top metrics row
    metric_cols = get_metric_columns(df)
    if not metric_cols:
        st.warning("No recognized health metric columns found in this dataset.")
        return
    cols = st.columns(min(len(metric_cols), 6))
    for i, col_name in enumerate(metric_cols[:6]):
        meta = KNOWN_COLUMNS[col_name]
        clean = df[col_name].dropna()
        if len(clean) == 0:
            continue
        recent_val = float(clean.iloc[-1])
        avg_val = float(clean.mean())
        delta = recent_val - avg_val

        with cols[i]:
            # Integer-valued metrics display without decimals
            int_metrics = {"heart_rate_bpm", "bp_systolic", "bp_diastolic", "steps", "calories_burned", "active_minutes", "water_intake_glasses"}
            display_val = f"{recent_val:.0f}" if col_name in int_metrics else f"{recent_val:.1f}"
            st.metric(
                label=meta["label"],
                value=display_val,
                delta=f"{delta:+.1f} vs avg",
                delta_color="inverse" if col_name in {"heart_rate_bpm", "bp_systolic", "bp_diastolic", "stress_score"} else "normal",
            )

    st.markdown("---")

    # Key charts
    col1, col2 = st.columns(2)

    with col1:
        if len(metric_cols) >= 2:
            selected = metric_cols[:4]
            fig = multi_metric_chart(df, selected)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = correlation_heatmap(df)
        st.plotly_chart(fig, use_container_width=True)

    # Summary stats table
    with st.expander("Summary Statistics", expanded=False):
        display_stats = summary.stats.copy()
        display_stats.index = [KNOWN_COLUMNS.get(c, {}).get("label", c) for c in display_stats.index]
        display_cols = ["mean", "std", "min", "25%", "50%", "75%", "max", "healthy_low", "healthy_high"]
        available_cols = [c for c in display_cols if c in display_stats.columns]
        st.dataframe(
            display_stats[available_cols].round(1),
            use_container_width=True,
        )


def render_risk_tab(df: pd.DataFrame, assessment: "RiskAssessment") -> None:  # noqa: F821
    """Render the risk assessment tab."""
    st.markdown("### Health Risk Assessment")
    st.caption("Powered by Random Forest + Logistic Regression ensemble model")

    # Risk gauge
    col1, col2 = st.columns([1, 2])
    with col1:
        fig = risk_gauge(assessment.overall_risk_score)
        st.plotly_chart(fig, use_container_width=True)

        # Risk level banner
        color_map = {"Low": "green", "Moderate": "orange", "High": "red", "Critical": "red"}
        color = color_map.get(assessment.risk_level, "gray")
        st.markdown(
            f'<div style="text-align:center; padding:10px; background-color:{color}; '
            f'color:white; border-radius:8px; font-weight:bold; font-size:1.1rem;">'
            f'{assessment.risk_level} Risk</div>',
            unsafe_allow_html=True,
        )

    with col2:
        fig = risk_heatmap(assessment)
        st.plotly_chart(fig, use_container_width=True)

        fig = feature_importance_chart(assessment.feature_importances)
        st.plotly_chart(fig, use_container_width=True)

    # Detailed per-metric breakdown
    st.markdown("---")
    st.markdown("#### Per-Metric Breakdown")

    metric_data = []
    for col_name, mr in assessment.metric_risks.items():
        meta = KNOWN_COLUMNS.get(col_name, {})
        metric_data.append({
            "Metric": mr.label,
            "Current": f"{mr.current_value} {meta.get('unit', '')}",
            "Healthy Range": f"{mr.healthy_range[0]}-{mr.healthy_range[1]}",
            "Status": mr.status,
            "Trend": mr.trend,
            "Risk Score": mr.risk_score,
        })

    if metric_data:
        st.dataframe(
            pd.DataFrame(metric_data).set_index("Metric"),
            use_container_width=True,
        )

    # Anomaly Detection section
    st.markdown("---")
    st.markdown("#### Anomaly Detection")
    st.caption("Powered by Isolation Forest — identifies days with unusual health patterns")

    df_anomalies = detect_anomalies(df)

    anomaly_count = int(df_anomalies["is_anomaly"].sum())
    if anomaly_count > 0:
        st.warning(f"Detected {anomaly_count} anomalous day(s) in your health data")

        # Show anomaly timeline
        metric_cols = get_metric_columns(df)
        selected = st.selectbox(
            "View anomalies for metric:",
            ["Overall Score"] + metric_cols,
            format_func=lambda c: "Overall Anomaly Score" if c == "Overall Score" else KNOWN_COLUMNS.get(c, {}).get("label", c),
            key="anomaly_metric_select",
        )

        metric_to_plot = None if selected == "Overall Score" else selected
        fig = anomaly_timeline_chart(df_anomalies, metric_to_plot)
        st.plotly_chart(fig, use_container_width=True)

        # Show anomaly details table
        with st.expander("Anomaly Details", expanded=False):
            anomaly_rows = df_anomalies[df_anomalies["is_anomaly"]][["date", "anomaly_score", "anomaly_explanation"]].copy()
            anomaly_rows.columns = ["Date", "Anomaly Score", "Explanation"]
            anomaly_rows = anomaly_rows.sort_values("Anomaly Score", ascending=False)
            st.dataframe(anomaly_rows, use_container_width=True, hide_index=True)
    else:
        st.success("No anomalous days detected — your health patterns are consistent!")


def render_trends_tab(df: pd.DataFrame) -> None:
    """Render the detailed trends/visualizations tab."""
    st.markdown("### Detailed Trends & Distributions")

    metric_cols = get_metric_columns(df)
    if not metric_cols:
        st.warning("No recognized health metric columns found.")
        return

    selected_metric = st.selectbox(
        "Select metric to explore:",
        metric_cols,
        format_func=lambda c: KNOWN_COLUMNS.get(c, {}).get("label", c),
    )

    if selected_metric:
        col1, col2 = st.columns(2)
        with col1:
            fig = metric_trend_chart(df, selected_metric)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = distribution_chart(df, selected_metric)
            st.plotly_chart(fig, use_container_width=True)

        fig = weekly_comparison_chart(df, selected_metric)
        st.plotly_chart(fig, use_container_width=True)

    # Multi-metric comparison
    st.markdown("---")
    st.markdown("#### Multi-Metric Comparison")
    selected_multi = st.multiselect(
        "Select metrics to compare:",
        metric_cols,
        default=metric_cols[:3],
        format_func=lambda c: KNOWN_COLUMNS.get(c, {}).get("label", c),
    )
    if selected_multi:
        fig = multi_metric_chart(df, selected_multi)
        st.plotly_chart(fig, use_container_width=True)


def render_recommendations_tab(assessment: "RiskAssessment") -> None:  # noqa: F821
    """Render the recommendations tab."""
    st.markdown("### Personalized Health Recommendations")

    for i, rec in enumerate(assessment.recommendations):
        # Determine icon based on content
        if "elevated" in rec.lower() or "high" in rec.lower() or "below" in rec.lower() or "low" in rec.lower():
            icon = "⚠️"
        elif "great work" in rec.lower() or "normal" in rec.lower():
            icon = "✅"
        elif "consult" in rec.lower() or "doctor" in rec.lower():
            icon = "🏥"
        else:
            icon = "💡"

        st.markdown(
            f"""<div style="background:#f8f9fa; border-left:4px solid #3498db;
            padding:12px 16px; margin:8px 0; border-radius:4px;">
            <span style="font-size:1.1rem;">{icon}</span>&nbsp;&nbsp;{rec}</div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.caption(
        "These recommendations are generated by AI analysis and are for informational purposes only. "
        "Always consult a healthcare professional for medical advice."
    )


def render_simulator_tab(df: pd.DataFrame, assessment: "RiskAssessment") -> None:  # noqa: F821
    """Render the What-If Health Simulator tab."""
    st.markdown("### What-If Health Simulator")
    st.caption(
        "Drag the sliders to simulate different health scenarios "
        "and see how your risk score changes in real-time"
    )

    recent = df.tail(7)
    col_sliders, col_results = st.columns([1, 1])

    slider_configs: dict[str, dict[str, float]] = {
        "heart_rate_bpm": {"min": 40, "max": 150, "step": 1},
        "bp_systolic": {"min": 80, "max": 200, "step": 1},
        "bp_diastolic": {"min": 40, "max": 120, "step": 1},
        "sleep_hours": {"min": 3.0, "max": 12.0, "step": 0.5},
        "steps": {"min": 0, "max": 25000, "step": 500},
        "active_minutes": {"min": 0, "max": 180, "step": 5},
        "stress_score": {"min": 1.0, "max": 10.0, "step": 0.5},
        "spo2_percent": {"min": 88.0, "max": 100.0, "step": 0.5},
        "weight_kg": {"min": 30.0, "max": 200.0, "step": 0.5},
        "water_intake_glasses": {"min": 0, "max": 16, "step": 1},
    }

    with col_sliders:
        st.markdown("#### Adjust Your Metrics")
        simulated_values: dict[str, float] = {}

        for col_name, config in slider_configs.items():
            if col_name not in df.columns:
                continue
            meta = KNOWN_COLUMNS[col_name]
            current_val = float(recent[col_name].mean())
            lo, hi = meta["healthy_range"]

            simulated_values[col_name] = st.slider(
                f"{meta['label']} ({meta['unit']})",
                min_value=float(config["min"]),
                max_value=float(config["max"]),
                value=float(round(current_val, 1)),
                step=float(config["step"]),
                help=f"Healthy range: {lo}-{hi} {meta['unit']}",
                key=f"sim_{col_name}",
            )

    with col_results:
        st.markdown("#### Simulated Risk Assessment")

        model, scaler, model_features = build_ensemble_model()

        defaults: dict[str, float] = {
            "heart_rate_bpm": 72, "bp_systolic": 122, "bp_diastolic": 78,
            "sleep_hours": 7.0, "steps": 8000, "active_minutes": 45,
            "stress_score": 5, "spo2_percent": 97, "weight_kg": 75,
            "water_intake_glasses": 7,
        }

        feature_vector = [
            simulated_values.get(feat, defaults.get(feat, 0))
            for feat in model_features
        ]

        X_input = np.array(feature_vector).reshape(1, -1)
        X_scaled = scaler.transform(X_input)
        probas = model.predict_proba(X_scaled)[0]
        sim_risk = float(probas[1] * 40 + probas[2] * 100) if len(probas) == 3 else 50.0
        sim_risk = round(min(100.0, max(0.0, sim_risk)), 1)

        fig = risk_gauge(sim_risk, title="Simulated Risk Score")
        st.plotly_chart(fig, use_container_width=True)

        actual_risk = assessment.overall_risk_score
        delta = sim_risk - actual_risk

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Current Risk", f"{actual_risk}/100")
        with col_b:
            st.metric(
                "Simulated Risk",
                f"{sim_risk}/100",
                delta=f"{delta:+.1f}",
                delta_color="inverse",
            )

        st.markdown("#### Metric Status")
        for col_name, val in simulated_values.items():
            meta = KNOWN_COLUMNS.get(col_name, {})
            lo, hi = meta.get("healthy_range", (0, 100))
            label = meta.get("label", col_name)
            unit = meta.get("unit", "")
            if lo <= val <= hi:
                st.markdown(f"**{label}**: {val} {unit} — in range")
            else:
                st.markdown(f"**{label}**: {val} {unit} — outside {lo}-{hi}")

    # Keep health tools as an expander below the simulator
    with st.expander("Health Tools (BMI Calculator & Calorie Estimator)"):
        _render_health_tools()


def _render_health_tools() -> None:
    """Render BMI calculator and calorie estimator."""
    tool_tab1, tool_tab2 = st.tabs(["BMI Calculator", "Calorie Estimator"])

    with tool_tab1:
        st.markdown("#### Body Mass Index Calculator")
        col1, col2 = st.columns(2)
        with col1:
            weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=70.0, step=0.5)
        with col2:
            height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.5)

        if st.button("Calculate BMI", type="primary"):
            bmi_val, bmi_cat = calculate_bmi(weight, height)

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Your BMI", f"{bmi_val}")
            with col_b:
                color = {"Underweight": "orange", "Normal weight": "green", "Overweight": "orange", "Obese": "red"}
                st.markdown(
                    f'<div style="padding:12px; background-color:{color[bmi_cat]}; '
                    f'color:white; border-radius:8px; text-align:center; font-weight:bold; margin-top:8px;">'
                    f'{bmi_cat}</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("**BMI Categories:**")
            st.markdown("- **Underweight:** < 18.5")
            st.markdown("- **Normal weight:** 18.5 - 24.9")
            st.markdown("- **Overweight:** 25 - 29.9")
            st.markdown("- **Obese:** 30+")

    with tool_tab2:
        st.markdown("#### Daily Calorie Needs Estimator")
        st.caption("Based on the Mifflin-St Jeor equation")

        col1, col2, col3 = st.columns(3)
        with col1:
            cal_weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=70.0, step=0.5, key="cal_w")
            cal_height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.5, key="cal_h")
        with col2:
            cal_age = st.number_input("Age (years)", min_value=15, max_value=100, value=30)
            cal_sex = st.selectbox("Sex", ["Male", "Female"])
        with col3:
            cal_activity = st.selectbox(
                "Activity Level",
                ["sedentary", "light", "moderate", "active", "very_active"],
                index=2,
                format_func=lambda x: {
                    "sedentary": "Sedentary (little/no exercise)",
                    "light": "Light (1-3 days/week)",
                    "moderate": "Moderate (3-5 days/week)",
                    "active": "Active (6-7 days/week)",
                    "very_active": "Very Active (intense daily)",
                }[x],
            )

        if st.button("Estimate Calories", type="primary"):
            calories = estimate_daily_calories(cal_weight, cal_height, cal_age, cal_sex.lower(), cal_activity)

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Maintenance", f"{calories} kcal/day")
            with col_b:
                st.metric("Weight Loss", f"{calories - 500} kcal/day", delta="-500 kcal", delta_color="inverse")
            with col_c:
                st.metric("Weight Gain", f"{calories + 500} kcal/day", delta="+500 kcal")


def render_report_tab(
    df: pd.DataFrame,
    summary: "DataSummary",  # noqa: F821
    assessment: "RiskAssessment",  # noqa: F821
) -> None:
    """Render the PDF report download tab."""
    st.markdown("### Download Health Report")
    st.markdown("Generate a comprehensive PDF report with all your health analysis results.")

    # Sleep analysis for report
    sleep_analysis = None
    if "sleep_hours" in df.columns:
        sleep_analysis = analyze_sleep_quality(df["sleep_hours"])

    if st.button("Generate PDF Report", type="primary"):
        with st.spinner("Generating report..."):
            pdf_bytes = generate_report(summary, assessment, sleep_analysis)

        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name="health_insight_report.pdf",
            mime="application/pdf",
        )
        st.success("Report generated successfully!")

    # Sleep quality section
    if sleep_analysis and sleep_analysis.get("label") != "No data":
        st.markdown("---")
        st.markdown("### Sleep Quality Analysis")

        col1, col2 = st.columns(2)
        with col1:
            score = sleep_analysis["score"]
            label = sleep_analysis["label"]
            color = "#2ecc71" if score >= 80 else "#f39c12" if score >= 60 else "#e74c3c"
            st.markdown(
                f'<div style="text-align:center; padding:20px; border-radius:12px; '
                f'border: 2px solid {color};">'
                f'<div style="font-size:2.5rem; font-weight:bold; color:{color};">{score}/100</div>'
                f'<div style="font-size:1.2rem; color:{color};">{label}</div></div>',
                unsafe_allow_html=True,
            )

        with col2:
            details = sleep_analysis["details"]
            st.metric("Average Duration", f"{details['average_hours']} hrs")
            st.metric("Consistency Score", f"{details['consistency_score']}/100")
            st.metric("Nights in 7-9h Range", f"{details['pct_in_range']}%")

        if sleep_analysis["recommendations"]:
            st.markdown("#### Sleep Recommendations")
            for rec in sleep_analysis["recommendations"]:
                st.info(rec)


def main() -> None:
    """Main application entry point."""
    configure_page()

    # Hero section
    st.markdown(
        '''
        <div style="text-align:center; padding: 2rem 0 1rem 0;">
            <div style="font-size:3rem; margin-bottom:0.3rem;">🏥</div>
            <p class="main-header">AI Health Insight Agent</p>
            <p class="sub-header">
                ML-powered health data analysis, risk scoring &amp; personalized recommendations
            </p>
            <div style="display:flex; justify-content:center; gap:24px; flex-wrap:wrap; margin-top:8px;">
                <span style="background:#667eea15; color:#667eea; padding:4px 14px; border-radius:20px; font-size:0.82rem; font-weight:600;">
                    RF + LR Ensemble
                </span>
                <span style="background:#764ba215; color:#764ba2; padding:4px 14px; border-radius:20px; font-size:0.82rem; font-weight:600;">
                    11 Health Metrics
                </span>
                <span style="background:#10b98115; color:#10b981; padding:4px 14px; border-radius:20px; font-size:0.82rem; font-weight:600;">
                    100% Local &amp; Private
                </span>
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    # Load data
    df = render_sidebar()

    if df is None:
        st.info("Upload a CSV file or select the demo dataset from the sidebar to get started.")
        st.markdown("---")
        st.markdown("#### Expected CSV Format")
        st.markdown("Your CSV should contain a `date` column and one or more health metric columns:")
        sample_cols = pd.DataFrame({
            "Column": list(KNOWN_COLUMNS.keys()),
            "Description": [v["label"] for v in KNOWN_COLUMNS.values()],
            "Unit": [v["unit"] for v in KNOWN_COLUMNS.values()],
            "Healthy Range": [f"{v['healthy_range'][0]} - {v['healthy_range'][1]}" for v in KNOWN_COLUMNS.values()],
        })
        st.dataframe(sample_cols, use_container_width=True, hide_index=True)
        return

    # Compute summary and run risk assessment once
    summary = compute_summary(df)

    with st.spinner("Running ML risk analysis..."):
        assessment = assess_health_risk(df)

    # Tabs
    tab_overview, tab_risk, tab_trends, tab_recs, tab_sim, tab_report = st.tabs([
        "Overview",
        "Risk Assessment",
        "Trends & Charts",
        "Recommendations",
        "What-If Simulator",
        "Report & Sleep",
    ])

    with tab_overview:
        render_overview_tab(df, summary)

    with tab_risk:
        render_risk_tab(df, assessment)

    with tab_trends:
        render_trends_tab(df)

    with tab_recs:
        render_recommendations_tab(assessment)

    with tab_sim:
        render_simulator_tab(df, assessment)

    with tab_report:
        render_report_tab(df, summary, assessment)


if __name__ == "__main__":
    main()
