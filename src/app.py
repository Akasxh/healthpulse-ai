"""AI Health Insight Agent - Main Streamlit Application."""

from __future__ import annotations

import streamlit as st
import pandas as pd

from .data_utils import (
    analyze_sleep_quality,
    calculate_bmi,
    compute_summary,
    estimate_daily_calories,
    get_metric_columns,
    load_csv,
    KNOWN_COLUMNS,
)
from .models import assess_health_risk
from .report import generate_report
from .sample_data import generate_sample_data, get_sample_csv_bytes
from .visualizations import (
    correlation_heatmap,
    distribution_chart,
    feature_importance_chart,
    metric_trend_chart,
    multi_metric_chart,
    risk_gauge,
    risk_heatmap,
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

    st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #7f8c8d;
        margin-top: -10px;
        margin-bottom: 20px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
    }
    .stMetric > div {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 10px 15px;
        border-left: 4px solid #3498db;
    }
    div[data-testid="stSidebar"] {
        background-color: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)


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


def render_overview_tab(df: pd.DataFrame, summary: "DataSummary") -> None:  # noqa: F821
    """Render the overview dashboard tab."""
    st.markdown("### Dashboard Overview")

    # Top metrics row
    metric_cols = get_metric_columns(df)
    cols = st.columns(min(len(metric_cols), 6))
    for i, col_name in enumerate(metric_cols[:6]):
        meta = KNOWN_COLUMNS[col_name]
        recent_val = df[col_name].dropna().iloc[-1] if df[col_name].notna().any() else 0
        avg_val = df[col_name].mean()
        delta = recent_val - avg_val

        with cols[i]:
            st.metric(
                label=meta["label"],
                value=f"{recent_val:.0f}" if recent_val == int(recent_val) else f"{recent_val:.1f}",
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


def render_risk_tab(df: pd.DataFrame) -> "RiskAssessment":  # noqa: F821
    """Render the risk assessment tab. Returns the assessment for reuse."""
    st.markdown("### Health Risk Assessment")
    st.caption("Powered by Random Forest + Logistic Regression ensemble model")

    with st.spinner("Running ML risk analysis..."):
        assessment = assess_health_risk(df)

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

    return assessment


def render_trends_tab(df: pd.DataFrame) -> None:
    """Render the detailed trends/visualizations tab."""
    st.markdown("### Detailed Trends & Distributions")

    metric_cols = get_metric_columns(df)

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


def render_tools_tab() -> None:
    """Render the health tools tab (BMI calculator, calorie estimator, etc.)."""
    st.markdown("### Health Tools")

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

    # Header
    st.markdown('<p class="main-header">AI Health Insight Agent</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">ML-powered health data analysis, risk scoring, and personalized recommendations</p>',
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

    # Compute summary
    summary = compute_summary(df)

    # Run risk assessment once
    assessment = assess_health_risk(df)

    # Tabs
    tab_overview, tab_risk, tab_trends, tab_recs, tab_tools, tab_report = st.tabs([
        "Overview",
        "Risk Assessment",
        "Trends & Charts",
        "Recommendations",
        "Health Tools",
        "Report & Sleep",
    ])

    with tab_overview:
        render_overview_tab(df, summary)

    with tab_risk:
        render_risk_tab(df)

    with tab_trends:
        render_trends_tab(df)

    with tab_recs:
        render_recommendations_tab(assessment)

    with tab_tools:
        render_tools_tab()

    with tab_report:
        render_report_tab(df, summary, assessment)


if __name__ == "__main__":
    main()
