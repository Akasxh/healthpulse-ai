# AI Health Insight Agent

ML-powered health data analysis, risk scoring, and personalized recommendations. Upload your health data or use the built-in demo to get instant insights — all processing happens locally with no external APIs.

![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.40+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

<!-- Screenshots -->
<!-- ![Dashboard](screenshots/dashboard.png) -->
<!-- ![Risk Assessment](screenshots/risk.png) -->
<!-- ![Trends](screenshots/trends.png) -->

## Features

- **Health Data Analysis** — Upload CSV or use synthetic demo data (heart rate, blood pressure, sleep, steps, SpO2, stress, weight, and more)
- **ML Risk Scoring** — Random Forest + Logistic Regression ensemble model provides per-metric and overall health risk assessment
- **Interactive Visualizations** — Plotly charts including trend lines, correlation heatmaps, risk gauges, distributions, and weekly comparisons
- **Personalized Recommendations** — AI-generated health insights based on your specific data patterns
- **Sleep Quality Analyzer** — Comprehensive sleep analysis with consistency scoring and targeted recommendations
- **BMI Calculator** — Instant BMI calculation with category classification
- **Calorie Estimator** — Daily calorie needs based on Mifflin-St Jeor equation with activity level adjustment
- **PDF Report Generation** — Download a professional health report with all findings and recommendations
- **Privacy First** — All processing runs locally. No data leaves your machine. No API keys required.

## Supported Health Metrics

| Metric | Column Name | Unit | Healthy Range |
|--------|------------|------|---------------|
| Heart Rate | `heart_rate_bpm` | bpm | 60-100 |
| Systolic BP | `bp_systolic` | mmHg | 90-120 |
| Diastolic BP | `bp_diastolic` | mmHg | 60-80 |
| Sleep Duration | `sleep_hours` | hours | 7-9 |
| Daily Steps | `steps` | steps | 7,000-15,000 |
| Calories Burned | `calories_burned` | kcal | 1,800-2,800 |
| Active Minutes | `active_minutes` | min | 30-120 |
| Weight | `weight_kg` | kg | 50-100 |
| Blood Oxygen | `spo2_percent` | % | 95-100 |
| Stress Score | `stress_score` | /10 | 1-4 |
| Water Intake | `water_intake_glasses` | glasses | 8-12 |

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

```bash
# Clone and enter the project
cd p1-health-agent

# Install dependencies
uv sync

# Run the app
uv run streamlit run src/app.py
```

The app opens at `http://localhost:8501`. Click **Demo Dataset** in the sidebar to explore with synthetic data.

### Using Your Own Data

1. Prepare a CSV with a `date` column and any of the metric columns listed above
2. Click **Upload CSV** in the sidebar
3. Upload your file and explore the analysis

## Tech Stack

- **Frontend**: Streamlit
- **ML Models**: scikit-learn (RandomForestClassifier + LogisticRegression ensemble via VotingClassifier)
- **Visualizations**: Plotly
- **PDF Generation**: fpdf2
- **Data Processing**: pandas, NumPy

## Architecture

```
src/
  app.py             # Streamlit UI and tab routing
  models.py          # ML ensemble model, risk scoring, recommendations
  data_utils.py      # CSV loading, validation, BMI/calorie calculators, sleep analysis
  visualizations.py  # Plotly chart generators
  report.py          # PDF report builder
  sample_data.py     # Synthetic health data generator
```

## How It Works

1. **Data Ingestion** — CSV is validated, parsed, and normalized. Missing values are handled gracefully.
2. **Risk Analysis** — Each metric is scored against healthy ranges. An ensemble ML model (trained on synthetic population data) predicts overall risk class.
3. **Visualization** — Interactive Plotly charts show trends, distributions, correlations, and risk heatmaps.
4. **Recommendations** — Rule-based engine generates personalized advice based on metric values, trends, and risk levels.
5. **Report** — All findings are compiled into a downloadable PDF.

## License

MIT
