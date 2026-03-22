# AI Health Insight Agent

> **ML-powered health data analysis, risk scoring, and personalized recommendations** -- all running locally on your machine with zero data leaving your environment.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/streamlit-1.40+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Plotly](https://img.shields.io/badge/plotly-5.24+-3F4F75?logo=plotly&logoColor=white)](https://plotly.com)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![No API Keys](https://img.shields.io/badge/API%20keys-none%20required-brightgreen)]()

Upload your health data CSV or instantly explore with the built-in demo dataset -- no setup, no API keys, no cloud. The app analyzes 11 health metrics through a **Random Forest + Logistic Regression ensemble model**, generates interactive visualizations, and produces personalized recommendations with a downloadable PDF report.

## Key Features

| Feature | Description |
|---------|-------------|
| **ML Risk Scoring** | Random Forest + Logistic Regression ensemble (VotingClassifier) scores each metric and computes overall health risk |
| **11 Health Metrics** | Heart rate, blood pressure (sys/dia), sleep, steps, calories, active minutes, weight, SpO2, stress, water intake |
| **Interactive Dashboards** | Plotly trend lines, correlation heatmaps, risk gauges, distributions, weekly comparisons |
| **Smart Recommendations** | Rule-based engine generates personalized advice based on metric values, trends, and risk levels |
| **Sleep Quality Analyzer** | Consistency scoring, duration analysis, and targeted sleep improvement recommendations |
| **Health Tools** | BMI calculator and Mifflin-St Jeor calorie estimator with activity levels |
| **PDF Reports** | Professional downloadable report with all findings, charts summary, and recommendations |
| **Privacy First** | 100% local processing. No external APIs. No data leaves your machine |
| **Zero-Config Demo** | Works out of the box with synthetic data -- no file upload needed |

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

```bash
# 1. Clone and enter the project
cd p1-health-agent

# 2. Install dependencies (requires Python 3.12+ and uv)
uv sync

# 3. Launch the app
uv run streamlit run src/app.py
```

The app opens at **http://localhost:8501**. The **Demo Dataset** is selected by default -- you'll see the full dashboard immediately with 90 days of synthetic health data. No file upload needed.

### Using Your Own Data

1. Click **Upload CSV** in the sidebar
2. Upload a CSV with a `date` column and any of the metric columns from the table above
3. Explore your personalized analysis, risk scores, and recommendations

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

1. **Data Ingestion** -- CSV is validated, parsed, and normalized. Dates are auto-detected, column names are normalized, and missing values are handled gracefully.
2. **Per-Metric Risk Scoring** -- Each metric is scored by deviation from its healthy range, with trend detection (improving/stable/worsening) via 7-day rolling windows.
3. **Ensemble ML Model** -- A `VotingClassifier` combining `RandomForestClassifier` (60% weight) and `LogisticRegression` (40% weight) is trained on 2,000 synthetic population samples. The model predicts risk class (Low/Moderate/High) from the most recent 7 days of data.
4. **Combined Risk Score** -- Overall risk = 40% average metric risk + 60% ML model probability-weighted score.
5. **Visualization** -- Interactive Plotly charts: trend lines with healthy range bands, correlation heatmaps, risk gauges, box+histogram distributions, and weekly comparisons.
6. **Recommendations** -- Rule-based engine generates personalized advice based on current values, trends, and risk levels.
7. **Report** -- All findings compiled into a professional PDF with tables, risk badges, and formatted recommendations.

## License

MIT
