# AI Health Insight Agent

ML-powered health data analysis app that scores risk across 11 health metrics using a Random Forest + Logistic Regression ensemble (VotingClassifier), generates interactive Plotly dashboards, and produces downloadable PDF reports. Runs 100% locally with zero API keys.

## How to Run

```bash
uv sync && uv run streamlit run src/app.py
```

Opens at http://localhost:8501 with a built-in 90-day demo dataset.

## Key Files

| File | Purpose |
|------|---------|
| `src/app.py` | Streamlit UI with tab routing (dashboard, metrics, recommendations, tools, report) |
| `src/models.py` | ML ensemble model (VotingClassifier), per-metric risk scoring, recommendation engine |
| `src/data_utils.py` | CSV loading/validation, BMI calculator, calorie estimator, sleep analysis |
| `src/visualizations.py` | Plotly chart generators (trend lines, heatmaps, gauges, distributions) |
| `src/report.py` | PDF report builder using fpdf2 |
| `src/sample_data.py` | Synthetic health data generator for demo mode |

## Testing

```bash
uv run python -c "from src.sample_data import *; print('Imports OK')"
```

## Architecture

Data flow: CSV upload or demo data -> validation/normalization (data_utils) -> per-metric risk scoring + ML ensemble prediction (models) -> interactive Plotly visualizations (visualizations) -> personalized recommendations (models) -> PDF report (report).

The ML model trains on 2,000 synthetic population samples at runtime. Risk score = 40% average metric deviation + 60% ML probability-weighted score.
