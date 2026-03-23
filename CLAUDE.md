# HealthPulse AI — AI Health Insight Agent

ML-powered Streamlit app that analyzes 11 health metrics using a Random Forest + Logistic Regression ensemble (VotingClassifier), generates interactive Plotly dashboards, per-metric risk scoring, personalized recommendations, and downloadable PDF reports. Runs 100% locally with zero API keys.

## How to Run

```bash
# Install dependencies
uv sync

# Launch the app (opens at http://localhost:8501)
uv run streamlit run src/app.py

# With auto-reload and debug logging
STREAMLIT_SERVER_RUN_ON_SAVE=true uv run streamlit run src/app.py --logger.level=debug
```

Docker alternative:
```bash
docker build -t healthpulse-ai . && docker run --rm -p 8501:8501 healthpulse-ai
```

## Key Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/app.py` | 553 | Streamlit UI — 6 tabs (Overview, Risk, Trends, Recommendations, Tools, Report & Sleep), custom CSS with gradient sidebar, hero section |
| `src/models.py` | 374 | VotingClassifier ensemble (RF + LR), per-metric risk scoring via deviation from healthy ranges, trend detection (7-day windows), recommendation engine |
| `src/data_utils.py` | 244 | CSV loading/validation, column normalization, `compute_summary()`, BMI calculator, Mifflin-St Jeor calorie estimator, sleep quality analysis |
| `src/visualizations.py` | 340 | 8 Plotly chart generators: trend lines, multi-metric overlay, correlation heatmap, risk gauge, risk heatmap, feature importance bar, distribution (box+histogram), weekly comparison |
| `src/report.py` | 212 | PDF builder using fpdf2 — custom `HealthReport(FPDF)` class with headers, footers, risk badges, metric tables, section formatting |
| `src/sample_data.py` | 101 | Synthetic data generator — 11 metrics with weekly seasonality (`is_weekend`), gradual improvement trends, and realistic noise |
| `tests/test_data_utils.py` | — | Unit tests for data utilities |
| `tests/conftest.py` | — | Shared pytest fixtures |

## Architecture

Data flow: CSV upload or demo data -> `load_csv()` / `generate_sample_data()` -> `compute_summary()` + `assess_health_risk()` -> per-metric `_compute_metric_risk()` + `build_ensemble_model()` -> `VotingClassifier.predict_proba()` -> combined risk score (40% metric deviation + 60% ML probability) -> Plotly visualizations + `_generate_recommendations()` -> `generate_report()` PDF.

Key data structures:
- `DataSummary` (dataclass): n_rows, n_days, date_range, available_metrics, missing_pct, stats DataFrame
- `RiskAssessment` (dataclass): overall_risk_score (0-100), risk_level (Low/Moderate/High/Critical), metric_risks dict, recommendations list, feature_importances dict
- `MetricRisk` (dataclass): name, label, current_value, healthy_range, risk_score, status, trend
- `KNOWN_COLUMNS` (dict): Registry of all 11 metrics with label, healthy_range, unit

ML model trains on 2,000 synthetic samples at runtime (deterministic seed=123). StandardScaler applied before fitting. Risk classes: 0=Low, 1=Moderate, 2=High.

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Or via Makefile
make test

# Quick import check
uv run python -c "from src.sample_data import *; print('Imports OK')"

# Lint
uv run ruff check src/ tests/
```

## Common Issues

- **"No recognized health metric columns"**: CSV column names must match `KNOWN_COLUMNS` keys exactly (lowercase, underscored). The loader normalizes spaces and case but won't guess arbitrary column names.
- **Model training on every request**: `build_ensemble_model()` is called per `assess_health_risk()` invocation. No caching. For production, wrap with `@st.cache_resource`.
- **Large CSV performance**: No pagination or chunked loading. DataFrames are held in memory. Practical limit ~10K rows on default Streamlit deployment.
- **PDF generation fails**: fpdf2 requires no external fonts. If `generate_report()` errors, check that the `DataSummary.stats` DataFrame has expected columns.

## Environment Variables

All optional. See `.env.example`.

| Variable | Default | Purpose |
|----------|---------|---------|
| `STREAMLIT_SERVER_PORT` | `8501` | Server port |
| `STREAMLIT_SERVER_HEADLESS` | `true` | No browser auto-open |
| `STREAMLIT_BROWSER_GATHER_USAGE_STATS` | `false` | Disable telemetry |
| `STREAMLIT_LOGGER_LEVEL` | `info` | Log verbosity |

## Makefile Targets

`make install` | `make run` | `make dev` | `make test` | `make lint` | `make format` | `make clean` | `make docker-build` | `make docker-run` | `make docker-up` | `make docker-down`
