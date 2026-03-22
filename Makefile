.PHONY: install run test clean lint

install:
	uv sync

run:
	uv run streamlit run src/app.py

test:
	uv run python -c "from src.sample_data import *; print('Imports OK')"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

lint:
	uv run ruff check src/
