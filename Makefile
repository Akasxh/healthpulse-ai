.PHONY: install run dev test lint clean docker-build docker-run docker-up docker-down help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies with uv
	uv sync

run: ## Run the Streamlit app
	uv run streamlit run src/app.py

dev: ## Run with auto-reload and debug logging
	STREAMLIT_SERVER_RUN_ON_SAVE=true uv run streamlit run src/app.py --logger.level=debug

test: ## Run tests with pytest
	uv run pytest tests/ -v

lint: ## Lint with ruff
	uv run ruff check src/ tests/

format: ## Format with ruff
	uv run ruff format src/ tests/

clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov .ruff_cache .mypy_cache dist build *.egg-info

docker-build: ## Build Docker image
	docker build -t health-agent .

docker-run: ## Run Docker container
	docker run --rm -p 8501:8501 --name health-agent health-agent

docker-up: ## Start with docker compose
	docker compose up -d

docker-down: ## Stop docker compose
	docker compose down
