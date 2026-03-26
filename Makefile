.PHONY: install install-dev data train evaluate quantize benchmark serve demo lint test clean help

# ─── Setup ────────────────────────────────────────────────────────────────────

install:  ## Install production dependencies
	pip install -e .

install-dev:  ## Install with dev dependencies
	pip install -e ".[dev]"
	pre-commit install

# ─── Data ─────────────────────────────────────────────────────────────────────

data:  ## Download and prepare the dataset
	python scripts/prepare_data.py

# ─── Training ─────────────────────────────────────────────────────────────────

train:  ## Fine-tune with default config
	python scripts/train.py --config configs/training/base.yaml

train-sweep:  ## Run hyperparameter sweep
	python scripts/run_sweep.py

# ─── Evaluation ───────────────────────────────────────────────────────────────

evaluate:  ## Evaluate the fine-tuned model
	python scripts/evaluate.py

baseline:  ## Run base model baseline (zero-shot + few-shot)
	python scripts/evaluate.py --baseline

# ─── Optimization ─────────────────────────────────────────────────────────────

quantize:  ## Quantize model (GPTQ + AWQ)
	python scripts/quantize.py

benchmark:  ## Run inference benchmarks (latency, throughput, memory)
	python scripts/benchmark_inference.py

# ─── Serving ──────────────────────────────────────────────────────────────────

serve:  ## Start the FastAPI inference server
	uvicorn src.serving.api:app --host 0.0.0.0 --port 8000

demo:  ## Launch the Gradio demo UI
	python demo/app.py

# ─── Dev Tools ────────────────────────────────────────────────────────────────

lint:  ## Run linter
	ruff check src/ scripts/ tests/
	ruff format --check src/ scripts/ tests/

format:  ## Auto-format code
	ruff check --fix src/ scripts/ tests/
	ruff format src/ scripts/ tests/

test:  ## Run tests
	pytest tests/

test-cov:  ## Run tests with coverage report
	pytest tests/ --cov=src --cov-report=term-missing

# ─── Charts ───────────────────────────────────────────────────────────────────

charts:  ## Generate result visualizations
	python scripts/generate_charts.py

# ─── Cleanup ──────────────────────────────────────────────────────────────────

clean:  ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info __pycache__ .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

# ─── Help ─────────────────────────────────────────────────────────────────────

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
