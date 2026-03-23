# AILS Makefile
# Artificial Intelligence Learning System
# Created by Cherry Computer Ltd.

.PHONY: help install install-dev test lint format clean run-api docker-build docker-up docker-down

PYTHON   := python3
PIP      := pip3
SRC_DIR  := src
TEST_DIR := tests

# ── Default target ────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════╗"
	@echo "║   AILS — Artificial Intelligence Learning System     ║"
	@echo "║   Created by Cherry Computer Ltd.                    ║"
	@echo "╠══════════════════════════════════════════════════════╣"
	@echo "║  make install        Install all dependencies        ║"
	@echo "║  make install-dev    Install + dev tools             ║"
	@echo "║  make test           Run all unit tests              ║"
	@echo "║  make test-cov       Run tests with coverage report  ║"
	@echo "║  make lint           Run flake8 linter               ║"
	@echo "║  make format         Auto-format with black          ║"
	@echo "║  make run-api        Start the AILS REST API         ║"
	@echo "║  make docker-build   Build Docker image              ║"
	@echo "║  make docker-up      Start all Docker services       ║"
	@echo "║  make docker-down    Stop all Docker services        ║"
	@echo "║  make clean          Remove build artifacts          ║"
	@echo "╚══════════════════════════════════════════════════════╝"
	@echo ""

# ── Installation ─────────────────────────────────────────────────────────────
install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev: install
	$(PIP) install -e ".[dev]"
	$(PIP) install black flake8 mypy isort pytest pytest-cov bandit safety

# ── Testing ───────────────────────────────────────────────────────────────────
test:
	$(PYTHON) -m pytest $(TEST_DIR)/ -v --tb=short

test-cov:
	$(PYTHON) -m pytest $(TEST_DIR)/ -v --tb=short \
		--cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing
	@echo "Coverage report: htmlcov/index.html"

test-fast:
	$(PYTHON) -m pytest $(TEST_DIR)/ -v -x --tb=short -q

# ── Code Quality ──────────────────────────────────────────────────────────────
lint:
	$(PYTHON) -m flake8 $(SRC_DIR)/ $(TEST_DIR)/ \
		--max-line-length=100 \
		--ignore=E501,W503 \
		--exclude=__pycache__,.git

format:
	$(PYTHON) -m black $(SRC_DIR)/ $(TEST_DIR)/ examples/ \
		--line-length=88
	$(PYTHON) -m isort $(SRC_DIR)/ $(TEST_DIR)/ examples/

type-check:
	$(PYTHON) -m mypy $(SRC_DIR)/ --ignore-missing-imports

security:
	$(PYTHON) -m bandit -r $(SRC_DIR)/ -ll
	$(PYTHON) -m safety check -r requirements.txt

# ── Running ───────────────────────────────────────────────────────────────────
run-api:
	$(PYTHON) -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

run-demo:
	$(PYTHON) examples/sentiment_analysis_pipeline.py

# ── Docker ───────────────────────────────────────────────────────────────────
docker-build:
	docker build -t cherrycomputerltd/ails:latest .

docker-up:
	docker-compose up -d
	@echo "✅ AILS services started. API at http://localhost:8000"

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f ails-api

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ htmlcov/ .coverage
	@echo "✅ Cleaned build artifacts."

# ── All checks before PR ─────────────────────────────────────────────────────
pre-pr: format lint test-cov security
	@echo "✅ All checks passed — ready for PR!"
