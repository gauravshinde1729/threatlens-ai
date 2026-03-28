.PHONY: install test lint format run

install:
	python -m pip install --upgrade pip
	pip install -e ".[dev]"

test:
	pytest --cov=src --cov-report=term-missing --cov-fail-under=80

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

run:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
