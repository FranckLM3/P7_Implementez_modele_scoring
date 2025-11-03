# Makefile - Credit Scoring

PYTHON := .venv/bin/python

.PHONY: help install train test

help:
	@echo "Available commands:"
	@echo "  make install    - Install package"
	@echo "  make train      - Train LightGBM model"
	@echo "  make test       - Run tests"

install:
	@echo "Installing package..."
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -e .
	@echo "Installation complete!"

train:
	@echo "Training LightGBM model..."
	$(PYTHON) scripts/train.py --data data/model_dataset.csv --out-dir artifacts
	@echo "Training complete! Model in: artifacts/pipeline.joblib"

test:
	@echo "Running tests..."
	$(PYTHON) -m pytest tests/ -v
	@echo "Tests complete!"
