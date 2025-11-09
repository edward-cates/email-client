.PHONY: up lint test dataset-preview train help

help:
	@echo "Available commands:"
	@echo "  make up           - Run the application"
	@echo "  make lint         - Run pyright type checker"
	@echo "  make test         - Run pytest tests"
	@echo "  make dataset-preview      - Preview dataset from emails"
	@echo "  make train          - Train a BERT email classification model"

up:
	uvicorn src.web.main:app --host 0.0.0.0 --port 9000 --reload

lint:
	pyright src/

test:
	python -m pytest tests/

dataset-preview:
	python -m src.classification.dataset

train:
	python -m src.classification.training
