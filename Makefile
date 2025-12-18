.PHONY: up lint test classifier-dataset classifier prioritizer-dataset prioritizer help

help:
	@echo "Available commands:"
	@echo "  make up                  - Run the application"
	@echo "  make lint                - Run pyright type checker"
	@echo "  make test                - Run pytest tests"
	@echo "  make classifier-dataset  - Preview classification dataset from emails"
	@echo "  make classifier          - Train a BERT email classification model"
	@echo "                             Usage: make classifier MAX_EPOCHS=10 PATIENCE=5"
	@echo "  make prioritizer-dataset - Preview prioritization dataset from emails"
	@echo "  make prioritizer         - Train a BERT email prioritization model"
	@echo "                             Usage: make prioritizer MAX_EPOCHS=10 PATIENCE=5"

up:
	uvicorn src.web.main:app --host 0.0.0.0 --port 9000 --reload

lint:
	pyright src/

test:
	python -m pytest tests/

classifier-dataset:
	python -m src.classification.dataset

classifier:
	python -m src.classification.training $(if $(MAX_EPOCHS),--max_epochs $(MAX_EPOCHS)) $(if $(PATIENCE),--patience $(PATIENCE))

prioritizer-dataset:
	python -m src.prioritization.dataset

prioritizer:
	python -m src.prioritization.training $(if $(MAX_EPOCHS),--max_epochs $(MAX_EPOCHS)) $(if $(PATIENCE),--patience $(PATIENCE))

