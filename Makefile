.PHONY: up lint help

help:
	@echo "Available commands:"
	@echo "  make up           - Run the application"
	@echo "  make lint         - Run ruff linter"

up:
	uvicorn src.web.main:app --host 0.0.0.0 --port 9000 --reload

lint:
	ruff check src/

