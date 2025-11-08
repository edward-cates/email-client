.PHONY: up lint build test help

help:
	@echo "Available commands:"
	@echo "  make build        - Build the Docker image"
	@echo "  make up           - Run the application"
	@echo "  make lint         - Run ruff linter"
	@echo "  make test         - Run pytest tests"

build:
	docker compose build

up:
	docker compose up

lint:
	docker compose run --rm app ruff check src/

test:
	docker compose run --rm app python -m pytest tests/

