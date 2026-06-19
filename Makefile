.PHONY: lint format-check type test coverage

lint:
	uv run ruff check .

format-check:
	uv run ruff format --check .

type:
	uv run mypy src/kerrdisk

test:
	uv run pytest -q

coverage:
	uv run pytest --cov=src/kerrdisk --cov-report=term-missing
