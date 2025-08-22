.PHONY: setup run docker test clean export

setup:
	bash ./setup.sh

run:
	uv run --env-file .env python -m app.main

docker:
	docker build -t app_template .
	docker run --rm -p 7860:7860 --env-file .env app_template

test:
	uv run --env-file .env pytest tests

export:
	uv export --format requirements-txt > requirements.txt

clean:
	rm -rf .venv __pycache__ */__pycache__ uv.lock .pytest_cache .mypy_cache
