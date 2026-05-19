.PHONY: install dev test lint format build clean publish

install:
	pip install -e .

dev:
	pip install -e ".[dev,all]"

test:
	pytest tests/ -v --tb=short

lint:
	ruff check druid/ tests/

format:
	ruff format druid/ tests/

build: clean
	python -m build

clean:
	rm -rf dist/ build/ *.egg-info druid/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +

publish: build
	twine check dist/*
	twine upload dist/*
