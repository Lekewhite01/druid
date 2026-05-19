# Contributing to DRUID

We welcome contributions. Here's how to get started.

## Setup

```bash
git clone https://github.com/Lekewhite01/druid.git
cd druid
python -m venv venv
source venv/bin/activate
pip install -e ".[dev,all]"
```

## Running Tests

```bash
pytest tests/ -v
```

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
ruff check druid/ tests/
ruff format druid/ tests/
```

## Pull Requests

1. Fork the repo and create a feature branch.
2. Write tests for new functionality.
3. Ensure all tests pass and ruff reports no issues.
4. Submit a PR with a clear description of the change.

## Reporting Issues

Open an issue on GitHub with:
- What you expected to happen.
- What actually happened.
- A minimal reproducible example.
- Your Python version and OS.
