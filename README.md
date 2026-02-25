# CREDO: Epistemic-Aware Conformalized Credal Envelopes for Regression
This repository contains the implementation of the methods described in the paper...

## Installation

### Prerequisites
- Python 3.8+
- Git
- Poetry (recommended) or venv + pip

### Clone
```bash
git clone <repo-url>
cd CREDO
```

### With Poetry (recommended)
```bash
pip install --user poetry        # if needed
poetry install                   # create env and install deps
```

### With virtualenv
```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
pip install -e .
pip install -r requirements-dev.txt   # if present
pytest
```

## Development
- Add deps:
```bash
poetry add <pkg>
poetry add --dev <pkg>
```
- Run checks:
```bash
poetry run pytest -q
poetry run black --check .
poetry run isort --check-only .
poetry run flake8 .
```
- Auto-format:
```bash
poetry run black .
poetry run isort .
```
