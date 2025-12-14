# Local Epistemic Uncertainty in Regression Conformal Prediction via Imprecise Probabilities and Credal Sets.
This repository contains the implementation of the methods described in the paper:

## Installation

### Prerequisites
- Python 3.8+
- Git
- Poetry (recommended) or venv + pip

### Clone
```bash
git clone <repo-url>
cd credal_CP
```

### With Poetry (recommended)
```bash
pip install --user poetry        # if needed
poetry install                   # create env and install deps
poetry run pytest                # run tests
poetry run jupyter lab           # run notebooks
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

## Contributing
Open issues/PRs, follow existing style and add tests for new features. If you want a tailored example for a specific script/notebook, specify the file and desired output.

