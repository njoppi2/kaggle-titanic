# Contributing

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Before Opening a PR

Run:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Pull Request Guidelines

- Keep changes scoped (data prep, modeling, or validation).
- Document metric impact and dataset assumptions in PR description.
- Update `README.md` when execution steps or outputs change.
- Do not commit Kaggle secrets or local-only generated artifacts.
