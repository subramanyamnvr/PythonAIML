# Notebook-To-Package, Testing, And CI

## Focus
Converting notebook logic into reusable package code with tests and a simple CI workflow.

## Included Here
- `sample_notebook_refactor.py`: a starter refactor example that mirrors notebook-to-module cleanup
- `src/ml_package/`: reusable package code extracted from notebook logic
- `tests/test_features.py`: unit tests for the packaged feature code
- `.github/workflows/ci.yml`: a basic CI pipeline for automated checks

## Run
- `python sample_notebook_refactor.py`
- `python -m unittest discover -s tests -v`

## Expand With
- packaging via `pyproject.toml`
- linting, formatting, and notebook smoke tests in CI
