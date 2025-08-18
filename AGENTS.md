# Repository Guidelines

## Project Structure & Module Organization
- `data/`: Parquet files per instrument (created by `download_data.py`).
- `download_data.py`: Data acquisition from Yahoo Finance via `yfinance`.
- `momentum_strategy.py`: Core logic (data loading, signals, allocation, backtest, metrics).
- `*.ipynb`: Analysis notebooks (e.g., `momentum_analysis.ipynb`).
- `README.md`: Strategy overview and usage examples.

## Build, Test, and Development Commands
- Install deps: `pip install pandas numpy pyarrow yfinance matplotlib seaborn`
  - If a `requirements.txt` is added, prefer: `pip install -r requirements.txt`.
- Download data: `python download_data.py --outdir data --start 2000-01-01 --interval 1d`
  - Tip: The script’s default `--outdir` may be absolute on your machine; override to `data`.
- Run strategy (Python REPL):
  ```python
  from momentum_strategy import run_full_strategy
  res = run_full_strategy(data_dir='data', lookback_months=12, top_quantile=0.25)
  ```

## Coding Style & Naming Conventions
- Follow PEP 8; use 4‑space indentation and `snake_case` for functions/variables.
- Add docstrings to public functions; include parameter/return types where reasonable.
- Prefer type hints for new/modified code.
- Keep functions focused; avoid side effects in analysis utilities.

## Testing Guidelines
- Framework: `pytest` (recommended). Place tests under `tests/` with files named `test_*.py`.
- Example run: `python -m pytest -q`.
- Aim for unit tests around signal generation, selection, weighting, and metrics.
- Use small synthetic DataFrames for deterministic tests; avoid network calls.

## Commit & Pull Request Guidelines
- Commit messages: use imperative mood and clear scope.
  - Recommended Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.
- Pull Requests should include:
  - Summary of changes and motivation.
  - Linked issue (if applicable) and screenshots/plots for analysis impacts.
  - Notes on data/behavior changes and any breaking API changes.

## Security & Configuration Tips
- Never commit credentials or large raw datasets; `data/` should remain generated artifacts.
- Pin dependencies in `requirements.txt` when introduced to ensure reproducibility.
- For offline/CI tests, mock `yfinance` and use cached Parquet fixtures under `tests/fixtures/`.

