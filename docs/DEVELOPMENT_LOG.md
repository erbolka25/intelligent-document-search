# Development Log

## 2026-11-24 — Phase 0 audit

- Audit of the existing RAG project as the base for DocuMind KG: [docs/AUDIT_PHASE0.md](AUDIT_PHASE0.md).

## 2026-11-24 — Task 1: dev environment repair and lint

- Removed stale `src/*.egg-info` folders and reinstalled the project in the existing venv (`pip install -e ".[dev]"`) after the move to `D:\Projects\python_in30days`.
- `pytest -q` without `PYTHONPATH`: 291 passed.
- `ruff check .`: all checks passed (2 auto-fixes in `tests/test_core.py`).
- Commit `70e6e2f`.

## 2026-11-24 — Task 2: housekeeping and CI

- Tagged the learning code as `learning-archive` (local, not pushed).
- Moved screenshots into `docs/screenshots/`.
- TASK2_RESULTS_PENDING
