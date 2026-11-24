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
- Commit `62604ef`: DocuMind KG brief, screenshots and this log.
- Commit `cec008a`: removed learning leftovers (`src/document_search/core.py`, `src/document_search/utils.py`, `src/main.py`, `tests/test_core.py`). Moved `clean_text` into `ingestion.py` and its 3 tests into `tests/test_ingestion.py`.
- Local results after cleanup: `pytest -q` 262 passed (was 291; 32 exercise tests removed, 3 `clean_text` tests kept). `ruff check .` all checks passed.
- Added the GitHub Actions workflow `.github/workflows/ci.yml` (ruff + pytest, Python 3.13, CPU-only torch) and a README badge.
- CI result on GitHub: not known when this entry was committed; check the Actions page.
