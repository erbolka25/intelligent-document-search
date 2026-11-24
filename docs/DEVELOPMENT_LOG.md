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
- CI result on GitHub: run 45407202962 passed (ruff clean, 262 passed).

## 2026-11-24 — Task 3: Kyrgyz baseline

- Commit `a7d6171`: sentence splitting now also accepts sentences that start with Ң, Ө or Ү (3 new tests).
- Added the Kyrgyz corpus `data/samples_ky/` (4 policies). Merged 24 Kyrgyz-scenario questions into `data/eval/eval_set.json` (now 53 questions, each with `q_lang` and `corpus`). The Kyrgyz texts and questions were reviewed by a native speaker.
- Commit `0121086`: evaluation builds one index per corpus and reports metrics and top-1 score distributions per scenario. `pytest -q`: 274 passed. `ruff check .`: all checks passed.
- Offline retrieval run with `multilingual-e5-small`, without `--llm`. Results are in [docs/EVALUATION.md](EVALUATION.md):
  - Hit@1: ky→ky 8/8, ky→ru 5/8, ru→ky 3/4, ru→ru 21/21.
  - FalseRefusal 0.000 everywhere, but the lowest answerable score is 0.751 (ru→ky).
  - CorrectRefusal 0.000: unanswerable Kyrgyz questions score up to 0.854.
- On a Windows cp1251 console the eval CLI crashes when printing Kyrgyz letters. Run it with `PYTHONIOENCODING=utf-8`.
