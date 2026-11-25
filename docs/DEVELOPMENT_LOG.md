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

## 2026-11-25 — Task 4: embedding model comparison

- Commit `845f03e`:
  - The 4 unanswerable Kyrgyz questions now also run against the ru corpus (13 unanswerable cases in total).
  - The eval CLI prints UTF-8 itself, so the cp1251 crash above is fixed.
  - The query/passage prefixes are explicit settings (`EMBEDDING_QUERY_PREFIX` / `EMBEDDING_PASSAGE_PREFIX`), plus `--model` / `--query-prefix` / `--passage-prefix` flags for eval runs.
  - `pytest -q`: 281 passed.
- Commit `ba991d9`: compared `multilingual-e5-small`, `multilingual-e5-base` and `BAAI/bge-m3` offline, retrieval only, without `--llm`. Details are in [docs/EVALUATION.md](EVALUATION.md#embedding-model-comparison):
  - Hit@1 overall: 39/44, 41/44 and 41/44; ky→ru: 5/8, 5/8 and 6/8.
  - bge-m3 scores about 0.25 lower, so `MIN_SCORE=0.75` refuses every question. Using it would mean re-tuning the threshold.
  - No model lets the threshold refuse more than 2 of 13 unanswerable questions.
  - Decision: e5-base. Same score scale as e5-small, 3.5× faster per query than bge-m3, half its size.
- CI run 45538969647 passed.

## 2026-11-25 — Task 5: switch to multilingual-e5-base

- Commit `7bea259`: `ruff format .` (13 files, formatting only).
- Commit `a5157bf`: CI also runs `ruff format --check .`. Bumped `actions/checkout` v4 → v7 and `actions/setup-python` v5 → v7 (latest releases v7.0.1 / v7.1.0), off the deprecated Node 20.
- Behaviour before the switch, with the 384-d index and a 768-d model:
  - Startup succeeded.
  - Every `/search`, `/ask` and upload failed with a dimension `ValueError`, returned as HTTP 500.
  - Two models with the same dimension would have mixed silently.
- Commit `685c087`:
  - The index saves `index_meta.json` next to its vectors: model, dimension and prefixes.
  - On startup, when the stored identity differs from the configured embedder, it logs a warning, re-embeds every chunk from its saved text and saves the index.
  - A missing, unreadable or inconsistent metadata file counts as a mismatch. If re-embedding fails, startup fails and the old files stay untouched.
  - Default `EMBEDDING_MODEL` is now `intfloat/multilingual-e5-base` in `config.py`, `.env.example`, the Dockerfile and the README.
  - 11 new tests in `tests/test_reembed.py` (× numpy/faiss). `pytest -q`: 303 passed; `ruff check .` and `ruff format --check .` are clean.
- The local `.env` had `EMBEDDING_MODEL=...e5-small`; only that line was changed to e5-base.
- Verification, offline, without `--llm`:
  - The eval with the new default gives Hit@1 41/44, MRR 0.951, with every top-1 chunk and score identical to the e5-base comparison run.
  - `uvicorn` (1 worker) against the existing `data/index/faiss` logged "re-embedding 13 chunks" and then "Re-embedded 13 chunks (dim 384 -> 768) and saved the index". `GET /health` returned 200 `{"status":"ok"}`, and the server was then stopped.
