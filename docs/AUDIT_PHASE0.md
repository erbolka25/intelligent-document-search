# DocuMind KG: Phase 0 audit of "Intelligent Document Search"

- **Date:** 2026-11-24
- **Branch:** `documind/phase0-audit` (HEAD `0ab14d2`)
- **Spec:** `DOCUMIND_KG_BRIEF.md`. No `MASTER_PROJECT_BRIEF.md` exists in the repo.
- **Scope:** a read-only audit. No project file was changed, nothing was installed, and the venv was left as it was. No database exists to touch. No paid API was called. This report is the only file created.

Every finding below is labelled with how it was checked:

- **[RUN]**: I checked it by running something (tests, ruff, a small script).
- **[CODE]**: I read it in the code but did not run it.
- **[DOC]**: it comes from README / sprint_status and was not re-checked.

---

## 1. Project structure

Skipped: `venv/`, `.git/`, caches and `__pycache__`.

```
.
├── .dockerignore  .env (git-ignored)  .env.example  .gitignore
├── DOCUMIND_KG_BRIEF.md        (untracked)  target spec
├── Dockerfile  docker-compose.yml  pyproject.toml
├── README.md                   current product docs (EN)
├── sprint_status.md            local only: excluded via .git/info/exclude
├── RAG.png                     (untracked)  UI screenshot
├── docs/                       (untracked)  RAG.png, RAG2.png, RAG3.png (UI screenshots)
├── data/
│   ├── eval/        eval_set.json (tracked), report.json (git-ignored, stale)
│   ├── index/faiss/ chunks.json, vectors.npy (git-ignored runtime index, 13 chunks)
│   ├── samples/     4 RU policy .txt files (fictional company)
│   ├── samples_en/  the same 4 policies in EN
│   └── uploads/     4 uploaded .txt files stored under their content hash (git-ignored)
├── src/
│   ├── main.py                 leftover learning script: broken import
│   ├── document_search/
│   │   ├── config.py           pydantic-settings
│   │   ├── ingestion.py        parse (.txt/.pdf), clean, sentence-aware chunking
│   │   ├── retrieval/          embeddings.py (BaseEmbedder, e5), index.py (NumPy/FAISS)
│   │   ├── generation.py       prompt, LLM clients, NO_ANSWER, citations, language detection
│   │   ├── service.py          RagService: orchestration + fallbacks
│   │   ├── evaluation.py       retrieval / refusal metrics CLI
│   │   ├── i18n.py             server-side translations
│   │   ├── core.py             learning exercises; only clean_text is used in production
│   │   ├── utils.py            empty
│   │   └── api/                app.py, auth.py, ratelimit.py, schemas.py, static/ (vanilla JS UI)
│   ├── inteligent_document_searcher.egg-info/   stale build metadata (git-ignored)
│   └── intelligent_document_searcher.egg-info/  second, differently spelled copy
└── tests/  14 test modules + js/ (Node harnesses for the frontend)
```

Size: about 2,050 lines of Python in `src/`, about 2,000 lines of tests, and about 850 lines of static frontend. **[RUN]** (`wc -l`)

There is no `.github/` folder, so there is no CI. **[RUN]**

---

## 2. Actual tech stack

The versions below are what the existing venv has installed. **[RUN]** (`pip list`)

| Layer | Technology | Installed version | Declared in |
|---|---|---|---|
| Runtime | CPython | 3.13.2 (venv); `python:3.13-slim` in Docker | pyproject `>=3.12` |
| API | FastAPI / Starlette / Uvicorn | 0.143.3 / 1.8.0 / 0.54.0 | pyproject |
| Validation / config | Pydantic / pydantic-settings | 2.14.1 / 2.16.0 | pyproject |
| Uploads | python-multipart | 0.0.32 | pyproject |
| Sessions | itsdangerous | 2.2.0 | pyproject |
| PDF parsing | PyMuPDF | 1.28.2 | pyproject |
| Embeddings | sentence-transformers / transformers / torch (CPU) | 6.1.0 / 5.20.1 / 2.14.1+cpu | pyproject (torch installed separately) |
| Embedding model | `intfloat/multilingual-e5-small` (384-d) | in the HF cache at `D:\Cache\huggingface` | config default |
| Vector index | NumPy / faiss-cpu (`IndexFlatIP`) | 2.5.4 / 1.15.1 | pyproject |
| LLM SDKs | openai / anthropic | 3.33.0 / 1.15.0 | pyproject |
| Frontend | Vanilla HTML + JS + CSS, no build step | – | `api/static/` |
| Dev | pytest / ruff / httpx / mypy / black / pytest-cov | 9.1.1 / 0.16.6 / 0.28.1 / … | `[dev]` extra |
| Container | Dockerfile + docker-compose (one `app` service, `app-data` volume) | – | – |

Not present: PostgreSQL, pgvector, SQLAlchemy, Alembic, React/Vite/TypeScript/Tailwind, python-docx, CI.

The package is named `inteligent-document-searcher` (with a typo), and `src/` holds egg-info for both spellings.

---

## 3. What is implemented

| Area | What exists | Source |
|---|---|---|
| **Parsing** | `.txt` is decoded as UTF-8 (BOM-aware), falling back to cp1251. `.pdf` goes through PyMuPDF **page by page with 1-based page numbers**. Pages with no readable text (scans, broken fonts) are detected with a heuristic, skipped, and reported as `skipped_pages`. Password-protected and corrupt PDFs are rejected with stable error codes. Text gets NFKC normalization and control/garbage characters removed. **No DOCX, no Markdown, no OCR.** | [CODE] `ingestion.py`; [RUN] tests pass |
| **Chunking** | Sentence-aware: a rule-based regex splits sentences, whole sentences are packed into chunks of at most `CHUNK_MAX_CHARS` (500), and consecutive chunks overlap by whole sentences up to `CHUNK_OVERLAP_CHARS` (100). Over-long sentences are split on word boundaries. Chunks never cross pages. Each chunk stores `doc_id, filename, page, chunk_index, char_start, char_end, text`. **No token count and no language field.** Offsets point into the *normalized* page text, not the original file. | [CODE] |
| **Embeddings** | `BaseEmbedder` ABC with `SentenceTransformerEmbedder`. Default model `intfloat/multilingual-e5-small`, CPU, loaded lazily, local cache first (works offline). The `query:` / `passage:` prefixes are added when the model name contains `"e5"`. Vectors are L2-normalized. | [CODE]; [RUN] the model loads offline (see §9) |
| **Vector store** | `BaseVectorIndex` ABC with `NumpyIndex` (brute force) and `FaissIndex` (`IndexFlatIP`). Both are exact cosine search. Everything is in memory and persisted to `data/index/<backend>/vectors.npy` + `chunks.json` (each file replaced atomically, but not the pair together). Deleting a document rebuilds the index. Corrupt files are renamed `*.corrupt-<ts>` and the app starts with an empty index. **No database.** | [CODE]; [RUN] tests |
| **Retrieval** | Embed the query, search the top-k (1–20, default 5), then drop hits below `MIN_SCORE` (0.75). **There is no filter by document or collection**: every search covers all documents. No hybrid search, no reranking. | [CODE] |
| **LLM generation** | `BaseLLMClient` ABC with `OpenAIClient` (Chat Completions) and `AnthropicClient` (beta Messages with server-side fallback), selected by `LLM_PROVIDER`. Keys come only from env/.env. Timeout and 2 retries. SDK errors map to stable codes (`auth`, `quota`, `rate_limit`, `network`, `model`, `refusal`, `empty`, `api`). Without a key, or when the LLM fails, `/ask` falls back to **retrieval-only** mode with a localized notice. | [CODE]; [RUN] unit tests with fake SDK clients |
| **Refusal** | Three layers: (1) no chunk clears `MIN_SCORE`, so the LLM is not called and `answer_found=false`; (2) the model replies `NO_ANSWER: …`; (3) **a reply without a valid `[n]` citation is discarded** as unsupported. A stray `NO_ANSWER` in mid-text is stripped. | [CODE]; [RUN] tests |
| **Citations** | Sources go to the LLM numbered `<source id="n" file=… page=…>`. `[n]` markers are mapped back to the retrieved chunks, and out-of-range markers are deleted from the text. **Every returned citation is therefore a chunk that was actually retrieved.** The API returns filename, page (PDF) or chunk index (TXT), offsets, score and full text. | [CODE] |
| **Citation validation** | `check_citations()` runs a lexical check of whether the cited passage shares 5-character-prefix key terms with the sentence citing it. Results are `ok` / `unsupported` / `cross_language` / `no_terms` / `out_of_range`. **The results are only logged.** They do not affect the answer or the UI. Stopwords cover RU and EN only. | [CODE] |
| **Kyrgyz detection** | Only for the *question*, to choose the answer language. A question counts as Kyrgyz if it contains ң/ө/ү or one of 22 common Kyrgyz words. Otherwise the language is decided by Cyrillic vs Latin character counts. Document language is **not** detected or stored. | [CODE]; [RUN] probe (§9) |
| **API** | FastAPI with unversioned routes: `/auth/login`, `/auth/logout`, `/health`, `/stats`, `GET/POST/DELETE /documents`, `DELETE /documents/{id}`, `/search`, `/ask`. Every error has the shape `{"error", "message"}` with a localized message and a request id on 500s, and responses never include stack traces. Logs are JSON lines. Security headers and CSP are set. A session cookie is required for everything except the login flow, `/health` and static files. | [CODE]; [RUN] `test_api.py`, `test_auth.py` |
| **Auth / limits** | One shared password checked with `hmac.compare_digest`. Sessions are signed with itsdangerous and tracked in memory, so they can be revoked. In-memory sliding-window rate limits for login, ask/search and upload. | [CODE] |
| **Frontend** | One static page plus a login page in vanilla JS. Tabs for asking and searching, drag-and-drop upload, a document list with delete, clickable `[n]` citations that jump to the passage card, and an RU/EN switch (`i18n.json`). The DOM is built with `textContent` and `append` only, with no `innerHTML`, so it is XSS-safe. | [CODE]; [RUN] Node-based render tests ran (Node is installed) |
| **Tests** | 14 pytest modules plus 2 JS harnesses. They use a deterministic `HashingEmbedder` and fake LLM/SDK clients, so they need **no network, no model and no API key**. | [RUN] |
| **Docker** | `python:3.13-slim` image with CPU torch. The model is baked in at build time. Runs as a non-root user, has a healthcheck, and stores data in the `app-data` volume. Compose defines one service and no database. | [CODE]. Not built (that would download about 1.5 GB). README says it was not re-tested after the auth/UI changes. [DOC] |
| **CI** | None. | [RUN] |
| **Evaluation** | `python -m document_search.evaluation` indexes `data/samples` and computes Recall/Precision@1,3,5, MRR, CorrectRefusal and FalseRefusal, plus a latency benchmark. `--llm` runs the full RAG pipeline, which costs money. `eval_set.json` has **29 questions: 21 RU + 3 EN answerable, 5 unanswerable (4 RU, 1 EN), and 0 Kyrgyz**. | [CODE]; I did not run it. `data/eval/report.json` is stale: 24 questions, from before the refusal metrics were added. |

---

## 4. What works and what is broken

### Verified by running

| Check | Result |
|---|---|
| `venv\Scripts\python -m pytest -q` (the README command) | ❌ **Fails at collection:** `ModuleNotFoundError: No module named 'document_search'`. The venv's editable install (`__editable__.inteligent_document_searcher-0.1.0.pth`) points to `C:\Users\user\Desktop\python_in30days\src`, which no longer exists, because the project was moved to `D:\Projects\`. |
| The same command with `PYTHONPATH=src` set for that process only | ✅ **291 passed, 0 failed, 0 skipped**, in 32.8 s. The Node-based frontend tests also ran. No test calls a paid API. |
| `ruff check .` | ⚠️ **3 errors**, all in `tests/test_core.py`: I001 (import order), F401 (unused import), F811 (redefinition). All three can be auto-fixed. `src/` is clean. |
| `python src/main.py` | ❌ `ImportError: cannot import name 'search_word' from 'document_search.utils'`, because `utils.py` is empty. This is dead learning code, not part of the app. |
| Sentence splitting on Kyrgyz text | ❌ `"Иш күнү 9:00дө башталат. Үй-бүлөлүк өргүү 3 күн."` comes back as **one** sentence. The boundary regex only allows `[A-ZА-ЯЁ0-9]` after a period, so sentences that start with **Ү, Ө or Ң** are never split. The same pattern in Russian splits correctly. |
| Question language detection | ✅ `"Канча күн өргүү берилет?"` → Kyrgyz. ✅ Kyrgyz typed without special letters but with a listed word (`канча`) → Kyrgyz. ❌ `"Командировкага акча"` → **Russian**: no special letters, no listed word. |
| Cross-lingual retrieval probe (offline, local model) | See §9. |
| Secret scan of tracked files | ✅ No API-key patterns, private keys or hardcoded passwords found (`git grep` with key/password regexes). `.env` is git-ignored (`git check-ignore` → `.gitignore:13`) and has **never been committed** (`git log --all -- .env` is empty). `.dockerignore` also excludes `.env`. |

Not run: the server itself (`uvicorn`), Docker, `evaluation.py`, and any real LLM call.

Because the server imports `document_search` through the same broken editable install, **`uvicorn document_search.api.app:app` will very likely fail the same way from this venv** until the package is reinstalled. This is inferred, not run.

### Read in code only (likely issues)

- **Prompt-delimiter spoofing:** `format_source()` puts the raw chunk text between `<source …>` and `</source>` without escaping. A document containing `</source><source id="1" …>` can forge source boundaries. The system prompt does say to treat sources as untrusted, which helps, but the boundary itself is not protected. The filename is safe because `sanitize_filename` strips quotes.
- **`delete_document` / `clear_documents`** change the in-memory index before `save()` and have no rollback. sprint_status lists this as known.
- **Index persistence** writes two files with two separate `os.replace` calls. A crash between them leaves the files mismatched, which `load()` then renames as corrupt. This is acknowledged in the code.
- **Upload processing is synchronous** inside the request: parse, embed and save. There is no limit on PDF page count, so a 10 MB PDF with thousands of pages ties up a worker. There is no "processing/failed" status: an upload either succeeds or returns an error.
- **Single-process design:** the index, sessions and rate limits all live in memory, so it must run as one uvicorn worker. This is documented.
- **The e5 prefix rule** (`"e5" in model_name`) is a heuristic. A different model whose name happens to contain "e5" would get the wrong prefixes.

---

## 5. Code quality

**Strengths**

- **Clear module boundaries.** Ingestion, retrieval (embedder and index behind ABCs), generation (LLM behind an ABC), orchestration (`RagService` with injected dependencies) and the API are separate modules. `create_app(service=…)` makes the API testable with fakes. This already matches the brief's "provider abstraction" requirement in spirit.
- **Error handling is above average for a student project.** Errors have stable codes, there is a fallback mode, rollback on failed indexing, recovery from a corrupt index, no stack traces in responses, and request IDs.
- **Configuration** comes entirely from pydantic-settings, and secrets are `SecretStr`.
- **Tests** are meaningful and fast and run offline.

**Weaknesses**

- **`generation.py` does four jobs:** prompts, two provider clients, citation parsing and validation, and language detection. It should be split as the project grows.
- **Learning leftovers:** `core.py` holds about 30 exercise functions plus a stub `search()`, and only `clean_text` is used. `utils.py` is empty. `src/main.py` is broken. `tests/test_core.py` tests the exercises. There are two egg-info directories.
- **Hardcoded values:** the Kyrgyz word list and the RU/EN stopwords in `generation.py`; `MIN_SCORE=0.75`, calibrated on the 4 Russian sample documents only; default model IDs in `config.py`; the sentence regex alphabet; and `_STEM_LENGTH=5`.
- **Metadata model:** `Chunk` is a frozen dataclass that is serialized to JSON, and there is no `Document` entity. Document metadata (upload date, size, mime type, language, status, page count) is never stored. `documents()` derives the list from the chunks.

**Security**

| Topic | Finding |
|---|---|
| Path traversal | ✅ Files are stored as `<sha256[:16]><ext>`, so user input never reaches the path. `doc_id` is checked against `^[0-9a-f]{16}$` before deletion, and the delete glob uses only that hash. [CODE] |
| Upload validation | ✅ Extension allow-list, Content-Length pre-check, a size cap when the body is read, a `%PDF-` magic check, and a NUL-byte check for text files. ⚠️ No page-count or CPU limit for PDFs. PyMuPDF parses untrusted PDFs in-process, so keep it up to date. |
| XSS | ✅ The frontend only uses `textContent`/`append`, and there is a strict CSP with no inline scripts. CSP is relaxed only on `/docs` and `/redoc`. |
| Prompt injection | ⚠️ Partly handled: the system prompt says to ignore instructions inside sources, and answers without citations are dropped. Source delimiters are not escaped (see §4). There is no test with adversarial document content. |
| Auth | ✅ Constant-time password compare, HttpOnly SameSite=Lax cookie, revocable sessions, login rate limit. ⚠️ One shared password and no per-user data separation, which the brief accepts for the MVP. |
| CORS | No CORS middleware, so the app is same-origin only. That is fine for now, but a separate React dev server will need a restricted CORS configuration. |
| Secrets | ✅ See §4. |

---

## 6. Untracked files: `docs/`, `RAG.png`, `DOCUMIND_KG_BRIEF.md`

| File | Contents | Recommendation |
|---|---|---|
| `docs/RAG.png` (62 KB) | English UI screenshot: header and status pills, 4 sample documents uploaded, the party-budget question answered with **"The documents do not answer this question."** (the refusal demo). | **Commit** it. The brief and portfolio requirements ask for screenshots. Consider renaming it to something like `docs/screenshots/refusal.png`. |
| `docs/RAG2.png` (97 KB) | Answer to "daily allowance abroad" → "40 USD per day [1][2]", with the source list and the retrieved passages. | **Commit** (`citations.png`). |
| `docs/RAG3.png` (91 KB) | Answer to "How often must I change my password?" → "every 90 days [1]". | **Commit** (`answer.png`). |
| `RAG.png` (root, 64 KB) | Almost the same image as `docs/RAG.png` (same screen, the upload results expanded, different bytes). | **Don't commit it at the root.** Keep one copy under `docs/`. |
| `DOCUMIND_KG_BRIEF.md` | The target spec. | **Commit** it, since it is the project's spec from here on. |

The screenshots show only fictional sample data. No secrets or personal data are visible.

`sprint_status.md` is kept out of git through `.git/info/exclude`. It contains personal, non-project notes alongside the RAG task list, so **it should stay out of the repository**. The useful RAG section can later move into `docs/DEVELOPMENT_LOG.md`.

---

## 7. Gap analysis against the brief

Reuse = keep as is. Refactor = keep the logic but reshape or extend it. Missing = build it new.

| Feature | Brief requirement | Current state | Verdict |
|---|---|---|---|
| Formats | PDF, DOCX, TXT, MD | PDF and TXT only | **Reuse** PDF/TXT; **Missing** DOCX (python-docx) and MD |
| Page metadata | Page-level for PDF; structure for DOCX | ✅ PDF pages, skipped-page report | **Reuse** |
| Scanned PDFs | Report them as unsupported | ✅ `no_text` error + `skipped_pages` | **Reuse** |
| Upload validation and safe storage | Type, size, safe names, no traversal | ✅ Solid | **Reuse** (move into a storage service) |
| Processing status | Uploaded / Processing / Ready / Failed | ❌ Synchronous upload; success or error only | **Missing** (needs a `documents` table) |
| Document entity | filename, mime, size, language, pages, dates, error | ❌ Only `doc_id`, `filename`, chunk count, derived from the chunks | **Missing** |
| Chunking | Sentence/paragraph-aware, configurable, with metadata | ✅ Sentence-aware, configurable, offsets, page | **Refactor**: add token count and language, fix the Kyrgyz capitals, add a paragraph-first strategy |
| Chunking experiments | Compare strategies | Only one strategy; the eval harness exists | **Refactor** (the eval harness is reusable) |
| Multilingual embeddings | Tested multilingual model, behind an abstraction | ✅ `BaseEmbedder` + e5-small; ❌ no Kyrgyz testing | **Reuse** the abstraction; **Missing** a model comparison and Kyrgyz evaluation |
| Vector store | PostgreSQL + pgvector | NumPy/FAISS files in memory | **Refactor**: new pgvector store; keep NumPy for tests |
| Retrieval filtering | By documents or collections | ❌ Always searches everything | **Missing** |
| Retrieval diagnostics | Configurable, diagnosable | Scores returned, top-k, threshold | **Reuse** + extend |
| LLM abstraction | Provider interface | ✅ `BaseLLMClient`, OpenAI and Anthropic | **Reuse** (move into `providers/llm`) |
| Grounded prompt | Only the sources, refuse, citations, untrusted content | ✅ All four are present | **Reuse**; **Refactor** to escape the delimiters |
| Insufficient evidence | Explicit refusal flag | ✅ `answer_found=false` (the brief calls it `insufficient_evidence`) | **Reuse** (rename it in the v1 API) |
| Citation correctness | Each citation = a retrieved chunk; no fake pages | ✅ Enforced by construction | **Reuse** |
| Citation validation | Validate support | ⚠️ Lexical check, logged only | **Refactor** (surface it, or use it to gate the answer) |
| Answer language | Same as the question, or user-selected | ✅ Detected from the question; ❌ no `answer_language` parameter | **Refactor** |
| Document language | Detect or record | ❌ | **Missing** |
| API | `/api/v1/...`, documents CRUD + status, `/chat/ask` with `document_ids`, health with a DB check | Unversioned; no status, document detail or `document_ids` | **Refactor** (add v1 routers and keep the old routes during the migration) |
| Error responses | Consistent, safe | ✅ `{error, message}` + i18n | **Reuse** |
| Database | Postgres, FKs, indexes, Alembic, cascading delete | ❌ None | **Missing** |
| Collections, chat sessions, history | Tables and endpoints | ❌ | **Missing** (collections can wait; the brief lists them in Phase 9) |
| Frontend | React + Vite + TS + Tailwind; dashboard, documents, document detail, chat with a citations panel | Vanilla JS single page: upload, list, ask/search, clickable citations | **Reuse** for now; **Missing** the React app (a rewrite, Phase 6) |
| UI localization | RU + EN now, KY later | ✅ RU/EN via `i18n.json`, with a test checking that keys stay consistent | **Reuse** (adding KY is just one more block) |
| Security | Validation, CORS, secrets, untrusted content, resource limits | Mostly ✅; no PDF resource limit, no CORS config | **Reuse** + small additions |
| Tests | Unit, integration, API, evaluation | ✅ 291 unit/API tests; ❌ no DB integration tests | **Reuse** |
| Evaluation dataset | 20–30 questions in KY/RU/EN, cross-lingual, with expected page | 29 questions, RU/EN only, no page field, written together with the documents | **Refactor**: add KY docs and questions and cross-lingual pairs |
| Docker / local Postgres | Compose for Postgres | Compose for the app only; Docker not re-tested | **Refactor** (add a `db` service with the pgvector image) |
| CI | (implied by "tests pass", portfolio) | ❌ | **Missing** |
| Docs set | PRODUCT_SPECIFICATION, ARCHITECTURE, ROADMAP, DATABASE_SCHEMA, RAG_METHODOLOGY, DEVELOPMENT_LOG | README only (good quality) | **Missing** (Phase 0/1 deliverable) |

---

## 8. Architecture recommendation

**Recommendation: evolve this codebase in place.** Add the FastAPI + PostgreSQL/pgvector + SQLAlchemy + Alembic layer *inside the existing `src/document_search` package*, reusing the current modules. Do not create a separate `backend/` project.

### Why

1. **The core is already shaped the way the brief wants.** It has provider ABCs for embeddings, LLM and the index, a service with injected dependencies, and stable error codes. The brief's `services/` and `providers/` layout is mostly a matter of moving and renaming things, not rewriting them.
2. **291 offline tests are a safety net.** A parallel backend would start at zero tests and copy the parsing, chunking, refusal and citation logic, which is the hardest part to get right. The brief also says: *"Do not create a second duplicate project without a strong technical reason."*
3. **The current demo works and is documented.** Moving step by step keeps it working at every commit.
4. **What's missing is mostly additive:** a database, a document lifecycle, v1 routes, DOCX/MD, and Kyrgyz evaluation. None of it requires throwing away existing code.

The one real design change is persistence. `BaseVectorIndex` assumes all chunks live in memory: `documents()` walks the chunk list, and deletion rebuilds the index. pgvector needs a **repository-style interface** that also owns documents and their status, with search accepting a `document_ids` filter. Introduce that as a new interface and keep `NumpyIndex` behind it as the zero-dependency backend for tests and the offline demo.

### Target layout (adapted from brief §8, inside the current package)

```
src/document_search/
  config.py               + DATABASE_URL, STORAGE_BACKEND=file|postgres
  ingestion/              parsers/{pdf,txt,docx,md}.py, chunking.py   (split from ingestion.py)
  providers/embeddings/   from retrieval/embeddings.py
  providers/llm/          OpenAI/Anthropic clients from generation.py
  rag/                    prompt.py, citations.py, language.py         (split from generation.py)
  store/                  base.py (DocumentStore protocol), memory.py (NumPy/FAISS), pgvector.py
  db/                     session.py, models.py (documents, document_chunks, …)
  services/               document_service.py, rag_service.py           (from service.py)
  api/                    app.py, routes/v1/{documents,chat,health}.py, legacy routes kept
migrations/               Alembic
```

### Migration path that keeps the current app working

1. **Stabilize.** Repair the venv install, add CI (ruff + pytest), remove the learning leftovers. No behaviour change.
2. **Introduce the store interface.** Create `DocumentStore` and wrap the current NumPy/FAISS index as `MemoryStore`. `RagService` talks to the interface. All existing tests must pass unchanged.
3. **Add the DB in parallel.** Add a `db` service to docker-compose (a `pgvector/pgvector` image), SQLAlchemy models, an Alembic baseline, and a `/api/v1/health` endpoint that checks the database. The app still runs on `STORAGE_BACKEND=file` by default.
4. **Build `PgVectorStore`** behind the same interface, with document status, cascading delete and a `document_ids` filter. Run one set of contract tests against both stores, skipping the Postgres run when no database is reachable.
5. **Add the v1 API.** Add `/api/v1/documents` (upload, list, detail, status, delete) and `/api/v1/chat/ask` (with `document_ids`, `answer_language` and `insufficient_evidence`). Keep `/documents`, `/ask` and `/search` as thin aliases until the frontend moves over.
6. **Switch the default** to `postgres` once parity tests and evaluation pass. Keep `file` for tests and the offline demo.
7. **Frontend:** keep the vanilla UI working against v1. Build React/Vite/TS in a new `frontend/` folder (Phase 6), and remove the static UI only after the React app reaches feature parity.

---

## 9. Risks for Kyrgyz / Russian / English cross-lingual retrieval

### Measured: offline probe [RUN]

I indexed the 4 Russian sample documents with the real `multilingual-e5-small` model. The model was loaded from the local cache with `HF_HUB_OFFLINE=1`, into a throwaway in-memory index in a scratch script, so no project files were touched. I then compared the top-1 hit and its score for equivalent questions in Russian, English and Kyrgyz. The Kyrgyz questions are my own translations and have not been checked by a native speaker, so treat this as indicative only.

Not recorded — the audit session ended before the probe finished. Will be measured properly in the Kyrgyz baseline task (step 3).

### Structural risks [CODE]

1. **No Kyrgyz evidence at all.** The eval set has 0 Kyrgyz questions, there are no Kyrgyz documents, and `MIN_SCORE=0.75` was calibrated only on Russian and English questions against Russian documents. Kyrgyz is low-resource in e5's training data, so cross-lingual similarity scores are usually lower, and Kyrgyz questions risk being **cut by the threshold before the LLM ever sees them** (false refusals).
2. **One global threshold** for every language pair. Same-language and cross-language score distributions differ, so a single cut-off can't fit both.
3. **Chunking bug:** sentences that start with Ү, Ө or Ң are never split [RUN]. This mainly affects chunk boundaries and overlap in Kyrgyz documents, and can produce chunks up to `max_chars` that cut through the middle of a sentence.
4. **Language detection** is a script and word-list heuristic [RUN]. Kyrgyz typed without ң/ө/ү and without a listed word is detected as Russian, so the answer comes back in Russian. Since Russian and Kyrgyz share the Cyrillic alphabet, script counting cannot tell them apart.
5. **Citation check is weak for Kyrgyz.** Kyrgyz is agglutinative, so 5-character prefix matching both over-matches and under-matches. The stopwords are RU/EN only. Kyrgyz↔Russian pairs with no shared terms are marked `unsupported`, not `cross_language`, because `detect_language` returns the same broad script result for both. It is only logged for now, but any future gating on it would wrongly reject correct Kyrgyz answers.
6. **The model is small:** `e5-small` has 384 dimensions and 12 layers. `multilingual-e5-base`/`-large` or `bge-m3` are the obvious candidates to compare, but only on a real KY/RU/EN eval set. Changing the model means re-embedding everything, which is easy now and harder once pgvector holds production data.
7. **The LLM writes the Kyrgyz answer.** Answer quality depends on the provider's Kyrgyz ability, which is untested here. The source excerpts are shown in their original language, which is correct per the brief.
8. **The `e5` prefix heuristic** is tied to the model name. Swapping to a non-e5 model silently changes how queries are encoded. Make it an explicit setting.

---

## 10. Top 5 next steps (smallest first, one session each)

1. **Repair the dev environment and lint (≈30 min).** Run `venv\Scripts\python -m pip install -e ".[dev]"` so the editable install points at `D:\Projects\python_in30days\src`. Confirm `venv\Scripts\python -m pytest -q` shows 291 passed without `PYTHONPATH`. Run `ruff check . --fix` (3 import errors in `tests/test_core.py`). Delete the stale `src/*.egg-info` folders.
   *This changes the venv, so it needs your OK. I did not do it during the audit.*
2. **Housekeeping and CI.** Commit `DOCUMIND_KG_BRIEF.md` and the screenshots (moved to `docs/screenshots/`, without the root duplicate). Move `clean_text` into `ingestion.py` and remove `core.py`'s exercises, `utils.py`, `src/main.py` and `tests/test_core.py`'s exercise tests. Add a GitHub Actions workflow that runs `ruff` and `pytest` (no model download is needed thanks to `HashingEmbedder`).
3. **Kyrgyz baseline.** Fix the sentence regex to include `ҢӨҮ` (with a test). Add Kyrgyz versions of the 4 sample policies (`data/samples_ky/`). Extend `eval_set.json` with KY→KY, KY→RU, RU→EN and EN→RU questions, plus an `expected_page`/`lang` field. Run `evaluation.py` (free, local) and record the real numbers, including whether `MIN_SCORE` needs to depend on the language pair.
4. **Ingestion completeness.** Add DOCX (python-docx, keeping paragraph and heading structure) and Markdown parsers. Add `language` and `token_count` to `Chunk`, defaulted so existing `chunks.json` files still load. Escape `<source>` delimiters in chunk text, and add an adversarial prompt-injection test.
5. **Phase 1 foundation, no RAG rewiring yet.** Add `DATABASE_URL` to the config, a `db` service in docker-compose (pgvector image), SQLAlchemy 2.x models for `documents` and `document_chunks` (with `vector(384)`, foreign keys, cascading delete and indexes), an Alembic baseline migration, and `GET /api/v1/health` with a database check. Write `docs/ARCHITECTURE.md` and `docs/DATABASE_SCHEMA.md` alongside it. The existing app keeps running on the file store.

---

## Appendix: commands run during the audit

All were read-only.

| Command | Result |
|---|---|
| `ls`, `find` (tree), `cat` of configs and sources, `wc -l` | ✅ |
| `git ls-files`, `git check-ignore -v .env sprint_status.md`, `git log --all -- .env`, `git grep` (secret patterns) | ✅ No secrets; `.env` ignored and never committed |
| `venv\Scripts\python -m pytest -q` | ❌ ModuleNotFoundError (stale editable install) |
| `PYTHONPATH=src venv\Scripts\python -m pytest -q -p no:cacheprovider` | ✅ 291 passed, 0 failed, 0 skipped (32.8 s) |
| `venv\Scripts\ruff check . --no-cache` | ⚠️ 3 errors in `tests/test_core.py` |
| `venv\Scripts\python -m pip list` | ✅ versions in §2 |
| `python src/main.py` | ❌ ImportError (`search_word`) |
| Inline probe of `split_sentence_spans` and `detect_language` | ✅ ran; bugs as listed in §4 |
| Offline e5 cross-lingual probe (scratch script, in-memory index, `HF_HUB_OFFLINE=1`) | see §9 |
