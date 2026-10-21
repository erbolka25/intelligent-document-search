# Intelligent Document Searcher

Semantic search and question answering (RAG) over your own `.txt` and `.pdf` documents, with a password-protected web UI in Russian and English.

- **Semantic search** finds passages by meaning, not exact words. It works across languages, so an English query finds Russian text.
- **Ask** answers a question using **only** the uploaded documents. Every claim cites a numbered source (file and page or chunk), and clicking `[n]` jumps to that passage. If the documents don't contain the answer, it says so instead of guessing. The answer is written in the language of the question.
- **Never-dies demo**: without an LLM key, or if the LLM call fails, `/ask` falls back to showing the best-matching passages with a short notice.

```
upload ─► parse (.txt/.pdf) ─► clean ─► sentence-aware chunks ─► embeddings ─► vector index (FAISS or NumPy) ─► disk
question ─► embedding ─► top-k chunks ─► LLM (OpenAI or Anthropic) with numbered sources ─► answer + citations
```

| Component | Choice |
|---|---|
| API | FastAPI + Pydantic. Static HTML/JS UI at `/`, sign-in page at `/login` |
| Embeddings | `intfloat/multilingual-e5-small`: multilingual (incl. Russian), 384-dim, runs on CPU |
| Vector index | FAISS `IndexFlatIP` or NumPy brute force (both exact cosine search), saved to `data/index/<backend>/` |
| LLM | OpenAI (default provider) or Anthropic, chosen with `LLM_PROVIDER`. The key comes from `.env` only |
| UI languages | Russian (default) and English, from `src/document_search/api/static/i18n.json` |

## Quick start (Windows, PowerShell)

Requires Python 3.12+ and about 2.5 GB of free disk space. The embedding model (~470 MB) is downloaded once, so do it while online, before the demo. After that, the app loads it from the local cache and needs the internet only for LLM answers.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# CPU-only PyTorch (~200 MB download)
pip install torch --index-url https://download.pytorch.org/whl/cpu
# --only-binary avoids pip trying to compile numpy from source on Python 3.13
pip install --only-binary=:all: -e ".[dev]"

# Download the embedding model once (~470 MB)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"

Copy-Item .env.example .env
# Edit .env: set APP_PASSWORD, SESSION_SECRET and OPENAI_API_KEY (see "Configuration")

uvicorn document_search.api.app:app --port 8000
```

If PowerShell refuses to run `Activate.ps1`, run `Set-ExecutionPolicy -Scope Process Bypass` in the same window first. Alternatively, skip activation and start the server with `.\venv\Scripts\python.exe -m uvicorn document_search.api.app:app --port 8000`.

Open http://localhost:8000 and sign in with `APP_PASSWORD`. The server is ready when the log prints `Application startup complete`, which takes about 35–70 seconds. Start it a few minutes before a demo. Stop the server with `Ctrl+C`. Interactive API docs are at http://localhost:8000/docs (after sign-in).

On Linux/macOS, use `source venv/bin/activate` and `cp .env.example .env`. Everything else is the same.

### Docker

```bash
cp .env.example .env        # then set APP_PASSWORD, SESSION_SECRET and an API key
docker compose up --build   # first build downloads ~1.5 GB (CPU torch + model baked into the image)
```

The index and uploads persist in the `app-data` volume. `data/` is not copied into the image, so upload the demo documents through the UI. The Docker setup was not re-tested after the sign-in and UI changes.

## Using the web UI

- **Upload**: drag `.txt`/`.pdf` files onto the upload area. Each document shows its chunk count and a **×** button that deletes it (after confirmation). Re-uploading the same file is detected as a duplicate.
- **Ask a question / Semantic search**: two tabs. *Ask* returns an answer with clickable `[n]` citations plus the retrieved passages. *Search* returns the closest passages with their cosine similarity.
- **Header**: status pills (documents, chunks, index, whether the AI is connected), language (Русский / English), theme (system / light / dark) and **Sign out**. Language and theme are remembered in the browser.

## Demo script (5 steps)

`data/samples/` holds four fictional policies of a fictional company, ОсОО «Ала-Тоо Сервис», in Russian, with amounts in som: working time and vacation, information security, business trips, and benefits and training.

1. **Upload.** Drag the four files from `data/samples/` (or the client's documents) onto the upload area. Point out the chunk counts. Upload one file again to show duplicate detection.
2. **Search by meaning.** On the *Семантический поиск* tab, search `Как часто менять пароль?`. The IT security passage comes first. Then search `how often should I change my password`: the same Russian passage is still on top.
3. **Ask a question.** On the *Задать вопрос* tab, ask `Сколько платят суточные за границей?`. The answer (3 500 сом в день) cites `[n]`. Click the citation to jump to the passage.
4. **Show honesty and languages.** Ask `Какой бюджет на корпоратив?`: the system says the documents don't contain this. Ask `How much is the daily allowance for trips inside Kyrgyzstan?`: the answer comes in English from the Russian documents. Switch the UI to English to show the interface translation.
5. **Show resilience.** Restart the server: the documents are still indexed. Optionally remove the API key and restart: `/ask` still returns the relevant passages, with a notice that AI answers are off.

## API

Everything except `/login`, `/auth/login`, `/auth/logout`, `/health` and `/static/*` needs a session cookie from `/auth/login`. Without it, API calls return `401` and pages redirect to `/login`.

| Method | Path | Body | Purpose |
|---|---|---|---|
| `POST` | `/auth/login` | `{"password": str}` | Sign in. Sets an HttpOnly session cookie |
| `POST` | `/auth/logout` | – | Sign out. Revokes the session |
| `GET` | `/health` | – | Liveness probe: `{"status": "ok"}` |
| `GET` | `/stats` | – | Document/chunk counts, index backend, document limit, whether the LLM is enabled, provider and model |
| `GET` | `/documents` | – | Indexed documents: `doc_id`, `filename`, `chunks` |
| `POST` | `/documents` | multipart `file` | Upload `.txt`/`.pdf` (≤ `MAX_UPLOAD_MB`). Returns `chunks`, `duplicate` and `skipped_pages` (PDF pages without readable text) |
| `DELETE` | `/documents/{doc_id}` | – | Delete one document from the index and disk |
| `DELETE` | `/documents` | – | Delete all documents |
| `POST` | `/search` | `{"query": str, "top_k": 1..20}` | Top-k chunks with cosine score and source |
| `POST` | `/ask` | `{"question": str, "top_k": 1..20}` | Answer with citations (`mode: "rag"`), or passages only (`mode: "retrieval_only"` plus a `notice` code) |

Errors always look like `{"error": "<code>", "message": "<localized text>"}`; `422` responses add `"fields"`. Responses never include stack traces. Unexpected errors return `500` with a request ID that matches the server log. The message language comes from `?lang=`, then the `X-Lang` header, then `Accept-Language`, and defaults to Russian.

With curl (bash on Linux/macOS; on Windows, Git Bash and `curl.exe` send Cyrillic in `-d` in the wrong encoding, and the server answers `400 bad_request`):

```bash
curl -c cookies.txt -H "Content-Type: application/json" \
     -d '{"password": "YOUR_PASSWORD"}' http://localhost:8000/auth/login

curl -b cookies.txt -F "file=@data/samples/komandirovki.txt" http://localhost:8000/documents

curl -b cookies.txt -H "Content-Type: application/json" \
     -d '{"query": "как часто менять пароль", "top_k": 3}' http://localhost:8000/search

curl -b cookies.txt -H "Content-Type: application/json" \
     -d '{"question": "Сколько платят суточные за границей?"}' http://localhost:8000/ask
```

PowerShell (tested on Windows PowerShell 5.1). Send the body as UTF-8 bytes so Cyrillic survives:

```powershell
Invoke-RestMethod http://localhost:8000/auth/login -Method Post -ContentType "application/json" `
  -Body '{"password": "YOUR_PASSWORD"}' -SessionVariable s
Invoke-RestMethod http://localhost:8000/ask -Method Post -WebSession $s -ContentType "application/json; charset=utf-8" `
  -Body ([Text.Encoding]::UTF8.GetBytes('{"question": "Сколько платят суточные за границей?"}'))
```

Shape of an `/ask` response (abridged):

```json
{
  "question": "Сколько платят суточные за границей?",
  "mode": "rag",
  "answer": "Суточные при командировках за границу составляют 3 500 сом в день [1].",
  "answer_found": true,
  "citations": [{"rank": 1, "doc_id": "…", "filename": "komandirovki.txt", "page": null, "chunk_index": 0, "score": 0.846, "text": "…"}],
  "results": ["… all retrieved chunks …"],
  "notice": null,
  "notice_message": null
}
```

## Configuration

All settings come from environment variables or `.env` in the project root (see `.env.example`). `.env` is git-ignored. Never commit it.

| Variable | Default | Notes |
|---|---|---|
| `APP_PASSWORD` | empty | **Required.** Shared sign-in password. If empty, nobody can sign in (`/auth/login` returns 503) |
| `SESSION_SECRET` | random per start | Signs session cookies. Set a long random string, e.g. `python -c "import secrets; print(secrets.token_urlsafe(32))"`. If unset, every restart signs everyone out |
| `SESSION_MAX_AGE_S` | `28800` | Session lifetime (8 h) |
| `COOKIE_SECURE` | `auto` | `auto` marks the cookie Secure when the request came over HTTPS |
| `LLM_PROVIDER` | `openai` | `openai`, `anthropic` or `none` (`none` = passages only) |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | empty | The key for the chosen provider. Missing key = passages only |
| `LLM_MODEL` | provider default | Empty means `gpt-6-luna` (OpenAI) or `claude-opus-5-5` (Anthropic). The demo was tested with `gpt-4o-mini` |
| `LLM_EFFORT` | `low` | `low`…`max`, for models that support reasoning effort. Higher means slower but more thorough |
| `LLM_MAX_TOKENS` / `LLM_TIMEOUT_S` | `4000` / `60` | |
| `LOGIN_RATE_PER_MIN` | `5` | Sign-in attempts per IP per minute |
| `ASK_RATE_PER_MIN` | `20` | `/ask` and `/search` calls per session per minute |
| `UPLOAD_RATE_PER_MIN` | `30` | Uploads and deletes per session per minute |
| `INDEX_BACKEND` | `faiss` | `faiss` or `numpy`. Each backend keeps its own index under `data/index/<backend>/` |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | Changing it requires re-uploading documents |
| `TOP_K` / `MIN_SCORE` | `5` / `0.0` | e5 scores cluster around 0.7–0.9, so tune `MIN_SCORE` carefully |
| `MAX_UPLOAD_MB` / `MAX_DOCUMENTS` | `10` / `50` | |
| `CHUNK_MAX_CHARS` / `CHUNK_OVERLAP_CHARS` | `500` / `100` | |
| `DATA_DIR` | `data` | Uploads and index location |
| `LOG_LEVEL` | `INFO` | Logs are JSON lines, one per request |

## Security

- One shared password per installation. Sessions are random IDs in a signed, HttpOnly, SameSite=Lax cookie and are revoked on sign-out.
- In-memory rate limits for sign-in, questions and uploads.
- Uploads: extension allow-list (`.txt`, `.pdf`), size limit checked before the body is read, content check (`%PDF-` header, no binary in text files), sanitized filenames. Files are stored under a content hash, never under the user's filename.
- Security headers on every response: Content-Security-Policy (no inline scripts or styles), `X-Frame-Options: DENY`, `nosniff`. HTML pages are sent with `Cache-Control: no-store`.
- The LLM prompt treats document text as untrusted and tells the model to ignore instructions found inside it.
- There is no HTTPS in the app itself. To expose it beyond your own machine, put it behind a reverse proxy with TLS and run uvicorn with `--proxy-headers`.

## Adding a UI language

All UI, error and notice texts are in `src/document_search/api/static/i18n.json`, one top-level block per language (`"ru"`, `"en"`). To add Kyrgyz, copy the `"en"` block as `"ky"`, set `"meta": {"name": "Кыргызча"}` and translate the values. Keep the `{placeholders}` as they are. The language appears in the switcher automatically. `pytest tests/test_i18n.py` checks that the new block has exactly the same keys and placeholders as Russian.

## Development

```powershell
pytest          # no network, no model download: the tests use a hashing embedder and a fake LLM
ruff check .
```

Layout:

- `src/document_search/`
  - `config.py`: settings from `.env`.
  - `ingestion.py`: parse, clean and chunk.
  - `retrieval/`: embeddings and indexes.
  - `generation.py`: prompt, OpenAI and Anthropic clients, citations, question-language detection.
  - `service.py`: orchestration and fallback.
  - `i18n.py`: server-side translations.
  - `evaluation.py`
  - `api/`: FastAPI app, auth, rate limiter, schemas, static UI.

## Evaluation

```powershell
python -m document_search.evaluation --out data/eval/report.json
```

This command indexes `data/samples/` and runs the 24 questions in `data/eval/eval_set.json` (21 Russian, 3 English) against both backends. It reports Recall@K, Precision@K and MRR. It also runs a separate latency benchmark on random vectors. A chunk counts as relevant if it comes from the expected file and contains the expected text. This measures **retrieval only**, not the quality of LLM answers.

Results from one run (`multilingual-e5-small`, 500/100 chunking, 13 chunks; Windows laptop CPU):

| backend | MRR | Recall@1 | Recall@3 | Recall@5 | Precision@1 | Precision@3 | Precision@5 |
|---|---|---|---|---|---|---|---|
| numpy | 0.958 | 0.917 | 0.958 | 0.958 | 0.958 | 0.347 | 0.208 |
| faiss | 0.958 | 0.917 | 0.958 | 0.958 | 0.958 | 0.347 | 0.208 |

Latency on 50,000 random 384-d vectors: NumPy 6.1 ms/query, FAISS 8.6 ms/query.

How to read this:

- Both backends do **exact** search, so their quality is identical by design. The choice between them is about speed and operations, not accuracy.
- At this scale, brute force with NumPy is as fast as FAISS `IndexFlatIP`. FAISS pays off with approximate indexes (IVF/HNSW) at millions of vectors, which this demo doesn't need.
- The eval set is tiny and was written together with the sample documents. Treat it as a smoke test, not as a promise of accuracy on a client's documents. Precision@3/5 is capped low because most questions have only one relevant chunk.

## Known limitations

- **Scanned PDFs** (images without a text layer) are rejected. There is no OCR. Pages with broken fonts are skipped, and the upload result lists them.
- **One process, one shared password.** The index, sessions and rate limits live in memory. Run a single uvicorn worker. A restart keeps documents but signs everyone out if `SESSION_SECRET` is unset. This is fine for a demo and small teams, not for high load or per-user accounts.
- **No HTTPS** in the app itself (see Security).
- **UI languages**: Russian and English only. Kyrgyz is planned, as one more block in `i18n.json`.
- **Answer language** is detected from the question's alphabet (Latin → English, Cyrillic → Russian, ң/ө/ү → Kyrgyz). Mixed-language questions follow whichever alphabet dominates.
- **Small embedding model.** It's fast on CPU, but on close paraphrases the right passage sometimes ranks 2nd–3rd instead of 1st. `/ask` sends the top 5 to the LLM, which compensates. A larger model (`multilingual-e5-base`/`-large`) is a one-line change, after which documents must be re-uploaded.
- **Sentence splitting is rule-based.** It can split on some abbreviations, and a chunk may end with a dangling section number such as `3.`. The effect on retrieval is minor.
- **Startup takes about 35–70 s** on a laptop CPU. Most of it is importing PyTorch/transformers (~40 s). Loading the model itself takes ~8 s. After the first download the model loads from the local cache with no network calls, so search works offline. Without internet, `/ask` shows passages with a "could not reach the AI service" notice.
