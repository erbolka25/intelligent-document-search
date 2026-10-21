# Intelligent Document Searcher

Semantic search and question answering (RAG) over your own `.txt` and `.pdf` documents.

- **Semantic search**: finds passages by meaning, not exact words. It works across languages, so an English query finds Russian text.
- **Ask**: answers a question using **only** the uploaded documents. Every claim cites a numbered source (file and page or chunk). If the documents don't contain the answer, the reply is "I don't know."
- **Never-dies demo**: without an LLM key, or if the LLM call fails, `/ask` falls back to showing the best-matching passages.

```
upload ─► parse (.txt/.pdf) ─► clean ─► sentence-aware chunks ─► embeddings ─► vector index (FAISS or NumPy) ─► disk
question ─► embedding ─► top-k chunks ─► Claude with numbered sources ─► answer + citations
```

| Component | Choice |
|---|---|
| API | FastAPI + Pydantic, single static HTML page at `/` |
| Embeddings | `intfloat/multilingual-e5-small`: multilingual (incl. Russian), 384-dim, 512-token input |
| Vector index | FAISS `IndexFlatIP` or NumPy brute force (both exact cosine search), saved to `data/index/` |
| LLM | Anthropic Claude (`claude-opus-5-5` by default), key from env only |

## Quick start (Windows, PowerShell, about 10 minutes)

Requires Python 3.12+ and about 2.5 GB of free disk space.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# CPU-only PyTorch (~200 MB download)
pip install torch --index-url https://download.pytorch.org/whl/cpu
# --only-binary avoids pip trying to compile numpy from source on Python 3.13
pip install --only-binary=:all: -e ".[dev]"

# Download the embedding model once (~470 MB). Do this BEFORE the demo, while online.
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"

# Optional: enable LLM answers
Copy-Item .env.example .env
# edit .env and set ANTHROPIC_API_KEY=...

uvicorn document_search.api.app:app --port 8000
```

Open http://localhost:8000. Interactive API docs are at http://localhost:8000/docs.

On Linux/macOS, use `source venv/bin/activate` and `cp .env.example .env`. Everything else is the same.

### Docker

```bash
cp .env.example .env        # optionally set ANTHROPIC_API_KEY
docker compose up --build   # first build downloads ~1.5 GB (torch CPU + model baked into the image)
```

The index and uploads persist in the `app-data` volume.

## API

| Method | Path | Body | Purpose |
|---|---|---|---|
| `GET` | `/health` | – | Status, document/chunk counts, index backend, whether the LLM is enabled |
| `GET` | `/documents` | – | Indexed documents |
| `POST` | `/documents` | multipart `file` | Upload `.txt`/`.pdf` (≤ 10 MB). Re-uploading the same content is detected by hash |
| `POST` | `/search` | `{"query": str, "top_k": 1..20}` | Top-k chunks with cosine score and source |
| `POST` | `/ask` | `{"question": str, "top_k": 1..20}` | Answer with citations (`mode: "rag"`), or passages only (`mode: "retrieval_only"`) |

Errors always look like `{"error": "<code>", "detail": ...}`. Responses never include stack traces.

```bash
curl -F "file=@data/samples/hr_policy.txt" http://localhost:8000/documents

curl -X POST http://localhost:8000/search -H "Content-Type: application/json" \
     -d '{"query": "как часто менять пароль", "top_k": 3}'

curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "Сколько дней ежегодного отпуска?"}'
```

PowerShell equivalents. On Windows, `curl.exe -d '...'` mangles non-ASCII text such as Cyrillic, so send the body as UTF-8 bytes:

```powershell
curl.exe -F "file=@data/samples/hr_policy.txt" http://localhost:8000/documents
Invoke-RestMethod http://localhost:8000/ask -Method Post -ContentType "application/json; charset=utf-8" `
  -Body ([Text.Encoding]::UTF8.GetBytes('{"question": "Сколько дней ежегодного отпуска?"}'))
```

Shape of an `/ask` response (illustrative, abridged):

```json
{
  "mode": "rag",
  "answer": "Ежегодный оплачиваемый отпуск составляет 28 календарных дней [1].",
  "answer_found": true,
  "citations": [{"rank": 1, "filename": "hr_policy.txt", "page": null, "chunk_index": 1, "score": 0.885, "text": "..."}],
  "results": ["... all retrieved chunks ..."]
}
```

## Demo script (5 steps)

The `data/samples/` folder has three fictional company policies (HR, IT security, travel expenses) in Russian.

1. **Upload.** Open http://localhost:8000 and drag the three files from `data/samples/` (or the client's documents) onto the upload area. Point out the chunk counts and that re-uploading a file is detected as a duplicate.
2. **Search by meaning.** On the *Семантический поиск* tab, search `Как часто менять пароль?`. The IT security passage comes first, with its score and source. Then search the same in English, `how often should I change my password`. All top results still come from the Russian IT security policy.
3. **Ask a question.** On the *Задать вопрос* tab, ask `Сколько платят суточные за границей?`. The answer cites `[n]`. Click the citation to jump to the exact passage and file.
4. **Show honesty.** Ask something the documents don't cover, e.g. `Какой бюджет на корпоратив?`. The system says it doesn't know instead of inventing an answer.
5. **Show resilience.** Restart the server: the documents are still indexed. (Optional: remove the API key and restart, and `/ask` still returns the relevant passages in retrieval-only mode.)

## Configuration

All settings come from environment variables or `.env` (see `.env.example`):

| Variable | Default | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | empty | Empty means retrieval-only `/ask` |
| `LLM_MODEL` | `claude-opus-5-5` | |
| `LLM_EFFORT` | `low` | `low`…`max`. Higher means slower but more thorough answers |
| `INDEX_BACKEND` | `faiss` | `faiss` or `numpy`. Each backend keeps its own index under `data/index/<backend>/` |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | Changing it requires re-uploading documents |
| `CHUNK_MAX_CHARS` / `CHUNK_OVERLAP_CHARS` | `500` / `100` | |
| `MAX_UPLOAD_MB` | `10` | |
| `MIN_SCORE` | `0.0` | e5 scores cluster around 0.7–0.9, so tune carefully |
| `DATA_DIR` | `data` | Uploads and index location |

## Development

```powershell
pytest          # no network, no model download: the tests use a hashing embedder and a fake LLM
ruff check .
```

Layout: `src/document_search/` has `ingestion.py` (parse and chunk), `retrieval/` (embeddings and indexes), `generation.py` (prompt, Claude client, citations), `service.py` (orchestration and fallback), `api/` (FastAPI app, schemas, static page), `evaluation.py`.

## Evaluation

```powershell
python -m document_search.evaluation --out data/eval/report.json
```

This command indexes `data/samples/`, runs the 18 questions in `data/eval/eval_set.json` (16 Russian, 2 English) against both backends, and reports Recall@K, Precision@K and MRR. It also runs a separate latency benchmark on random vectors. A chunk counts as relevant if it comes from the expected file and contains the expected text.

Results from one run (`multilingual-e5-small`, 500/100 chunking, 10 chunks; Windows laptop CPU):

| backend | MRR | Recall@1 | Recall@3 | Recall@5 | Precision@1 | Precision@3 | Precision@5 |
|---|---|---|---|---|---|---|---|
| numpy | 0.852 | 0.694 | 0.972 | 0.972 | 0.722 | 0.333 | 0.200 |
| faiss | 0.852 | 0.694 | 0.972 | 0.972 | 0.722 | 0.333 | 0.200 |

Latency on 50,000 random 384-d vectors: numpy 3.6 ms/query, FAISS 4.8 ms/query.

How to read this:
- Both backends do **exact** search, so their quality is identical by design. The choice between them is about speed and operations, not accuracy.
- At this scale, single-query brute force with NumPy is as fast as FAISS `IndexFlatIP`. FAISS pays off with approximate indexes (IVF/HNSW) at millions of vectors, which this demo doesn't need.
- The eval set is tiny and was written by the same person who wrote the sample documents. Treat it as a smoke test, not a benchmark. Precision@3/5 is capped low because most questions have only one relevant chunk.

## Known limitations

- **Scanned PDFs** (images without a text layer) are rejected. There is no OCR.
- **Single process.** The index lives in memory, guarded by a lock. Run one uvicorn worker. This is fine for a demo and for small teams, not for high load.
- **No delete/update endpoint yet.** To reset, stop the server and delete `data/index/`.
- **Small embedding model.** It's fast on CPU, but on close paraphrases (e.g. "поездка" vs "командировка") the right passage sometimes ranks 2nd–3rd instead of 1st. `/ask` sends the top 5 to the LLM, which compensates. A larger model (`multilingual-e5-base`/`-large`) is a one-line change.
- **Sentence splitting is rule-based.** It can split on some abbreviations. This has a minor effect on retrieval.
- **No authentication.** Don't expose it to the internet as is.
