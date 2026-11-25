# Retrieval evaluation: Kyrgyz baseline

- **Date:** 2026-11-24
- **Model:** `intfloat/multilingual-e5-small`, loaded offline (`HF_HUB_OFFLINE=1`)
- **Chunking:** `max_chars=500`, `overlap_chars=100`
- **Threshold:** `MIN_SCORE=0.75`, unchanged
- **Mode:** retrieval only, without `--llm`. A refusal here means that no chunk cleared `MIN_SCORE`.
- **Command:** `python -m document_search.evaluation --bench-n 0`

## Data

| Corpus | Folder | Documents | Chunks |
|---|---|---|---|
| ru | `data/samples` | 4 | 13 |
| en | `data/samples_en` | 4 | 13 |
| ky | `data/samples_ky` | 4 | 13 |

The eval set (`data/eval/eval_set.json`) has 53 questions: 44 answerable and 9 unanswerable. Each question is searched only in its own corpus's index. No questions target the en corpus yet.

## Metrics per scenario (question language → corpus)

| Scenario | Answerable | Unanswerable | Hit@1 | Recall@1 | Recall@3 | Recall@5 | MRR | CorrectRefusal | FalseRefusal |
|---|---|---|---|---|---|---|---|---|---|
| ru→ru | 21 | 4 | 21/21 | 0.952 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| en→ru | 3 | 1 | 2/3 | 0.667 | 0.667 | 0.667 | 0.667 | 0.000 | 0.000 |
| ky→ky | 8 | 4 | 8/8 | 0.938 | 0.938 | 0.938 | 1.000 | 0.000 | 0.000 |
| ky→ru | 8 | 0 | 5/8 | 0.625 | 1.000 | 1.000 | 0.771 | – | 0.000 |
| ru→ky | 4 | 0 | 3/4 | 0.625 | 0.875 | 0.875 | 0.875 | – | 0.000 |
| **overall** | 44 | 9 | 39/44 | 0.841 | 0.955 | 0.955 | 0.924 | 0.000 | 0.000 |

**How to read these columns.** Recall@K is the share of a question's relevant chunks that appear in the top K. When an answer falls in the overlap of two chunks, both chunks count as relevant, so a correct top hit scores only 0.5 at K=1. Hit@1 counts the questions whose top-ranked chunk is relevant.

## Top-1 similarity score

| Scenario | Answerable: min | Answerable: median | Answerable: max | Unanswerable: max |
|---|---|---|---|---|
| ru→ru | 0.803 | 0.846 | 0.896 | 0.842 |
| en→ru | 0.778 | 0.785 | 0.825 | 0.783 |
| ky→ky | 0.800 | 0.846 | 0.868 | 0.854 |
| ky→ru | 0.790 | 0.797 | 0.826 | – |
| ru→ky | 0.751 | 0.788 | 0.808 | – |
| overall | 0.751 | 0.835 | 0.896 | 0.854 |

## Observations

- **Kyrgyz within Kyrgyz works.** All 8 ky→ky questions hit at rank 1, with a median top-1 score of 0.846, the same as ru→ru. That includes the 3 ky→ky questions typed without ң/ө/ү (scores 0.800–0.847).
- **Cross-lingual Kyrgyz is the weak spot.**
  - **ky→ru** has Hit@1 of 5/8 and MRR 0.771. The relevant chunk was at rank 2–3 for the foreign per-diem, travel-to-Osh and sick-leave questions. Recall@3 is still 1.000, so with `top_k ≥ 3` the LLM would see the right passage.
  - **ru→ky** missed "Можно ли разделить отпуск на части?", whose top-1 chunk came from the business-travel document.
- **`MIN_SCORE=0.75` causes no false refusals yet, but only just.** FalseRefusal is 0.000 in every scenario. However, the lowest answerable score is 0.751 (ru→ky), only 0.001 above the threshold. Kyrgyz cross-lingual medians (0.788 ru→ky, 0.797 ky→ru) are about 0.05 lower than same-language medians (0.846). Raising `MIN_SCORE` to reject off-topic questions would refuse Kyrgyz cross-lingual questions first.
- **The threshold refuses none of the unanswerable questions (CorrectRefusal 0.000).** The highest-scoring unanswerable Kyrgyz question (0.854, on maternity leave) scores above the ky→ky median and above every cross-lingual answerable question. No single threshold, and no per-pair threshold, can separate these. Refusal has to stay with the LLM step.
- **Recommended next tests:**
  - ~~Run the same eval with `multilingual-e5-base` and `bge-m3`~~: done, see [Embedding model comparison](#embedding-model-comparison).
  - Only after that, consider per-pair thresholds.
  - Run `--llm` on the Kyrgyz questions to measure real refusals.
  - Add en-corpus questions so the en index gets tested.

## Limitations

- **The dataset is very small.** It has 4 short documents per language and 8 or 4 questions per Kyrgyz scenario, so one question moves Hit@1 by 12–25 points. Treat these numbers as indicative, not as accuracy on real documents.
- **The Kyrgyz texts and questions were written by AI.** One native speaker reviewed them, so wording may still differ from how employees actually write.
- **The questions and documents were written together.** This favours lexical overlap between them.

## Embedding model comparison

- **Date:** 2026-11-25
- **Models:** `intfloat/multilingual-e5-small` (app default, unchanged), `intfloat/multilingual-e5-base`, `BAAI/bge-m3` (dense output only, revision `5617a9f`)
- **Prefixes:** `query: ` / `passage: ` for both e5 models; none for bge-m3
- **Setup:** offline (`HF_HUB_OFFLINE=1`), CPU, sentence-transformers 6.1.0, torch 2.14.1+cpu. Same chunking, same `MIN_SCORE=0.75`, retrieval only, without `--llm`.
- **Eval set:** the same 53 questions. Since 845f03e, the 4 Kyrgyz unanswerable questions are also run against the ru corpus, so there are 13 unanswerable cases (9 before). The e5-small ranking numbers match the baseline above.
- **Commands:**
  ```
  python -m document_search.evaluation --bench-n 0 --model intfloat/multilingual-e5-small
  python -m document_search.evaluation --bench-n 0 --model intfloat/multilingual-e5-base
  python -m document_search.evaluation --bench-n 0 --model BAAI/bge-m3 --query-prefix "" --passage-prefix ""
  ```

### Hit@1 per scenario

| Scenario | n | e5-small | e5-base | bge-m3 |
|---|---|---|---|---|
| ru→ru | 21 | **21/21** | **21/21** | 20/21 |
| en→ru | 3 | 2/3 | **3/3** | **3/3** |
| ky→ky | 8 | **8/8** | **8/8** | **8/8** |
| ky→ru | 8 | 5/8 | 5/8 | **6/8** |
| ru→ky | 4 | 3/4 | **4/4** | **4/4** |
| **overall** | 44 | 39/44 | **41/44** | **41/44** |

### Overall metrics and cost

| | e5-small | e5-base | bge-m3 |
|---|---|---|---|
| Recall@1 / @3 / @5 | 0.841 / 0.955 / 0.955 | 0.875 / 0.977 / 0.977 | 0.875 / 0.977 / **1.000** |
| MRR | 0.924 | 0.951 | **0.960** |
| ky→ru MRR | 0.771 | 0.729 | **0.844** |
| Top-1 score, answerable: min / median | 0.751 / 0.835 | 0.765 / 0.820 | 0.463 / 0.598 |
| Top-1 score, unanswerable: max | 0.854 | 0.848 | 0.614 |
| Separability (share of answerable/unanswerable pairs where the answerable question has the higher top-1 score) | 0.652 | 0.698 | **0.774** |
| FalseRefusal at `MIN_SCORE=0.75` | 0.000 | 0.000 | **1.000** (every question refused) |
| CorrectRefusal at the highest threshold with no false refusals | 0/13 (0.751) | **2/13** (0.765) | 1/13 (0.463) |
| Vector dimension | **384** | 768 | 1024 |
| Model size on disk | **~0.47 GB** | ~1.1 GB | ~2.3 GB |
| Indexing all 39 chunks | **1.1 s** | 3.4 s | 13.2 s |
| Embed + search per question | **19 ms** | 37 ms | 126 ms |

Times are single wall-clock runs on this machine (CPU). Model load time is left out because it depends on the disk cache. The first, cold e5-small load took 83 s, while the warm e5-base and bge-m3 loads took about 20 s.

### Observations

- **Both larger models fix the same-family misses.** e5-base and bge-m3 both get en→ru 3/3 and ru→ky 4/4. That includes "Можно ли разделить отпуск на части?", which e5-small sent to the business-travel document.
- **ky→ru stays the weak spot for every model.** The best result is bge-m3 with 6/8. "Ооруп калсам эмне кылышым керек?" misses with all three models, and "Майрам күнү иштесем, эмне болот?" misses with both larger models. Recall@3 stays at 0.875–1.000, so with `top_k ≥ 3` the LLM still sees the right passage.
- **bge-m3 regresses once on ru→ru.** "Что будет, если я работал в праздник?" is no longer at rank 1, the same holiday-work topic that bge-m3 misses in ky→ru.
- **bge-m3's scores are on a different scale.** Its cosine similarities are about 0.25 lower than e5's, so `MIN_SCORE=0.75` refuses every question. Switching to it would require re-tuning `MIN_SCORE` to about 0.45. Even then, its lowest answerable score (0.463, ky→ru) sits below 6 of the 13 unanswerable scores.
- **No model makes a retrieval threshold a useful refusal filter.** bge-m3 separates answerable from unanswerable best (0.774). But in absolute terms, the best zero-false-refusal threshold rejects only 0–2 of 13 unanswerable questions with any model. Refusal stays with the LLM step, as the baseline concluded.
- **e5-base costs 2× e5-small, bge-m3 costs 6–12×.** That applies to index and query time, and the vector dimension grows by the same order (768 and 1024 vs 384).

### Recommendation

**`intfloat/multilingual-e5-base` is the best candidate to replace e5-small.** This run does not change the default.

- It matches bge-m3 on overall Hit@1 (41/44) and on every non-ky→ru scenario.
- It keeps the e5 prefixes and score scale, so `MIN_SCORE=0.75` still causes no false refusals: its lowest answerable score is 0.765, a wider margin than e5-small's 0.751.
- It is about 3.5× faster per query than bge-m3, half its vector size, and half its disk size.

bge-m3's advantages (ky→ru 6/8, MRR 0.960, Recall@5 1.000) amount to one ky→ru question on this small set. Taking them would also mean a 2.3 GB model and recalibrating the threshold. Revisit bge-m3 if a larger Kyrgyz cross-lingual set confirms the gap.

Switching the default is a separate change:
- set `EMBEDDING_MODEL=intfloat/multilingual-e5-base`;
- re-index the stored documents, since vectors change from 384 to 768 dimensions;
- re-check `MIN_SCORE` with `--llm`.

The same small-dataset limitations listed above apply: one question moves a Kyrgyz scenario's Hit@1 by 12–25 points.
