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
  - Run the same eval with `multilingual-e5-base` and `bge-m3`, and compare cross-lingual Hit@1 and the gap between answerable and unanswerable scores.
  - Only after that, consider per-pair thresholds.
  - Run `--llm` on the Kyrgyz questions to measure real refusals.
  - Add en-corpus questions so the en index gets tested.

## Limitations

- **The dataset is very small.** It has 4 short documents per language and 8 or 4 questions per Kyrgyz scenario, so one question moves Hit@1 by 12–25 points. Treat these numbers as indicative, not as accuracy on real documents.
- **The Kyrgyz texts and questions were written by AI.** One native speaker reviewed them, so wording may still differ from how employees actually write.
- **The questions and documents were written together.** This favours lexical overlap between them.
