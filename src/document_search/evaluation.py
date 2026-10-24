"""Retrieval evaluation: Recall@K, Precision@K and MRR for each index backend, plus refusals.

Usage:
    python -m document_search.evaluation --docs data/samples --eval-set data/eval/eval_set.json
    python -m document_search.evaluation --llm   # also ask the LLM (uses the API key, costs money)

Relevance is defined by text, not by chunk id, so the eval set survives changes to
chunking: a chunk is relevant if it comes from the given file and contains the
given substring.

Questions marked ``"answerable": false`` have no answer in the documents. They are left
out of the ranking metrics and measure refusals instead: CorrectRefusal is the share of
them that get no answer, FalseRefusal the share of answerable questions that get none.
Without --llm a refusal means no chunk cleared MIN_SCORE; with --llm it means the full
RAG pipeline returned no answer.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from document_search.config import get_settings
from document_search.generation import BaseLLMClient, LLMError, build_llm_client, generate_answer
from document_search.ingestion import Chunk, ingest_document
from document_search.retrieval import (
    BaseEmbedder,
    SentenceTransformerEmbedder,
    create_index,
)
from document_search.retrieval.index import INDEX_BACKENDS


def chunk_key(chunk: Chunk) -> str:
    return f"{chunk.filename}#{chunk.chunk_index}"


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / k


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for rank, key in enumerate(retrieved, 1):
        if key in relevant:
            return 1.0 / rank
    return 0.0


@dataclass
class EvalCase:
    question: str
    relevant: set[str]
    answerable: bool = True


def load_cases(path: Path, chunks: list[Chunk]) -> list[EvalCase]:
    cases = []
    for item in json.loads(path.read_text(encoding="utf-8")):
        if item.get("answerable", True) is False:
            if item.get("relevant"):
                raise ValueError(f"Unanswerable question lists relevant chunks: {item['question']!r}")
            cases.append(EvalCase(item["question"], set(), answerable=False))
            continue
        relevant = {
            chunk_key(c)
            for spec in item["relevant"]
            for c in chunks
            if c.filename == spec["filename"] and spec["contains"] in c.text
        }
        if not relevant:
            raise ValueError(f"No chunk matches the relevance spec for: {item['question']!r}")
        cases.append(EvalCase(item["question"], relevant))
    return cases


def refusal_metrics(refused: list[bool], cases: list[EvalCase]) -> dict:
    """CorrectRefusal over unanswerable cases, FalseRefusal over answerable ones."""
    by_kind = {
        kind: [r for r, c in zip(refused, cases, strict=True) if c.answerable is not kind]
        for kind in (True, False)
    }
    metrics = {}
    if by_kind[True]:
        metrics["CorrectRefusal"] = float(np.mean(by_kind[True]))
    if by_kind[False]:
        metrics["FalseRefusal"] = float(np.mean(by_kind[False]))
    return metrics


def evaluate(
    embedder: BaseEmbedder,
    chunks: list[Chunk],
    cases: list[EvalCase],
    ks: list[int],
    backends: list[str],
    min_score: float | None = None,
) -> dict:
    """Ranking metrics over answerable cases; with ``min_score``, retrieval-level refusals."""
    vectors = embedder.embed_documents([c.text for c in chunks])
    query_vectors = [embedder.embed_query(case.question) for case in cases]
    max_k = max(ks)
    answerable = [c for c in cases if c.answerable]
    report: dict = {
        "chunks": len(chunks),
        "questions": len(answerable),
        "unanswerable": len(cases) - len(answerable),
        "min_score": min_score,
        "backends": {},
    }

    for backend in backends:
        index = create_index(backend)
        index.add(vectors, chunks)
        start = time.perf_counter()
        hits = [index.search(q, max_k) for q in query_vectors]
        elapsed_ms = (time.perf_counter() - start) * 1000
        rankings = [
            [chunk_key(r.chunk) for r in h] for h, c in zip(hits, cases, strict=True) if c.answerable
        ]

        metrics = {
            "MRR": float(
                np.mean(
                    [reciprocal_rank(r, c.relevant) for r, c in zip(rankings, answerable, strict=True)]
                )
            )
        }
        for k in ks:
            metrics[f"Recall@{k}"] = float(
                np.mean(
                    [recall_at_k(r, c.relevant, k) for r, c in zip(rankings, answerable, strict=True)]
                )
            )
            metrics[f"Precision@{k}"] = float(
                np.mean(
                    [
                        precision_at_k(r, c.relevant, k)
                        for r, c in zip(rankings, answerable, strict=True)
                    ]
                )
            )
        not_refused = []
        if min_score is not None:
            # Refused at retrieval level: no chunk clears the threshold, so the LLM never runs.
            refused = [not any(r.score >= min_score for r in h) for h in hits]
            metrics.update(refusal_metrics(refused, cases))
            not_refused = [
                c.question for r, c in zip(refused, cases, strict=True) if not c.answerable and not r
            ]
        report["backends"][backend] = {
            "metrics": metrics,
            "avg_search_ms": elapsed_ms / len(cases),
            "misses_at_1": [
                c.question
                for r, c in zip(rankings, answerable, strict=True)
                if r[:1] and r[0] not in c.relevant
            ],
            "not_refused": not_refused,
        }
    return report


def evaluate_answers(
    llm: BaseLLMClient,
    embedder: BaseEmbedder,
    chunks: list[Chunk],
    cases: list[EvalCase],
    top_k: int,
    min_score: float | None,
) -> dict:
    """Run the full RAG pipeline per question and count refusals (answer_found False)."""
    index = create_index("numpy")
    index.add(embedder.embed_documents([c.text for c in chunks]), chunks)
    scored_cases, refused, errors = [], [], []
    for case in cases:
        results = index.search(embedder.embed_query(case.question), top_k, min_score=min_score)
        try:
            answer = generate_answer(llm, case.question, results)
        except LLMError as exc:
            errors.append({"question": case.question, "error": exc.code})
            continue
        scored_cases.append(case)
        refused.append(not answer.answer_found)
    pairs = list(zip(refused, scored_cases, strict=True))
    return {
        "model": f"{llm.provider}/{llm.model}",
        "metrics": refusal_metrics(refused, scored_cases),
        "not_refused": [c.question for r, c in pairs if not c.answerable and not r],
        "falsely_refused": [c.question for r, c in pairs if c.answerable and r],
        "errors": errors,
    }


def benchmark_latency(backends: list[str], n: int, dim: int, queries: int = 100) -> dict:
    """Search latency on random unit vectors; measures speed only, not quality."""
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(n, dim)).astype(np.float32)
    query_vectors = rng.normal(size=(queries, dim)).astype(np.float32)
    dummy = [Chunk("x", "x", None, i, 0, 0, "") for i in range(n)]
    result = {}
    for backend in backends:
        index = create_index(backend)
        index.add(vectors, dummy)
        index.search(query_vectors[0], 5)  # warm-up
        start = time.perf_counter()
        for q in query_vectors:
            index.search(q, 5)
        result[backend] = (time.perf_counter() - start) * 1000 / queries
    return result


def _print_report(report: dict, ks: list[int]) -> None:
    print(f"\nChunks: {report['chunks']}, questions: {report['questions']}")
    columns = ["MRR"] + [f"{m}@{k}" for k in ks for m in ("Recall", "Precision")]
    print(f"{'backend':8} " + " ".join(f"{c:>12}" for c in columns) + f" {'ms/query':>9}")
    for backend, data in report["backends"].items():
        values = " ".join(f"{data['metrics'][c]:12.3f}" for c in columns)
        print(f"{backend:8} {values} {data['avg_search_ms']:9.3f}")
    for backend, data in report["backends"].items():
        if data["misses_at_1"]:
            print(f"\n[{backend}] top-1 misses:")
            for question in data["misses_at_1"]:
                print(f"  - {question}")
    if report.get("unanswerable") and report.get("min_score") is not None:
        # Exact search: every backend refuses the same questions, so show one.
        data = next(iter(report["backends"].values()))
        _print_refusals(
            f"Retrieval refusals (MIN_SCORE={report['min_score']}, "
            f"{report['unanswerable']} unanswerable questions)",
            data["metrics"],
            data["not_refused"],
        )
    if "llm_refusals" in report:
        llm = report["llm_refusals"]
        _print_refusals(f"RAG refusals with {llm['model']}", llm["metrics"], llm["not_refused"])
        for question in llm["falsely_refused"]:
            print(f"  refused but answerable: {question}")
        for error in llm["errors"]:
            print(f"  LLM error ({error['error']}): {error['question']}")
    if "latency_benchmark" in report:
        bench = report["latency_benchmark"]
        print(f"\nLatency on {bench['n']} random {bench['dim']}-d vectors (ms/query):")
        for backend, ms in bench["ms_per_query"].items():
            print(f"  {backend:8} {ms:.3f}")


def _print_refusals(title: str, metrics: dict, not_refused: list[str]) -> None:
    print(f"\n{title}:")
    for name in ("CorrectRefusal", "FalseRefusal"):
        if name in metrics:
            print(f"  {name:15} {metrics[name]:.3f}")
    for question in not_refused:
        print(f"  not refused: {question}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--docs", type=Path, default=Path("data/samples"))
    parser.add_argument("--eval-set", type=Path, default=Path("data/eval/eval_set.json"))
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument(
        "--bench-n", type=int, default=50_000, help="0 disables the latency benchmark"
    )
    parser.add_argument("--out", type=Path, default=None, help="optional JSON report path")
    parser.add_argument(
        "--llm",
        action="store_true",
        help="also run every question through the configured LLM (one API call each)",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    chunks: list[Chunk] = []
    for path in sorted(args.docs.iterdir()):
        if path.suffix.lower() in {".txt", ".pdf"}:
            doc_chunks = ingest_document(
                path.read_bytes(),
                path.name,
                max_chars=settings.chunk_max_chars,
                overlap_chars=settings.chunk_overlap_chars,
            ).chunks
            chunks.extend(doc_chunks)

    cases = load_cases(args.eval_set, chunks)
    backends = list(INDEX_BACKENDS)
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    report = evaluate(embedder, chunks, cases, args.k, backends, min_score=settings.min_score)
    if args.llm:
        llm = build_llm_client(
            settings.llm_provider,
            settings.llm_api_key(),
            settings.resolved_llm_model,
            settings.llm_effort,
            settings.llm_max_tokens,
            settings.llm_timeout_s,
        )
        if llm is None:
            print("--llm: no LLM provider or API key configured; skipping RAG refusals.")
        else:
            report["llm_refusals"] = evaluate_answers(
                llm, embedder, chunks, cases, settings.top_k, settings.min_score
            )
    report["embedding_model"] = settings.embedding_model
    report["chunking"] = {
        "max_chars": settings.chunk_max_chars,
        "overlap_chars": settings.chunk_overlap_chars,
    }
    if args.bench_n > 0:
        dim = int(embedder.embed_query("dim").shape[0])
        report["latency_benchmark"] = {
            "n": args.bench_n,
            "dim": dim,
            "ms_per_query": benchmark_latency(backends, args.bench_n, dim),
        }

    _print_report(report, args.k)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
