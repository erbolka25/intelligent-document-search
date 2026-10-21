"""Retrieval evaluation: Recall@K, Precision@K and MRR for each index backend.

Usage:
    python -m document_search.evaluation --docs data/samples --eval-set data/eval/eval_set.json

Relevance is defined by text, not by chunk id, so the eval set survives changes to
chunking: a chunk is relevant if it comes from the given file and contains the
given substring.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from document_search.config import get_settings
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


def load_cases(path: Path, chunks: list[Chunk]) -> list[EvalCase]:
    cases = []
    for item in json.loads(path.read_text(encoding="utf-8")):
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


def evaluate(
    embedder: BaseEmbedder,
    chunks: list[Chunk],
    cases: list[EvalCase],
    ks: list[int],
    backends: list[str],
) -> dict:
    vectors = embedder.embed_documents([c.text for c in chunks])
    query_vectors = [embedder.embed_query(case.question) for case in cases]
    max_k = max(ks)
    report: dict = {"chunks": len(chunks), "questions": len(cases), "backends": {}}

    for backend in backends:
        index = create_index(backend)
        index.add(vectors, chunks)
        start = time.perf_counter()
        rankings = [[chunk_key(r.chunk) for r in index.search(q, max_k)] for q in query_vectors]
        elapsed_ms = (time.perf_counter() - start) * 1000

        metrics = {
            "MRR": float(
                np.mean(
                    [reciprocal_rank(r, c.relevant) for r, c in zip(rankings, cases, strict=True)]
                )
            )
        }
        for k in ks:
            metrics[f"Recall@{k}"] = float(
                np.mean(
                    [recall_at_k(r, c.relevant, k) for r, c in zip(rankings, cases, strict=True)]
                )
            )
            metrics[f"Precision@{k}"] = float(
                np.mean(
                    [precision_at_k(r, c.relevant, k) for r, c in zip(rankings, cases, strict=True)]
                )
            )
        report["backends"][backend] = {
            "metrics": metrics,
            "avg_search_ms": elapsed_ms / len(cases),
            "misses_at_1": [
                c.question
                for r, c in zip(rankings, cases, strict=True)
                if r[:1] and r[0] not in c.relevant
            ],
        }
    return report


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
    if "latency_benchmark" in report:
        bench = report["latency_benchmark"]
        print(f"\nLatency on {bench['n']} random {bench['dim']}-d vectors (ms/query):")
        for backend, ms in bench["ms_per_query"].items():
            print(f"  {backend:8} {ms:.3f}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--docs", type=Path, default=Path("data/samples"))
    parser.add_argument("--eval-set", type=Path, default=Path("data/eval/eval_set.json"))
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument(
        "--bench-n", type=int, default=50_000, help="0 disables the latency benchmark"
    )
    parser.add_argument("--out", type=Path, default=None, help="optional JSON report path")
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
    report = evaluate(embedder, chunks, cases, args.k, backends)
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
