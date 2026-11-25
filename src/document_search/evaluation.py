"""Retrieval evaluation: Recall@K, Precision@K and MRR for each index backend, plus refusals.

Usage:
    python -m document_search.evaluation --docs data/samples --eval-set data/eval/eval_set.json
    python -m document_search.evaluation --llm   # also ask the LLM (uses the API key, costs money)

Relevance is defined by text, not by chunk id, so the eval set survives changes to
chunking: a chunk is relevant if it comes from the given file and contains the
given substring.

Each question has a language (``q_lang``) and a target corpus (``corpus``: ru, en or
ky, defaulting to ru). Every corpus folder gets its own index, and metrics are also
reported per scenario ``q_lang->corpus`` (e.g. ``ky->ru``) with the distribution of
top-1 similarity scores, to check whether one MIN_SCORE fits every language pair.
The backend comparison and --llm run on the ru corpus questions only.

Questions marked ``"answerable": false`` have no answer in the documents. They are left
out of the ranking metrics and measure refusals instead: CorrectRefusal is the share of
them that get no answer, FalseRefusal the share of answerable questions that get none.
Without --llm a refusal means no chunk cleared MIN_SCORE; with --llm it means the full
RAG pipeline returned no answer.
"""

from __future__ import annotations

import argparse
import json
import sys
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


DEFAULT_CORPUS = "ru"
DEFAULT_CORPORA = {
    "ru": Path("data/samples"),
    "en": Path("data/samples_en"),
    "ky": Path("data/samples_ky"),
}


@dataclass
class EvalCase:
    question: str
    relevant: set[str]
    answerable: bool = True
    q_lang: str = DEFAULT_CORPUS
    corpus: str = DEFAULT_CORPUS

    @property
    def scenario(self) -> str:
        """Question language -> corpus language, e.g. ``ky->ru``."""
        return f"{self.q_lang}->{self.corpus}"


def load_cases(path: Path, chunks: list[Chunk] | dict[str, list[Chunk]]) -> list[EvalCase]:
    """Load the eval set. ``chunks`` is one corpus (a list, taken as ``ru``) or
    a mapping corpus -> chunks; each question is matched against its own corpus.
    Entries without ``q_lang``/``corpus`` default to ``ru``. An unanswerable question
    may list several corpora (``"corpus": ["ky", "ru"]``): it then becomes one case per
    corpus, i.e. one per scenario."""
    by_corpus = chunks if isinstance(chunks, dict) else {DEFAULT_CORPUS: chunks}
    cases = []
    for item in json.loads(path.read_text(encoding="utf-8")):
        q_lang = item.get("q_lang", DEFAULT_CORPUS)
        corpus = item.get("corpus", DEFAULT_CORPUS)
        if item.get("answerable", True) is False:
            if item.get("relevant"):
                raise ValueError(
                    f"Unanswerable question lists relevant chunks: {item['question']!r}"
                )
            for name in corpus if isinstance(corpus, list) else [corpus]:
                cases.append(EvalCase(item["question"], set(), False, q_lang, name))
            continue
        targets = corpus if isinstance(corpus, list) else [corpus]
        if len(targets) != 1:
            raise ValueError(f"An answerable question targets one corpus: {item['question']!r}")
        corpus = targets[0]
        if corpus not in by_corpus:
            raise ValueError(f"Unknown corpus {corpus!r} for: {item['question']!r}")
        relevant = {
            chunk_key(c)
            for spec in item["relevant"]
            for c in by_corpus[corpus]
            if c.filename == spec["filename"] and spec["contains"] in c.text
        }
        if not relevant:
            raise ValueError(f"No chunk matches the relevance spec for: {item['question']!r}")
        cases.append(EvalCase(item["question"], relevant, True, q_lang, corpus))
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
            [chunk_key(r.chunk) for r in h]
            for h, c in zip(hits, cases, strict=True)
            if c.answerable
        ]

        metrics = {
            "MRR": float(
                np.mean(
                    [
                        reciprocal_rank(r, c.relevant)
                        for r, c in zip(rankings, answerable, strict=True)
                    ]
                )
            )
        }
        for k in ks:
            metrics[f"Recall@{k}"] = float(
                np.mean(
                    [
                        recall_at_k(r, c.relevant, k)
                        for r, c in zip(rankings, answerable, strict=True)
                    ]
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
                c.question
                for r, c in zip(refused, cases, strict=True)
                if not c.answerable and not r
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


def _score_stats(scores: list[float]) -> dict | None:
    if not scores:
        return None
    return {
        "min": float(np.min(scores)),
        "median": float(np.median(scores)),
        "max": float(np.max(scores)),
    }


def _group_metrics(rows: list[dict], ks: list[int], min_score: float | None) -> dict:
    """Ranking, refusal and top-1 score metrics for one group of per-question rows."""
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    metrics: dict = {}
    if answerable:
        metrics["MRR"] = float(np.mean([r["rr"] for r in answerable]))
        for k in ks:
            metrics[f"Recall@{k}"] = float(np.mean([r[f"recall@{k}"] for r in answerable]))
    if min_score is not None:
        cases = [EvalCase("", set(), r["answerable"]) for r in rows]
        metrics.update(refusal_metrics([r["refused"] for r in rows], cases))
    return {
        "questions": len(answerable),
        "unanswerable": len(unanswerable),
        "metrics": metrics,
        "top1_score_answerable": _score_stats([r["top1_score"] for r in answerable]),
        "top1_score_unanswerable_max": (
            max(r["top1_score"] for r in unanswerable) if unanswerable else None
        ),
    }


def evaluate_scenarios(
    embedder: BaseEmbedder,
    chunks_by_corpus: dict[str, list[Chunk]],
    cases: list[EvalCase],
    ks: list[int],
    min_score: float | None = None,
) -> dict:
    """Search every question in its own corpus' index and report metrics per
    scenario (``q_lang->corpus``) and overall, plus top-1 score distributions,
    so one MIN_SCORE can be checked against every language pair."""
    max_k = max(ks)
    indexes = {}
    start = time.perf_counter()
    for corpus, chunks in chunks_by_corpus.items():
        if chunks:
            index = create_index("numpy")
            index.add(embedder.embed_documents([c.text for c in chunks]), chunks)
            indexes[corpus] = index
    index_s = time.perf_counter() - start

    rows = []
    query_s = 0.0
    for case in cases:
        if case.corpus not in indexes:
            raise ValueError(f"No documents for corpus {case.corpus!r}: {case.question!r}")
        start = time.perf_counter()
        hits = indexes[case.corpus].search(embedder.embed_query(case.question), max_k)
        query_s += time.perf_counter() - start
        ranking = [chunk_key(h.chunk) for h in hits]
        row = {
            "question": case.question,
            "scenario": case.scenario,
            "answerable": case.answerable,
            "top1": ranking[0] if ranking else None,
            "top1_score": float(hits[0].score) if hits else 0.0,
            "refused": min_score is not None and not any(h.score >= min_score for h in hits),
        }
        if case.answerable:
            row["rr"] = reciprocal_rank(ranking, case.relevant)
            for k in ks:
                row[f"recall@{k}"] = recall_at_k(ranking, case.relevant, k)
            row["hit_at_1"] = bool(ranking[:1]) and ranking[0] in case.relevant
        rows.append(row)

    scenarios = sorted({r["scenario"] for r in rows})
    return {
        "min_score": min_score,
        "corpus_chunks": {c: len(ch) for c, ch in chunks_by_corpus.items()},
        # Wall-clock on this machine: embedding every corpus, and embed + search per question.
        "timing": {
            "index_all_corpora_s": index_s,
            "query_mean_ms": query_s * 1000 / len(cases) if cases else 0.0,
        },
        "scenarios": {
            s: _group_metrics([r for r in rows if r["scenario"] == s], ks, min_score)
            for s in scenarios
        },
        "overall": _group_metrics(rows, ks, min_score),
        "cases": rows,
    }


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


def use_utf8_output() -> None:
    """Print UTF-8 regardless of the console code page: on a cp1251 Windows console
    or pipe, Kyrgyz letters (ң ө ү) would otherwise raise UnicodeEncodeError."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def load_corpus(folder: Path, max_chars: int, overlap_chars: int) -> list[Chunk]:
    """Chunk every .txt/.pdf in ``folder``; a missing folder is an empty corpus."""
    if not folder.is_dir():
        return []
    chunks: list[Chunk] = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() in {".txt", ".pdf"}:
            chunks.extend(
                ingest_document(
                    path.read_bytes(),
                    path.name,
                    max_chars=max_chars,
                    overlap_chars=overlap_chars,
                ).chunks
            )
    return chunks


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
    if "by_scenario" in report:
        _print_scenarios(report["by_scenario"], ks)
    if "latency_benchmark" in report:
        bench = report["latency_benchmark"]
        print(f"\nLatency on {bench['n']} random {bench['dim']}-d vectors (ms/query):")
        for backend, ms in bench["ms_per_query"].items():
            print(f"  {backend:8} {ms:.3f}")


def _print_scenarios(data: dict, ks: list[int]) -> None:
    def fmt(value: float | None) -> str:
        return f"{value:.3f}" if value is not None else "-"

    sizes = ", ".join(f"{c}={n}" for c, n in data["corpus_chunks"].items())
    print(
        f"\nPer scenario (question lang -> corpus), chunks: {sizes}, MIN_SCORE={data['min_score']}"
    )
    columns = [f"Recall@{k}" for k in ks] + ["MRR", "CorrectRefusal", "FalseRefusal"]
    print(f"{'scenario':10} {'n':>3} {'unans':>5} " + " ".join(f"{c:>14}" for c in columns))
    groups = {**data["scenarios"], "overall": data["overall"]}
    for name, group in groups.items():
        values = " ".join(f"{fmt(group['metrics'].get(c)):>14}" for c in columns)
        print(f"{name:10} {group['questions']:3} {group['unanswerable']:5} {values}")
    print("\nTop-1 similarity score:")
    print(f"{'scenario':10} {'ans min':>8} {'median':>8} {'max':>8} {'unans max':>10}")
    for name, group in groups.items():
        stats = group["top1_score_answerable"] or {}
        print(
            f"{name:10} {fmt(stats.get('min')):>8} {fmt(stats.get('median')):>8} "
            f"{fmt(stats.get('max')):>8} {fmt(group['top1_score_unanswerable_max']):>10}"
        )


def _print_refusals(title: str, metrics: dict, not_refused: list[str]) -> None:
    print(f"\n{title}:")
    for name in ("CorrectRefusal", "FalseRefusal"):
        if name in metrics:
            print(f"  {name:15} {metrics[name]:.3f}")
    for question in not_refused:
        print(f"  not refused: {question}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--docs", type=Path, default=DEFAULT_CORPORA["ru"], help="the ru corpus folder"
    )
    parser.add_argument(
        "--corpus",
        action="append",
        default=[],
        metavar="LANG=DIR",
        help="corpus folder per language (repeatable); defaults: "
        + ", ".join(f"{lang}={path.as_posix()}" for lang, path in DEFAULT_CORPORA.items()),
    )
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
    parser.add_argument(
        "--model", default=None, help="embedding model to evaluate (default: EMBEDDING_MODEL)"
    )
    parser.add_argument(
        "--query-prefix",
        default=None,
        help='query prefix for --model (default: EMBEDDING_QUERY_PREFIX); "" for none',
    )
    parser.add_argument(
        "--passage-prefix",
        default=None,
        help='passage prefix for --model (default: EMBEDDING_PASSAGE_PREFIX); "" for none',
    )
    args = parser.parse_args(argv)
    use_utf8_output()

    settings = get_settings()
    # Overrides apply to this evaluation run only; the app keeps its configured model.
    model = args.model or settings.embedding_model
    query_prefix = (
        settings.embedding_query_prefix if args.query_prefix is None else args.query_prefix
    )
    passage_prefix = (
        settings.embedding_passage_prefix if args.passage_prefix is None else args.passage_prefix
    )
    corpus_dirs = {**DEFAULT_CORPORA, DEFAULT_CORPUS: args.docs}
    for spec in args.corpus:
        lang, sep, folder = spec.partition("=")
        if not sep or not lang or not folder:
            parser.error(f"--corpus expects LANG=DIR, got {spec!r}")
        corpus_dirs[lang] = Path(folder)
    chunks_by_corpus = {
        lang: load_corpus(folder, settings.chunk_max_chars, settings.chunk_overlap_chars)
        for lang, folder in corpus_dirs.items()
    }

    all_cases = load_cases(args.eval_set, chunks_by_corpus)
    # Backend comparison and --llm run on the ru corpus, as before the per-language split.
    chunks = chunks_by_corpus[DEFAULT_CORPUS]
    cases = [c for c in all_cases if c.corpus == DEFAULT_CORPUS]
    backends = list(INDEX_BACKENDS)
    embedder = SentenceTransformerEmbedder(
        model, query_prefix=query_prefix, passage_prefix=passage_prefix
    )
    start = time.perf_counter()
    dim = int(embedder.embed_query("warm-up").shape[0])  # loads the model
    model_load_s = time.perf_counter() - start
    report = evaluate(embedder, chunks, cases, args.k, backends, min_score=settings.min_score)
    report["by_scenario"] = evaluate_scenarios(
        embedder, chunks_by_corpus, all_cases, args.k, min_score=settings.min_score
    )
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
    report["embedding_model"] = model
    report["embedding_prefixes"] = {"query": query_prefix, "passage": passage_prefix}
    report["embedding_dim"] = dim
    report["model_load_s"] = model_load_s
    report["chunking"] = {
        "max_chars": settings.chunk_max_chars,
        "overlap_chars": settings.chunk_overlap_chars,
    }
    if args.bench_n > 0:
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
