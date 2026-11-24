import json

import pytest

from document_search.evaluation import (
    DEFAULT_CORPORA,
    EvalCase,
    benchmark_latency,
    evaluate,
    evaluate_answers,
    evaluate_scenarios,
    load_cases,
    load_corpus,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from document_search.ingestion import Chunk

RETRIEVED = ["a", "b", "c", "d"]


def test_recall_at_k():
    assert recall_at_k(RETRIEVED, {"b", "z"}, 1) == 0.0
    assert recall_at_k(RETRIEVED, {"b", "z"}, 2) == 0.5
    assert recall_at_k(RETRIEVED, {"a", "c"}, 3) == 1.0
    assert recall_at_k(RETRIEVED, set(), 3) == 0.0


def test_precision_at_k():
    assert precision_at_k(RETRIEVED, {"a", "c"}, 1) == 1.0
    assert precision_at_k(RETRIEVED, {"a", "c"}, 4) == 0.5
    assert precision_at_k(RETRIEVED, {"a"}, 0) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(RETRIEVED, {"a"}) == 1.0
    assert reciprocal_rank(RETRIEVED, {"c", "d"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(RETRIEVED, {"z"}) == 0.0


CHUNKS = [
    Chunk("1", "hr.txt", None, 0, 0, 0, "Отпуск 28 календарных дней."),
    Chunk("1", "hr.txt", None, 1, 0, 0, "Удалённая работа два дня в неделю."),
    Chunk("2", "it.txt", None, 0, 0, 0, "Пароль менять каждые 90 дней."),
]


def test_load_cases_matches_by_text(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(
        json.dumps(
            [{"question": "q", "relevant": [{"filename": "it.txt", "contains": "90 дней"}]}]
        ),
        encoding="utf-8",
    )
    assert load_cases(path, CHUNKS)[0].relevant == {"it.txt#0"}


def test_load_cases_rejects_stale_spec(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(
        json.dumps([{"question": "q", "relevant": [{"filename": "it.txt", "contains": "нет"}]}]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_cases(path, CHUNKS)


def test_evaluate_both_backends_agree(embedder):
    pytest.importorskip("faiss")
    cases = [
        EvalCase("пароль менять 90 дней", {"it.txt#0"}),
        EvalCase("отпуск 28 дней", {"hr.txt#0"}),
    ]
    report = evaluate(embedder, CHUNKS, cases, [1, 3], ["numpy", "faiss"])
    numpy_metrics = report["backends"]["numpy"]["metrics"]
    assert numpy_metrics == report["backends"]["faiss"]["metrics"]
    assert numpy_metrics["MRR"] == 1.0 and numpy_metrics["Recall@1"] == 1.0


def test_load_cases_accepts_unanswerable_questions(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(
        json.dumps(
            [
                {"question": "q", "relevant": [{"filename": "it.txt", "contains": "90 дней"}]},
                {"question": "бюджет на корпоратив?", "answerable": False, "relevant": []},
            ]
        ),
        encoding="utf-8",
    )
    cases = load_cases(path, CHUNKS)
    assert [c.answerable for c in cases] == [True, False]
    assert cases[1].relevant == set()


def test_load_cases_rejects_unanswerable_with_relevant_spec(tmp_path):
    path = tmp_path / "eval.json"
    spec = [{"filename": "it.txt", "contains": "90 дней"}]
    path.write_text(
        json.dumps([{"question": "q", "answerable": False, "relevant": spec}]), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        load_cases(path, CHUNKS)


MIXED_CASES = [
    EvalCase("пароль менять 90 дней", {"it.txt#0"}),
    EvalCase("отпуск 28 дней", {"hr.txt#0"}),
    EvalCase("квантовая хромодинамика", set(), answerable=False),
]


def test_evaluate_scores_refusals_and_keeps_ranking_on_answerable(embedder):
    report = evaluate(embedder, CHUNKS, MIXED_CASES, [1], ["numpy"], min_score=0.3)
    data = report["backends"]["numpy"]
    assert report["questions"] == 2 and report["unanswerable"] == 1
    # the unanswerable case does not drag the ranking metrics down
    assert data["metrics"]["MRR"] == 1.0
    assert data["metrics"]["CorrectRefusal"] == 1.0
    assert data["metrics"]["FalseRefusal"] == 0.0
    assert data["not_refused"] == []


def test_evaluate_without_min_score_reports_no_refusals(embedder):
    data = evaluate(embedder, CHUNKS, MIXED_CASES, [1], ["numpy"])["backends"]["numpy"]
    assert "CorrectRefusal" not in data["metrics"] and data["not_refused"] == []


def test_evaluate_answers_counts_rag_refusals(embedder):
    from tests.test_generation import FakeLLM

    # Answers every question with a citation, so only empty retrieval can refuse.
    report = evaluate_answers(FakeLLM("Ответ [1]."), embedder, CHUNKS, MIXED_CASES, 3, 0.3)
    assert report["metrics"] == {"CorrectRefusal": 1.0, "FalseRefusal": 0.0}
    assert report["not_refused"] == [] and report["falsely_refused"] == []

    # A model that never cites anything refuses everything, answerable or not.
    report = evaluate_answers(FakeLLM("Без ссылок."), embedder, CHUNKS, MIXED_CASES, 3, 0.0)
    assert report["metrics"] == {"CorrectRefusal": 1.0, "FalseRefusal": 1.0}
    assert len(report["falsely_refused"]) == 2


def test_evaluate_answers_records_llm_errors(embedder):
    from document_search.generation import LLMError
    from tests.test_generation import FakeLLM

    llm = FakeLLM(error=LLMError("quota", "no credits"))
    report = evaluate_answers(llm, embedder, CHUNKS, MIXED_CASES, 3, 0.0)
    assert [e["error"] for e in report["errors"]] == ["quota"] * 3
    assert report["metrics"] == {}


def test_benchmark_latency_runs():
    result = benchmark_latency(["numpy"], n=100, dim=8, queries=3)
    assert result["numpy"] >= 0


KY_CHUNKS = [
    Chunk("3", "hr_ky.txt", None, 0, 0, 0, "Өргүү 28 календардык күндү түзөт."),
    Chunk("3", "hr_ky.txt", None, 1, 0, 0, "Сырсөздү ар 90 күн сайын алмаштыруу керек."),
]
BY_CORPUS = {"ru": CHUNKS, "ky": KY_CHUNKS}


def test_eval_case_defaults_to_ru_scenario():
    assert EvalCase("q", set()).scenario == "ru->ru"
    assert EvalCase("q", set(), q_lang="ky", corpus="ru").scenario == "ky->ru"


def test_load_cases_matches_each_question_in_its_own_corpus(tmp_path):
    path = tmp_path / "eval.json"
    items = [
        # old-style entry without q_lang/corpus -> ru
        {"question": "q1", "relevant": [{"filename": "it.txt", "contains": "90 дней"}]},
        {
            "question": "q2",
            "q_lang": "ky",
            "corpus": "ky",
            "relevant": [{"filename": "hr_ky.txt", "contains": "90 күн"}],
        },
        {"question": "q3", "q_lang": "ky", "corpus": "ky", "answerable": False, "relevant": []},
    ]
    path.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    cases = load_cases(path, BY_CORPUS)
    assert [(c.scenario, c.relevant) for c in cases] == [
        ("ru->ru", {"it.txt#0"}),
        ("ky->ky", {"hr_ky.txt#1"}),
        ("ky->ky", set()),
    ]


def test_load_cases_spec_does_not_match_other_corpus(tmp_path):
    path = tmp_path / "eval.json"
    # hr_ky.txt exists only in the ky corpus, so a ru-corpus question cannot use it.
    spec = [{"filename": "hr_ky.txt", "contains": "90 күн"}]
    path.write_text(json.dumps([{"question": "q", "relevant": spec}]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_cases(path, BY_CORPUS)


def test_load_cases_rejects_unknown_corpus(tmp_path):
    path = tmp_path / "eval.json"
    spec = [{"filename": "it.txt", "contains": "90 дней"}]
    path.write_text(json.dumps([{"question": "q", "corpus": "xx", "relevant": spec}]), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown corpus"):
        load_cases(path, BY_CORPUS)


SCENARIO_CASES = [
    EvalCase("пароль менять 90 дней", {"it.txt#0"}),
    EvalCase("отпуск 28 дней", {"hr.txt#0"}),
    EvalCase("квантовая хромодинамика", set(), answerable=False),
    EvalCase("сырсөздү ар 90 күн сайын алмаштыруу", {"hr_ky.txt#1"}, q_lang="ky", corpus="ky"),
    # Russian question against the ky corpus: lexical hashing finds nothing relevant.
    EvalCase("отпуск 28 дней", {"hr_ky.txt#0"}, q_lang="ru", corpus="ky"),
]


def test_evaluate_scenarios_groups_by_language_pair(embedder):
    report = evaluate_scenarios(embedder, BY_CORPUS, SCENARIO_CASES, [1, 3], min_score=0.3)
    assert set(report["scenarios"]) == {"ru->ru", "ky->ky", "ru->ky"}
    assert report["corpus_chunks"] == {"ru": 3, "ky": 2}

    ru = report["scenarios"]["ru->ru"]
    assert (ru["questions"], ru["unanswerable"]) == (2, 1)
    assert ru["metrics"]["Recall@1"] == 1.0 and ru["metrics"]["CorrectRefusal"] == 1.0
    assert report["scenarios"]["ky->ky"]["metrics"]["Recall@1"] == 1.0
    assert "CorrectRefusal" not in report["scenarios"]["ky->ky"]["metrics"]

    overall = report["overall"]
    assert (overall["questions"], overall["unanswerable"]) == (4, 1)
    assert len(report["cases"]) == len(SCENARIO_CASES)


def test_evaluate_scenarios_searches_only_the_target_corpus(embedder):
    report = evaluate_scenarios(embedder, BY_CORPUS, SCENARIO_CASES, [1])
    tops = {(r["scenario"], r["question"]): r["top1"] for r in report["cases"]}
    assert tops[("ky->ky", "сырсөздү ар 90 күн сайын алмаштыруу")].startswith("hr_ky.txt")
    assert tops[("ru->ky", "отпуск 28 дней")].startswith("hr_ky.txt")
    assert tops[("ru->ru", "отпуск 28 дней")] == "hr.txt#0"


def test_evaluate_scenarios_score_distribution(embedder):
    report = evaluate_scenarios(embedder, BY_CORPUS, SCENARIO_CASES, [1])
    ru = report["scenarios"]["ru->ru"]
    rows = [r for r in report["cases"] if r["scenario"] == "ru->ru"]
    answerable_scores = sorted(r["top1_score"] for r in rows if r["answerable"])
    stats = ru["top1_score_answerable"]
    assert stats["min"] == pytest.approx(answerable_scores[0])
    assert stats["max"] == pytest.approx(answerable_scores[-1])
    assert stats["min"] <= stats["median"] <= stats["max"]
    unanswerable = [r["top1_score"] for r in rows if not r["answerable"]]
    assert ru["top1_score_unanswerable_max"] == pytest.approx(max(unanswerable))
    # no unanswerable questions in ky->ky; without min_score there are no refusal metrics
    assert report["scenarios"]["ky->ky"]["top1_score_unanswerable_max"] is None
    assert "FalseRefusal" not in report["overall"]["metrics"]


def test_evaluate_scenarios_rejects_question_for_empty_corpus(embedder):
    case = EvalCase("q", set(), answerable=False, q_lang="en", corpus="en")
    with pytest.raises(ValueError, match="No documents"):
        evaluate_scenarios(embedder, {**BY_CORPUS, "en": []}, [case], [1])


def test_repository_eval_set_matches_its_corpora():
    """Every relevance spec in data/eval/eval_set.json resolves in its own corpus."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    by_corpus = {
        lang: load_corpus(root / folder, 500, 100) for lang, folder in DEFAULT_CORPORA.items()
    }
    cases = load_cases(root / "data/eval/eval_set.json", by_corpus)
    assert {c.scenario for c in cases} >= {"ru->ru", "en->ru", "ky->ky", "ky->ru", "ru->ky"}
