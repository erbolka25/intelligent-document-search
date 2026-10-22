import logging

import pytest

from document_search.generation import check_citations, generate_answer, key_terms
from document_search.ingestion import Chunk
from document_search.retrieval import SearchResult
from tests.test_generation import FakeLLM

TRIPS = "Суточные при командировках за границу составляют 3 500 сом в день."
PASSWORD = "Пароль нужно менять каждые 90 дней и он должен содержать не менее 12 символов."


def _results(*texts: str) -> list[SearchResult]:
    return [
        SearchResult(Chunk("d", "f.txt", None, i, 0, len(t), t), 0.8, i + 1)
        for i, t in enumerate(texts)
    ]


def statuses(text: str, *passages: str) -> list[tuple[int, str]]:
    return [(c.source, c.status) for c in check_citations(text, _results(*passages))]


def test_key_terms_join_digit_groups_and_match_inflections():
    assert "3500" in key_terms("3 500 сом")
    assert key_terms("суточных") == key_terms("суточные")
    assert key_terms("это если может") == set()


def test_matching_citation_is_ok():
    assert statuses("За границей суточные составляют 3 500 сом [1].", TRIPS) == [(1, "ok")]


def test_citation_to_an_unrelated_passage_is_unsupported():
    text = "За границей суточные составляют 3 500 сом [2]."
    assert statuses(text, TRIPS, PASSWORD) == [(2, "unsupported")]


def test_each_sentence_is_checked_against_its_own_citations():
    text = "Суточные за границей 3 500 сом [1]. Пароль меняют каждые 90 дней [1][2]."
    assert statuses(text, TRIPS, PASSWORD) == [(1, "ok"), (1, "unsupported"), (2, "ok")]


def test_marker_after_the_full_stop_belongs_to_the_previous_sentence():
    text = "Суточные за границей 3 500 сом. [1] Пароль меняют каждые 90 дней. [2]"
    assert statuses(text, TRIPS, PASSWORD) == [(1, "ok"), (2, "ok")]


def test_english_sentence_on_russian_passage_is_cross_language_not_a_failure():
    assert statuses("The daily allowance abroad is set by policy [1].", TRIPS) == [
        (1, "cross_language")
    ]


@pytest.mark.parametrize(
    "text, expected",
    [("См. источник [5].", [(5, "out_of_range")]), ("Да [1].", [(1, "no_terms")])],
)
def test_edge_cases(text, expected):
    assert statuses(text, TRIPS) == expected


def test_generate_answer_logs_the_check_and_keeps_the_answer(caplog):
    reply = "Суточные за границей 3 500 сом [1]. Пароль меняют каждые 90 дней [1]."
    with caplog.at_level(logging.INFO, logger="document_search.generation"):
        answer = generate_answer(FakeLLM(reply), "Сколько суточные?", _results(TRIPS))
    assert answer.text == reply
    record = next(r for r in caplog.records if r.getMessage() == "Citation check")
    assert record.fields["citations"] == 2
    assert record.fields["citations_ok"] == 1
    assert record.fields["citations_unsupported"] == 1
