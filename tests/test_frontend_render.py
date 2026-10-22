"""How the UI renders /ask and /search results: no raw scores, and no-answer passages collapsed."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from document_search.i18n import I18N_FILE, translate

NODE = shutil.which("node")
HARNESS = Path(__file__).parent / "js" / "render_harness.js"
STATIC_DIR = I18N_FILE.parent

pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")


def chunk(rank: int, score: float) -> dict:
    return {
        "rank": rank,
        "score": score,
        "doc_id": "d1",
        "filename": "komandirovki.txt",
        "page": None,
        "chunk_index": rank - 1,
        "char_start": 0,
        "char_end": 10,
        "text": f"passage {rank}",
    }


RESULTS = [chunk(1, 0.8731), chunk(2, 0.8512)]


def render(tmp_path, kind: str, data: dict, lang: str = "en") -> dict:
    response = tmp_path / "response.json"
    response.write_text(json.dumps(data), encoding="utf-8")
    proc = subprocess.run(
        [NODE, str(HARNESS), str(STATIC_DIR), lang, kind, str(response)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=True,
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def walk(node):
    yield node
    if isinstance(node, dict):
        for child in node["children"]:
            yield from walk(child)


def text_of(node) -> str:
    if isinstance(node, str):
        return node
    return "".join(text_of(c) for c in node["children"])


def ask(answer_found: bool, **extra) -> dict:
    return {
        "question": "q",
        "mode": "rag",
        "answer": "3 500 som a day [1]." if answer_found else None,
        "answer_found": answer_found,
        "citations": [RESULTS[0]] if answer_found else [],
        "results": RESULTS,
        **extra,
    }


@pytest.mark.parametrize(
    "kind, data",
    [("ask", ask(True)), ("ask", ask(False)), ("search", {"query": "q", "results": RESULTS})],
)
def test_raw_score_is_never_shown(tmp_path, kind, data):
    text = text_of(render(tmp_path, kind, data))
    for raw in ("0.873", "0.851", "87%", "85%"):
        assert raw not in text


def test_passages_carry_a_rank_label_with_an_explaining_tooltip(tmp_path):
    out = render(tmp_path, "search", {"query": "q", "results": RESULTS})
    labels = [n for n in walk(out) if isinstance(n, dict) and n["class"] == "relevance"]
    assert [text_of(n) for n in labels] == [
        translate("en", "ui.query.rank_label", n=1, total=2),
        translate("en", "ui.query.rank_label", n=2, total=2),
    ]
    assert all(n["attrs"]["title"] == translate("en", "ui.query.rank_title") for n in labels)


@pytest.mark.parametrize("lang", ["ru", "en"])
def test_no_answer_collapses_passages_behind_a_toggle(tmp_path, lang):
    out = render(tmp_path, "ask", ask(False), lang)
    top_level = [c for c in out["children"] if isinstance(c, dict)]
    assert [c["tag"] for c in top_level] == ["div", "details"]
    assert text_of(top_level[0]) == translate(lang, "ui.query.no_answer")

    details = top_level[1]
    assert "open" not in details["attrs"]
    summary = details["children"][0]
    assert summary["tag"] == "summary"
    assert translate(lang, "ui.query.show_closest") in text_of(summary)
    assert translate(lang, "ui.query.may_be_unrelated") in text_of(summary)
    cards = [c for c in details["children"][1:] if c["class"] == "result"]
    assert len(cards) == len(RESULTS)


def test_answer_found_shows_passages_open(tmp_path):
    out = render(tmp_path, "ask", ask(True))
    assert not [n for n in walk(out) if isinstance(n, dict) and n["tag"] == "details"]
    assert len([n for n in walk(out) if isinstance(n, dict) and n["class"] == "result"]) == 2
