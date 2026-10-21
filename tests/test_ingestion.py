from itertools import pairwise

import pytest

from document_search.ingestion import (
    IngestionError,
    chunk_text,
    clean_unicode,
    compute_doc_id,
    ingest_document,
    is_readable_text,
    normalize_text,
    split_sentence_spans,
)


def _sentences(text: str) -> list[str]:
    return [text[s:e] for s, e in split_sentence_spans(text)]


def test_split_sentences_russian_and_english():
    text = "Первое предложение. Второе! Third one? 4 пункт."
    assert _sentences(text) == ["Первое предложение.", "Второе!", "Third one?", "4 пункт."]


def test_split_sentences_keeps_decimals_and_lowercase_continuations():
    text = "Ставка 3.5 процента. Например, т. е. так."
    assert _sentences(text) == ["Ставка 3.5 процента.", "Например, т. е. так."]


def test_normalize_text_keeps_paragraphs_and_joins_wrapped_lines():
    raw = "4. Заголовок\nпервая строка\r\nвторая  строка\n\n\n  Новый абзац  \n \n"
    assert normalize_text(raw) == "4. Заголовок первая строка вторая строка\nНовый абзац"


def test_paragraph_break_is_a_sentence_boundary():
    text = normalize_text("Раздел без точки\n\nследующий абзац с маленькой буквы")
    assert _sentences(text) == ["Раздел без точки", "следующий абзац с маленькой буквы"]


def test_chunk_text_respects_max_chars_and_sentence_boundaries():
    sentences = [f"Предложение номер {i} про тестирование." for i in range(30)]
    text = " ".join(sentences)
    chunks = chunk_text(text, max_chars=200, overlap_chars=50)

    assert len(chunks) > 1
    for start, end in chunks:
        assert end - start <= 200
        piece = text[start:end]
        assert piece.startswith("Предложение") and piece.endswith(".")
    assert chunks[0][0] == 0
    assert chunks[-1][1] == len(text)


def test_chunk_text_overlap_shares_sentences():
    text = " ".join(f"Sentence {i} is here." for i in range(20))
    chunks = chunk_text(text, max_chars=100, overlap_chars=40)
    for (_, prev_end), (next_start, _) in pairwise(chunks):
        assert next_start < prev_end  # overlap exists
        assert prev_end - next_start <= 40


def test_chunk_text_without_overlap_is_contiguous():
    text = " ".join(f"Sentence {i} is here." for i in range(20))
    chunks = chunk_text(text, max_chars=100, overlap_chars=0)
    for (_, prev_end), (next_start, _) in pairwise(chunks):
        assert next_start == prev_end + 1  # separated by one space


def test_chunk_text_splits_oversized_sentence():
    text = "слово " * 100  # one 600-char "sentence" with no punctuation
    chunks = chunk_text(text.strip(), max_chars=100, overlap_chars=0)
    assert len(chunks) >= 6
    assert all(end - start <= 100 for start, end in chunks)


def test_chunk_text_empty():
    assert chunk_text("") == []


@pytest.mark.parametrize("max_chars, overlap", [(0, 0), (100, 100), (100, -1)])
def test_chunk_text_invalid_params(max_chars, overlap):
    with pytest.raises(ValueError):
        chunk_text("text", max_chars=max_chars, overlap_chars=overlap)


def test_ingest_txt_metadata():
    data = "Первый абзац.\n\n  Второй   абзац.".encode()
    chunks = ingest_document(data, "notes.txt").chunks
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.text == "Первый абзац.\nВторой абзац."
    assert chunk.doc_id == compute_doc_id(data)
    assert chunk.filename == "notes.txt"
    assert chunk.page is None
    assert (chunk.chunk_index, chunk.char_start, chunk.char_end) == (0, 0, len(chunk.text))


def test_ingest_txt_cp1251():
    chunks = ingest_document("Привет из Windows.".encode("cp1251"), "legacy.TXT").chunks
    assert chunks[0].text == "Привет из Windows."


def test_ingest_pdf_pages(make_pdf):
    data = make_pdf(["Hello from page one.", "Second page text."])
    chunks = ingest_document(data, "doc.pdf").chunks
    assert [(c.page, c.chunk_index) for c in chunks] == [(1, 0), (2, 1)]
    assert "page one" in chunks[0].text
    assert "Second page" in chunks[1].text


@pytest.mark.parametrize(
    "data, filename",
    [
        (b"", "empty.txt"),
        (b"   \n\t ", "blank.txt"),
        (b"some text", "script.exe"),
        (b"some text", "no_extension"),
        (b"not a pdf at all", "broken.pdf"),
    ],
)
def test_ingest_rejects_bad_input(data, filename):
    with pytest.raises(IngestionError):
        ingest_document(data, filename)


CYRILLIC_PAGE = "Ежегодный оплачиваемый отпуск составляет 28 календарных дней.\nЖылдык эс алуу 28 күн."


def test_ingest_cyrillic_pdf_is_readable(make_unicode_pdf):
    result = ingest_document(make_unicode_pdf([CYRILLIC_PAGE]), "отчёт.pdf")
    text = " ".join(c.text for c in result.chunks)
    assert "Ежегодный оплачиваемый отпуск" in text
    assert "күн" in text  # Kyrgyz letters survive
    assert "�" not in text and "■" not in text
    assert result.skipped_pages == []


def test_ingest_pdf_skips_garbage_page_and_reports_it(make_unicode_pdf):
    garbage = "IIIIII IIIII IIIIIIII IIII IIIIIII IIIIII IIIII IIII"
    data = make_unicode_pdf([CYRILLIC_PAGE, garbage, None, CYRILLIC_PAGE])
    result = ingest_document(data, "mixed.pdf")
    assert result.skipped_pages == [2]  # blank page 3 is skipped silently
    assert {c.page for c in result.chunks} == {1, 4}


def test_ingest_pdf_with_only_garbage_is_rejected(make_unicode_pdf):
    with pytest.raises(IngestionError) as info:
        ingest_document(make_unicode_pdf(["��� �� ���"] * 2), "s.pdf")
    assert info.value.code == "no_text"


@pytest.mark.parametrize(
    "text, readable",
    [
        ("Обычный русский текст про отпуск и командировки.", True),
        ("Plain English text about travel expenses.", True),
        ("Жөнөкөй кыргыз тексти, иш сапарлары жөнүндө.", True),
        ("IIIII � IIII 2026 � IIII IIIIIIII IIIII", False),
        ("■■■■ ■■■ abc defgh ijklm", False),
        ("12 34 56", False),
        ("", False),
    ],
)
def test_is_readable_text(text, readable):
    assert is_readable_text(text) is readable


def test_clean_unicode_normalizes_ligatures_and_soft_hyphens():
    assert clean_unicode("ﬁnal o­tpusk\x07") == "final otpusk "


@pytest.mark.parametrize(
    "data, filename, code",
    [
        (b"", "empty.txt", "empty_file"),
        (b"x", "a.exe", "unsupported_type"),
        (b"not a pdf", "a.pdf", "corrupted_pdf"),
        (b"   ", "blank.txt", "no_text"),
    ],
)
def test_ingestion_error_codes(data, filename, code):
    with pytest.raises(IngestionError) as info:
        ingest_document(data, filename)
    assert info.value.code == code
