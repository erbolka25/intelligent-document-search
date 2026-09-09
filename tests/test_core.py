import pytest
from document_search.core import clean_text, count_tokens_approx, parse_chunk_size , word_count
@pytest.mark.parametrize("dirty_text, expected_clean_text", [
    ("  привет    эрбол ", "привет эрбол"),
    ("ПРИВЕТ", "ПРИВЕТ"),
    ("привет\n\nмир", "привет мир")
])
def test_clean_text(dirty_text, expected_clean_text):
    assert clean_text(dirty_text) == expected_clean_text


@pytest.mark.parametrize("text, expected_word_count", [
    ("привет эрбол", 2),
    ("привет\n\nмир", 2),
    ("   ", 0)
])
def test_word_count(text, expected_word_count):
    assert word_count(text) == expected_word_count
def test_parse_chunk_size_valid():
    assert parse_chunk_size("10") == 10
    assert parse_chunk_size("0") == 0
    assert parse_chunk_size("-5") == -5
def test_parse_chunk_size_invalid():
    with pytest.raises(ValueError):
        parse_chunk_size("abc")
    with pytest.raises(ValueError):
        parse_chunk_size("10.5")
    with pytest.raises(ValueError):
        parse_chunk_size("")

def test_is_valid_similarity_score():
    from document_search.core import is_valid_similarity_score
    assert is_valid_similarity_score(0.0) == True
    assert is_valid_similarity_score(0.5) == True
    assert is_valid_similarity_score(1.0) == True
    assert is_valid_similarity_score(-0.1) == False
    assert is_valid_similarity_score(1.1) == False
def test_is_valid_similarity_score_invalid_type():
    from document_search.core import is_valid_similarity_score
    with pytest.raises(TypeError):
        is_valid_similarity_score("0.5")
    with pytest.raises(TypeError):
        is_valid_similarity_score(None)
def test_format_time():
    from document_search.core import format_time
    assert format_time(45) == "00:00:45"
    assert format_time(75) == "00:01:15"
    assert format_time(3605) == "01:00:05"
def test_average():
    from document_search.core import average
    assert average([1, 2, 3, 4, 5]) == 3.0
    assert average([10.5, 20.5, 30.5]) == 20.5
    assert average([-1, -2, -3]) == -2.0
    assert average([]) == 0.0
def test_truncate():
    from document_search.core import truncate
    assert truncate("Hello, World!", 5) == "He..."
    assert truncate("Short text", 20) == "Short text"
    assert truncate("Exact length", 12) == "Exact length"
def test_is_probably_sentence_end():
    from document_search.core import is_probably_sentence_end
    assert is_probably_sentence_end('.') == True
    assert is_probably_sentence_end('!') == True
    assert is_probably_sentence_end('?') == True
    assert is_probably_sentence_end(',') == False
    assert is_probably_sentence_end('a') == False
def test_count_tokens_approx():
    from document_search.core import count_tokens_approx
    assert count_tokens_approx("Hello, World!") == 3
    assert count_tokens_approx("This is a test sentence.") == 6
    assert count_tokens_approx("") == 0
def test_split_sentences():
    from document_search.core import split_sentences
    assert split_sentences("This is a sentence. This is another.") == ["This is a sentence", "This is another"]
    assert split_sentences("No punctuation here") == ["No punctuation here"]
    assert split_sentences("Multiple...dots...here.") == ["Multiple", "dots", "here"]
def test_get_snippet():
    from document_search.core import get_snippet
    text = "This is a sample text for testing the get_snippet function."
    assert get_snippet(text, 5, 10) == "This is a sample"
    assert get_snippet(text, 0, 5) == "This i"
    assert get_snippet(text, 50, 5) == "ppet functi"
    