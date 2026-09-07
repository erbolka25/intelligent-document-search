import pytest
from document_search.core import clean_text , word_count
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
