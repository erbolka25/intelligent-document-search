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
def test_top_n():
    from document_search.core import top_n
    items = ['a', 'b', 'c', 'd']
    scores = [0.1, 0.4, 0.3, 0.2]
    assert top_n(items, scores, 2) == ['b', 'c']
    assert top_n(items, scores, 0) == []
    assert top_n(items, scores, 5) == ['b', 'c', 'd', 'a']
def test_chunking_list():
    from document_search.core import chunking_list
    items = [1, 2, 3, 4, 5, 6, 7]
    assert chunking_list(items, 3) == [[1, 2, 3], [4, 5, 6], [7]]
def test_dedupe_preserve_order():
    from document_search.core import dedupe_preserve_order
    items = [1, 2, 3, 2, 1, 4, 5]
    assert dedupe_preserve_order(items) == [1, 2, 3, 4, 5]
    items = ['a', 'b', 'a', 'c', 'b']
    assert dedupe_preserve_order(items) == ['a', 'b', 'c']
    items = []
    assert dedupe_preserve_order(items) == []
def test_long_chunks():
    from document_search.core import long_chunks
    chunks = ["short", "this is a long chunk of text that exceeds fifty characters in length", "another short one", "yet another long chunk that should be included in the result"]
    assert long_chunks(chunks) == ["this is a long chunk of text that exceeds fifty characters in length", "yet another long chunk that should be included in the result"]
    chunks = ["short1", "short2", "short3"]
    assert long_chunks(chunks) == []
def test_flatten():
    from document_search.core import flatten
    list_of_lists = [[1, 2], [3, 4], [5]]
    assert flatten(list_of_lists) == [1, 2, 3, 4, 5]
    list_of_lists = [["a", "b"], ["c"], ["d", "e", "f"]]
    assert flatten(list_of_lists) == ["a", "b", "c", "d", "e", "f"]
    list_of_lists = []
    assert flatten(list_of_lists) == []
def test_build_doc_index():
    from document_search.core import build_doc_index
    docs = [
        {'id': 'doc1', 'content': 'This is the first document.'},
        {'id': 'doc2', 'content': 'This is the second document.'},
        {'id': 'doc3', 'content': 'This is the third document.'}
    ]
    expected_index = {
    'doc1': 'This is the first document.',
    'doc2': 'This is the second document.',
    'doc3': 'This is the third document.'
   }
    assert build_doc_index(docs) == expected_index
def test_word_frequency():
    from document_search.core import word_frequency
    text = "This is a test. This test is only a test."
    expected_freq = {
        'this': 2,
        'is': 2,
        'a': 2,
        'test': 3,
        'only': 1
    }
    
    assert word_frequency(text) == expected_freq
def test_find_common_words():
    from document_search.core import find_common_words
    text1 = "This is a test."
    text2 = "This test is only a test."
    expected_common = {'this', 'is', 'a', 'test'}
    assert find_common_words(text1, text2) == expected_common
def test_is_duplicate():
    from document_search.core import is_duplicate
    hashes_storage = set()
    text1 = "This is a test."
    text2 = "This is a test."
    text3 = "Completely different text."
    assert is_duplicate(text1, hashes_storage) == False
    assert is_duplicate(text2, hashes_storage) == True
    assert is_duplicate(text3, hashes_storage) == False
def test_inverted_index():
    from document_search.core import inverted_index
    docs = [
        {'id': 'doc1', 'content': 'This is the first document.'},
        {'id': 'doc2', 'content': 'This is the second document.'},
        {'id': 'doc3', 'content': 'This is the third document.'}
    ]
    expected_index = {
        'this': {'doc1', 'doc2', 'doc3'},
        'is': {'doc1', 'doc2', 'doc3'},
        'the': {'doc1', 'doc2', 'doc3'},
        'first': {'doc1'},
        'document': {'doc1', 'doc2', 'doc3'},
        'second': {'doc2'},
        'third': {'doc3'}
    }
    assert inverted_index(docs) == expected_index