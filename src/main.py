from document_search.utils import search_word
my_text="i am learning python everyday"
def test_search_word():
    assert search_word(my_text, "python") == True
    assert search_word(my_text, "java") == False    

test_search_word()
print("All tests passed!")