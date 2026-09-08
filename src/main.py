from document_search.utils import search_word
my_text="i am learning python everyday"
def test_search_word():
    assert search_word(my_text, "python") == True
    assert search_word(my_text, "java") == False    

test_search_word()
print("All tests passed!")
# print("-7 / 3 =", -7 / 3)
# print("-7 // 3 =", -7 // 3)
# Задание 5: Разница деления для отрицательных чисел
# -7 / 3 дает точный результат -2.33
# -7 // 3 дает -3, потому что Python ВСЕГДА округляет результат ВНИЗ (влево по числовой прямой). 
# На градуснике -3 градуса меньше (холоднее), чем -2.33.