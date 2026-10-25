import pytest

from document_search.generation import detect_language


@pytest.mark.parametrize(
    "question",
    [
        "Саламатсызбы! Ноутбукка кепилдик канча убакытка берилет?",  # no ң/ө/ү
        "Бишкекке жеткирүү канча турат?".replace("ү", "у"),  # the letter typed as Russian у
        "Заказды кантип кайтарсам болот?",
        "Накталай акча менен төлөсө болобу?".replace("ө", "о"),
        "Буйрутма качан келет?",
        "Дүкөн кайда жайгашкан?".replace("ү", "у").replace("ө", "о"),
        "Рахмат, дагы бир суроо: кандай кепилдик бар?",
        "КАНЧА ТУРАТ?",  # case-insensitive
        "Өргүү канча күн?",  # the Kyrgyz letters still decide first
    ],
)
def test_kyrgyz_without_special_letters(question):
    assert detect_language(question) == "Kyrgyz"


@pytest.mark.parametrize(
    "question",
    [
        "Сколько стоит доставка в Ош?",
        "Можно ли вернуть товар, если я уже открыл упаковку?",
        (
            "Здравствуйте! Я заказал ноутбук на прошлой неделе, курьер так и не приехал, "
            "а в личном кабинете статус заказа до сих пор «в обработке». Подскажите, "
            "пожалуйста, когда ждать доставку и можно ли получить компенсацию за задержку?"
        ),
        # Mostly Cyrillic: the script count (unchanged) decides Russian vs English.
        "Подойдёт ли оригинальная зарядка USB-C к моему ноутбуку MacBook?",
        "Работает ли оплата через Visa и Mastercard в приложении iOS?",
        "Жокей выиграл скачки, а менеджер Болот кайдановски не знал об этом.",  # word boundaries
        "Канчаловский снял новый фильм.",
    ],
)
def test_russian_has_no_false_kyrgyz(question):
    assert detect_language(question) == "Russian"


@pytest.mark.parametrize(
    "question",
    [
        "How much is delivery to Osh?",
        "Can I return opened headphones?",
        "Does the warranty cover a cracked screen on my iPhone 15?",
    ],
)
def test_english(question):
    assert detect_language(question) == "English"
