"""The translation catalog stays complete: adding a language is one more JSON block."""

import re
from pathlib import Path

import pytest
from starlette.requests import Request

from document_search import generation, ingestion
from document_search.api import app as app_module
from document_search.i18n import (
    DEFAULT_LANGUAGE,
    I18N_FILE,
    catalog,
    request_language,
    translate,
)

STATIC_DIR = I18N_FILE.parent
KEY_LITERAL = re.compile(r"""["'`]((?:ui|errors|notices)\.[\w.]+)["'`]""")


def flat_keys(tree: dict, prefix: str = "") -> set[str]:
    keys = set()
    for name, value in tree.items():
        path = f"{prefix}{name}"
        keys |= flat_keys(value, f"{path}.") if isinstance(value, dict) else {path}
    return keys


def placeholders(text: str) -> set[str]:
    return set(re.findall(r"\{(\w+)\}", text))


def make_request(headers: dict[str, str] | None = None, query: str = "") -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    return Request({"type": "http", "headers": raw, "query_string": query.encode()})


def test_default_language_is_russian_and_present():
    assert DEFAULT_LANGUAGE == "ru"
    assert {"ru", "en"} <= set(catalog())


@pytest.mark.parametrize("lang", [lang for lang in catalog() if lang != DEFAULT_LANGUAGE])
def test_every_language_has_the_same_keys_and_placeholders(lang):
    reference, other = catalog()[DEFAULT_LANGUAGE], catalog()[lang]
    assert flat_keys(other) == flat_keys(reference)
    for key in flat_keys(reference):
        assert placeholders(translate(lang, key)) == placeholders(translate(DEFAULT_LANGUAGE, key)), key


def test_ui_keys_used_in_html_and_js_exist():
    known = flat_keys(catalog()[DEFAULT_LANGUAGE])
    used = set()
    for path in STATIC_DIR.glob("*.*"):
        if path.suffix in {".html", ".js"}:
            used |= set(KEY_LITERAL.findall(path.read_text(encoding="utf-8")))
    # Built dynamically in common.js: `ui.controls.theme_${theme}`.
    used |= {f"ui.controls.theme_{theme}" for theme in ("auto", "light", "dark")}
    assert used, "no keys found: the regex is broken"
    assert used - known == set()


def test_html_and_js_contain_no_hardcoded_cyrillic():
    for path in STATIC_DIR.glob("*.*"):
        if path.suffix in {".html", ".js"}:
            text = path.read_text(encoding="utf-8")
            assert not re.search("[а-яё]", text, re.IGNORECASE), f"{path.name} has untranslated text"


def _codes(module, pattern: str) -> set[str]:
    return set(re.findall(pattern, Path(module.__file__).read_text(encoding="utf-8")))


def test_every_server_error_and_notice_code_is_translated():
    errors = _codes(app_module, r'ApiError\(\d+, "(\w+)"') | _codes(
        app_module, r'error_response\(\s*request, \d+, "(\w+)"'
    )
    errors |= _codes(ingestion, r'IngestionError\(\s*"(\w+)"')
    notices = _codes(generation, r'LLMError\(\s*"(\w+)"') | {"llm_disabled", "quota", "api"}
    assert errors and notices
    known = flat_keys(catalog()[DEFAULT_LANGUAGE])
    assert {f"errors.{c}" for c in errors} - known == set()
    assert {f"notices.{c}" for c in notices} - known == set()


@pytest.mark.parametrize(
    "headers, query, expected",
    [
        ({}, "", "ru"),
        ({"X-Lang": "en"}, "", "en"),
        ({"Accept-Language": "en-US,en;q=0.9"}, "", "en"),
        ({"X-Lang": "en"}, "lang=ru", "ru"),
        ({"X-Lang": "de"}, "", "ru"),
    ],
)
def test_request_language(headers, query, expected):
    assert request_language(make_request(headers, query)) == expected


def test_translate_falls_back_to_default_language_then_key():
    assert translate("xx", "errors.not_found") == translate("ru", "errors.not_found")
    assert translate("en", "errors.no_such_key") == "errors.no_such_key"
