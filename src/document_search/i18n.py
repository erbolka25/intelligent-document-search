"""Translations shared by the API and the UI, loaded from api/static/i18n.json.

Each top-level key of the JSON file is a language. Adding a language means adding
one more block with the same keys; tests/test_i18n.py checks that the keys match.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

from starlette.requests import Request

logger = logging.getLogger(__name__)

I18N_FILE = Path(__file__).parent / "api" / "static" / "i18n.json"
DEFAULT_LANGUAGE = "ru"


@lru_cache
def catalog() -> dict[str, dict]:
    return json.loads(I18N_FILE.read_text(encoding="utf-8"))


def languages() -> tuple[str, ...]:
    return tuple(catalog())


def _lookup(tree: dict, key: str) -> str | None:
    node: object = tree
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node if isinstance(node, str) else None


def translate(lang: str, key: str, **params: object) -> str:
    """Translate a dotted key, falling back to the default language, then to the key itself."""
    strings = catalog()
    text = _lookup(strings.get(lang, {}), key) or _lookup(strings.get(DEFAULT_LANGUAGE, {}), key)
    if text is None:
        logger.warning("Missing translation for %s", key)
        return key
    try:
        return text.format(**params)
    except (KeyError, IndexError, ValueError):
        return text


def request_language(request: Request) -> str:
    """?lang= wins, then the X-Lang header the UI sends, then Accept-Language."""
    candidates = [request.query_params.get("lang"), request.headers.get("x-lang")]
    for part in request.headers.get("accept-language", "").split(","):
        candidates.append(part.split(";")[0].strip())
    supported = languages()
    for value in candidates:
        if value:
            base = value.lower().split("-")[0]
            if base in supported:
                return base
    return DEFAULT_LANGUAGE
