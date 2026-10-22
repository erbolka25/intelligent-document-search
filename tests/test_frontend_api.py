"""The UI shows localized messages, not browser text, when the server cannot be reached."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from document_search.i18n import I18N_FILE, translate

NODE = shutil.which("node")
HARNESS = Path(__file__).parent / "js" / "api_harness.js"
COMMON_JS = I18N_FILE.parent / "common.js"

pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")


def run_api(lang: str, scenario: str) -> dict:
    proc = subprocess.run(
        [NODE, str(HARNESS), str(COMMON_JS), str(I18N_FILE), lang, scenario],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=True,
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("lang", ["ru", "en"])
@pytest.mark.parametrize(
    "scenario, key",
    [
        ("network", "ui.network.unreachable"),
        ("proxy_down", "ui.network.unreachable"),
        ("timeout", "ui.network.timeout"),
        ("not_json", "ui.network.bad_response"),
    ],
)
def test_fetch_failures_show_a_localized_message(lang, scenario, key):
    message = run_api(lang, scenario)["message"]
    assert message == translate(lang, key)
    assert "fetch" not in message.lower()


def test_server_message_is_passed_through():
    assert run_api("en", "server_message")["message"] == "SERVER MESSAGE"


def test_successful_json_is_returned():
    assert run_api("ru", "ok") == {"body": {"ok": True}}
