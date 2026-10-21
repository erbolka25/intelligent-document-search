"""LLM provider clients with the SDK transport replaced: no network, no real keys."""

from types import SimpleNamespace

import httpx2
import openai
import pytest

from document_search.config import ENV_FILE, PROJECT_ROOT, Settings
from document_search.generation import (
    AnthropicClient,
    LLMError,
    OpenAIClient,
    build_llm_client,
)

_REQUEST = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")


def _status_error(cls, status: int, code: str | None = None):
    body = {"code": code} if code else None
    return cls("error", response=httpx2.Response(status, request=_REQUEST), body=body)


def _reply(content: str | None, refusal: str | None = None, finish_reason: str = "stop"):
    message = SimpleNamespace(content=content, refusal=refusal)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)])


class FakeCompletions:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result, self.error, self.kwargs = result, error, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.result


def _openai(model: str = "gpt-6-luna", **fake) -> tuple[OpenAIClient, FakeCompletions]:
    client = OpenAIClient(api_key="sk-test", model=model, max_tokens=123, timeout_s=5)
    completions = FakeCompletions(**fake)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


def test_openai_success_sends_limits_and_messages():
    client, fake = _openai(result=_reply("Ответ [1]."))
    assert client.complete("system", "user") == "Ответ [1]."
    assert fake.kwargs["model"] == "gpt-6-luna"
    assert fake.kwargs["max_completion_tokens"] == 123
    assert fake.kwargs["reasoning_effort"] == "low"
    assert [m["role"] for m in fake.kwargs["messages"]] == ["system", "user"]


def test_openai_client_configures_timeout_and_retries():
    client = OpenAIClient(api_key="sk-test", timeout_s=7)
    assert client._client.max_retries == 2
    assert client._client.timeout == 7


def test_openai_legacy_model_skips_reasoning_effort():
    client, fake = _openai(model="gpt-4o-mini", result=_reply("ok"))
    client.complete("s", "u")
    assert "reasoning_effort" not in fake.kwargs


@pytest.mark.parametrize(
    "error, code",
    [
        (_status_error(openai.AuthenticationError, 401), "auth"),
        (_status_error(openai.RateLimitError, 429), "rate_limit"),
        (_status_error(openai.RateLimitError, 429, "insufficient_quota"), "quota"),
        (_status_error(openai.NotFoundError, 404), "model"),
        (_status_error(openai.InternalServerError, 500), "api"),
        (openai.APIConnectionError(request=_REQUEST), "network"),
        (openai.APITimeoutError(request=_REQUEST), "network"),
    ],
)
def test_openai_errors_map_to_codes(error, code):
    client, _ = _openai(error=error)
    with pytest.raises(LLMError) as info:
        client.complete("s", "u")
    assert info.value.code == code
    assert "sk-test" not in str(info.value)


@pytest.mark.parametrize(
    "reply, code",
    [(_reply(None, refusal="no"), "refusal"), (_reply("", finish_reason="length"), "empty")],
)
def test_openai_refusal_and_empty_reply(reply, code):
    client, _ = _openai(result=reply)
    with pytest.raises(LLMError) as info:
        client.complete("s", "u")
    assert info.value.code == code


def test_build_llm_client_selects_provider():
    common = {"model": "m", "effort": "low", "max_tokens": 10, "timeout_s": 1.0}
    assert isinstance(build_llm_client("openai", "k", **common), OpenAIClient)
    assert isinstance(build_llm_client("anthropic", "k", **common), AnthropicClient)
    assert build_llm_client("none", "k", **common) is None
    assert build_llm_client("openai", None, **common) is None


def _settings(**values) -> Settings:
    return Settings(_env_file=None, **values)


def test_settings_pick_key_and_default_model_per_provider():
    s = _settings(llm_provider="openai", openai_api_key=" sk-o ", anthropic_api_key="sk-a")
    assert s.llm_api_key() == "sk-o" and s.resolved_llm_model == "gpt-6-luna"
    s = _settings(llm_provider="anthropic", anthropic_api_key="sk-a", llm_model="claude-x")
    assert s.llm_api_key() == "sk-a" and s.resolved_llm_model == "claude-x"
    assert _settings(llm_provider="openai", openai_api_key="  ").llm_api_key() is None
    assert _settings(llm_provider="none", openai_api_key="sk-o").llm_api_key() is None


def test_env_file_and_data_dir_resolve_from_project_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # a different working directory must not matter
    assert ENV_FILE == PROJECT_ROOT / ".env"
    assert (PROJECT_ROOT / "pyproject.toml").exists()
    assert _settings().data_dir == PROJECT_ROOT / "data"
    assert _settings(data_dir=tmp_path).data_dir == tmp_path


def test_env_file_values_are_loaded(tmp_path):
    env = tmp_path / ".env"
    env.write_text("LLM_PROVIDER=anthropic\nANTHROPIC_API_KEY=sk-from-file\n", encoding="utf-8")
    s = Settings(_env_file=env)
    assert s.llm_provider == "anthropic" and s.llm_api_key() == "sk-from-file"
    assert "sk-from-file" not in repr(s)
