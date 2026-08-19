"""LLM backend tests — OpenRouter/Bedrock request shapes, auth and credits
failures, cache behavior, and vision capability flags.

Gemini is legacy; the default stack is OpenRouter (BYO key) with Bedrock
(founder credits, boto3 default chain) as the credit-backed alternative.
"""

import json
from unittest.mock import MagicMock, patch

import httpx

from publikclip_pipeline.scoring import llm


def _patch_home(tmp_path, monkeypatch):
    monkeypatch.setenv("PUBLIKCLIP_HOME", str(tmp_path))
    monkeypatch.delenv("PUBLIKCLIP_OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("PUBLIKCLIP_BEDROCK_MODEL", raising=False)


def test_openrouter_requires_key(tmp_path, monkeypatch):
    _patch_home(tmp_path, monkeypatch)
    try:
        llm.OpenRouterClient()
        raise AssertionError("expected LlmError")
    except llm.LlmError as err:
        assert "OpenRouter API key" in str(err)


def test_openrouter_request_shape_and_cache(tmp_path, monkeypatch):
    _patch_home(tmp_path, monkeypatch)
    monkeypatch.setenv("PUBLIKCLIP_OPENROUTER_API_KEY", "sk-or-test")
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        req = httpx.Request("POST", url)
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"hook": 7}'}}]}, request=req)

    with patch.object(llm.httpx, "post", side_effect=fake_post):
        client = llm.OpenRouterClient()
        result = client.generate_json(
            "score this", {"type": "object"}, images=[b"\xff\xd8fakejpeg"]
        )

    assert result == {"hook": 7}
    assert captured["url"] == llm.OPENROUTER_URL
    assert captured["kwargs"]["headers"]["Authorization"] == "Bearer sk-or-test"
    body = captured["kwargs"]["json"]
    assert body["response_format"] == {"type": "json_object"}
    content = body["messages"][0]["content"]
    assert content[0]["type"] == "text"
    assert "Emit ONLY valid JSON" in content[0]["text"]
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    # Cache: an identical second call must not POST again.
    captured.clear()
    again = client.generate_json("score this", {"type": "object"}, images=[b"\xff\xd8fakejpeg"])
    assert again == {"hook": 7}
    assert "kwargs" not in captured


def test_openrouter_credits_and_auth_failures(tmp_path, monkeypatch):
    _patch_home(tmp_path, monkeypatch)
    monkeypatch.setenv("PUBLIKCLIP_OPENROUTER_API_KEY", "sk-or-test")
    with patch.object(llm.httpx, "post", return_value=httpx.Response(402, json={})):
        try:
            llm.OpenRouterClient().generate_json("p", {})
            raise AssertionError("expected LlmError")
        except llm.LlmError as err:
            assert "credits" in str(err).lower()
    with patch.object(llm.httpx, "post", return_value=httpx.Response(401, json={})):
        try:
            llm.OpenRouterClient().generate_json("p", {})
            raise AssertionError("expected LlmError")
        except llm.LlmError as err:
            assert "rejected" in str(err)


def test_openrouter_error_inside_200_body(tmp_path, monkeypatch):
    _patch_home(tmp_path, monkeypatch)
    monkeypatch.setenv("PUBLIKCLIP_OPENROUTER_API_KEY", "sk-or-test")
    with patch.object(
        llm.httpx,
        "post",
        return_value=httpx.Response(200, json={"error": "insufficient credits"}, request=httpx.Request("POST", llm.OPENROUTER_URL)),
    ):
        try:
            llm.OpenRouterClient().generate_json("p", {})
            raise AssertionError("expected LlmError")
        except llm.LlmError as err:
            assert "insufficient credits" in str(err)


def test_bedrock_requires_boto3(tmp_path, monkeypatch):
    _patch_home(tmp_path, monkeypatch)
    with patch.dict("sys.modules", {"boto3": None}):
        try:
            llm.BedrockClient()
            raise AssertionError("expected LlmError")
        except llm.LlmError as err:
            assert "boto3" in str(err)


def test_bedrock_converse_shape_and_cache(tmp_path, monkeypatch):
    _patch_home(tmp_path, monkeypatch)
    fake_client = MagicMock()
    fake_client.converse.return_value = {
        "output": {"message": {"content": [{"text": '{"genre": "lo-fi hip hop"}'}]}}
    }
    fake_boto3 = MagicMock()
    fake_boto3.client.return_value = fake_client
    with patch.dict("sys.modules", {"boto3": fake_boto3}):
        client = llm.BedrockClient()
        result = client.generate_json("music brief", {"type": "object"}, images=[b"jpegbytes"])

    assert result == {"genre": "lo-fi hip hop"}
    call = fake_client.converse.call_args
    assert call.kwargs["modelId"] == llm.BEDROCK_MODEL
    messages = call.kwargs["messages"]
    assert messages[0]["content"][0] == {"text": "music brief"}
    assert messages[0]["content"][1]["image"]["format"] == "jpeg"
    assert messages[0]["content"][1]["image"]["source"]["bytes"] == b"jpegbytes"
    assert call.kwargs["inferenceConfig"]["temperature"] == 0.2

    # Cache hit: identical inputs must not call Bedrock again.
    fake_client.converse.reset_mock()
    again = client.generate_json("music brief", {"type": "object"}, images=[b"jpegbytes"])
    assert again == {"genre": "lo-fi hip hop"}
    fake_client.converse.assert_not_called()


def test_openrouter_schema_echo_retries_then_fails(tmp_path, monkeypatch):
    """GLM-4.5V echoed the schema on image calls; the client must retry once
    with an anti-echo instruction, then fail cleanly instead of returning a
    schema-shaped dict that crashes composite()."""
    _patch_home(tmp_path, monkeypatch)
    monkeypatch.setenv("PUBLIKCLIP_OPENROUTER_API_KEY", "sk-or-test")
    schema = {
        "type": "object",
        "properties": {"visual_interest": {"type": "integer"}},
        "required": ["visual_interest"],
    }
    echo = json.dumps(schema)
    calls = []

    def fake_post(url, **kwargs):
        calls.append(kwargs["json"])
        body = kwargs["json"]
        # First call: model echoes the schema. After the anti-echo retry: valid.
        if "never repeat the schema" not in body["messages"][0]["content"][0]["text"]:
            content = echo
        else:
            content = '{"visual_interest": 7}'
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]}, request=httpx.Request("POST", url))

    with patch.object(llm.httpx, "post", side_effect=fake_post):
        client = llm.OpenRouterClient()
        result = client.generate_json("rate frames", schema, images=[b"img"])
    assert result == {"visual_interest": 7}
    assert len(calls) == 2

    # If the model keeps echoing, the client must raise, not return garbage.
    def fake_post_always_echo(url, **kwargs):
        return httpx.Response(200, json={"choices": [{"message": {"content": echo}}]}, request=httpx.Request("POST", url))

    with patch.object(llm.httpx, "post", side_effect=fake_post_always_echo):
        try:
            # Different prompt → no cache hit → the model keeps echoing → must raise.
            llm.OpenRouterClient().generate_json("rate frames again", schema, images=[b"img"])
            raise AssertionError("expected LlmError")
        except llm.LlmError as err:
            assert "missing required keys" in str(err)


def test_vision_flags():
    assert llm.OpenRouterClient.supports_vision
    assert llm.BedrockClient.supports_vision
    assert not llm.OllamaClient.supports_vision


def test_make_client_routing(tmp_path, monkeypatch):
    _patch_home(tmp_path, monkeypatch)
    monkeypatch.setenv("PUBLIKCLIP_OPENROUTER_API_KEY", "sk-or-test")
    assert llm.make_client("openrouter").backend == "openrouter"
    with patch.dict("sys.modules", {"boto3": MagicMock()}):
        assert llm.make_client("bedrock").backend == "bedrock"
    try:
        llm.make_client("bogus")
        raise AssertionError("expected LlmError for unknown mode")
    except llm.LlmError as err:
        assert "Unknown llm_mode" in str(err)
    assert llm.make_client("gemini").backend == "openrouter"  # legacy jobs map to openrouter
    fake_tags = httpx.Response(
        200,
        json={"models": [{"name": "llama3.1:8b"}]},
        request=httpx.Request("GET", llm.OLLAMA_URL),
    )
    with patch.object(llm.httpx, "get", return_value=fake_tags):
        assert llm.make_client("ollama").backend == "ollama"
