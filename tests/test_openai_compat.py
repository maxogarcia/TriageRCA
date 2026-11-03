from __future__ import annotations

import json
from types import SimpleNamespace

from src.llm.base import Message, ToolCall
from src.llm.openai_compat import OpenAICompatClient, _to_openai_messages
from src.schemas import Hypothesis


def test_echoed_tool_call_arguments_are_json():
    messages = [
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="c1", name="metric_query", args={"service": "cartservice", "top_n": None})],
        )
    ]

    arguments = _to_openai_messages(messages)[0]["tool_calls"][0]["function"]["arguments"]

    assert json.loads(arguments) == {"service": "cartservice", "top_n": None}


def test_response_schema_is_sent_as_json_schema(monkeypatch):
    captured: dict = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        message = SimpleNamespace(content="{}", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)

    client = OpenAICompatClient(model="qwen2.5:7b-instruct", base_url="http://localhost:11434/v1")
    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)

    client._raw_generate([Message(role="user", content="hi")], response_schema=Hypothesis)

    response_format = captured["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "Hypothesis"
    assert "ranked_services" in response_format["json_schema"]["schema"]["properties"]
