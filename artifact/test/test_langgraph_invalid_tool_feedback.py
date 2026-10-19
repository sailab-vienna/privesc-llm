from typing import Any, cast
from types import SimpleNamespace

import pytest

from src.gym.agent import got_root, nudge_for_invalid_tool_call, nudge_for_tools
from src.gym.tools import parse_xml_tool_calls


class DummyAIMessage:
    def __init__(self, content: str, tool_calls=None, additional_kwargs=None):
        self.type = "ai"
        self.content = content
        self.tool_calls = tool_calls or []
        self.additional_kwargs = additional_kwargs or {}


@pytest.mark.parametrize(
    ("content", "expected_kind"),
    [
        ('x<tool_call>{"name":"exec_command"</tool_call>', "invalid_json"),
        ("x<tool_call>[]</tool_call>", "invalid_shape"),
        ('x<tool_call>{"command":"id"}</tool_call>', "missing_name"),
        ('x<tool_call>{"name":"exec_command"}</tool_call>', "invalid_arguments"),
    ],
)
def test_parse_xml_tool_calls_marks_invalid_errors(
    content: str, expected_kind: str
) -> None:
    msg = DummyAIMessage(content)
    out = parse_xml_tool_calls(msg)

    assert out.tool_calls == []
    errors = out.additional_kwargs.get("invalid_tool_call_errors")
    assert isinstance(errors, list)
    assert errors and errors[0]["kind"] == expected_kind


def test_parse_xml_tool_calls_keeps_valid_calls_with_mixed_errors() -> None:
    msg = DummyAIMessage(
        'x<tool_call>{"name":"exec_command","arguments":{"command":"id"}}</tool_call>'
        '<tool_call>{"command":"whoami"}</tool_call>'
    )
    out = parse_xml_tool_calls(msg)

    assert len(out.tool_calls) == 1
    assert out.tool_calls[0]["name"] == "exec_command"
    errors = out.additional_kwargs.get("invalid_tool_call_errors")
    assert isinstance(errors, list)
    assert errors and errors[0]["kind"] == "missing_name"


def test_nudge_for_tools_includes_invalid_tool_call_feedback() -> None:
    state = {
        "messages": [
            DummyAIMessage(
                "x",
                tool_calls=[],
                additional_kwargs={
                    "invalid_tool_call_errors": [
                        {"kind": "missing_name", "message": "m", "raw": "{}"}
                    ]
                },
            )
        ]
    }
    config = {"configurable": {"no_tool_calls_nudge": "use tools", "logger": None}}

    out = nudge_for_tools(cast(Any, state), cast(Any, config))
    content = out["messages"][0].content
    assert "Error: Invalid tool call format" in content
    assert "missing_name" in content
    assert content.endswith("use tools")


def test_got_root_routes_to_invalid_tool_call_nudge_without_root() -> None:
    state = {
        "messages": [
            DummyAIMessage(
                "x",
                tool_calls=[{"name": "exec_command", "args": {"command": "id"}}],
                additional_kwargs={
                    "invalid_tool_call_errors": [
                        {"kind": "missing_name", "message": "m", "raw": "{}"}
                    ]
                },
            ),
            SimpleNamespace(type="tool", content='{"got_root": false}'),
        ]
    }
    config = {"configurable": {}}
    assert got_root(cast(Any, state), cast(Any, config)) == "invalid_tool_call_nudge"


def test_nudge_for_invalid_tool_call_includes_feedback_and_nudge() -> None:
    state = {
        "messages": [
            DummyAIMessage(
                "x",
                tool_calls=[{"name": "exec_command"}],
                additional_kwargs={
                    "invalid_tool_call_errors": [
                        {"kind": "invalid_json", "message": "m", "raw": "{"}
                    ]
                },
            )
        ]
    }
    config = {"configurable": {"no_tool_calls_nudge": "use tools", "logger": None}}
    out = nudge_for_invalid_tool_call(cast(Any, state), cast(Any, config))
    content = out["messages"][0].content
    assert "Error: Invalid tool call format" in content
    assert "invalid_json" in content
    assert content.endswith("use tools")
