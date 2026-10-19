import json
from pathlib import Path

import pytest

from src.parsers.qwen3 import (
    parse_tool_calls_detailed,
    parse_trace_history,
    summarize_tool_call_parse_errors,
    tool_call_parse_error_payload,
)

TRAJ_DIR = Path(__file__).parent / "trajectories"


def _read_chat(name: str) -> str:
    return (TRAJ_DIR / name).read_text(encoding="utf-8")


def _load_trace(i: int) -> dict:
    return json.loads((TRAJ_DIR / f"trace_{i}.json").read_text(encoding="utf-8"))


def _remove_think_tags(text: str) -> str:
    return (
        text.replace("<think>\n", "")
        .replace("\n</think>\n\n", "")
        .replace("</think>", "")
    )


def _assert_tools_against_trace(i: int):
    chat = _read_chat(f"chat_{i}.txt")
    expected_trace = _load_trace(i)
    parsed_trace = parse_trace_history(chat)

    assert parsed_trace["success"] == expected_trace["success"]
    for i, msg in enumerate(parsed_trace["messages"]):
        if msg["role"] != "system":
            expected_msg = expected_trace["messages"][i]
            # Qwen3 chat template adds some think tags that were not in the original
            # messages, so we need to remove them for comparison.
            msg_without_think = _remove_think_tags(msg["content"])
            expected_without_think = _remove_think_tags(expected_msg["content"])
            assert msg_without_think == expected_without_think
            expected_tool_calls = expected_msg.get("tool_calls", [])
            for j, tool_call in enumerate(msg.get("tool_calls", [])):
                expected_tool_call = expected_tool_calls[j]
                expected_tool_call.pop("id")
                assert tool_call == expected_tool_call


@pytest.mark.parametrize("i", range(6))
def test_chat_against_trace(i):
    _assert_tools_against_trace(i)


def test_parse_tool_calls_detailed_reports_missing_name_error() -> None:
    content, calls, errors = parse_tool_calls_detailed(
        '<tool_call>{"command":"id"}</tool_call>'
    )
    assert content == ""
    assert calls == []
    assert len(errors) == 1
    assert errors[0].kind == "missing_name"


def test_parse_trace_history_emits_tool_error_for_invalid_tool_call() -> None:
    chat = '<|im_start|>assistant\nabc<tool_call>{"command":"id"}</tool_call><|im_end|>'
    parsed = parse_trace_history(chat)
    assert parsed["messages"][0] == {"role": "assistant", "content": "abc"}
    tool_msg = parsed["messages"][1]
    assert tool_msg["role"] == "tool"
    payload = json.loads(tool_msg["content"])
    assert payload["invalid_tool_call_kind"] == "missing_name"
    assert payload["error"].startswith("Invalid tool call")


def test_tool_call_error_helpers_produce_stable_feedback() -> None:
    _, _, errors = parse_tool_calls_detailed('<tool_call>{"command":"id"}</tool_call>')
    payload = tool_call_parse_error_payload(errors[0])
    assert set(payload.keys()) == {
        "error",
        "invalid_tool_call_kind",
        "detail",
        "raw_tool_call",
    }
    summary = summarize_tool_call_parse_errors(errors)
    assert "missing_name" in summary


@pytest.mark.parametrize(
    ("raw", "expected_kind"),
    [
        ('<tool_call>{"name":"exec_command"</tool_call>', "invalid_json"),
        ("<tool_call>[]</tool_call>", "invalid_shape"),
        ('<tool_call>{"command":"id"}</tool_call>', "missing_name"),
        ('<tool_call>{"name":"exec_command"}</tool_call>', "invalid_arguments"),
    ],
)
def test_parse_tool_calls_detailed_covers_all_error_kinds(
    raw: str, expected_kind: str
) -> None:
    _, _, errors = parse_tool_calls_detailed(raw)
    assert len(errors) == 1
    assert errors[0].kind == expected_kind


@pytest.mark.parametrize(
    ("raw", "expected_kind"),
    [
        (
            '<|im_start|>assistant\n<tool_call>{"name":"exec_command"</tool_call><|im_end|>',
            "invalid_json",
        ),
        ("<|im_start|>assistant\n<tool_call>[]</tool_call><|im_end|>", "invalid_shape"),
        (
            '<|im_start|>assistant\n<tool_call>{"command":"id"}</tool_call><|im_end|>',
            "missing_name",
        ),
        (
            '<|im_start|>assistant\n<tool_call>{"name":"exec_command"}</tool_call><|im_end|>',
            "invalid_arguments",
        ),
    ],
)
def test_parse_trace_history_emits_specific_invalid_tool_call_kind(
    raw: str, expected_kind: str
) -> None:
    parsed = parse_trace_history(raw)
    tool_msgs = [m for m in parsed["messages"] if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    payload = json.loads(tool_msgs[0]["content"])
    assert payload["invalid_tool_call_kind"] == expected_kind
