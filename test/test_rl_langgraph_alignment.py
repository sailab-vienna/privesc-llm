import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from src.parsers.qwen3 import parse_trace_history
from src.parsers.tool_calls import parse_tool_calls_detailed
from src.vf_envs.reward import calculate_privesc_reward


def _extract_functions(
    path: str, names: list[str], globals_ns: dict[str, Any]
) -> dict[str, Any]:
    source = Path(path).read_text(encoding="utf-8")
    module = ast.parse(source)
    selected = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name in set(names)
    ]
    exec(
        compile(ast.Module(body=cast(list[ast.stmt], selected), type_ignores=[]), path, "exec"),
        globals_ns,
    )
    return {name: globals_ns[name] for name in names}


class DummyAIMessage:
    def __init__(self, content: str, tool_calls=None, additional_kwargs=None):
        self.type = "ai"
        self.content = content
        self.tool_calls = tool_calls or []
        self.additional_kwargs = additional_kwargs or {}


def _load_langgraph_functions() -> dict[str, Any]:
    tools_fns = _extract_functions(
        "src/gym/tools.py",
        ["parse_xml_tool_calls"],
        {
            "parse_tool_calls_detailed": parse_tool_calls_detailed,
            "summarize_tool_call_parse_errors": lambda errors: "",
            "json": __import__("json"),
            "uuid": __import__("uuid"),
            "log": SimpleNamespace(
                debug=lambda *a, **k: None, warning=lambda *a, **k: None
            ),
        },
    )
    agent_fns = _extract_functions(
        "src/gym/agent.py",
        ["should_continue", "_invalid_tool_call_feedback_from_state", "got_root"],
        {
            "END": "__END__",
            "MessagesState": dict,
            "RunnableConfig": dict,
            "json": __import__("json"),
            "_assistant_turns": lambda config: int(
                config.get("configurable", {})
                .get("timing_stats", {})
                .get("assistant_turns", 0)
            ),
            "log": SimpleNamespace(
                debug=lambda *a, **k: None, warning=lambda *a, **k: None
            ),
        },
    )
    return {**tools_fns, **agent_fns}


def test_invalid_only_tool_call_alignment_between_langgraph_and_rl() -> None:
    fns = _load_langgraph_functions()
    parse_xml_tool_calls = fns["parse_xml_tool_calls"]
    should_continue = fns["should_continue"]

    content = 'thought<tool_call>{"command":"id"}</tool_call>'

    # LangGraph: invalid-only call -> no tool_calls, invalid metadata, nudge path.
    ai_msg = DummyAIMessage(content)
    ai_msg = parse_xml_tool_calls(ai_msg)
    assert ai_msg.tool_calls == []
    errors = ai_msg.additional_kwargs.get("invalid_tool_call_errors")
    assert isinstance(errors, list) and errors and errors[0]["kind"] == "missing_name"
    route = should_continue(
        {"messages": [ai_msg]},
        {"configurable": {"max_user_turns": None, "max_assistant_turns": None}},
    )
    assert route == "nudge"

    # RL: same output should produce invalid-tool-call signal and not no-tool-calls.
    transcript = f"<|im_start|>assistant\n{content}<|im_end|>"
    parsed = parse_trace_history(transcript)
    result = calculate_privesc_reward(
        trace_history=parsed["messages"], got_root=False, max_turns=50
    )
    assert result.metadata["total_invalid_tool_calls"] == 1
    assert result.metadata["total_invalid_tool_calls_missing_name"] == 1
    assert result.metadata["no_tool_call_count"] == 0


def test_mixed_valid_invalid_alignment_between_langgraph_and_rl() -> None:
    fns = _load_langgraph_functions()
    parse_xml_tool_calls = fns["parse_xml_tool_calls"]
    should_continue = fns["should_continue"]
    got_root = fns["got_root"]

    content = (
        'thought<tool_call>{"name":"exec_command","arguments":{"command":"id"}}</tool_call>'
        '<tool_call>{"command":"whoami"}</tool_call>'
    )

    # LangGraph: valid calls still execute, malformed call is surfaced for feedback.
    ai_msg = DummyAIMessage(content)
    ai_msg = parse_xml_tool_calls(ai_msg)
    assert len(ai_msg.tool_calls) == 1
    assert ai_msg.tool_calls[0]["name"] == "exec_command"
    errors = ai_msg.additional_kwargs.get("invalid_tool_call_errors")
    assert isinstance(errors, list) and errors and errors[0]["kind"] == "missing_name"

    route = should_continue(
        {"messages": [ai_msg]},
        {"configurable": {"max_user_turns": None, "max_assistant_turns": None}},
    )
    assert route == "tools"

    # After tool execution without root, LangGraph routes to invalid-tool-call nudge.
    post_tools_route = got_root(
        {
            "messages": [
                ai_msg,
                SimpleNamespace(type="tool", content='{"got_root": false}'),
            ]
        },
        {"configurable": {}},
    )
    assert post_tools_route == "invalid_tool_call_nudge"

    # RL: same output keeps valid tool calls and tracks invalid kind.
    transcript = f"<|im_start|>assistant\n{content}<|im_end|>"
    parsed = parse_trace_history(transcript)
    result = calculate_privesc_reward(
        trace_history=parsed["messages"], got_root=False, max_turns=50
    )
    assert result.metadata["total_invalid_tool_calls"] == 1
    assert result.metadata["total_invalid_tool_calls_missing_name"] == 1
