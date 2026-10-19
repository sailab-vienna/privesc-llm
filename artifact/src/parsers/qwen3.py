import json
import logging
import re
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel

log = logging.getLogger("reward_parser")


class ToolCall(BaseModel):
    """Tool call extracted from a transcript."""

    name: str
    arguments: str  # JSON-encoded string of the call arguments


class ChatMessage(BaseModel):
    """Chat message with role and content."""

    role: str
    content: str


class ToolResponse(BaseModel):
    """Tool response payload (JSON-encoded)."""

    content: str


class ToolCallParseError(BaseModel):
    """Structured error for malformed tool call blocks."""

    kind: str
    message: str
    raw: str


def tool_call_parse_error_payload(error: ToolCallParseError) -> dict[str, str]:
    """Build a tool-style error payload for malformed tool calls."""
    return {
        "error": f"Invalid tool call ({error.kind})",
        "invalid_tool_call_kind": error.kind,
        "detail": error.message,
        "raw_tool_call": error.raw,
    }


def summarize_tool_call_parse_errors(
    errors: list[ToolCallParseError], max_items: int = 2
) -> str:
    """Return a compact summary suitable for nudges/logging."""
    if not errors:
        return ""
    shown = ", ".join(error.kind for error in errors[:max_items])
    if len(errors) > max_items:
        shown += f", +{len(errors) - max_items} more"
    return f"Invalid tool call blocks: {shown}."


# Tokens
im_start_token: str = "<|im_start|>"
im_end_token: str = "<|im_end|>"
tool_call_start_token: str = "<tool_call>"
tool_call_end_token: str = "</tool_call>"
tool_response_start_token: str = "<tool_response>"
tool_response_end_token: str = "</tool_response>"

# Regex
im_block_regex = re.compile(
    r"<\|im_start\|>([^\r\n]+)\r?\n(.*?)<\|im_end\|>", re.DOTALL
)
tool_call_regex = re.compile(r"\n?<tool_call>(.*?)</tool_call>\n?", re.DOTALL)
tool_response_regex = re.compile(
    r"<tool_response>\n?(.*?)\n?</tool_response>", re.DOTALL
)


def parse_chat_messages(text: str) -> List[ChatMessage]:
    """Split a Qwen3 transcript into messages; tool responses become tool-role messages."""
    messages: List[ChatMessage] = []
    for role, content in im_block_regex.findall(text):
        role = role.strip()
        if tool_response_start_token in content and tool_response_end_token in content:
            _, raws = _parse_tool_responses_raw(content)
            for raw in raws:
                messages.append(ChatMessage(role="tool", content=raw))
        else:
            messages.append(ChatMessage(role=role, content=content))

    return messages


def parse_tool_calls_detailed(
    text: str,
) -> Tuple[str, List[ToolCall], List[ToolCallParseError]]:
    """Extract tool call blocks and return (content_without_calls, calls, errors)."""
    if tool_call_start_token not in text or tool_call_end_token not in text:
        return text, [], []

    matches = tool_call_regex.findall(text)
    function_calls: List[ToolCall] = []
    errors: List[ToolCallParseError] = []
    for match in matches:
        raw = match.strip()
        try:
            function_call = json.loads(raw)
        except Exception as e:
            errors.append(
                ToolCallParseError(kind="invalid_json", message=str(e), raw=raw)
            )
            continue

        if not isinstance(function_call, dict):
            errors.append(
                ToolCallParseError(
                    kind="invalid_shape",
                    message=f"Expected JSON object, got {type(function_call).__name__}",
                    raw=raw,
                )
            )
            continue

        name = function_call.get("name")
        if not isinstance(name, str) or not name:
            errors.append(
                ToolCallParseError(
                    kind="missing_name",
                    message="Tool call missing string field 'name'",
                    raw=raw,
                )
            )
            continue

        try:
            arguments = _extract_tool_call_arguments(function_call)
            function_calls.append(
                ToolCall(name=name, arguments=json.dumps(arguments, ensure_ascii=False))
            )
        except Exception as e:
            errors.append(
                ToolCallParseError(kind="invalid_arguments", message=str(e), raw=raw)
            )

    content = tool_call_regex.sub("", text)
    return content, function_calls, errors


def parse_tool_responses(text: str) -> Tuple[str, List[ToolResponse]]:
    """Extract tool response blocks as ToolResponse objects."""
    if tool_response_start_token not in text or tool_response_end_token not in text:
        return text, []

    matches = tool_response_regex.findall(text)
    responses: List[ToolResponse] = []
    for match in matches:
        try:
            obj = json.loads(match)
            responses.append(ToolResponse(content=json.dumps(obj, ensure_ascii=False)))
        except Exception as e:
            log.error(f"Failed to decode tool response: {e} - {match}")

    content = tool_response_regex.sub("", text)
    return content, responses


def _parse_tool_responses_raw(text: str) -> Tuple[str, List[str]]:
    """Extract raw tool response blocks without JSON validation."""
    if tool_response_start_token not in text or tool_response_end_token not in text:
        return text, []

    raws = tool_response_regex.findall(text)
    content = tool_response_regex.sub("", text)
    return content, raws


def parse_trace_history(text: str) -> Dict[str, Any]:
    """Build an OpenAI-style message list and infer success from a transcript."""
    trace_history: List[Dict[str, Any]] = []
    got_root = False

    for m in parse_chat_messages(text):
        if m.role == "assistant":
            content_wo_calls, calls, errors = parse_tool_calls_detailed(m.content)
            if calls:
                tool_calls = [
                    {
                        "type": "function",
                        "function": {"name": c.name, "arguments": c.arguments},
                    }
                    for c in calls
                ]
                trace_history.append(
                    {
                        "role": "assistant",
                        "content": content_wo_calls,
                        "tool_calls": tool_calls,
                    }
                )
            else:
                trace_history.append({"role": "assistant", "content": content_wo_calls})

            for error in errors:
                trace_history.append(
                    {
                        "role": "tool",
                        "content": json.dumps(
                            tool_call_parse_error_payload(error), ensure_ascii=False
                        ),
                    }
                )
        elif m.role == "tool":
            content_str = m.content
            try:
                payload = json.loads(content_str)
                if bool(payload.get("got_root")):
                    got_root = True
            except Exception:
                pass
            trace_history.append({"role": "tool", "content": content_str})
        else:
            trace_history.append({"role": m.role, "content": m.content})

    return {"messages": trace_history, "success": got_root}


def _extract_tool_call_arguments(function_call: dict[str, Any]) -> Any:
    """Read tool call arguments from either `arguments` or `args`.

    For backward compatibility with existing traces, an explicitly provided empty
    `arguments` object/list without `args` is normalized to `None`.
    """

    if "arguments" in function_call:
        arguments = function_call["arguments"]
        if arguments:
            return arguments
        if "args" in function_call:
            return function_call["args"]
        return None

    if "args" in function_call:
        return function_call["args"]

    raise ValueError("Tool call missing 'arguments' or 'args'")
