import json
from collections.abc import Mapping
from typing import Any

from datasets import Dataset


def _json_object_from_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    if not value.lstrip().startswith("{"):
        return value
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value
    return parsed if isinstance(parsed, Mapping) else value


def _normalize_tool_call(tool_call: Any) -> Any:
    if not isinstance(tool_call, Mapping):
        return tool_call

    function = tool_call.get("function")
    if isinstance(function, Mapping):
        if "arguments" not in function:
            return tool_call
        arguments = _json_object_from_string(function["arguments"])
        if arguments is function["arguments"]:
            return tool_call
        return {**tool_call, "function": {**function, "arguments": arguments}}

    if "arguments" not in tool_call:
        return tool_call
    arguments = _json_object_from_string(tool_call["arguments"])
    if arguments is tool_call["arguments"]:
        return tool_call
    return {**tool_call, "arguments": arguments}


def _copy_on_change(items: list[Any], transform) -> list[Any]:
    copied: list[Any] | None = None
    for index, item in enumerate(items):
        transformed = transform(item)
        if transformed is not item and copied is None:
            copied = items[:index]
        if copied is not None:
            copied.append(transformed)
    return copied if copied is not None else items


def _normalize_conversation(messages: list[Any]) -> list[Any]:
    def normalize_message(message: Any) -> Any:
        if not isinstance(message, Mapping):
            return message

        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            return message

        normalized_tool_calls = _copy_on_change(tool_calls, _normalize_tool_call)
        if normalized_tool_calls is tool_calls:
            return message
        return {**message, "tool_calls": normalized_tool_calls}

    return _copy_on_change(messages, normalize_message)


def normalize_chat_template_messages(messages: list[Any]) -> list[Any]:
    return _normalize_conversation(messages)


def normalize_chat_template_batch(conversations: list[list[Any]]) -> list[list[Any]]:
    return _copy_on_change(conversations, normalize_chat_template_messages)


def apply_chat_templates(
    dataset: Dataset,
    tokenizer,
    tools: list,
    *,
    num_proc: int | None = None,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> Dataset:
    """Apply chat template to all messages and add a 'text' field."""
    map_kwargs: dict[str, Any] = {}
    template_kwargs = dict(chat_template_kwargs or {})
    if num_proc is not None:
        if num_proc < 1:
            raise ValueError("num_proc must be greater than or equal to 1")
        resolved_num_proc = min(num_proc, len(dataset))
        if resolved_num_proc > 1:
            map_kwargs["num_proc"] = resolved_num_proc
    return dataset.map(
        lambda batch: {
            "text": tokenizer.apply_chat_template(
                normalize_chat_template_batch(batch["messages"]),
                tokenize=False,
                tools=tools,
                **template_kwargs,
            )
        },
        batched=True,
        remove_columns=dataset.column_names,
        **map_kwargs,
    )
