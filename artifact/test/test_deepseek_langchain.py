from typing import cast

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.utils.langchain import (
    OpenRouterReasoningChatOpenAI,
    ReasoningContentChatOpenAI,
    build_chat_openai,
    build_llm_kwargs,
    convert_to_trace_messages,
)


def _client() -> ReasoningContentChatOpenAI:
    return cast(
        ReasoningContentChatOpenAI,
        build_chat_openai(
            api_base="https://api.deepseek.com",
            api_key="test",
            model="deepseek-v4-flash",
        ),
    )


def test_build_llm_kwargs_passes_reasoning_effort_explicitly() -> None:
    kwargs = build_llm_kwargs(
        {
            "reasoning_effort": "high",
            "extra_body": {"thinking": {"type": "enabled"}},
        }
    )

    assert kwargs["reasoning_effort"] == "high"
    assert kwargs["extra_body"] == {"thinking": {"type": "enabled"}}
    assert "model_kwargs" not in kwargs


def test_reasoning_payload_adapters_match_deepseek_route() -> None:
    direct = build_chat_openai(
        api_base="https://api.deepseek.com",
        api_key="test",
        model="deepseek-v4-flash",
    )
    routed = build_chat_openai(
        api_base="https://openrouter.ai/api/v1",
        api_key="test",
        model="deepseek/deepseek-v4-flash",
    )

    assert isinstance(direct, ReasoningContentChatOpenAI)
    assert isinstance(routed, OpenRouterReasoningChatOpenAI)


def test_deepseek_payload_preserves_reasoning_content_for_tool_turn() -> None:
    assistant = AIMessage(
        content="",
        additional_kwargs={"reasoning_content": "I need to inspect privileges."},
        tool_calls=[
            {
                "name": "exec_command",
                "args": {"command": "id"},
                "id": "call_1",
                "type": "tool_call",
            }
        ],
    )

    payload = _client()._get_request_payload(
        [
            HumanMessage(content="start"),
            assistant,
            ToolMessage(
                content='{"output": "uid=1000(user)"}',
                tool_call_id="call_1",
            ),
        ]
    )

    assert payload["messages"][1]["reasoning_content"] == (
        "I need to inspect privileges."
    )


def test_deepseek_response_preserves_reasoning_content_on_ai_message() -> None:
    result = _client()._create_chat_result(
        {
            "id": "chatcmpl-test",
            "model": "deepseek-v4-flash",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Visible plan.",
                        "reasoning_content": "Hidden reasoning.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )

    message = result.generations[0].message
    assert message.content == "Visible plan."
    assert message.additional_kwargs["reasoning_content"] == "Hidden reasoning."


def test_openrouter_response_preserves_reasoning_on_ai_message() -> None:
    client = build_chat_openai(
        api_base="https://openrouter.ai/api/v1",
        api_key="test",
        model="deepseek/deepseek-v4-flash",
    )

    result = client._create_chat_result(
        {
            "id": "chatcmpl-test",
            "model": "deepseek/deepseek-v4-flash",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "exec_command",
                                    "arguments": '{"command": "id"}',
                                },
                            }
                        ],
                        "reasoning": "Hidden OpenRouter reasoning.",
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )

    message = result.generations[0].message
    assert message.additional_kwargs["reasoning_content"] == (
        "Hidden OpenRouter reasoning."
    )
    assert message.tool_calls[0]["name"] == "exec_command"


def test_openrouter_payload_preserves_prior_reasoning_content() -> None:
    client = build_chat_openai(
        api_base="https://openrouter.ai/api/v1",
        api_key="test",
        model="deepseek/deepseek-v4-flash",
    )
    assistant = AIMessage(
        content="",
        additional_kwargs={"reasoning_content": "I need to inspect privileges."},
        tool_calls=[
            {
                "name": "exec_command",
                "args": {"command": "id"},
                "id": "call_1",
                "type": "tool_call",
            }
        ],
    )

    payload = client._get_request_payload(
        [
            HumanMessage(content="start"),
            assistant,
            ToolMessage(
                content='{"output": "uid=1000(user)"}',
                tool_call_id="call_1",
            ),
        ]
    )

    assert payload["messages"][1]["reasoning"] == "I need to inspect privileges."
    assert "reasoning_content" not in payload["messages"][1]


def test_trace_messages_store_deepseek_reasoning_as_content_block() -> None:
    messages = convert_to_trace_messages(
        [
            AIMessage(
                content="Visible summary.",
                additional_kwargs={"reasoning_content": "Hidden reasoning."},
            )
        ]
    )

    assert messages == [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "reasoning",
                    "content": [
                        {
                            "type": "reasoning_text",
                            "text": "Hidden reasoning.",
                        }
                    ],
                },
                {"type": "text", "text": "Visible summary."},
            ],
        }
    ]
