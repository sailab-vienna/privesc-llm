from typing import Any, cast

from src.sft.chat_template import (
    apply_chat_templates,
    normalize_chat_template_batch,
    normalize_chat_template_messages,
)


def test_normalize_chat_template_messages_converts_json_string_arguments() -> None:
    messages: list[Any] = [
        {"role": "user", "content": "check identity"},
        {
            "role": "assistant",
            "content": "I will run id.",
            "tool_calls": [
                {
                    "function": {
                        "name": "exec_command",
                        "arguments": '{"command":"id"}',
                    }
                }
            ],
        },
    ]

    normalized = normalize_chat_template_messages(messages)

    original_arguments = messages[1]["tool_calls"][0]["function"]["arguments"]
    normalized_arguments = normalized[1]["tool_calls"][0]["function"]["arguments"]
    assert original_arguments == '{"command":"id"}'
    assert normalized_arguments == {"command": "id"}


def test_normalize_chat_template_messages_supports_batched_conversations() -> None:
    batch: list[Any] = [
        [
            {"role": "assistant", "content": "", "tool_calls": []},
        ],
        [
            {
                "role": "assistant",
                "content": "Trying credentials.",
                "tool_calls": [
                    {
                        "function": {
                            "name": "test_credentials",
                            "arguments": '{"user":"root","password":"toor"}',
                        }
                    }
                ],
            }
        ],
    ]

    normalized = normalize_chat_template_batch(batch)

    assert normalized[1][0]["tool_calls"][0]["function"]["arguments"] == {
        "user": "root",
        "password": "toor",
    }


def test_apply_chat_templates_passes_mapping_arguments_to_tokenizer() -> None:
    class FakeDataset:
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            self.rows = rows
            self.column_names = list(rows[0]) if rows else []

        def map(self, func, *, batched: bool, remove_columns: list[str]):
            assert batched is True
            assert remove_columns == ["messages"]
            batch = {"messages": [row["messages"] for row in self.rows]}
            mapped = func(batch)
            return FakeDataset([{"text": text} for text in mapped["text"]])

        def __getitem__(self, index: int) -> dict[str, Any]:
            return self.rows[index]

    class MappingOnlyTokenizer:
        def apply_chat_template(self, messages, *, tokenize: bool, tools: list) -> list[str]:
            assert tokenize is False
            assert tools == []
            rendered = []
            for conversation in messages:
                tool_calls = conversation[1]["tool_calls"]
                arguments = tool_calls[0]["function"]["arguments"]
                assert arguments == {"command": "id"}
                rendered.append("rendered")
            return rendered

    dataset = FakeDataset(
        [
            {
                "messages": [
                    {"role": "user", "content": "check identity"},
                    {
                        "role": "assistant",
                        "content": "I will run id.",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "exec_command",
                                    "arguments": '{"command":"id"}',
                                }
                            }
                        ],
                    },
                ]
            }
        ]
    )

    rendered = apply_chat_templates(cast(Any, dataset), MappingOnlyTokenizer(), [])

    assert rendered[0]["text"] == "rendered"
