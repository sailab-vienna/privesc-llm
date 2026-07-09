import numpy as np
import pytest

from langchain_core.messages import HumanMessage

from src.gym import context


class _StaticTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return {"input_ids": np.array([[1, 2, 3]])}


def test_get_num_tokens_from_messages_uses_chat_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(context, "_get_tokenizer", lambda: _StaticTokenizer())
    tokens = context.get_num_tokens_from_messages(
        [HumanMessage(content="hi")], tools=[]
    )
    assert tokens == 3


def test_get_num_tokens_from_messages_passes_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ToolAwareTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert "tools" in kwargs
            return {"input_ids": np.array([[1, 2, 3, 4]])}

    monkeypatch.setattr(context, "_get_tokenizer", lambda: _ToolAwareTokenizer())
    tokens = context.get_num_tokens_from_messages(
        [HumanMessage(content="hi")],
        tools=[{"type": "function", "function": {"name": "x", "description": "d"}}],
    )
    assert tokens == 4
