from typing import Any, cast

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState

from src.gym.agent import (
    LiveTraceRejected,
    _new_timing_stats,
    assess_agent_success_and_turns,
    call_model,
    nudge_for_tools,
    should_continue,
)
from src.gym.context import (
    ContextPolicy,
    is_active,
    trim_to_budget,
)


def test_context_trim_drops_old_messages_when_budget_is_small(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "src.gym.context.get_num_tokens_from_messages",
        lambda messages, tools=None: len(messages),
    )
    messages = [
        SystemMessage(content="system"),
        HumanMessage(content="user"),
        AIMessage(content="ai-1"),
        HumanMessage(content="user-2"),
        AIMessage(content="ai-2"),
        HumanMessage(content="user-3"),
    ]

    result = trim_to_budget(
        messages,
        tools=[],
        max_len=3,
        reserve_output_tokens=0,
    )

    assert [m.content for m in result] == ["system", "user", "user-3"]


def test_context_trim_preserves_system_and_initial_user_when_budget_allows(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "src.gym.context.get_num_tokens_from_messages",
        lambda messages, tools=None: len(messages),
    )
    messages = [
        SystemMessage(content="system"),
        HumanMessage(content="user"),
        AIMessage(content="ai-1"),
        HumanMessage(content="user-2"),
        AIMessage(content="ai-2"),
    ]

    result = trim_to_budget(
        messages,
        tools=[],
        max_len=4,
        reserve_output_tokens=0,
    )

    assert [m.content for m in result][:2] == ["system", "user"]


def test_context_is_disabled_for_trace_collection_mode():
    policy = ContextPolicy(enabled=True)
    assert is_active(policy, "trace_collection") is False


def _timing_stats(turns: int) -> dict[str, object]:
    stats = _new_timing_stats()
    stats["assistant_turns"] = turns
    return stats


class _RejectFirstLiveQuality:
    sft_tokenizer_model = "test-tokenizer"

    def sft_num_tokens(self, messages: list[Any]) -> int:
        return 0

    def result_fields(self, messages: list[Any]) -> dict[str, Any]:
        return {
            "sft_num_tokens": self.sft_num_tokens(messages),
            "sft_tokenizer_model": self.sft_tokenizer_model,
        }

    def rejection_reasons(
        self,
        *,
        messages: list[Any],
        timing_stats: dict[str, Any],
        usage_stats: dict[str, Any],
        assistant_turns: int | None = None,
    ) -> list[str]:
        if getattr(messages[-1], "content", "") == "":
            return ["too_many_empty_reasoning (1 > 0)"]
        return []


def test_reserve_output_tokens_uses_model_max_tokens_by_default():
    from src.gym.context import resolve_reserve_tokens

    reserve = resolve_reserve_tokens(
        configured_reserve_ratio=None,
        max_len=32768,
        agent_params={"max_tokens": 4096},
    )
    assert reserve == 4096


def test_should_continue_uses_assistant_turn_counter_when_history_trimmed():
    state = cast(
        MessagesState,
        {
            "messages": [
            HumanMessage(content="start"),
            AIMessage(content="latest"),
            ]
        },
    )
    config: RunnableConfig = {
        "configurable": {
            "max_user_turns": None,
            "max_assistant_turns": 3,
            "timing_stats": _timing_stats(3),
        }
    }

    route = should_continue(state, config)

    assert route == "__end__"


def test_assess_turns_uses_assistant_turn_counter_when_history_trimmed():
    state = cast(
        MessagesState,
        {
            "messages": [
            HumanMessage(content="start"),
            AIMessage(content="latest", tool_calls=[]),
            ]
        },
    )
    config: RunnableConfig = {"configurable": {"timing_stats": _timing_stats(7)}}

    success, turns = assess_agent_success_and_turns(state, config)

    assert success is False
    assert turns == 7


def test_should_continue_ignores_stale_graph_turn_counter():
    state = cast(
        MessagesState,
        {
            "messages": [
                HumanMessage(content="start"),
                AIMessage(content="latest"),
            ],
            "assistant_turns": 1,
        },
    )
    config: RunnableConfig = {
        "configurable": {
            "max_user_turns": None,
            "max_assistant_turns": 3,
            "timing_stats": _timing_stats(3),
        }
    }

    route = should_continue(state, config)

    assert route == "__end__"


def test_assess_turns_ignores_stale_graph_turn_counter():
    state = cast(
        MessagesState,
        {
            "messages": [
                HumanMessage(content="start"),
                AIMessage(content="latest", tool_calls=[]),
            ],
            "assistant_turns": 1,
        },
    )
    config: RunnableConfig = {"configurable": {"timing_stats": _timing_stats(7)}}

    success, turns = assess_agent_success_and_turns(state, config)

    assert success is False
    assert turns == 7


@pytest.mark.asyncio
async def test_call_model_updates_timing_turn_counter():
    class FakeModel:
        async def ainvoke(self, messages):
            return AIMessage(content="thinking")

    state = cast(
        MessagesState,
        {
            "messages": [HumanMessage(content="start")],
            "assistant_turns": 0,
        },
    )
    config: RunnableConfig = {
        "configurable": {
            "model": FakeModel(),
            "scenario_name": "test",
            "model_invoke_timeout": 1,
            "timing_stats": _timing_stats(0),
        }
    }

    result = await call_model(state, config)

    assert list(result.keys()) == ["messages"]
    assert config["configurable"]["timing_stats"]["assistant_turns"] == 1


@pytest.mark.asyncio
async def test_call_model_retries_tool_call_free_response_without_nudge():
    class FakeModel:
        def __init__(self):
            self.calls = 0

        async def ainvoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content="I should inspect the system first.")
            return AIMessage(
                content="I will inspect identity now.",
                tool_calls=[
                    {
                        "name": "exec_command",
                        "args": {"command": "id"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            )

    metadata: dict[str, object] = {}
    state = cast(MessagesState, {"messages": [HumanMessage(content="start")]})
    config: RunnableConfig = {
        "configurable": {
            "model": FakeModel(),
            "scenario_name": "test",
            "model_invoke_timeout": 1,
            "timing_stats": _timing_stats(0),
            "usage_stats": {"llm_usage_by_turn": []},
            "runner_mode": "trace_collection",
            "reject_tool_call_free_turns": True,
            "live_turn_max_attempts": 3,
            "trace_metadata": metadata,
        }
    }

    result = await call_model(state, config)

    assert len(result["messages"]) == 1
    accepted = result["messages"][0]
    assert accepted.content == "I will inspect identity now."
    assert accepted.tool_calls
    assert config["configurable"]["model"].calls == 2
    assert config["configurable"]["timing_stats"]["assistant_turns"] == 1
    assert len(config["configurable"]["usage_stats"]["llm_usage_by_turn"]) == 2
    assert metadata["model_retries_for_live_quality"] == [
        {
            "turn": 1,
            "rejected_attempt_count": 1,
            "rejected_attempts": [
                {"attempt": 1, "reasons": ["tool_call_free_turn"]}
            ],
        }
    ]


@pytest.mark.asyncio
async def test_call_model_rejects_after_tool_call_free_retries():
    class FakeModel:
        def __init__(self):
            self.calls = 0

        async def ainvoke(self, messages):
            self.calls += 1
            return AIMessage(content="I am still thinking without a tool.")

    metadata: dict[str, object] = {}
    state = cast(MessagesState, {"messages": [HumanMessage(content="start")]})
    config: RunnableConfig = {
        "configurable": {
            "model": FakeModel(),
            "scenario_name": "test",
            "model_invoke_timeout": 1,
            "timing_stats": _timing_stats(0),
            "usage_stats": {"llm_usage_by_turn": []},
            "runner_mode": "trace_collection",
            "reject_tool_call_free_turns": True,
            "live_turn_max_attempts": 3,
            "trace_metadata": metadata,
        }
    }

    with pytest.raises(LiveTraceRejected) as exc_info:
        await call_model(state, config)

    assert exc_info.value.reasons == ["tool_call_free_turn"]
    assert config["configurable"]["model"].calls == 3
    assert config["configurable"]["timing_stats"]["assistant_turns"] == 0
    assert len(config["configurable"]["usage_stats"]["llm_usage_by_turn"]) == 3
    assert metadata["model_retries_for_live_quality"] == [
        {
            "turn": 1,
            "rejected_attempt_count": 3,
            "rejected_attempts": [
                {"attempt": 1, "reasons": ["tool_call_free_turn"]},
                {"attempt": 2, "reasons": ["tool_call_free_turn"]},
                {"attempt": 3, "reasons": ["tool_call_free_turn"]},
            ],
        }
    ]


@pytest.mark.asyncio
async def test_call_model_retries_full_live_quality_violation():
    class FakeModel:
        def __init__(self):
            self.calls = 0

        async def ainvoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "exec_command",
                            "args": {"command": "id"},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                )
            return AIMessage(
                content="I will inspect identity and privileges before choosing a privilege-escalation path.",
                tool_calls=[
                    {
                        "name": "exec_command",
                        "args": {"command": "id"},
                        "id": "call_2",
                        "type": "tool_call",
                    }
                ],
            )

    metadata: dict[str, object] = {}
    state = cast(MessagesState, {"messages": [HumanMessage(content="start")]})
    config: RunnableConfig = {
        "configurable": {
            "model": FakeModel(),
            "turn_gate": _RejectFirstLiveQuality(),
            "tools": [],
            "scenario_name": "test",
            "model_invoke_timeout": 1,
            "timing_stats": _timing_stats(0),
            "usage_stats": {"llm_usage_by_turn": []},
            "runner_mode": "trace_collection",
            "live_turn_max_attempts": 3,
            "trace_metadata": metadata,
        }
    }

    result = await call_model(state, config)

    accepted = result["messages"][0]
    assert accepted.content.startswith("I will inspect identity")
    assert config["configurable"]["model"].calls == 2
    retry_meta = cast(
        list[dict[str, object]], metadata["model_retries_for_live_quality"]
    )
    assert retry_meta[0]["rejected_attempts"] == [
        {"attempt": 1, "reasons": ["too_many_empty_reasoning (1 > 0)"]}
    ]


@pytest.mark.asyncio
async def test_call_model_does_not_retry_tool_call_free_evaluation_response():
    class FakeModel:
        def __init__(self):
            self.calls = 0

        async def ainvoke(self, messages):
            self.calls += 1
            return AIMessage(content="I am thinking without a tool.")

    metadata: dict[str, object] = {}
    state = cast(MessagesState, {"messages": [HumanMessage(content="start")]})
    config: RunnableConfig = {
        "configurable": {
            "model": FakeModel(),
            "scenario_name": "benchmark",
            "model_invoke_timeout": 1,
            "timing_stats": _timing_stats(0),
            "usage_stats": {"llm_usage_by_turn": []},
            "runner_mode": "evaluation",
            "reject_tool_call_free_turns": True,
            "live_turn_max_attempts": 3,
            "trace_metadata": metadata,
        }
    }

    result = await call_model(state, config)

    assert len(result["messages"]) == 1
    assert config["configurable"]["model"].calls == 1
    assert config["configurable"]["timing_stats"]["assistant_turns"] == 1
    assert len(config["configurable"]["usage_stats"]["llm_usage_by_turn"]) == 1
    assert "model_retries_for_live_quality" not in metadata


@pytest.mark.asyncio
async def test_no_tool_call_nudge_loop_stops_at_max_turns_with_stale_graph_state():
    class FakeModel:
        async def ainvoke(self, messages):
            return AIMessage(content="Still thinking about sudo tar without a tool call.")

    state = cast(
        MessagesState,
        {
            "messages": [HumanMessage(content="start")],
            "assistant_turns": 0,
        },
    )
    config: RunnableConfig = {
        "configurable": {
            "model": FakeModel(),
            "scenario_name": "05_vuln_sudo_gtfo",
            "model_invoke_timeout": 1,
            "timing_stats": _timing_stats(0),
            "max_assistant_turns": 3,
            "max_user_turns": None,
            "no_tool_calls_nudge": "Use exec_command or test_credentials.",
        }
    }

    ended = False
    for _ in range(10):
        result = await call_model(state, config)
        state = cast(
            MessagesState,
            {
                "messages": [*state["messages"], *result["messages"]],
                # Reproduce the production failure mode where the graph-state turn
                # counter lags while nudges keep re-entering the agent.
                "assistant_turns": 0,
            },
        )

        route = should_continue(state, config)
        if route == "__end__":
            ended = True
            break

        assert route == "nudge"

        nudge = nudge_for_tools(state, config)
        state = cast(
            MessagesState,
            {
                "messages": [*state["messages"], *nudge["messages"]],
                "assistant_turns": 0,
            },
        )

    assert ended is True
    assert config["configurable"]["timing_stats"]["assistant_turns"] == 3
