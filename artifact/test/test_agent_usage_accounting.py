from types import SimpleNamespace
from typing import Any, cast

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from src.config import (
    AgentConfig,
    AppConfig,
    PromptsConfig,
    RunnerConfig,
    ScenarioConfig,
    SSHConfig,
)
from src.dataset.privesc.sft_preprocessing import SFTPromptNormalizer
from src.gym import agent as agent_module
from src.gym.agent import (
    _extract_message_usage,
    _generate_assistant_message,
    _usage_payload,
)
from src.runner.live_trace_quality import LiveTraceQuality


def test_extract_message_usage_prefers_usage_metadata() -> None:
    message = AIMessage(
        content="ok",
        usage_metadata={
            "input_tokens": 11,
            "output_tokens": 7,
            "total_tokens": 18,
        },
    )

    assert _extract_message_usage(message) == (11, 7, 18)


def test_extract_message_usage_falls_back_to_response_metadata() -> None:
    message = AIMessage(
        content="ok",
        response_metadata={
            "token_usage": {
                "prompt_tokens": 13,
                "completion_tokens": 5,
                "total_tokens": 18,
            }
        },
    )

    assert _extract_message_usage(message) == (13, 5, 18)


def test_extract_message_usage_sums_prompt_and_completion_when_total_missing() -> None:
    message = AIMessage(
        content="ok",
        response_metadata={
            "usage": {
                "prompt_tokens": 17,
                "completion_tokens": 6,
            }
        },
    )

    assert _extract_message_usage(message) == (17, 6, 23)


@pytest.mark.parametrize(
    "model",
    [
        "anthropic/claude-opus-4.6",
        "anthropic/claude-opus-4.7",
        "deepseek/deepseek-v4-pro",
    ],
)
def test_usage_payload_adds_per_turn_costs(model: str) -> None:
    payload = _usage_payload(
        {
            "llm_usage_by_turn": [
                {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                {"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23},
            ]
        },
        model,
    )

    turns = payload["llm_usage_by_turn"]
    assert len(turns) == 2
    assert turns[0]["turn"] == 1
    assert turns[1]["turn"] == 2
    assert turns[0]["cost"] > 0.0
    assert turns[1]["cost"] > turns[0]["cost"]


def test_live_trace_payload_counts_normalized_sft_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = AppConfig(
        ssh=SSHConfig(user="root", key_path="key", servers="host:22"),
        scenario=ScenarioConfig(
            name="demo",
            container_user="lowpriv",
            container_password="trustno1",
        ),
        agent=AgentConfig(
            api_key="key",
            api_base="http://localhost:1234/v1",
            model="model",
            max_turns=12,
        ),
        runner=RunnerConfig(
            mode="trace_collection",
            runs_per_item=1,
            output_dir="outputs/test",
        ),
        prompts=PromptsConfig(
            system_template="trace_collection.jinja",
            start_instruction="trace start",
            no_tool_calls_nudge="nudge",
            template_vars={
                "user": "lowpriv",
                "password": "trustno1",
                "max_turns": 12,
                "term_cols": 80,
                "term_rows": 24,
            },
        ),
    )
    counted: dict[str, object] = {}

    def _fake_count(messages, tools=None):
        counted["messages"] = messages
        return 123

    monkeypatch.setattr(
        "src.dataset.privesc.sft_preprocessing.get_num_tokens_from_messages",
        _fake_count,
    )
    live_trace_quality = LiveTraceQuality(
        cfg=cfg,
        tools=[],
        prompt_normalizer=SFTPromptNormalizer(cfg),
        trace_metadata={},
        quality_filter=None,
    )
    payload = live_trace_quality.payload(
        messages=[
            SystemMessage(content="SECRET SOLUTION DATA\nleaked solution"),
            HumanMessage(content="trace start"),
            AIMessage(
                content="Visible answer omitted from SFT.",
                additional_kwargs={"reasoning_content": "I will enumerate first."},
            ),
        ],
        timing_stats={"assistant_turns": 1},
        usage_stats={"llm_usage_by_turn": []},
    )

    normalized = cast(list[dict[str, Any]], counted["messages"])
    assert isinstance(normalized, list)
    assert "SECRET SOLUTION DATA" not in normalized[0]["content"]
    assert normalized[1]["content"].startswith("Start privilege escalation now.")
    assert normalized[2]["content"] == "I will enumerate first."
    assert "SECRET SOLUTION DATA" in payload["history"][0]["content"]
    assert payload["sft_num_tokens"] == 123


@pytest.mark.asyncio
async def test_tool_call_free_retry_logs_info_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Model:
        def __init__(self) -> None:
            self.messages = [
                SimpleNamespace(content="thinking only", tool_calls=[]),
                SimpleNamespace(
                    content="tool",
                    tool_calls=[
                        {
                            "name": "exec_command",
                            "args": {"command": "id"},
                            "id": "call_1",
                        }
                    ],
                ),
            ]

        async def ainvoke(self, _messages):
            return self.messages.pop(0)

    class _Logger:
        def __init__(self) -> None:
            self.infos: list[str] = []

        def info(self, message: str) -> None:
            self.infos.append(message)

    logger = _Logger()
    debug_calls = []
    monkeypatch.setattr(
        agent_module.log, "debug", lambda *args: debug_calls.append(args)
    )
    trace_metadata: dict[str, object] = {}
    config = cast(
        RunnableConfig,
        {
            "configurable": {
                "runner_mode": "trace_collection",
                "reject_tool_call_free_turns": True,
                "live_turn_max_attempts": 3,
                "model_invoke_timeout": 10,
                "trace_metadata": trace_metadata,
                "logger": logger,
            },
        },
    )

    message = await _generate_assistant_message(
        state={"messages": []},
        config=config,
        model=_Model(),
        scenario="demo",
        turn=1,
    )

    assert message.tool_calls
    assert logger.infos == [
        "Live trace quality rejected model response; retrying (1/3): tool_call_free_turn"
    ]
    assert debug_calls == []
    assert trace_metadata["model_retries_for_live_quality"] == [
        {
            "turn": 1,
            "rejected_attempt_count": 1,
            "rejected_attempts": [
                {"attempt": 1, "reasons": ["tool_call_free_turn"]}
            ],
        }
    ]
