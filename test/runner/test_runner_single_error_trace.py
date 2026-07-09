"""Tests for error trace preservation in the single runner."""

from __future__ import annotations

import asyncio
from typing import Any, cast

import httpx
import openai
import pytest
from omegaconf import OmegaConf

from src.config import (
    AppConfig,
    RunnerConfig,
    SourceConfig,
    ScenarioConfig,
    AgentConfig,
    SSHConfig,
)
from src.gym.agent import ModelCallDeadlineExceeded, SessionTraceError
from src.runner import single
from src.scenarios.types import ScenarioInstance


def _cfg(max_retries_per_run: int = 0) -> AppConfig:
    return OmegaConf.structured(
        AppConfig(
            runner=RunnerConfig(
                mode="trace_collection",
                runs_per_item=1,
                output_dir="outputs/test",
                source=SourceConfig(type="static", scenarios=["demo"]),
                max_retries_per_run=max_retries_per_run,
            ),
            scenario=ScenarioConfig(
                name="demo",
                image="demo:latest",
                container_user="lowpriv",
                container_password="trustno1",
                term_cols=120,
                term_rows=40,
            ),
            agent=AgentConfig(
                model="openai/gpt-test",
                api_key="test-key",
                api_base="http://localhost:1234/v1",
                max_turns=60,
            ),
            ssh=SSHConfig(),
        )
    )


def _session_trace_error(
    original_error: BaseException | None = None,
) -> SessionTraceError:
    return SessionTraceError(
        "provider tool-call mismatch",
        history=[
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "thinking"},
            {"role": "tool", "content": '{"got_root": false}'},
        ],
        final_context=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "start"},
            {"role": "assistant", "content": "thinking"},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "exec_command",
                    "description": "Execute command",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        turns=1,
        context={"trimmed_messages": 2},
        timing={
            "assistant_turns": 1,
            "tool_calls_executed": 1,
            "total_llm_ms_raw": 200.0,
            "total_tool_ms_raw": 121.0,
            "llm_call_ms_raw": [200.0],
            "tool_call_ms_raw": [121.0],
            "rollout_wall_clock_ms": 350.0,
            "cost_ms_raw": 321.0,
        },
        usage={
            "llm_usage_by_turn": [
                {
                    "turn": 1,
                    "prompt_tokens": 100,
                    "completion_tokens": 23,
                    "total_tokens": 123,
                    "cost": 0.0042,
                }
            ]
        },
        original_error=original_error or RuntimeError("400"),
        total_tokens=123,
        prompt_tokens=100,
        completion_tokens=23,
        total_cost=0.0042,
    )


def test_error_result_keeps_partial_trace_shape() -> None:
    cfg = _cfg()
    err = _session_trace_error()

    result = single.error_result(cfg, run_index=7, error=err)

    assert result["scenario"] == "error_run_7"
    assert result["status"] == "error"
    assert result["success"] is False
    assert result["history"] == [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": [{"type": "text", "text": "thinking"}]},
        {"role": "tool", "content": '{"got_root": false}'},
    ]
    assert "messages" not in result
    assert "trace" not in result
    assert result["final_context"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "start"},
        {"role": "assistant", "content": [{"type": "text", "text": "thinking"}]},
    ]
    assert result["error"] == str(err)
    assert result["tools"] == err.tools
    assert result["turns"] == 1
    assert result["total_tokens"] == 123
    assert result["prompt_tokens"] == 100
    assert result["completion_tokens"] == 23
    assert result["total_cost"] == pytest.approx(0.0042)
    assert result["metadata"]["trace_incomplete"] is True
    assert result["metadata"]["benchmark_eligible"] is False
    assert result["metadata"]["context"] == err.context
    assert result["timing"] == err.timing
    assert "usage" not in result
    assert result["llm_usage_by_turn"] == err.usage["llm_usage_by_turn"]
    assert result["schema_version"] == 3


@pytest.mark.asyncio
async def test_run_once_emits_session_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg()
    instance = ScenarioInstance(
        id="demo",
        config=ScenarioConfig(
            name="demo",
            image="demo:latest",
            container_user="lowpriv",
            container_password="trustno1",
            term_cols=120,
            term_rows=40,
        ),
        metadata={},
    )

    class _Callback:
        total_tokens = 0
        prompt_tokens = 0
        completion_tokens = 0
        total_cost = 0.0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _Display:
        def session_start(self, *args, **kwargs):
            return None

        def set_token_source(self, source):
            return None

        def session_end(self, *args, **kwargs):
            return None

    async def _fake_run_react_session(*args, **kwargs):
        provider_content = [
            {
                "type": "reasoning",
                "content": [
                    {"type": "reasoning_text", "text": "I found a likely path."}
                ],
            },
            {"type": "text", "text": "I will run a command now."},
            {
                "type": "function_call",
                "name": "exec_command",
                "arguments": '{"command": "id"}',
            },
        ]
        return {
            "success": True,
            "history": [
                {
                    "role": "assistant",
                    "content": provider_content,
                    "tool_calls": [{"function": {"name": "exec_command"}}],
                },
                {"role": "assistant", "content": "Plain repair text."},
            ],
            "final_context": [{"role": "assistant", "content": provider_content}],
            "turns": 2,
            "tools": [],
            "context": {},
            "timing": {
                "assistant_turns": 2,
                "tool_calls_executed": 1,
                "total_llm_ms_raw": 150.0,
                "total_tool_ms_raw": 75.0,
                "llm_call_ms_raw": [70.0, 80.0],
                "tool_call_ms_raw": [75.0],
                "rollout_wall_clock_ms": 260.0,
                "cost_ms_raw": 225.0,
            },
            "usage": {
                "llm_usage_by_turn": [
                    {
                        "turn": 1,
                        "prompt_tokens": 11,
                        "completion_tokens": 7,
                        "total_tokens": 18,
                        "cost": 0.0,
                    },
                    {
                        "turn": 2,
                        "prompt_tokens": 13,
                        "completion_tokens": 5,
                        "total_tokens": 18,
                        "cost": 0.0,
                    },
                ]
            },
            "sft_num_tokens": 321,
            "sft_tokenizer_model": "Qwen/Qwen3-4B-Instruct-2507",
        }

    monkeypatch.setattr(single, "get_openai_callback", lambda: _Callback())
    monkeypatch.setattr(single, "run_react_session", _fake_run_react_session)
    monkeypatch.setattr(single, "_git_commit_hash", lambda: "deadbeef")

    result = await single.run_once(cfg, instance, display=cast(Any, _Display()))

    assert result["success"] is True
    assert result["timing"] == {
        "assistant_turns": 2,
        "tool_calls_executed": 1,
        "total_llm_ms_raw": 150.0,
        "total_tool_ms_raw": 75.0,
        "llm_call_ms_raw": [70.0, 80.0],
        "tool_call_ms_raw": [75.0],
        "rollout_wall_clock_ms": 260.0,
        "cost_ms_raw": 225.0,
    }
    assert result["history"][0]["content"] == [
        {
            "type": "reasoning",
            "content": [{"type": "reasoning_text", "text": "I found a likely path."}],
        },
        {"type": "text", "text": "I will run a command now."},
    ]
    assert result["history"][0]["tool_calls"] == [
        {"function": {"name": "exec_command"}}
    ]
    assert result["history"][1]["content"] == [
        {"type": "text", "text": "Plain repair text."}
    ]
    assert result["final_context"][0]["content"] == result["history"][0]["content"]
    assert "trace" not in result
    assert "messages" not in result
    assert "usage" not in result
    assert result["llm_usage_by_turn"][0]["prompt_tokens"] == 11
    assert result["sft_num_tokens"] == 321
    assert result["sft_tokenizer_model"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert "sft_tokenizer_model" not in result["metadata"]
    assert "error" not in result
    assert "failure_reason" not in result
    assert result["schema_version"] == 3


@pytest.mark.asyncio
async def test_run_with_retries_sets_source_scenario_on_final_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(max_retries_per_run=0)
    instance = ScenarioInstance(
        id="11_cron_calling_user_wildcard",
        config=cfg.scenario,
    )

    async def _raise_run_once(*args, **kwargs):
        raise _session_trace_error()

    monkeypatch.setattr(single, "run_once", _raise_run_once)

    result = await single.run_with_retries(cfg, instance, run_index=11)

    assert result["scenario"] == instance.id
    assert result["status"] == "error"
    assert result["success"] is False
    assert result["attempt"] == 1
    assert result["max_retries"] == 0
    assert result["metadata"]["trace_incomplete"] is True
    assert result["metadata"]["source_scenario"] == instance.id
    assert result["run_index"] == 11
    assert result["metadata"]["run_index"] == 11
    assert result["metadata"]["item_run_ordinal"] == 11
    assert result["metadata"]["item_index"] == 0
    assert result["history"]


@pytest.mark.asyncio
async def test_run_once_marks_live_pre_repair_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg()
    instance = ScenarioInstance(
        id="demo",
        config=ScenarioConfig(
            name="demo",
            image="demo:latest",
            container_user="lowpriv",
            container_password="trustno1",
            term_cols=120,
            term_rows=40,
        ),
        metadata={"generator_name": "sudo_gtfobins", "seed": 42},
    )

    class _Callback:
        total_tokens = 0
        prompt_tokens = 0
        completion_tokens = 0
        total_cost = 0.0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    async def _fake_run_react_session(*args, **kwargs):
        return {
            "success": False,
            "history": [],
            "final_context": [],
            "turns": 1,
            "tools": [],
            "context": {},
            "timing": {},
            "usage": {"llm_usage_by_turn": []},
            "trace_rejected": True,
            "trace_rejection_reasons": [
                "invalid_tool_call_schema (assistant[2].tool_calls[0] exec_command.command Field required)"
            ],
            "trace_metadata": {
                "generator_name": "sudo_gtfobins",
                "seed": 42,
                "model_retries_for_live_quality": [
                    {
                        "turn": 3,
                        "rejected_attempt_count": 1,
                        "rejected_attempts": [
                            {
                                "attempt": 1,
                                "reasons": [
                                    "invalid_tool_call_schema (assistant[2].tool_calls[0] exec_command.command Field required)"
                                ],
                            }
                        ],
                    }
                ],
            },
        }

    monkeypatch.setattr(single, "get_openai_callback", lambda: _Callback())
    monkeypatch.setattr(single, "run_react_session", _fake_run_react_session)
    monkeypatch.setattr(single, "_git_commit_hash", lambda: "deadbeef")

    result = await single.run_once(cfg, instance)

    assert result["status"] == "completed"
    assert result["success"] is False
    assert "error" not in result
    assert result["metadata"]["benchmark_eligible"] is False
    assert result["metadata"]["pre_repair_rejected"] is True
    assert result["metadata"]["pre_repair_rejection_reasons"] == [
        "invalid_tool_call_schema (assistant[2].tool_calls[0] exec_command.command Field required)"
    ]
    assert result["metadata"]["model_retries_for_live_quality"] == [
        {
            "turn": 3,
            "rejected_attempt_count": 1,
            "rejected_attempts": [
                {
                    "attempt": 1,
                    "reasons": [
                        "invalid_tool_call_schema (assistant[2].tool_calls[0] exec_command.command Field required)"
                    ],
                }
            ],
        }
    ]
    assert result["failure_reason"] == (
        "Rejected by live SFT quality filter: invalid_tool_call_schema "
        "(assistant[2].tool_calls[0] exec_command.command Field required)"
    )
    assert result["metadata"]["prompt_vars"]["max_turns"] == int(
        cfg.datasets.sft.quality.max_turns
    )


@pytest.mark.asyncio
async def test_run_with_retries_uses_instance_prompt_vars_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(max_retries_per_run=0)
    instance = ScenarioInstance(
        id="demo",
        config=ScenarioConfig(
            name="demo",
            image="demo:latest",
            container_user="generated_user",
            container_password="generated_pass",
            term_cols=132,
            term_rows=55,
        ),
    )

    async def _raise_run_once(*args, **kwargs):
        raise _session_trace_error()

    monkeypatch.setattr(single, "run_once", _raise_run_once)

    result = await single.run_with_retries(cfg, instance, run_index=0)

    assert result["metadata"]["prompt_vars"] == {
        "user": "generated_user",
        "password": "generated_pass",
        "max_turns": int(cfg.datasets.sft.quality.max_turns),
        "term_cols": 132,
        "term_rows": 55,
    }


@pytest.mark.asyncio
async def test_run_with_retries_routes_retry_message_to_display(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(max_retries_per_run=1)
    instance = ScenarioInstance(id="demo", config=cfg.scenario)
    attempts = 0

    class _Display:
        def __init__(self) -> None:
            self.infos: list[str] = []
            self.reset_attempts = 0

        def reset_attempt(self) -> None:
            self.reset_attempts += 1

        def info(self, message: str) -> None:
            self.infos.append(message)

    async def _fake_run_once(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary failure")
        return {
            "status": "completed",
            "success": True,
            "turns": 1,
            "history": [],
            "metadata": {},
        }

    monkeypatch.setattr(single, "run_once", _fake_run_once)
    display = _Display()

    result = await single.run_with_retries(
        cfg, instance, run_index=0, display=cast(Any, display)
    )

    assert result["success"] is True
    assert display.infos == ["Retry 1/2: temporary failure"]
    assert display.reset_attempts == 2


@pytest.mark.asyncio
async def test_run_with_retries_keeps_console_retry_without_display(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(max_retries_per_run=1)
    instance = ScenarioInstance(id="demo", config=cfg.scenario)
    attempts = 0
    printed: list[str] = []

    async def _fake_run_once(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary failure")
        return {
            "status": "completed",
            "success": True,
            "turns": 1,
            "history": [],
            "metadata": {},
        }

    monkeypatch.setattr(single, "run_once", _fake_run_once)
    monkeypatch.setattr(single.console, "print", lambda message: printed.append(message))

    result = await single.run_with_retries(cfg, instance, run_index=0)

    assert result["success"] is True
    assert printed == ["[yellow]Retry 1/2: temporary failure[/]"]


@pytest.mark.asyncio
async def test_run_with_retries_does_not_log_final_error_to_display(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(max_retries_per_run=0)
    instance = ScenarioInstance(id="demo", config=cfg.scenario)

    class _Display:
        def __init__(self) -> None:
            self.infos: list[str] = []

        def reset_attempt(self) -> None:
            pass

        def info(self, message: str) -> None:
            self.infos.append(message)

    async def _fake_run_once(*args, **kwargs):
        raise RuntimeError("permanent failure")

    monkeypatch.setattr(single, "run_once", _fake_run_once)
    display = _Display()

    result = await single.run_with_retries(
        cfg, instance, run_index=0, display=cast(Any, display)
    )

    assert result["success"] is False
    assert result["error"] == "permanent failure"
    assert display.infos == []


def test_retryable_infra_error_treats_transient_openai_status_as_infra() -> None:
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    response = httpx.Response(429, request=request)
    err = openai.RateLimitError("rate limit", response=response, body=None)

    assert single.is_retryable_infra_error(err) is True


def test_retryable_infra_error_treats_model_call_deadline_as_infra() -> None:
    err = _session_trace_error(ModelCallDeadlineExceeded("deadline"))

    assert single.is_retryable_infra_error(err) is True


def test_retryable_infra_error_treats_ready_command_timeout_as_infra() -> None:
    err = single.ReadyCommandError(
        "Ready command failed with exit 124: timed out",
        exit_status=single.TIMEOUT_EXIT_CODE,
    )

    assert single.is_retryable_infra_error(err) is True


@pytest.mark.asyncio
async def test_run_with_retries_marks_model_call_deadline_not_benchmark_eligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(max_retries_per_run=0)
    cfg.runner.max_infra_retries_per_run = 0
    instance = ScenarioInstance(
        id="demo",
        config=cfg.scenario,
    )

    async def _raise_run_once(*args, **kwargs):
        raise _session_trace_error(ModelCallDeadlineExceeded("deadline"))

    monkeypatch.setattr(single, "run_once", _raise_run_once)

    result = await single.run_with_retries(cfg, instance, run_index=0)

    assert result["status"] == "error"
    assert result["metadata"]["benchmark_eligible"] is False
    assert result["attempt"] == 1
    assert result["max_retries"] == 0


def test_retryable_infra_error_keeps_nontransient_openai_status_benchmark_eligible() -> (
    None
):
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    response = httpx.Response(400, request=request)
    err = openai.BadRequestError("bad request", response=response, body=None)

    assert single.is_retryable_infra_error(err) is False


@pytest.mark.asyncio
async def test_run_with_retries_keeps_partial_trace_on_wall_clock_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(max_retries_per_run=0)
    cfg.runner.run_wall_clock_timeout = 1
    instance = ScenarioInstance(
        id="09_root_password_root",
        config=cfg.scenario,
    )

    async def _cancelled_run_once(*args, **kwargs):
        try:
            await asyncio.sleep(3600)
        except BaseException as exc:
            raise _session_trace_error(original_error=exc) from exc

    async def _fake_wait_for(task: asyncio.Task[dict[str, Any]], timeout: int):
        await asyncio.sleep(0)
        task.cancel()
        return await task

    monkeypatch.setattr(single, "run_once", _cancelled_run_once)
    monkeypatch.setattr(single.asyncio, "wait_for", _fake_wait_for)

    result = await single.run_with_retries(cfg, instance, run_index=9)

    assert result["scenario"] == instance.id
    assert result["status"] == "error"
    assert result["success"] is False
    assert result["attempt"] == 1
    assert result["max_retries"] == 0
    assert result["error"] == "Run timed out after 1s"
    assert result["history"]
    assert result["final_context"]
    assert result["tools"]
    assert result["timing"]
    assert "trace" not in result
    assert "messages" not in result
    assert result["metadata"]["trace_incomplete"] is True
    assert result["metadata"]["timed_out"] is True
    assert result["metadata"]["source_scenario"] == instance.id
    assert result["run_index"] == 9
    assert result["metadata"]["run_index"] == 9
