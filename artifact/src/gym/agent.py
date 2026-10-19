import asyncio
import contextlib
import json
import logging
import time
from typing import Any, Callable, Protocol, cast

import hydra
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from omegaconf import OmegaConf

from src.config import (
    MISSING,
    AppConfig,
    effective_max_assistant_turns,
    register_with_hydra,
)
from src.gym.context import (
    ContextPolicy,
    ContextStats,
    resolve_reserve_tokens,
    trim_context,
)
from src.gym.prompts import initial_messages
from src.gym.scenario import PrivEscScenario
from src.gym.tools import make_privesc_tools, parse_xml_tool_calls
from src.tui.agent_panels import AgentLogger
from src.utils.langchain import (
    build_chat_openai,
    build_llm_kwargs,
    convert_to_trace_messages,
)
from src.utils.pricing import calculate_cost

log = logging.getLogger("privesc_agent")


class TurnGate(Protocol):
    def rejection_reasons(
        self,
        *,
        messages: list[Any],
        timing_stats: dict[str, Any],
        usage_stats: dict[str, Any],
        assistant_turns: int | None = None,
    ) -> list[str]: ...

    def result_fields(self, messages: list[Any]) -> dict[str, Any]: ...


TurnGateFactory = Callable[[AppConfig, list[Any], dict[str, Any]], TurnGate]


class ModelCallDeadlineExceeded(RuntimeError):
    """The runner-owned deadline for one model call was exceeded."""

    pass


class EnvironmentSetupDeadlineExceeded(RuntimeError):
    """The runner-owned deadline for environment setup was exceeded."""

    def __init__(self, timeout: int, elapsed_ms: float) -> None:
        super().__init__(f"Environment setup exceeded {timeout}s deadline")
        self.timeout = timeout
        self.elapsed_ms = elapsed_ms


class LiveTraceRejected(RuntimeError):
    """Raised when a trace-collection prefix fails live SFT quality checks."""

    def __init__(self, reasons: list[str]) -> None:
        super().__init__("Trace rejected by live SFT quality filter")
        self.reasons = reasons


def _noop_logger(*args: Any, **kwargs: Any) -> None:
    return None


def _new_timing_stats() -> dict[str, Any]:
    return {
        "assistant_turns": 0,
        "tool_calls_executed": 0,
        "total_llm_ms_raw": 0.0,
        "total_tool_ms_raw": 0.0,
        "llm_call_ms_raw": [],
        "tool_call_ms_raw": [],
        "rollout_wall_clock_ms": 0.0,
        "time_to_root_wall_clock_ms": None,
        "time_to_root_interaction_ms_raw": None,
        "rollout_start_ns": None,
    }


def _new_usage_stats() -> dict[str, Any]:
    return {
        "llm_usage_by_turn": [],
    }


def _timing_stats(config: RunnableConfig) -> dict[str, Any]:
    configurable = config["configurable"]
    stats = configurable.get("timing_stats")
    if isinstance(stats, dict):
        return stats
    stats = _new_timing_stats()
    configurable["timing_stats"] = stats
    return stats


def _usage_stats(config: RunnableConfig) -> dict[str, Any]:
    configurable = config["configurable"]
    stats = configurable.get("usage_stats")
    if isinstance(stats, dict):
        return stats
    stats = _new_usage_stats()
    configurable["usage_stats"] = stats
    return stats


def _extract_token_usage(
    usage: dict[str, Any],
    *,
    prompt_keys: tuple[str, ...],
    completion_keys: tuple[str, ...],
) -> tuple[int, int, int] | None:
    prompt_tokens = next(
        (usage[key] for key in prompt_keys if isinstance(usage.get(key), int)),
        None,
    )
    completion_tokens = next(
        (usage[key] for key in completion_keys if isinstance(usage.get(key), int)),
        None,
    )
    if not isinstance(prompt_tokens, int) or not isinstance(completion_tokens, int):
        return None
    total_tokens = usage.get("total_tokens")
    if not isinstance(total_tokens, int):
        total_tokens = prompt_tokens + completion_tokens
    return int(prompt_tokens), int(completion_tokens), int(total_tokens)


def _extract_message_usage(message: Any) -> tuple[int, int, int]:
    usage = getattr(message, "usage_metadata", None)
    if isinstance(usage, dict):
        tokens = _extract_token_usage(
            usage,
            prompt_keys=("input_tokens", "prompt_tokens"),
            completion_keys=("output_tokens", "completion_tokens"),
        )
        if tokens is not None:
            return tokens

    response_metadata = getattr(message, "response_metadata", None)
    if isinstance(response_metadata, dict):
        token_usage = response_metadata.get("token_usage")
        if isinstance(token_usage, dict):
            tokens = _extract_token_usage(
                token_usage,
                prompt_keys=("prompt_tokens", "input_tokens"),
                completion_keys=("completion_tokens", "output_tokens"),
            )
            if tokens is not None:
                return tokens
        usage_dict = response_metadata.get("usage")
        if isinstance(usage_dict, dict):
            tokens = _extract_token_usage(
                usage_dict,
                prompt_keys=("prompt_tokens", "input_tokens"),
                completion_keys=("completion_tokens", "output_tokens"),
            )
            if tokens is not None:
                return tokens

    return 0, 0, 0


def _usage_payload(stats: dict[str, Any], model_name: str) -> dict[str, Any]:
    turns = []
    for idx, entry in enumerate(stats.get("llm_usage_by_turn", []), start=1):
        if not isinstance(entry, dict):
            continue
        prompt_tokens = int(entry.get("prompt_tokens", 0))
        completion_tokens = int(entry.get("completion_tokens", 0))
        total_tokens = int(entry.get("total_tokens", prompt_tokens + completion_tokens))
        available = bool(entry.get("available", total_tokens > 0))
        turns.append(
            {
                "turn": idx,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "available": available,
                "cost": (
                    calculate_cost(model_name, prompt_tokens, completion_tokens)
                    if available
                    else 0.0
                ),
            }
        )
    return {"llm_usage_by_turn": turns}


def _timing_payload(stats: dict[str, Any]) -> dict[str, Any]:
    # Export raw timing totals here; these calibration traces are used to choose
    # clipping thresholds offline before reward-time clipping is applied.
    total_llm_ms_raw = float(stats.get("total_llm_ms_raw", 0.0))
    total_tool_ms_raw = float(stats.get("total_tool_ms_raw", 0.0))
    llm_call_ms_raw = [float(value) for value in stats.get("llm_call_ms_raw", [])]
    tool_call_ms_raw = [float(value) for value in stats.get("tool_call_ms_raw", [])]
    return {
        "assistant_turns": int(stats.get("assistant_turns", 0)),
        "tool_calls_executed": int(stats.get("tool_calls_executed", 0)),
        "total_llm_ms_raw": total_llm_ms_raw,
        "total_tool_ms_raw": total_tool_ms_raw,
        "llm_call_ms_raw": llm_call_ms_raw,
        "tool_call_ms_raw": tool_call_ms_raw,
        "rollout_wall_clock_ms": float(stats.get("rollout_wall_clock_ms", 0.0)),
        "time_to_root_wall_clock_ms": stats.get("time_to_root_wall_clock_ms"),
        "time_to_root_interaction_ms_raw": stats.get("time_to_root_interaction_ms_raw"),
        "cost_ms_raw": total_llm_ms_raw + total_tool_ms_raw,
    }


class _TimingLogger:
    def __init__(
        self,
        delegate: AgentLogger | None,
        stats: dict[str, Any],
    ) -> None:
        self._delegate = delegate
        self._stats = stats

    def tool_result(
        self, name: str, result: dict[str, Any], duration_ms: int | None = None
    ) -> None:
        if duration_ms is not None:
            self._stats["tool_calls_executed"] = (
                int(self._stats.get("tool_calls_executed", 0)) + 1
            )
            self._stats["total_tool_ms_raw"] = float(
                self._stats.get("total_tool_ms_raw", 0.0)
            ) + float(duration_ms)
            cast(list[float], self._stats["tool_call_ms_raw"]).append(
                float(duration_ms)
            )
        if (
            bool(result.get("got_root"))
            and self._stats.get("time_to_root_wall_clock_ms") is None
        ):
            rollout_start_ns = self._stats.get("rollout_start_ns")
            if isinstance(rollout_start_ns, int):
                self._stats["time_to_root_wall_clock_ms"] = (
                    time.perf_counter_ns() - rollout_start_ns
                ) / 1_000_000.0
            self._stats["time_to_root_interaction_ms_raw"] = float(
                self._stats.get("total_llm_ms_raw", 0.0)
            ) + float(self._stats.get("total_tool_ms_raw", 0.0))
        if self._delegate:
            self._delegate.tool_result(name, result, duration_ms)

    def __getattr__(self, name: str) -> Any:
        if self._delegate is None:
            return _noop_logger
        return getattr(self._delegate, name)


def _timing_logs_enabled(config: RunnableConfig) -> bool:
    return bool(config["configurable"].get("debug_timing_logs", False))


def _log_timing(config: RunnableConfig, message: str, *args: Any) -> None:
    if _timing_logs_enabled(config):
        log.info(message, *args)


def _agent_runtime_timeout(cfg: AppConfig) -> int:
    runtime = getattr(cfg.agent, "runtime", None)
    if runtime is None:
        return 180
    return int(getattr(runtime, "model_invoke_timeout", 180))


def _agent_debug_timing_logs(cfg: AppConfig) -> bool:
    runtime = getattr(cfg.agent, "runtime", None)
    if runtime is None:
        return False
    return bool(getattr(runtime, "debug_timing_logs", False))


def _environment_setup_timeout(cfg: AppConfig) -> int | None:
    return getattr(cfg.runner, "environment_setup_timeout", None)


async def _cleanup_after_setup_timeout(
    scenario: PrivEscScenario,
    timeout: int,
) -> None:
    cleanup_timeout = min(float(timeout), 30.0)
    with contextlib.suppress(Exception):
        await asyncio.wait_for(scenario.close(), timeout=cleanup_timeout)


async def _enter_scenario_with_timeout(
    cfg: AppConfig,
    logger: AgentLogger,
) -> PrivEscScenario:
    timeout = _environment_setup_timeout(cfg)
    scenario = PrivEscScenario(cfg.ssh, cfg.scenario, logger=logger)
    if timeout is None:
        return await scenario.__aenter__()

    start_ns = time.perf_counter_ns()
    try:
        return await asyncio.wait_for(scenario.__aenter__(), timeout=timeout)
    except TimeoutError as exc:
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
        await _cleanup_after_setup_timeout(scenario, timeout)
        raise EnvironmentSetupDeadlineExceeded(timeout, elapsed_ms) from exc


def _assistant_turns(config: RunnableConfig) -> int:
    return int(_timing_stats(config).get("assistant_turns", 0))


async def call_model(state: MessagesState, config: RunnableConfig) -> dict[str, Any]:
    model = config["configurable"]["model"]
    logger: AgentLogger | None = config["configurable"].get("logger")
    scenario = str(config["configurable"].get("scenario_name", "unknown"))
    max_assistant_turns = config["configurable"].get("max_assistant_turns")
    completed_turns = _assistant_turns(config)
    if max_assistant_turns and completed_turns >= max_assistant_turns:
        log.debug(
            "Max assistant turns (%d) reached before model call, stopping agent",
            max_assistant_turns,
        )
        return {}

    turn = completed_turns + 1
    message = await _generate_assistant_message(
        state=state,
        config=config,
        model=model,
        scenario=scenario,
        turn=turn,
    )
    stats = _timing_stats(config)
    stats["assistant_turns"] = completed_turns + 1

    log.debug("Model response: %d tool_calls", len(message.tool_calls or []))

    if logger:
        content = message.content if isinstance(message.content, str) else ""
        tool_calls = [
            {"name": tc["name"], "args": tc["args"]}
            for tc in (message.tool_calls or [])
        ]
        logger.assistant(content, tool_calls)

    return {"messages": [message]}


async def _generate_assistant_message(
    *,
    state: MessagesState,
    config: RunnableConfig,
    model: Any,
    scenario: str,
    turn: int,
) -> Any:
    max_attempts = _live_turn_max_attempts(config)
    rejected_attempts: list[dict[str, Any]] = []

    for attempt in range(1, max_attempts + 1):
        message = await _invoke_model_attempt(
            state=state,
            config=config,
            model=model,
            scenario=scenario,
            turn=turn,
            attempt=attempt,
        )
        reasons = _live_assistant_turn_rejection_reasons(
            state=state,
            config=config,
            assistant_message=message,
            turn=turn,
        )
        if not reasons:
            _record_model_retries_for_live_quality(
                config,
                turn,
                rejected_attempts,
            )
            return message
        rejected_attempts.append({"attempt": attempt, "reasons": reasons})
        if attempt < max_attempts:
            logger: AgentLogger | None = config["configurable"].get("logger")
            message = (
                "Live trace quality rejected model response; retrying "
                f"({attempt}/{max_attempts}): {', '.join(reasons)}"
            )
            if logger:
                logger.info(message)
            else:
                log.debug(
                    "%s: scenario=%s turn=%s",
                    message,
                    scenario,
                    turn,
                )

    _record_model_retries_for_live_quality(config, turn, rejected_attempts)
    last_reasons = rejected_attempts[-1]["reasons"] if rejected_attempts else []
    raise LiveTraceRejected([str(reason) for reason in last_reasons])


async def _invoke_model_attempt(
    *,
    state: MessagesState,
    config: RunnableConfig,
    model: Any,
    scenario: str,
    turn: int,
    attempt: int,
) -> Any:
    timeout = int(config["configurable"].get("model_invoke_timeout", 180))
    _log_timing(
        config,
        "model_invoke_start scenario=%s turn=%s attempt=%s",
        scenario,
        turn,
        attempt,
    )
    try:
        start_ns = time.perf_counter_ns()
        message = await asyncio.wait_for(
            model.ainvoke(state["messages"]), timeout=timeout
        )
    except TimeoutError as exc:
        raise ModelCallDeadlineExceeded(
            f"Model call exceeded {timeout}s deadline"
        ) from exc

    elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
    _record_llm_attempt(config, message, elapsed_ms)
    message = parse_xml_tool_calls(message)
    _log_timing(
        config,
        "model_invoke_done scenario=%s turn=%s attempt=%s tool_calls=%s",
        scenario,
        turn,
        attempt,
        len(message.tool_calls or []),
    )
    return message


def _live_assistant_turn_rejection_reasons(
    *,
    state: MessagesState,
    config: RunnableConfig,
    assistant_message: Any,
    turn: int,
) -> list[str]:
    configurable = config["configurable"]
    if configurable.get("runner_mode") != "trace_collection":
        return []

    turn_gate = configurable.get("turn_gate")
    timing_stats = configurable.get("timing_stats")
    usage_stats = configurable.get("usage_stats")
    if (
        not _is_turn_gate(turn_gate)
        or not isinstance(timing_stats, dict)
        or not isinstance(usage_stats, dict)
    ):
        return _fallback_live_assistant_turn_rejection_reasons(
            config=config,
            assistant_message=assistant_message,
        )

    gate = cast(TurnGate, turn_gate)
    return gate.rejection_reasons(
        messages=[*state["messages"], assistant_message],
        timing_stats=timing_stats,
        usage_stats=usage_stats,
        assistant_turns=turn,
    )


def _is_turn_gate(value: Any) -> bool:
    return hasattr(value, "rejection_reasons") and hasattr(value, "result_fields")


def _turn_gate_result_fields(turn_gate: Any, messages: list[Any]) -> dict[str, Any]:
    if not _is_turn_gate(turn_gate):
        return {}
    return dict(cast(TurnGate, turn_gate).result_fields(messages))


def _fallback_live_assistant_turn_rejection_reasons(
    *,
    config: RunnableConfig,
    assistant_message: Any,
) -> list[str]:
    if not bool(config["configurable"].get("reject_tool_call_free_turns", False)):
        return []
    tool_calls = getattr(assistant_message, "tool_calls", None)
    return [] if isinstance(tool_calls, list) and tool_calls else ["tool_call_free_turn"]


def _live_turn_max_attempts(config: RunnableConfig) -> int:
    if config["configurable"].get("runner_mode") != "trace_collection":
        return 1
    raw_attempts = config["configurable"].get("live_turn_max_attempts", 1) or 1
    return max(1, int(raw_attempts))


def _record_llm_attempt(
    config: RunnableConfig, message: Any, elapsed_ms: float
) -> None:
    stats = _timing_stats(config)
    stats["total_llm_ms_raw"] = float(stats.get("total_llm_ms_raw", 0.0)) + elapsed_ms
    cast(list[float], stats["llm_call_ms_raw"]).append(elapsed_ms)
    usage_stats = _usage_stats(config)
    prompt_tokens, completion_tokens, total_tokens = _extract_message_usage(message)
    cast(list[dict[str, int | bool]], usage_stats["llm_usage_by_turn"]).append(
        {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "available": any(
                v > 0 for v in (prompt_tokens, completion_tokens, total_tokens)
            ),
        }
    )


def _record_model_retries_for_live_quality(
    config: RunnableConfig, turn: int, rejected_attempts: list[dict[str, Any]]
) -> None:
    if not rejected_attempts:
        return
    metadata = config["configurable"].get("trace_metadata")
    if not isinstance(metadata, dict):
        return
    retry_meta = metadata.get("model_retries_for_live_quality")
    if not isinstance(retry_meta, list):
        retry_meta = []
        metadata["model_retries_for_live_quality"] = retry_meta
    retry_meta.append(
        {
            "turn": turn,
            "rejected_attempt_count": len(rejected_attempts),
            "rejected_attempts": rejected_attempts,
        }
    )


def should_continue(state: MessagesState, config: RunnableConfig) -> str:
    max_user_turns = config["configurable"].get("max_user_turns")
    max_assistant_turns = config["configurable"].get("max_assistant_turns")
    num_user_turns = sum(1 for message in state["messages"] if message.type == "user")

    last_message = state["messages"][-1]

    # LLM call failed, e.g: max response length exceeded
    if last_message.type != "ai":
        log.debug("LLM call failed (last message not AI), stopping agent")
        return END

    if max_user_turns and num_user_turns >= max_user_turns:
        log.debug("Max user turns (%d) exceeded, stopping agent", max_user_turns)
        return END

    if getattr(last_message, "tool_calls", None):
        return "tools"

    if max_assistant_turns and _assistant_turns(config) >= max_assistant_turns:
        log.debug(
            "Max assistant turns (%d) exceeded, stopping agent", max_assistant_turns
        )
        return END

    log.debug("No tool calls detected, adding nudge")
    return "nudge"


def nudge_for_tools(state: MessagesState, config: RunnableConfig) -> dict[str, Any]:
    """
    Inject a short human nudge telling the model to either call tools or finish.
    """
    nudge_text = str(config["configurable"].get("no_tool_calls_nudge", ""))
    if not nudge_text:
        raise ValueError("Missing configurable.no_tool_calls_nudge")

    return _build_nudge_response(
        state=state,
        config=config,
        fallback_invalid_feedback="",
    )


def _invalid_tool_call_feedback_from_state(state: MessagesState) -> str:
    if not state.get("messages"):
        return ""

    last_ai = next(
        (
            msg
            for msg in reversed(state["messages"])
            if getattr(msg, "type", None) == "ai"
        ),
        None,
    )
    if last_ai is None:
        return ""

    additional = getattr(last_ai, "additional_kwargs", None)
    if not isinstance(additional, dict):
        return ""
    raw_errors = additional.get("invalid_tool_call_errors")
    if not isinstance(raw_errors, list) or not raw_errors:
        return ""

    kinds: list[str] = []
    for error in raw_errors:
        if isinstance(error, dict):
            kind = error.get("kind")
            if isinstance(kind, str) and kind:
                kinds.append(kind)

    kind_text = ", ".join(kinds[:2]) if kinds else "invalid_json"
    if len(kinds) > 2:
        kind_text += f", +{len(kinds) - 2} more"

    return (
        "Error: Invalid tool call format "
        f"({kind_text}). Use one JSON object inside each <tool_call> "
        "with keys 'name' and 'arguments'.\n"
    )


def nudge_for_invalid_tool_call(
    state: MessagesState, config: RunnableConfig
) -> dict[str, Any]:
    return _build_nudge_response(
        state=state,
        config=config,
        fallback_invalid_feedback=(
            "Error: Invalid tool call format. Use one JSON object inside each "
            "<tool_call> with keys 'name' and 'arguments'.\n"
        ),
    )


def _build_nudge_response(
    *,
    state: MessagesState,
    config: RunnableConfig,
    fallback_invalid_feedback: str,
) -> dict[str, Any]:
    nudge_text = str(config["configurable"].get("no_tool_calls_nudge", ""))
    if not nudge_text:
        raise ValueError("Missing configurable.no_tool_calls_nudge")

    invalid_feedback = _invalid_tool_call_feedback_from_state(state)
    if not invalid_feedback:
        invalid_feedback = fallback_invalid_feedback

    message_text = f"{invalid_feedback}{nudge_text}"
    logger: AgentLogger | None = config["configurable"].get("logger")
    if logger:
        logger.nudge(message_text)
    return {"messages": [HumanMessage(content=message_text)]}


def got_root(state: MessagesState, config: RunnableConfig):
    """Conditional for graph: return END if any tool result has got_root == True."""
    for message in state["messages"]:
        if message.type == "tool":
            try:
                payload = json.loads(getattr(message, "content", "{}") or "{}")
                if isinstance(payload, dict) and payload.get("got_root"):
                    return END
            except json.JSONDecodeError:
                log.warning(
                    "Failed to decode tool message as JSON: %s", message.content
                )
    if _invalid_tool_call_feedback_from_state(state):
        return "invalid_tool_call_nudge"
    return "agent"


def assess_agent_success_and_turns(
    state: MessagesState, config: RunnableConfig
) -> tuple[bool, int]:
    """Determine if the agent was successful and count turns."""
    success = got_root(state, config) == END
    turns = _assistant_turns(config)
    return success, turns


class SessionTraceError(RuntimeError):
    """Raised when a session fails after partial trace was produced."""

    def __init__(
        self,
        message: str,
        *,
        history: list[dict[str, Any]],
        final_context: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        turns: int,
        context: dict[str, Any],
        timing: dict[str, Any] | None,
        usage: dict[str, Any] | None,
        original_error: BaseException,
        total_tokens: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_cost: float = 0.0,
    ) -> None:
        super().__init__(message)
        self.history = history
        self.final_context = final_context
        self.tools = tools
        self.turns = turns
        self.context = context
        self.timing = timing or {}
        self.usage = usage or {}
        self.original_error = original_error
        self.total_tokens = total_tokens
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_cost = total_cost


async def run_react_session(
    cfg: AppConfig,
    logger: AgentLogger | None = None,
    trace_metadata: dict[str, Any] | None = None,
    turn_gate_factory: TurnGateFactory | None = None,
) -> dict[str, Any]:
    """Run a ReAct agent in an privilege escalation environment.

    Args:
        cfg: Application configuration
        logger: Optional rich logger for session output

    Returns a dict with success flag, messages, and turns.
    """
    log.debug(
        "Starting ReAct session: model=%s scenario=%s max_turns=%d",
        cfg.agent.model,
        cfg.scenario.name,
        effective_max_assistant_turns(cfg),
    )

    # Build ChatOpenAI kwargs from agent params
    params = (
        OmegaConf.to_container(cfg.agent.params, resolve=True)
        if cfg.agent.params and cfg.agent.params != MISSING
        else None
    )
    agent_params = (
        {str(k): v for k, v in params.items()} if isinstance(params, dict) else None
    )
    llm_kwargs = build_llm_kwargs(agent_params)
    # Disable client retries/timeouts so runner policy stays authoritative.
    llm_kwargs.update({"timeout": None, "max_retries": 0})

    llm = build_chat_openai(
        api_base=cfg.agent.api_base,
        api_key=cfg.agent.api_key,
        model=cfg.agent.model,
        **llm_kwargs,
    )

    timing_stats = _new_timing_stats()
    usage_stats = _new_usage_stats()
    display_logger = cast(AgentLogger, _TimingLogger(logger, timing_stats))
    live_trace_metadata = dict(trace_metadata or {})
    messages = initial_messages(cfg)
    captured_messages = list(messages)
    last_context_messages = list(messages)

    try:
        scenario = await _enter_scenario_with_timeout(cfg, display_logger)
    except EnvironmentSetupDeadlineExceeded as exc:
        timing = _timing_payload(timing_stats)
        timing["environment_setup_wall_clock_ms"] = exc.elapsed_ms
        raise SessionTraceError(
            str(exc),
            history=convert_to_trace_messages(captured_messages),
            final_context=convert_to_trace_messages(last_context_messages),
            tools=[],
            turns=0,
            context={
                "failure_phase": "environment_setup",
                "environment_setup_timeout": exc.timeout,
            },
            timing=timing,
            usage=_usage_payload(usage_stats, cfg.agent.model),
            original_error=exc,
        ) from exc

    try:
        tools = make_privesc_tools(scenario)
        turn_gate = (
            turn_gate_factory(cfg, tools, live_trace_metadata)
            if turn_gate_factory is not None
            else None
        )
        llm_with_tools = llm.bind_tools(tools, tool_choice="auto")
        base_tool_node = ToolNode(tools)

        def _capture_context(state: MessagesState) -> None:
            nonlocal last_context_messages
            msgs = state.get("messages", [])
            last_context_messages = list(msgs) if isinstance(msgs, list) else []

        def _capture_result(result: dict[str, Any]) -> dict[str, Any]:
            nonlocal last_context_messages
            result_messages = result.get("messages", [])
            if not isinstance(result_messages, list):
                result_messages = []
            captured_messages.extend(result_messages)
            next_messages = [*last_context_messages, *result_messages]
            last_context_messages = next_messages
            reasons = (
                turn_gate.rejection_reasons(
                    messages=next_messages,
                    timing_stats=timing_stats,
                    usage_stats=usage_stats,
                )
                if turn_gate is not None
                else []
            )
            if reasons:
                raise LiveTraceRejected(reasons)
            return result

        async def call_model_capture(
            state: MessagesState, config: RunnableConfig
        ) -> dict[str, Any]:
            _capture_context(state)
            result = await call_model(state, config)
            return _capture_result(result)

        async def tools_capture(
            state: MessagesState, config: RunnableConfig
        ) -> dict[str, Any]:
            _capture_context(state)
            result = await base_tool_node.ainvoke(state, config=config)
            return _capture_result(result)

        def nudge_capture(
            state: MessagesState, config: RunnableConfig
        ) -> dict[str, Any]:
            _capture_context(state)
            result = nudge_for_tools(state, config)
            return _capture_result(result)

        def invalid_tool_call_nudge_capture(
            state: MessagesState, config: RunnableConfig
        ) -> dict[str, Any]:
            _capture_context(state)
            result = nudge_for_invalid_tool_call(state, config)
            return _capture_result(result)

        graph = StateGraph(MessagesState)  # type: ignore[arg-type]
        graph.add_node("trim_context", trim_context)
        graph.add_node("agent", call_model_capture)
        graph.add_node("tools", tools_capture)
        graph.add_node("nudge", nudge_capture)
        graph.add_node("invalid_tool_call_nudge", invalid_tool_call_nudge_capture)
        graph.set_entry_point("trim_context")
        graph.add_edge("trim_context", "agent")
        graph.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                "nudge": "nudge",
                END: END,
            },
        )
        graph.add_edge("nudge", "trim_context")
        graph.add_conditional_edges(
            "tools",
            got_root,
            {
                "agent": "trim_context",
                "invalid_tool_call_nudge": "invalid_tool_call_nudge",
                END: END,
            },
        )
        graph.add_edge("invalid_tool_call_nudge", "trim_context")
        app = graph.compile()

        context_policy = ContextPolicy(
            enabled=cfg.agent.context_management.enabled,
            max_len=cfg.agent.context_management.max_len,
            reserve_output_tokens=resolve_reserve_tokens(
                cfg.agent.context_management.reserve_ratio,
                cfg.agent.context_management.max_len,
                agent_params,
            ),
            reserve_ratio=cfg.agent.context_management.reserve_ratio,
            agent_max_tokens=(
                agent_params.get("max_tokens")
                if isinstance(agent_params, dict)
                and isinstance(agent_params.get("max_tokens"), int)
                else None
            ),
        )
        context_stats = ContextStats.from_policy(context_policy)
        log.debug(
            "Context policy resolved: enabled=%s max_len=%d reserve_output=%d reserve_ratio=%s agent_max_tokens=%s",
            context_policy.enabled,
            context_policy.max_len,
            context_policy.reserve_output_tokens,
            context_policy.reserve_ratio,
            context_policy.agent_max_tokens,
        )

        config: RunnableConfig = {
            "configurable": {
                "model": llm_with_tools,
                "turn_gate": turn_gate,
                "tools": tools,
                "runner_mode": cfg.runner.mode,
                "scenario_name": cfg.scenario.name,
                "model_invoke_timeout": _agent_runtime_timeout(cfg),
                "reject_tool_call_free_turns": bool(
                    cfg.datasets.sft.quality.reject_tool_call_free_turns
                ),
                "live_turn_max_attempts": int(
                    cfg.datasets.sft.quality.live_turn_max_attempts
                ),
                "debug_timing_logs": _agent_debug_timing_logs(cfg),
                "context_policy": context_policy,
                "context_stats": context_stats,
                "max_assistant_turns": effective_max_assistant_turns(cfg),
                "logger": display_logger,
                "no_tool_calls_nudge": cfg.prompts.no_tool_calls_nudge,
                "trace_metadata": live_trace_metadata,
                "timing_stats": timing_stats,
                "usage_stats": usage_stats,
            },
            "recursion_limit": 1000,
        }
        oa_tools = [convert_to_openai_tool(tool) for tool in tools]

        # Log system prompt
        if messages:
            system_msg = messages[0]
            if hasattr(system_msg, "content"):
                display_logger.system_prompt(str(system_msg.content))

        rollout_start_ns = time.perf_counter_ns()
        timing_stats["rollout_start_ns"] = rollout_start_ns
        try:
            _log_timing(config, "session_graph_start scenario=%s", cfg.scenario.name)
            state = cast(
                MessagesState,
                await app.ainvoke(
                    input={"messages": messages},
                    config=config,
                ),
            )
            timing_stats["rollout_wall_clock_ms"] = (
                time.perf_counter_ns() - rollout_start_ns
            ) / 1_000_000.0
            _log_timing(config, "session_graph_done scenario=%s", cfg.scenario.name)
        except LiveTraceRejected as exc:
            timing_stats["rollout_wall_clock_ms"] = (
                time.perf_counter_ns() - rollout_start_ns
            ) / 1_000_000.0
            history = convert_to_trace_messages(captured_messages)
            final_context = convert_to_trace_messages(last_context_messages)
            partial_turns = _assistant_turns(config)
            return {
                "success": False,
                "history": history,
                "final_context": final_context,
                "turns": partial_turns,
                "tools": oa_tools,
                "context": context_stats.to_dict(),
                "timing": _timing_payload(timing_stats),
                "usage": _usage_payload(usage_stats, cfg.agent.model),
                "trace_rejected": True,
                "trace_rejection_reasons": exc.reasons,
                "trace_metadata": live_trace_metadata,
                **_turn_gate_result_fields(turn_gate, captured_messages),
            }
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            timing_stats["rollout_wall_clock_ms"] = (
                time.perf_counter_ns() - rollout_start_ns
            ) / 1_000_000.0
            history = convert_to_trace_messages(captured_messages)
            final_context = convert_to_trace_messages(last_context_messages)
            partial_turns = _assistant_turns(config)
            error_message = str(exc) or exc.__class__.__name__
            raise SessionTraceError(
                error_message,
                history=history,
                final_context=final_context,
                tools=oa_tools,
                turns=partial_turns,
                context=context_stats.to_dict(),
                timing=_timing_payload(timing_stats),
                usage=_usage_payload(usage_stats, cfg.agent.model),
                original_error=exc,
            ) from exc
    finally:
        await scenario.close()

    success, turns = assess_agent_success_and_turns(state, config)
    log.debug("Session complete: success=%s turns=%d", success, turns)

    oa_messages = convert_to_trace_messages(state["messages"])
    history = convert_to_trace_messages(captured_messages)
    context_stats: ContextStats = config["configurable"]["context_stats"]

    return {
        "success": success,
        "history": history,
        "final_context": oa_messages,
        "turns": turns,
        "tools": oa_tools,
        "context": context_stats.to_dict(),
        "timing": _timing_payload(timing_stats),
        "usage": _usage_payload(usage_stats, cfg.agent.model),
        "trace_metadata": live_trace_metadata,
        **_turn_gate_result_fields(turn_gate, captured_messages),
    }


async def run(cfg: AppConfig):
    await run_react_session(cfg)


@hydra.main(version_base=None, config_path="../../conf", config_name="app")
def main(cfg: AppConfig) -> None:
    logging.getLogger("asyncssh").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    log.info("Starting application with config: %s", cfg)
    asyncio.run(run_react_session(cfg))


if __name__ == "__main__":
    register_with_hydra()
    main()
