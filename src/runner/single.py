"""Single-worker runner with verbose panel output."""

import asyncio
from functools import lru_cache
from pathlib import Path
import subprocess
from typing import Any

import asyncssh
import httpx
import openai
from langchain_community.callbacks import get_openai_callback
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)

from src.config import AppConfig, effective_max_assistant_turns
from src.gym.backends.base import (
    TIMEOUT_EXIT_CODE,
    TRANSIENT_CONN_ERRORS,
    ReadyCommandError,
)
from src.gym.agent import (
    EnvironmentSetupDeadlineExceeded,
    ModelCallDeadlineExceeded,
    SessionTraceError,
    run_react_session,
)
from src.scenarios import build_scenario_source
from src.utils.pricing import calculate_cost
from src.tui.agent_panels import AgentDisplay, render_summary, AgentLogger
from src.tui.state import RunnerState
from .live_trace_quality import build_live_trace_quality
from .result_policy import is_exit_status_125_infra_error
from .schedule import build_schedule, save_result, update_cfg_with_instance

console = Console()

RETRYABLE_INFRA_ERRORS = TRANSIENT_CONN_ERRORS + (
    httpx.TimeoutException,
    httpx.NetworkError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    EnvironmentSetupDeadlineExceeded,
    ModelCallDeadlineExceeded,
)

_RETRYABLE_OPENAI_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


def _root_error(error: BaseException) -> BaseException:
    if isinstance(error, SessionTraceError):
        return _root_error(error.original_error)
    return error


def _is_retryable_openai_status_error(error: BaseException) -> bool:
    if not isinstance(error, openai.APIStatusError):
        return False
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        return status_code in _RETRYABLE_OPENAI_STATUS_CODES
    response = getattr(error, "response", None)
    response_status = getattr(response, "status_code", None)
    return (
        isinstance(response_status, int)
        and response_status in _RETRYABLE_OPENAI_STATUS_CODES
    )


def is_retryable_infra_error(error: Exception) -> bool:
    root = _root_error(error)
    if isinstance(root, asyncssh.PermissionDenied):
        return False
    if is_exit_status_125_infra_error(root):
        return True
    if (
        isinstance(root, ReadyCommandError)
        and root.exit_status == TIMEOUT_EXIT_CODE
    ):
        return True
    return isinstance(
        root, RETRYABLE_INFRA_ERRORS
    ) or _is_retryable_openai_status_error(root)


def is_wall_clock_timeout_error(error: Exception) -> bool:
    return isinstance(_root_error(error), asyncio.CancelledError)


def _max_infra_retries(cfg: AppConfig) -> int:
    return int(
        getattr(cfg.runner, "max_infra_retries_per_run", cfg.runner.max_retries_per_run)
    )


def _trace_rejection_message(reasons: list[str]) -> str:
    detail = "; ".join(reason for reason in reasons if reason)
    if not detail:
        return "Rejected by live SFT quality filter"
    return f"Rejected by live SFT quality filter: {detail}"


def _show_retry_notice(display: AgentLogger | None, message: str) -> None:
    if display is not None:
        display.info(message)
    else:
        console.print(f"[yellow]{message}[/]")


def _print_without_display(display: AgentLogger | None, console_message: str) -> None:
    if display is None:
        console.print(console_message)


def _attach_run_identity(
    cfg: AppConfig,
    result: dict[str, Any],
    instance: Any,
    run_index: int,
) -> dict[str, Any]:
    result["run_index"] = run_index
    metadata = result.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        result["metadata"] = metadata
    if isinstance(getattr(instance, "metadata", None), dict):
        for key, value in instance.metadata.items():
            metadata.setdefault(key, value)
    metadata["run_index"] = run_index

    source_cfg = getattr(cfg.runner, "source", None)
    if source_cfg is None:
        return result

    items: list[str] | None = None
    item_name: str | None = None
    if source_cfg.type == "static" and source_cfg.scenarios:
        items = list(source_cfg.scenarios)
        item_name = instance.id
    elif source_cfg.type == "procedural" and source_cfg.generators:
        items = list(source_cfg.generators)
        item_name = str(metadata.get("generator_name") or instance.id)

    if not items or not item_name:
        return result

    num_items = len(items)
    if num_items <= 0:
        return result

    metadata["item_run_ordinal"] = run_index // num_items
    metadata["item_index"] = run_index % num_items
    return result


def _llm_usage_by_turn(usage: Any) -> list[dict[str, Any]]:
    if not isinstance(usage, dict):
        return []
    turns = usage.get("llm_usage_by_turn")
    if not isinstance(turns, list):
        return []
    return [turn for turn in turns if isinstance(turn, dict)]


def _trace_content_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    if not isinstance(content, list):
        return []

    blocks: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "function_call":
            continue
        blocks.append(part)
    return blocks


def _normalize_trace_messages(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        return []
    normalized: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        item = dict(message)
        if item.get("role") == "assistant":
            item["content"] = _trace_content_blocks(item.get("content", ""))
        normalized.append(item)
    return normalized


async def run_once(
    cfg: AppConfig,
    instance,
    display: AgentLogger | None = None,
) -> dict[str, Any]:
    run_cfg = update_cfg_with_instance(cfg, instance)

    if display is None:
        generator_name = (
            instance.metadata.get("generator_name", "") if instance.metadata else ""
        )
        label = (
            f"{instance.id} ({generator_name})"
            if generator_name and generator_name != instance.id
            else instance.id
        )
        display = AgentDisplay(label, console=console)
        display.session_start(
            run_cfg.agent.model,
            effective_max_assistant_turns(run_cfg),
            instance.metadata,
        )

    with get_openai_callback() as cb:
        if display:
            display.set_token_source(cb)
        try:
            session = await run_react_session(
                run_cfg,
                display,
                trace_metadata=dict(instance.metadata or {}),
                turn_gate_factory=(
                    build_live_trace_quality
                    if run_cfg.runner.mode == "trace_collection"
                    else None
                ),
            )
        except SessionTraceError as exc:
            cost = cb.total_cost
            if cost == 0.0 and cb.total_tokens > 0:
                cost = calculate_cost(
                    run_cfg.agent.model, cb.prompt_tokens, cb.completion_tokens
                )
            raise SessionTraceError(
                str(exc),
                history=exc.history,
                final_context=exc.final_context,
                tools=exc.tools,
                turns=exc.turns,
                context=exc.context,
                timing=exc.timing,
                usage=exc.usage,
                original_error=exc.original_error,
                total_tokens=cb.total_tokens,
                prompt_tokens=cb.prompt_tokens,
                completion_tokens=cb.completion_tokens,
                total_cost=cost,
            ) from exc

    success = bool(session.get("success"))
    trace_rejected = bool(session.get("trace_rejected"))
    trace_rejection_reasons = [
        str(reason) for reason in session.get("trace_rejection_reasons", [])
    ]
    failure_reason = (
        _trace_rejection_message(trace_rejection_reasons) if trace_rejected else None
    )
    turns = int(session.get("turns", 0))
    cost = cb.total_cost
    if cost == 0.0 and cb.total_tokens > 0:
        cost = calculate_cost(
            run_cfg.agent.model, cb.prompt_tokens, cb.completion_tokens
        )

    if hasattr(display, "session_end"):
        display.session_end(
            success,
            turns,
            cb.total_tokens,
            cost,
            rejected=trace_rejected,
            message=failure_reason,
        )

    git_commit = _git_commit_hash()
    session_metadata = session.get("trace_metadata")
    metadata = (
        dict(session_metadata)
        if isinstance(session_metadata, dict)
        else dict(instance.metadata or {})
    )
    metadata["git_commit"] = git_commit
    metadata["scenario_backend"] = run_cfg.scenario.backend
    metadata["agent_api_base"] = run_cfg.agent.api_base
    metadata["run_wall_clock_timeout"] = getattr(
        run_cfg.runner, "run_wall_clock_timeout", None
    )
    metadata["prompt_vars"] = {
        "user": run_cfg.scenario.container_user,
        "password": run_cfg.scenario.container_password,
        "max_turns": effective_max_assistant_turns(run_cfg),
        "term_cols": run_cfg.scenario.term_cols,
        "term_rows": run_cfg.scenario.term_rows,
    }
    context_data = session.get("context", {})
    metadata["context"] = context_data
    metadata["benchmark_eligible"] = not trace_rejected
    if trace_rejected:
        metadata["pre_repair_rejected"] = True
        metadata["pre_repair_rejection_reasons"] = trace_rejection_reasons
    timing = session.get("timing", {})

    history = _normalize_trace_messages(session.get("history", []))
    usage = session.get("usage", {})
    llm_usage_by_turn = _llm_usage_by_turn(usage)
    sft_num_tokens = session.get("sft_num_tokens")
    sft_tokenizer_model = session.get("sft_tokenizer_model")

    result = {
        "model": run_cfg.agent.model,
        "scenario": instance.id,
        "mode": cfg.runner.mode,
        "status": "completed",
        "success": success,
        "turns": turns,
        "history": history,
        "final_context": _normalize_trace_messages(session.get("final_context", [])),
        "tools": session.get("tools", []),
        "total_tokens": cb.total_tokens,
        "prompt_tokens": cb.prompt_tokens,
        "completion_tokens": cb.completion_tokens,
        "total_cost": cost,
        "timing": timing,
        "schema_version": 3,
        "metadata": metadata,
    }
    if llm_usage_by_turn:
        result["llm_usage_by_turn"] = llm_usage_by_turn
    if failure_reason:
        result["failure_reason"] = failure_reason
    if isinstance(sft_num_tokens, int):
        result["sft_num_tokens"] = sft_num_tokens
    if isinstance(sft_tokenizer_model, str):
        result["sft_tokenizer_model"] = sft_tokenizer_model
    return result


def error_result(
    cfg: AppConfig,
    run_index: int,
    error: Exception,
    scenario_id: str | None = None,
    scenario_cfg: Any | None = None,
) -> dict[str, Any]:
    history: list[dict[str, Any]] = []
    final_context: list[dict[str, Any]] = []
    tools: list[dict[str, Any]] = []
    turns = 0
    total_tokens = 0
    prompt_tokens = 0
    completion_tokens = 0
    total_cost = 0.0
    context: dict[str, Any] = {}
    timing: dict[str, Any] = {}
    usage: dict[str, Any] = {}

    if isinstance(error, SessionTraceError):
        history = _normalize_trace_messages(error.history)
        final_context = _normalize_trace_messages(error.final_context)
        tools = error.tools
        turns = error.turns
        total_tokens = error.total_tokens
        prompt_tokens = error.prompt_tokens
        completion_tokens = error.completion_tokens
        total_cost = error.total_cost
        context = error.context
        timing = error.timing
        usage = error.usage

    llm_usage_by_turn = _llm_usage_by_turn(usage)
    prompt_scenario_cfg = scenario_cfg or cfg.scenario

    result = {
        "model": cfg.agent.model,
        "scenario": scenario_id or f"error_run_{run_index}",
        "mode": cfg.runner.mode,
        "status": "error",
        "success": False,
        "turns": turns,
        "history": history,
        "final_context": final_context,
        "tools": tools,
        "error": str(error),
        "total_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_cost": total_cost,
        "timing": timing,
        "schema_version": 3,
        "metadata": {
            "git_commit": _git_commit_hash(),
            "scenario_backend": prompt_scenario_cfg.backend,
            "agent_api_base": cfg.agent.api_base,
            "run_wall_clock_timeout": getattr(
                cfg.runner, "run_wall_clock_timeout", None
            ),
            "prompt_vars": {
                "user": prompt_scenario_cfg.container_user,
                "password": prompt_scenario_cfg.container_password,
                "max_turns": effective_max_assistant_turns(cfg),
                "term_cols": prompt_scenario_cfg.term_cols,
                "term_rows": prompt_scenario_cfg.term_rows,
            },
            "context": context,
            "trace_incomplete": isinstance(error, SessionTraceError),
            "source_scenario": scenario_id,
            "benchmark_eligible": False,
        },
    }
    if llm_usage_by_turn:
        result["llm_usage_by_turn"] = llm_usage_by_turn
    return result


@lru_cache(maxsize=1)
def _git_commit_hash() -> str | None:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


async def run_with_retries(
    cfg: AppConfig,
    instance,
    run_index: int,
    display: AgentLogger | None = None,
) -> dict[str, Any]:
    attempt = 0
    retry_budget = cfg.runner.max_retries_per_run
    wall_clock_timeout = getattr(cfg.runner, "run_wall_clock_timeout", None)

    while True:
        if display is not None:
            display.reset_attempt()
        task: asyncio.Task[dict[str, Any]] | None = None
        try:
            coro = run_once(cfg, instance, display)
            if wall_clock_timeout is not None:
                task = asyncio.create_task(coro)
                result = await asyncio.wait_for(task, timeout=wall_clock_timeout)
            else:
                result = await coro
            result["attempt"] = attempt + 1
            result["max_retries"] = retry_budget
            return _attach_run_identity(cfg, result, instance, run_index)
        except asyncio.TimeoutError as e:
            timeout_message = (
                f"Run timed out after {wall_clock_timeout}s (attempt {attempt + 1})"
            )
            _print_without_display(display, f"[red]{timeout_message}[/]")
            error: Exception = e
            if task is not None and task.done():
                try:
                    await task
                except SessionTraceError as task_error:
                    error = task_error
                except asyncio.CancelledError:
                    error = e
                except Exception as task_error:
                    error = task_error
            result = error_result(
                cfg,
                run_index,
                error,
                scenario_id=instance.id,
                scenario_cfg=instance.config,
            )
            result["error"] = f"Run timed out after {wall_clock_timeout}s"
            if isinstance(result.get("metadata"), dict):
                result["metadata"]["source_scenario"] = instance.id
                result["metadata"]["timed_out"] = True
            result["attempt"] = attempt + 1
            result["max_retries"] = retry_budget
            return _attach_run_identity(cfg, result, instance, run_index)
        except Exception as e:
            if wall_clock_timeout is not None and is_wall_clock_timeout_error(e):
                timeout_message = (
                    f"Run timed out after {wall_clock_timeout}s (attempt {attempt + 1})"
                )
                _print_without_display(display, f"[red]{timeout_message}[/]")
                result = error_result(
                    cfg,
                    run_index,
                    e,
                    scenario_id=instance.id,
                    scenario_cfg=instance.config,
                )
                result["error"] = f"Run timed out after {wall_clock_timeout}s"
                if isinstance(result.get("metadata"), dict):
                    result["metadata"]["source_scenario"] = instance.id
                    result["metadata"]["timed_out"] = True
                result["attempt"] = attempt + 1
                result["max_retries"] = retry_budget
                return _attach_run_identity(cfg, result, instance, run_index)
            retryable_infra = is_retryable_infra_error(e)
            retry_budget = (
                _max_infra_retries(cfg)
                if retryable_infra
                else cfg.runner.max_retries_per_run
            )
            if attempt >= retry_budget:
                _print_without_display(display, f"[red]Error: {e}[/]")
                result = error_result(
                    cfg,
                    run_index,
                    e,
                    scenario_id=instance.id,
                    scenario_cfg=instance.config,
                )
                if isinstance(result.get("metadata"), dict):
                    result["metadata"]["source_scenario"] = instance.id
                result["attempt"] = attempt + 1
                result["max_retries"] = retry_budget
                return _attach_run_identity(cfg, result, instance, run_index)
            retry_message = f"Retry {attempt + 1}/{retry_budget + 1}: {e}"
            _show_retry_notice(display, retry_message)

        attempt += 1


async def run_single(cfg: AppConfig) -> None:
    source = build_scenario_source(cfg.runner.source, generators_cfg=cfg.generators)
    schedule, desc = build_schedule(cfg, source)

    if not schedule:
        console.print(f"[dim]{desc}. Nothing to do.[/]")
        return

    console.print(f"[dim]{desc}[/]")
    state = RunnerState(
        total=len(schedule), max_turns=effective_max_assistant_turns(cfg)
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Running scenarios", total=len(schedule))

        for run_index in schedule:
            instance = None
            try:
                instance = source.build(run_index)
                generator_name = instance.metadata.get("generator_name", "")
                label = (
                    f"{instance.id} ({generator_name})"
                    if generator_name and generator_name != instance.id
                    else instance.id
                )
                state.set_pending(run_index, label)
                result = await run_with_retries(cfg, instance, run_index)
                state.add_retries(int(result.get("attempt", 1)) - 1)
                generator = instance.metadata.get("generator_name")
                success = result.get("success", False)
                turns = result.get("turns", 0)
                tokens = result.get("total_tokens", 0)
                cost = result.get("total_cost", 0.0)

                if success:
                    state.set_success(run_index, turns, 0.0, tokens, cost, generator)
                else:
                    msg = str(result.get("failure_reason") or result.get("error") or "")
                    state.set_failed(
                        run_index, msg, 0.0, turns, tokens, cost, generator
                    )
            except Exception as e:
                console.print(f"[red bold]Run failed:[/] {e}")
                result = error_result(
                    cfg,
                    run_index,
                    e,
                    scenario_id=instance.id if instance is not None else None,
                )
                if instance is not None:
                    result = _attach_run_identity(cfg, result, instance, run_index)
                state.set_failed(run_index, str(e), 0.0)
                state.add_retries(int(result.get("attempt", 1)) - 1)

            filename_prefix = (
                f"error_run_{run_index}" if result.get("status") == "error" else None
            )
            save_result(cfg, result, filename_prefix=filename_prefix)
            progress.advance(task)

    render_summary(state, cfg.agent.model, target_console=console)
