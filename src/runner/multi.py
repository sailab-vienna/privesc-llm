"""Multi-worker runner with live dashboard."""

import asyncio
import contextlib
import contextvars
import logging
import time
from collections.abc import Iterator
from typing import Any

from rich.live import Live
from rich.logging import RichHandler

from src.config import AppConfig, effective_max_assistant_turns
from src.scenarios import build_scenario_source
from src.tui import console
from src.tui.agent_panels import RunObserver, render_summary
from src.tui.dashboard import build_dashboard
from src.tui.state import RunnerState

from .schedule import build_schedule, save_result
from .single import _attach_run_identity, error_result, run_with_retries

CURRENT_TUI_RUN_INDEX: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "current_tui_run_index", default=None
)


class TuiLogHandler(logging.Handler):
    def __init__(self, state: RunnerState) -> None:
        super().__init__(level=logging.NOTSET)
        self._state = state

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
            run_index = CURRENT_TUI_RUN_INDEX.get()
            source = record.name or "runner"
            level = record.levelname.lower()
            self._state.log_info(run_index, message, level=level, source=source)
        except Exception:
            self.handleError(record)


@contextlib.contextmanager
def route_logs_to_tui(state: RunnerState) -> Iterator[None]:
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    tui_handler = TuiLogHandler(state)
    root_logger.handlers = [
        handler
        for handler in root_logger.handlers
        if not isinstance(handler, RichHandler)
    ]
    root_logger.addHandler(tui_handler)
    try:
        yield
    finally:
        root_logger.handlers = original_handlers


async def run_multi(cfg: AppConfig) -> None:
    source = build_scenario_source(cfg.runner.source, generators_cfg=cfg.generators)
    schedule, desc = build_schedule(cfg, source)

    if not schedule:
        console.print(f"[dim]{desc}. Nothing to do.[/]")
        return

    console.print(f"[dim]{desc}[/]")
    workers = int(cfg.runner.workers)
    model = cfg.agent.model

    state = RunnerState(
        total=len(schedule), max_turns=effective_max_assistant_turns(cfg)
    )
    results: dict[int, dict[str, Any]] = {}
    semaphore = asyncio.Semaphore(workers)

    # Build scenario names for display
    scenario_names: dict[int, str] = {}
    for run_index in schedule:
        try:
            instance = source.build(run_index)
            gen = (
                instance.metadata.get("generator_name", "") if instance.metadata else ""
            )
            scenario_names[run_index] = (
                f"{instance.id} ({gen})" if gen and gen != instance.id else instance.id
            )
        except Exception:
            scenario_names[run_index] = f"run_{run_index}"
        state.set_pending(run_index, scenario_names[run_index])

    async def run_with_semaphore(run_index: int) -> None:
        async with semaphore:
            token = CURRENT_TUI_RUN_INDEX.set(run_index)
            start = time.perf_counter()
            state.set_running(run_index)
            scenario_id: str | None = None

            instance = None
            try:
                instance = source.build(run_index)
                scenario_id = instance.id
                observer = RunObserver(run_index, state)
                result = await run_with_retries(
                    cfg, instance, run_index, display=observer
                )
                state.add_retries(int(result.get("attempt", 1)) - 1)
                duration = time.perf_counter() - start
                turns = result.get("turns", 0)
                tokens = result.get("total_tokens", 0)
                cost = result.get("total_cost", 0.0)
                generator = (
                    instance.metadata.get("generator_name")
                    if instance is not None and instance.metadata
                    else None
                )

                if result.get("success"):
                    state.set_success(
                        run_index, turns, duration, tokens, cost, generator
                    )
                else:
                    msg = str(result.get("failure_reason") or result.get("error") or "")
                    state.set_failed(
                        run_index, msg, duration, turns, tokens, cost, generator
                    )

                results[run_index] = result
                filename_prefix = (
                    f"error_run_{run_index}"
                    if result.get("status") == "error"
                    else None
                )
                save_result(cfg, result, filename_prefix=filename_prefix)

            except Exception as e:
                duration = time.perf_counter() - start
                generator = (
                    instance.metadata.get("generator_name")
                    if instance is not None and instance.metadata
                    else None
                )
                state.set_failed(run_index, str(e), duration, generator=generator)
                result = error_result(
                    cfg,
                    run_index,
                    e,
                    scenario_id=scenario_id,
                    scenario_cfg=instance.config if instance is not None else None,
                )
                if instance is not None:
                    result = _attach_run_identity(cfg, result, instance, run_index)
                results[run_index] = result
                state.add_retries(int(result.get("attempt", 1)) - 1)
                save_result(cfg, result, filename_prefix=f"error_run_{run_index}")
            finally:
                CURRENT_TUI_RUN_INDEX.reset(token)

    tasks = [run_with_semaphore(idx) for idx in schedule]

    with Live(
        build_dashboard(state, model, workers),
        console=console,
        refresh_per_second=4,
        transient=False,
    ) as live:
        with route_logs_to_tui(state):

            async def update_display():
                while state.completed < state.total:
                    live.update(build_dashboard(state, model, workers))
                    await asyncio.sleep(0.25)

            display_task = asyncio.create_task(update_display())

            try:
                await asyncio.gather(*tasks)
            finally:
                display_task.cancel()
                try:
                    await display_task
                except asyncio.CancelledError:
                    pass

            live.update(build_dashboard(state, model, workers))

    render_summary(state, model, target_console=console)
