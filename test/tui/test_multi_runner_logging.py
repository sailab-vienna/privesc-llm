import logging

from src.runner.multi import CURRENT_TUI_RUN_INDEX, TuiLogHandler
from src.tui.agent_panels import RunObserver
from src.tui.dashboard import build_dashboard
from src.tui.state import LogEntryType, RunnerState


def test_run_observer_info_logs_clean_message_without_last_action() -> None:
    state = RunnerState(total=1, max_turns=10)
    state.set_pending(0, "demo")
    state.set_running(0)
    state.log_tool_call(0, "exec", "id")
    observer = RunObserver(0, state)

    observer.info("[dim]  Launching container abc[/]")

    assert state.runs[0].last_action == "exec: id"
    entry = state.log_entries[-1]
    assert entry.entry_type == LogEntryType.INFO
    assert entry.content == "Launching container abc"
    assert entry.worker_id == 0


def test_tui_log_handler_attaches_records_to_current_run() -> None:
    state = RunnerState(total=1, max_turns=10)
    state.set_pending(7, "scenario")
    state.set_running(7)
    handler = TuiLogHandler(state)
    token = CURRENT_TUI_RUN_INDEX.set(7)
    try:
        handler.emit(
            logging.LogRecord(
                "privesc_agent",
                logging.WARNING,
                __file__,
                1,
                "Live repair failed: %s",
                ("boom",),
                None,
            )
        )
    finally:
        CURRENT_TUI_RUN_INDEX.reset(token)

    entry = state.log_entries[-1]
    assert entry.entry_type == LogEntryType.INFO
    assert entry.level == "warning"
    assert entry.source == "privesc_agent"
    assert entry.scenario == "scenario"
    assert entry.worker_id == 0
    assert entry.content == "Live repair failed: boom"


def test_tui_log_handler_records_workerless_system_records() -> None:
    state = RunnerState(total=0, max_turns=10)
    handler = TuiLogHandler(state)

    handler.emit(
        logging.LogRecord(
            "httpx",
            logging.ERROR,
            __file__,
            1,
            "request failed",
            (),
            None,
        )
    )

    entry = state.log_entries[-1]
    assert entry.entry_type == LogEntryType.INFO
    assert entry.level == "error"
    assert entry.scenario == "httpx"
    assert entry.worker_id == -1
    assert entry.content == "request failed"


def test_dashboard_renders_info_and_workerless_system_entries() -> None:
    state = RunnerState(total=1, max_turns=10)
    state.set_pending(0, "demo")
    state.set_running(0)
    state.log_info(0, "[green]Ready[/]")
    state.log_info(None, "parse warning", level="warning", source="privesc_tools")

    dashboard = build_dashboard(state, "model", workers=1)

    assert dashboard is not None
