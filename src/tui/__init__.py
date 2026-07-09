"""Reusable Rich TUI display components."""

from rich.console import Console

from .state import (
    RunnerState,
    RunState,
    LogEntryType,
)
from .agent_panels import AgentDisplay, RunObserver, render_summary
from .dashboard import build_dashboard

console = Console()

__all__ = [
    "console",
    "RunnerState",
    "RunState",
    "RunStatus",
    "LogEntry",
    "LogEntryType",
    "AgentDisplay",
    "RunObserver",
    "render_summary",
    "build_dashboard",
]
