"""Multi-worker dashboard with Rich Live display.

Renders a full-screen TUI with status, runs, and activity log panels.
Used by the multi-worker runner.
"""

import os
import time
from collections import defaultdict
from datetime import datetime
from urllib.parse import urlparse

from rich.layout import Layout
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text

from .formatting import (
    format_duration,
    format_elapsed,
    format_tokens,
    get_worker_color,
    single_line,
)
from .state import LogEntryType, RunnerState, RunStatus


def _build_header(state: RunnerState, model: str) -> Text:
    """Build the header line with key metrics."""
    header = Text(no_wrap=True, overflow="crop")

    # Title and model
    header.append("◆ ", style="bright_cyan bold")
    header.append("PrivEsc LLM Runner", style="bold bright_white")
    header.append("  │  ", style="white")
    header.append(model, style="bright_magenta bold")

    # Provider hostname from OPENAI_API_BASE
    api_base = os.environ.get("OPENAI_API_BASE", "")
    if api_base:
        try:
            hostname = urlparse(api_base).hostname or ""
            if hostname:
                header.append(" @ ", style="white")
                header.append(hostname, style="bright_cyan")
        except Exception:
            pass

    # Wall clock time
    header.append("  │  ", style="white")
    elapsed = time.perf_counter() - state.start_time
    header.append(f"⏱ {format_elapsed(elapsed)}", style="white")
    header.append(f"  {datetime.now().strftime('%H:%M:%S')}", style="white")

    return header


def _build_log_panel(state: RunnerState, max_lines: int | None = None) -> Panel:
    """Build the scrolling log panel."""
    if not state.log_entries:
        content = Text("Waiting for activity...", style="white italic")
        return Panel(
            content, title="[bold]Activity Log[/]", border_style="white", padding=(0, 1)
        )

    if max_lines is None:
        from . import console

        max_lines = max(10, console.height - 8)

    table = Table(box=None, show_header=False, padding=(0, 1), expand=True)
    table.add_column("W", width=4, no_wrap=True)
    table.add_column("Time", width=10, no_wrap=True, style="white", justify="right")
    table.add_column(
        "Scenario", ratio=2, no_wrap=True, overflow="ellipsis", style="white"
    )
    table.add_column("", width=2, no_wrap=True)
    table.add_column("Message", ratio=12)

    entries = list(state.log_entries)[-max_lines:]

    def icon_and_style(entry_type: LogEntryType) -> tuple[str, str]:
        if entry_type == LogEntryType.START:
            return "▶", "bright_cyan"
        if entry_type == LogEntryType.THINKING:
            return "◇", "white"
        if entry_type == LogEntryType.TOOL_CALL:
            return "→", "bright_yellow"
        if entry_type == LogEntryType.TOOL_RESULT:
            return "←", "bright_cyan"
        if entry_type == LogEntryType.INFO:
            return "·", "dim"
        if entry_type == LogEntryType.SUCCESS:
            return "✓", "bright_green bold"
        return "✗", "bright_red bold"

    for entry in entries:
        worker_text = "sys"
        worker_style = "white"
        if entry.worker_id >= 0:
            color = get_worker_color(entry.worker_id)
            worker_text = f"W{entry.worker_id}"
            worker_style = f"{color} bold"
        icon, icon_style = icon_and_style(entry.entry_type)

        msg_style = "white"
        if entry.entry_type == LogEntryType.THINKING:
            msg_style = "bright_white"
        elif entry.entry_type == LogEntryType.TOOL_CALL:
            msg_style = "bright_yellow"
        elif entry.entry_type == LogEntryType.SUCCESS:
            msg_style = "bright_green bold"
        elif entry.entry_type == LogEntryType.FAILED:
            msg_style = "bright_red"
        elif entry.entry_type == LogEntryType.INFO:
            level = entry.level.lower()
            if level in {"warning", "warn"}:
                icon = "!"
                icon_style = "bright_yellow"
                msg_style = "bright_yellow"
            elif level in {"error", "critical"}:
                icon = "!"
                icon_style = "bright_red bold"
                msg_style = "bright_red"
            else:
                icon_style = "dim"
                msg_style = "dim"

        table.add_row(
            Text(worker_text, style=worker_style, no_wrap=True),
            Text(
                time.strftime("%H:%M:%S", time.localtime(entry.timestamp)),
                style="white",
                no_wrap=True,
                justify="right",
            ),
            Text(entry.scenario, style="white", no_wrap=True, overflow="ellipsis"),
            Text(icon, style=icon_style, no_wrap=True),
            Text(
                single_line(entry.content),
                style=msg_style,
                no_wrap=True,
                overflow="ellipsis",
            ),
        )

    return Panel(
        table,
        title=f"[bold bright_white]◆ Log[/] [white]({len(entries)}/{len(state.log_entries)})[/]",
        border_style="bright_blue",
        padding=(0, 1),
    )


def _build_runs_panel(state: RunnerState, max_rows: int | None = None) -> Panel:
    """Build a compact per-run status panel."""
    if max_rows is None:
        from . import console

        available = console.height - 16
        max_rows = max(5, available // 2)

    table = Table(box=None, show_header=False, padding=(0, 1), expand=True)
    table.add_column("", width=2, no_wrap=True)
    table.add_column("W", width=4, no_wrap=True)
    table.add_column("Scenario", ratio=3, no_wrap=True, overflow="ellipsis")
    table.add_column("Time", width=6, no_wrap=True, style="white")
    table.add_column("Turns", width=6, no_wrap=True)
    table.add_column("Last", ratio=8, no_wrap=True, overflow="ellipsis", style="white")
    table.add_column("Tok", width=12, no_wrap=True, style="white")

    def row_for(run):
        color = get_worker_color(run.worker_id) if run.worker_id >= 0 else "white"

        status_icon = "○"
        status_style = "white"
        if run.status == RunStatus.RUNNING:
            status_icon, status_style = "●", "bright_yellow"
        elif run.status == RunStatus.SUCCESS:
            status_icon, status_style = "✓", "bright_green bold"
        elif run.status == RunStatus.FAILED:
            status_icon, status_style = "✗", "bright_red"

        worker = Text("", style="white")
        if run.worker_id >= 0:
            worker = Text(f"W{run.worker_id}", style=f"{color} bold")

        scenario_style = color if run.status == RunStatus.RUNNING else "white"
        scenario = Text(
            run.scenario, style=scenario_style, no_wrap=True, overflow="ellipsis"
        )

        if run.status == RunStatus.RUNNING:
            t = format_duration(time.perf_counter() - run.start_time)
        elif run.status in (RunStatus.SUCCESS, RunStatus.FAILED):
            t = format_duration(run.duration_s)
        else:
            t = ""
        time_cell = Text(t, style="white", no_wrap=True)

        turns = ""
        if state.max_turns and (run.turns or run.status != RunStatus.PENDING):
            turns = f"{run.turns}/{state.max_turns}"
        elif run.turns:
            turns = f"t{run.turns}"
        turns_style = "white"
        if run.status == RunStatus.RUNNING:
            turns_style = "yellow"
        elif run.status == RunStatus.SUCCESS:
            turns_style = "green"
        elif run.status == RunStatus.FAILED:
            turns_style = "red"
        turns_cell = Text(turns, style=turns_style, no_wrap=True)

        tok = ""
        if run.tokens > 0:
            tok = f"{format_tokens(run.prompt_tokens)}↓ {format_tokens(run.completion_tokens)}↑"
        tok_style = (
            "cyan" if run.status in (RunStatus.SUCCESS, RunStatus.FAILED) else "white"
        )
        tokens_cell = Text(tok, style=tok_style, no_wrap=True)

        last = ""
        last_style = "white"
        if run.status == RunStatus.FAILED and run.message:
            last = run.message
            last_style = "red"
        elif run.status == RunStatus.SUCCESS and run.last_action:
            last = run.last_action
            last_style = "green"
        elif run.status == RunStatus.RUNNING and run.last_action:
            last = run.last_action
        last_cell = Text(
            single_line(last), style=last_style, no_wrap=True, overflow="ellipsis"
        )

        return (
            Text(status_icon, style=status_style),
            worker,
            scenario,
            time_cell,
            turns_cell,
            last_cell,
            tokens_cell,
        )

    runs = list(state.runs.values())

    def sort_key(r):
        if r.status == RunStatus.RUNNING:
            return (0, -r.start_time if r.start_time else 0)
        elif r.status == RunStatus.PENDING:
            return (2, 0)
        else:
            # Sort completed runs by end time (recency)
            end_time = (r.start_time or 0) + (r.duration_s or 0)
            return (1, -end_time)

    runs.sort(key=sort_key)

    for run in runs[:max_rows]:
        table.add_row(*row_for(run))

    hidden = len(runs) - max_rows if len(runs) > max_rows else 0
    title = f"[bold bright_white]◆ Runs[/] [white]{len(runs)}"
    if hidden:
        title += f" (+{hidden})"
    title += "[/]"

    return Panel(table, title=title, border_style="bright_magenta", padding=(0, 1))


def _build_status_panel(state: RunnerState, workers: int) -> Panel:
    """Build the status summary panel with detailed metrics."""
    running = [r for r in state.runs.values() if r.status == RunStatus.RUNNING]
    pending_count = sum(1 for r in state.runs.values() if r.status == RunStatus.PENDING)
    failed_count = sum(1 for r in state.runs.values() if r.status == RunStatus.FAILED)

    table = Table(box=None, show_header=False, padding=(0, 1), expand=True)
    table.add_column("Label", style="white", ratio=2, no_wrap=True, overflow="ellipsis")
    table.add_column("Value", ratio=3, no_wrap=True, overflow="ellipsis")

    # Progress
    if state.total > 0:
        progress = Progress(
            TextColumn("[bold cyan]{task.completed}[/]/[white]{task.total}[/]"),
            BarColumn(
                bar_width=None,
                complete_style="bright_cyan",
                finished_style="bright_green",
            ),
            TextColumn("[bold white]{task.percentage:>3.0f}%[/]"),
            expand=True,
        )
        task_id = progress.add_task("runs", total=state.total)
        progress.update(task_id, completed=state.completed)
        table.add_row("◉ Progress", progress)

    # Time
    elapsed_total = time.perf_counter() - state.start_time
    time_text = Text()
    time_text.append(format_elapsed(elapsed_total), style="bold bright_white")
    if state.completed > 0:
        avg_duration = (
            sum(r.duration_s for r in state.runs.values() if r.duration_s)
            / state.completed
        )
        remaining = state.total - state.completed
        eta = (
            avg_duration * remaining / max(1, len(running))
            if running
            else avg_duration * remaining
        )
        rate = state.completed / elapsed_total * 60 if elapsed_total > 0 else 0
        time_text.append(" ~", style="white")
        time_text.append(format_elapsed(eta), style="bright_cyan")
        time_text.append(" left", style="white")
        time_text.append(f" ▸{rate:.1f}/m", style="bright_green")
    table.add_row("⏱ Time", time_text)

    # Retries
    retries_text = Text()
    retries_text.append(str(state.total_retries), style="bright_yellow bold")
    if state.runs_with_retry:
        retries_text.append(f" ({state.runs_with_retry})", style="bright_yellow")
    table.add_row("↻ Retries", retries_text)

    # Results
    results_text = Text()
    success_rate = None
    rate_color = "white"
    if state.completed > 0 or running:
        if running:
            results_text.append(f"{len(running)}", style="bright_cyan bold")
            results_text.append(" ●", style="cyan")

        if state.completed > 0:
            success_rate = state.successes / state.completed * 100
            rate_color = (
                "bright_green"
                if success_rate >= 50
                else "bright_yellow"
                if success_rate >= 25
                else "bright_red"
            )
            results_text.append(f" {state.successes}", style="bright_green bold")
            results_text.append(" ✓", style="green")
            if failed_count:
                results_text.append(f" {failed_count}", style="bright_red bold")
                results_text.append(" ✗", style="red")

        if pending_count:
            results_text.append(f" {pending_count}", style="white")
            results_text.append(" ◌", style="white")

        if success_rate is not None:
            results_text.append(f" [{success_rate:.0f}%]", style=f"{rate_color} bold")
    else:
        results_text.append("—", style="white")

    table.add_row("◈ Results", results_text)

    # Per-category stats
    cat_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"root": 0, "fail": 0, "run": 0, "turns": 0}
    )
    for run in state.runs.values():
        cat = run.scenario.split("[")[0].split("(")[0].strip()
        if run.status == RunStatus.SUCCESS:
            cat_stats[cat]["root"] += 1
            cat_stats[cat]["turns"] += run.turns
        elif run.status == RunStatus.FAILED:
            cat_stats[cat]["fail"] += 1
            cat_stats[cat]["turns"] += run.turns
        elif run.status == RunStatus.RUNNING:
            cat_stats[cat]["run"] += 1

    if cat_stats:
        sorted_cats = sorted(
            cat_stats.items(),
            key=lambda x: x[1]["root"] + x[1]["fail"] + x[1]["run"],
            reverse=True,
        )
        for i, (cat_name, stats) in enumerate(sorted_cats):
            is_last = i == len(sorted_cats) - 1
            tree_char = "└" if is_last else "├"

            cat_text = Text()
            total_done = stats["root"] + stats["fail"]
            if total_done > 0:
                cat_rate = stats["root"] / total_done * 100
                rate_style = (
                    "bright_green"
                    if cat_rate >= 50
                    else "bright_yellow"
                    if cat_rate >= 25
                    else "bright_red"
                )
                cat_text.append(f"{stats['root']}", style="bright_green")
                # Removed separator slash for cleaner look with spacing
                # cat_text.append("/", style="white")

                # Check if we have fails to display, otherwise just root count is distinct enough?
                # Actually stick to space separation
                cat_text.append(f" {stats['fail']}", style="red")

                cat_text.append(f" [{cat_rate:.0f}%]", style=rate_style)
                if stats["turns"] > 0:
                    avg_turns = stats["turns"] / total_done
                    cat_text.append(f" ~{avg_turns:.1f}t", style="white")
                # Also show running count if any
                if stats["run"] > 0:
                    cat_text.append(f" {stats['run']} ●", style="bright_cyan")
            elif stats["run"] > 0:
                cat_text.append(f"{stats['run']} ●", style="bright_cyan")
            else:
                continue
            table.add_row(f"  {tree_char} {cat_name}", cat_text)

    # Turns
    completed_runs = [r for r in state.runs.values() if r.turns > 0]
    turns_text = Text("—", style="white")
    if completed_runs and state.max_turns:
        turns_text = Text()
        avg_turns = sum(r.turns for r in completed_runs) / len(completed_runs)
        min_turns = min(r.turns for r in completed_runs)
        max_turns_val = max(r.turns for r in completed_runs)
        turns_text.append(f"{avg_turns:.1f}", style="bright_yellow bold")
        turns_text.append(f"/{state.max_turns}", style="white")
        if len(completed_runs) > 1:
            turns_text.append(f" ⟨{min_turns}–{max_turns_val}⟩", style="white")
    table.add_row("∿ Turns", turns_text)

    # Workers
    workers_text = Text()
    workers_text.append(f"{len(running)}", style="bright_cyan bold")
    workers_text.append(f"/{workers}", style="white")
    if pending_count:
        workers_text.append(f" +{pending_count}q", style="white")
    table.add_row("⊚ Workers", workers_text)

    # Tokens
    if state.total_tokens > 0 or state.total_cost > 0:
        tokens_text = Text()
        tokens_text.append(
            format_tokens(state.total_prompt_tokens), style="bright_cyan"
        )
        tokens_text.append("↓", style="white")
        tokens_text.append(" ", style="white")
        tokens_text.append(
            format_tokens(state.total_completion_tokens),
            style="bright_cyan bold",
        )
        tokens_text.append("↑", style="white")
        if state.total_cost > 0:
            tokens_text.append(f"  ${state.total_cost:.3f}", style="bright_yellow")
        table.add_row("◈ Tokens", tokens_text)

    # Dynamic border color
    border_style = "bright_cyan"
    if state.completed > 0:
        success_rate = state.successes / state.completed * 100
        if success_rate >= 50:
            border_style = "bright_green"
        elif success_rate >= 25:
            border_style = "bright_yellow"
        else:
            border_style = "bright_red"

    return Panel(
        table,
        title="[bold bright_white]◆ Status[/]",
        border_style=border_style,
        padding=(0, 1),
    )


def build_dashboard(state: RunnerState, model: str, workers: int) -> Layout:
    """Build the full dashboard display that fills the entire terminal."""
    from . import console

    term_height = console.height

    header_height = 3
    body_height = term_height - header_height

    top_height = max(8, body_height // 2)
    log_height = body_height - top_height

    runs_max_rows = max(5, top_height - 4)
    log_max_lines = max(5, log_height - 3)

    header = _build_header(state, model)
    status_panel = _build_status_panel(state, workers)
    runs_panel = _build_runs_panel(state, max_rows=runs_max_rows)
    log_panel = _build_log_panel(state, max_lines=log_max_lines)

    root = Layout(name="root")

    root.split_column(
        Layout(
            Panel(header, border_style="white", padding=(0, 1)),
            name="header",
            size=header_height,
        ),
        Layout(name="top", size=top_height),
        Layout(log_panel, name="log", ratio=1),
    )

    root["top"].split_row(
        Layout(status_panel, name="status", ratio=1),
        Layout(runs_panel, name="runs", ratio=2),
    )

    return root
