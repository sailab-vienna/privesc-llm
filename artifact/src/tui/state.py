"""Runner state management for TUI display."""

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from .formatting import format_tokens, plain_log_message, single_line


class RunStatus(Enum):
    """Status of a single run."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class LogEntryType(Enum):
    """Type of log entry for activity display."""

    START = "start"
    THINKING = "thinking"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    INFO = "info"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class LogEntry:
    """A single log entry for the TUI."""

    worker_id: int
    scenario: str
    entry_type: LogEntryType
    content: str
    level: str = "info"
    source: str | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class RunState:
    """State of a single run for TUI display."""

    scenario: str
    worker_id: int = -1
    status: RunStatus = RunStatus.PENDING
    turns: int = 0
    message: str = ""
    duration_s: float = 0.0
    start_time: float = 0.0
    last_action: str = ""
    tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0


@dataclass
class RunnerState:
    """Global state for runner TUI display."""

    runs: dict[int, RunState] = field(default_factory=dict)
    completed: int = 0
    total: int = 0
    successes: int = 0
    log_entries: deque[LogEntry] = field(default_factory=lambda: deque(maxlen=200))
    next_worker_id: int = 0
    active_workers: dict[int, int] = field(default_factory=dict)
    start_time: float = field(default_factory=time.perf_counter)
    max_turns: int = 0
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost: float = 0.0
    by_generator: dict[str, dict[str, int]] = field(default_factory=dict)
    total_turns: int = 0
    total_retries: int = 0
    runs_with_retry: int = 0

    def set_pending(self, run_index: int, scenario: str) -> None:
        """Initialize a run as pending."""
        self.runs[run_index] = RunState(scenario=scenario, status=RunStatus.PENDING)

    def set_running(self, run_index: int) -> int:
        """Mark run as running and assign a worker ID. Returns worker_id."""
        if run_index in self.runs:
            worker_id = self.next_worker_id
            self.next_worker_id += 1

            self.runs[run_index].status = RunStatus.RUNNING
            self.runs[run_index].worker_id = worker_id
            self.runs[run_index].start_time = time.perf_counter()
            self.active_workers[run_index] = worker_id

            self.add_log(run_index, LogEntryType.START, "Starting scenario")
            return worker_id
        return 0

    def _record_stats(
        self,
        success: bool,
        turns: int,
        tokens: int,
        cost: float,
        generator: str | None = None,
    ) -> None:
        """Record run statistics."""
        self.completed += 1
        self.total_tokens += tokens
        self.total_cost += cost
        self.total_turns += turns
        if success:
            self.successes += 1

        if generator:
            if generator not in self.by_generator:
                self.by_generator[generator] = {"success": 0, "fail": 0, "turns": 0}
            self.by_generator[generator]["turns"] += turns
            self.by_generator[generator]["success" if success else "fail"] += 1

    def add_retries(self, retries_used: int) -> None:
        if retries_used > 0:
            self.total_retries += retries_used
            self.runs_with_retry += 1

    def set_success(
        self,
        run_index: int,
        turns: int,
        duration_s: float,
        tokens: int = 0,
        cost: float = 0.0,
        generator: str | None = None,
    ) -> None:
        """Mark run as successful."""
        if run_index in self.runs:
            self.runs[run_index].status = RunStatus.SUCCESS
            self.runs[run_index].turns = turns
            self.runs[run_index].duration_s = duration_s
            self.runs[run_index].tokens = tokens
            self.runs[run_index].cost = cost

            tok_str = format_tokens(tokens) if tokens else ""
            self.add_log(
                run_index,
                LogEntryType.SUCCESS,
                f"Got root in {turns} turns" + (f" ({tok_str})" if tok_str else ""),
            )
            self.active_workers.pop(run_index, None)
            self._record_stats(True, turns, tokens, cost, generator)

    def set_failed(
        self,
        run_index: int,
        message: str,
        duration_s: float,
        turns: int = 0,
        tokens: int = 0,
        cost: float = 0.0,
        generator: str | None = None,
    ) -> None:
        """Mark run as failed."""
        if run_index in self.runs:
            self.runs[run_index].status = RunStatus.FAILED
            self.runs[run_index].message = message
            self.runs[run_index].duration_s = duration_s
            self.runs[run_index].turns = turns
            self.runs[run_index].tokens = tokens
            self.runs[run_index].cost = cost

            self.add_log(run_index, LogEntryType.FAILED, message or "Failed")
            self.active_workers.pop(run_index, None)
            self._record_stats(False, turns, tokens, cost, generator)

    def add_log(self, run_index: int, entry_type: LogEntryType, content: str) -> None:
        """Add a log entry."""
        if run_index in self.runs:
            run = self.runs[run_index]
            if entry_type == LogEntryType.TOOL_CALL:
                run.last_action = single_line(content, max_len=10000)
            self.log_entries.append(
                LogEntry(
                    worker_id=run.worker_id,
                    scenario=run.scenario,
                    entry_type=entry_type,
                    content=content,
                )
            )

    def log_info(
        self,
        run_index: int | None,
        content: str,
        *,
        level: str = "info",
        source: str | None = None,
    ) -> None:
        """Log runtime or system information without changing run action state."""
        message = plain_log_message(content)
        if not message:
            return

        normalized_level = level.lower()
        if run_index is not None and run_index in self.runs:
            run = self.runs[run_index]
            worker_id = run.worker_id
            scenario = run.scenario
        else:
            worker_id = -1
            scenario = source or "runner"

        self.log_entries.append(
            LogEntry(
                worker_id=worker_id,
                scenario=scenario,
                entry_type=LogEntryType.INFO,
                content=message,
                level=normalized_level,
                source=source,
            )
        )

    def log_thinking(self, run_index: int, content: str) -> None:
        """Log LLM thinking/reasoning."""
        self.add_log(run_index, LogEntryType.THINKING, content)

    def log_tool_call(self, run_index: int, tool_name: str, args: str) -> None:
        """Log a tool call."""
        self.add_log(run_index, LogEntryType.TOOL_CALL, f"{tool_name}: {args}")

    def log_tool_result(
        self,
        run_index: int,
        output: str,
        got_root: bool,
        exit_code: int | None,
        command: str | None = None,
    ) -> None:
        """Log a tool result."""
        parts = []
        if got_root:
            parts.append("root=true")
        if exit_code is not None:
            parts.append(f"exit={exit_code}")
        prefix = " ".join(parts)
        content = f"{prefix} {output}".strip()
        if run_index in self.runs and got_root and command:
            self.runs[run_index].last_action = single_line(command, max_len=10000)
        self.add_log(run_index, LogEntryType.TOOL_RESULT, content)

    def update_turn(self, run_index: int, turn: int) -> None:
        """Update the turn count for a run."""
        if run_index in self.runs:
            self.runs[run_index].turns = turn

    def update_tokens(
        self,
        run_index: int,
        tokens: int,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float,
    ) -> None:
        """Update running token count for a run."""
        if run_index in self.runs:
            run = self.runs[run_index]
            # Update live totals using deltas; final stats are recorded on completion.
            tokens_delta = tokens - run.tokens
            prompt_delta = prompt_tokens - run.prompt_tokens
            completion_delta = completion_tokens - run.completion_tokens
            cost_delta = cost - run.cost

            run.tokens = tokens
            run.prompt_tokens = prompt_tokens
            run.completion_tokens = completion_tokens
            run.cost = cost

            self.total_tokens += tokens_delta
            self.total_prompt_tokens += prompt_delta
            self.total_completion_tokens += completion_delta
            self.total_cost += cost_delta

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        return self.successes / self.completed * 100 if self.completed > 0 else 0.0

    @property
    def avg_turns(self) -> float:
        """Calculate average turns per completed run."""
        return self.total_turns / self.completed if self.completed > 0 else 0.0
