"""Single-run verbose panels for agent display.

Renders scrolling panels per turn, showing reasoning and tool calls.
Used by the single-worker runner for verbose output.
"""

from datetime import datetime
from typing import Any, Protocol

from rich.box import ROUNDED, SIMPLE
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .state import RunnerState


class AgentLogger(Protocol):
    """Protocol for agent logging."""

    def session_start(
        self, model: str, max_turns: int, metadata: dict[str, Any] | None = None
    ) -> None: ...

    def system_prompt(self, content: str) -> None: ...

    def assistant(
        self, content: str, tool_calls: list[dict[str, Any]] | None = None
    ) -> None: ...

    def nudge(self, message: str) -> None: ...

    def info(self, message: str) -> None: ...

    def tool_result(
        self, name: str, result: dict[str, Any], duration_ms: int | None = None
    ) -> None: ...

    def reset_attempt(self) -> None: ...

    def session_end(
        self,
        success: bool,
        turns: int,
        tokens: int = 0,
        cost: float = 0.0,
        *,
        rejected: bool = False,
        message: str | None = None,
    ) -> None: ...

    def set_token_source(self, source: Any) -> None: ...


class AgentDisplay:
    """Renders per-turn panels for a single agent session.

    Provides verbose output with full reasoning and tool call details.
    Used by single-worker runner.
    """

    def __init__(self, scenario: str, console: Console | None = None):
        from . import console as default_console

        self.scenario = scenario
        self.turn = 0
        self.max_turns = 10
        self._console = console or default_console
        self._last_cmd: str | None = None
        self._winning_cmd: str | None = None
        self._start: datetime | None = None

    def session_start(
        self, model: str, max_turns: int, metadata: dict[str, Any] | None = None
    ) -> None:
        """Log session start with metadata."""
        self._start = datetime.now()
        self.max_turns = max_turns
        model_short = model.split("/")[-1]

        self._console.print()
        self._console.print(
            f"[cyan bold]━━━ {self.scenario} ━━━[/] [white]({model_short}, max {max_turns} turns)[/]"
        )

        if metadata:
            parts = []
            for key in ["generator_name", "seed", "hint"]:
                if key in metadata:
                    val = (
                        str(metadata[key])[:50] + "..."
                        if len(str(metadata.get(key, ""))) > 50
                        else metadata.get(key)
                    )
                    parts.append(f"{key}={val}")
            if parts:
                self._console.print(f"[grey70]  {' · '.join(parts)}[/]")

    def system_prompt(self, content: str) -> None:
        """Log system prompt."""
        body = Text(content, style="grey70")
        self._console.print(
            Panel(
                body,
                title="[grey70]System Prompt[/]",
                border_style="grey70",
                box=ROUNDED,
                padding=(0, 1),
            )
        )
        self._console.print()

    def assistant(
        self, content: str, tool_calls: list[dict[str, Any]] | None = None
    ) -> None:
        """Log assistant message with tool calls."""
        self.turn += 1
        body = Text()

        # Reasoning
        if content:
            body.append(content.strip(), style="green")
            if tool_calls:
                body.append("\n")

        # Tool calls
        for call in tool_calls or []:
            name, args = call.get("name", "?"), call.get("args", {})
            body.append("\n→ ", style="grey70")
            body.append(name, style="yellow bold")
            if name == "exec_command" and "command" in args:
                self._last_cmd = args["command"]
                body.append(" ", style="grey70")
                body.append("$ ", style="grey70")
                body.append(args["command"], style="magenta")
            elif name == "test_credentials":
                body.append(
                    f" user={args.get('user', '')} pass={args.get('password', '')}",
                    style="grey70",
                )

        self._console.print(
            Panel(
                body,
                title=f"[blue bold]Turn {self.turn}/{self.max_turns}[/]",
                border_style="cyan",
                box=ROUNDED,
                padding=(0, 1),
            )
        )

    def nudge(self, message: str) -> None:
        """Log a nudge message sent to the model."""
        body = Text()
        body.append("⚠️ ", style="yellow")
        body.append(message, style="yellow italic")
        self._console.print(
            Panel(
                body,
                title="[yellow bold]Nudge (no tool call)[/]",
                border_style="yellow",
                box=ROUNDED,
                padding=(0, 1),
            )
        )

    def info(self, message: str) -> None:
        """Log general info message."""
        self._console.print(message)

    def set_token_source(self, source: Any) -> None:
        """No-op for single display (prints final stats only)."""
        pass

    def reset_attempt(self) -> None:
        """Reset per-attempt state for retries."""
        self.turn = 0
        self._last_cmd = None
        self._winning_cmd = None

    def tool_result(
        self, name: str, result: dict[str, Any], duration_ms: int | None = None
    ) -> None:
        """Log tool result."""
        output = result.get("output", "")
        got_root = result.get("got_root", False)
        exit_code = result.get("exit_code")
        command = result.get("command") if name == "exec_command" else None

        if got_root and command:
            self._winning_cmd = command

        body = Text()

        # For test_credentials, show user/pass/message
        if name == "test_credentials":
            user = result.get("user", "")
            password = result.get("password", "")
            message = result.get("message", "")
            success = result.get("success", False)
            status_icon = "✓" if success else "✗"
            cmd_synth = f"test: {user}:{password}"
            if got_root:
                self._winning_cmd = cmd_synth
            body.append(
                f"{status_icon} {user}:{password}",
                style="yellow" if success else "white",
            )
            if message:
                body.append(f" - {message}", style="grey70")
        elif output:
            body.append(str(output).rstrip(), style="white")
        else:
            body.append("(no output)", style="grey70 italic")

        status = (
            "[bright_green bold]✅ got_root[/]"
            if got_root
            else "[white]got_root=false[/]"
        )
        exit_code_text = f" [white]exit={exit_code}[/]" if exit_code is not None else ""
        timing = f" [cyan]{duration_ms}ms[/]" if duration_ms else ""
        border = "bright_green" if got_root else "white"

        self._console.print(
            Panel(
                body,
                title=f"[yellow bold]{name}[/]",
                subtitle=f"{status}{exit_code_text}{timing}",
                subtitle_align="right",
                border_style=border,
                box=ROUNDED,
                padding=(0, 1),
            )
        )

    def session_end(
        self,
        success: bool,
        turns: int,
        tokens: int = 0,
        cost: float = 0.0,
        *,
        rejected: bool = False,
        message: str | None = None,
    ) -> None:
        """Log session end with summary panel."""
        duration = (datetime.now() - self._start).total_seconds() if self._start else 0

        # Build stats line
        parts = [f"{turns} turns", f"{duration:.1f}s"]
        if tokens:
            parts.append(f"{tokens:,} tok")
        if cost > 0:
            parts.append(f"${cost:.4f}")

        body = Text()

        # Status line
        if success:
            body.append("✅ SUCCESS", style="bright_green bold")
        elif rejected:
            body.append("⚠ REJECTED", style="yellow bold")
        else:
            body.append("❌ FAILED", style="red bold")
        body.append(f"  {' · '.join(parts)}", style="white")

        if not success and message:
            body.append("\n")
            body.append(message, style="yellow" if rejected else "red")

        # Winning command (if success)
        if success and self._winning_cmd:
            cmd = (
                self._winning_cmd[:70] + "..."
                if len(self._winning_cmd) > 70
                else self._winning_cmd
            )
            body.append("\n")
            body.append("Winning: ", style="white")
            body.append(cmd, style="magenta")

        if success:
            border = "bright_green"
        elif rejected:
            border = "yellow"
        else:
            border = "red"
        self._console.print()
        self._console.print(
            Panel(
                body,
                title=f"[bold]{self.scenario}[/]",
                border_style=border,
                box=ROUNDED,
                padding=(0, 1),
            )
        )
        self._console.print()


def render_summary(
    state: RunnerState, model: str, target_console: Console | None = None
) -> None:
    """Render experiment summary table."""
    from . import console

    c = target_console or console
    c.print()
    c.print("[bold cyan]" + "═" * 50 + "[/]")
    c.print("[bold cyan]  EXPERIMENT SUMMARY[/]")
    c.print("[bold cyan]" + "═" * 50 + "[/]")

    model_short = model.split("/")[-1]
    rate_style = (
        "green"
        if state.success_rate >= 80
        else "yellow"
        if state.success_rate >= 50
        else "red"
    )

    c.print(f"\n[bold]Model:[/] {model_short}")
    c.print(
        f"[bold]Success:[/] [{rate_style}]{state.successes}/{state.completed} ({state.success_rate:.1f}%)[/]"
    )
    c.print(f"[bold]Avg Turns:[/] {state.avg_turns:.2f}")
    if state.total_tokens:
        c.print(f"[bold]Tokens:[/] {state.total_tokens:,}")
    if state.total_cost > 0:
        c.print(f"[bold]Cost:[/] ${state.total_cost:.4f}")
    if state.total_retries:
        c.print(
            f"[bold]Retries:[/] {state.total_retries} total"
            f" ({state.runs_with_retry} runs)"
        )

    if state.by_generator:
        c.print()
        table = Table(title="[bold]By Generator[/]", box=SIMPLE, header_style="bold")
        table.add_column("Generator", style="magenta")
        table.add_column("Rate", justify="right")
        table.add_column("Turns", justify="right")

        for gen, g in sorted(
            state.by_generator.items(), key=lambda x: -x[1]["success"]
        ):
            total = g["success"] + g["fail"]
            rate = g["success"] / total * 100 if total else 0
            style = "green" if rate >= 80 else "yellow" if rate >= 50 else "red"
            table.add_row(
                gen,
                f"[{style}]{g['success']}/{total}[/]",
                f"{g['turns'] / total:.1f}" if total else "-",
            )
        c.print(table)

    c.print("[bold cyan]" + "═" * 50 + "[/]")


class RunObserver:
    """Observer that sends run events directly to RunnerState.

    Used by multi-worker runner to update dashboard state.
    Implements same interface as AgentDisplay for interchangeability.
    """

    def __init__(self, run_index: int, state: RunnerState):
        self._run_index = run_index
        self._state = state
        self.turn = 0
        self.max_turns = 10
        self._last_cmd: str | None = None
        self._token_callback: Any = None

    def set_token_source(self, source: Any) -> None:
        """Set the OpenAI callback for continuous token tracking."""
        self._token_callback = source

    def reset_attempt(self) -> None:
        """Reset per-attempt state for retries."""
        self.turn = 0
        self._last_cmd = None
        self._state.update_turn(self._run_index, 0)

    def session_start(
        self, model: str, max_turns: int, metadata: dict[str, Any] | None = None
    ) -> None:
        """Initialize session (no-op for observer)."""
        self.max_turns = max_turns

    def system_prompt(self, content: str) -> None:
        """Log system prompt (no-op for observer)."""
        pass

    def assistant(
        self, content: str, tool_calls: list[dict[str, Any]] | None = None
    ) -> None:
        """Log assistant message - notify state of thinking and tool calls."""
        self.turn += 1
        self._state.update_turn(self._run_index, self.turn)

        # Report current token count if callback is set
        if self._token_callback is not None:
            self._state.update_tokens(
                self._run_index,
                self._token_callback.total_tokens,
                self._token_callback.prompt_tokens,
                self._token_callback.completion_tokens,
                self._token_callback.total_cost,
            )

        # Log thinking
        if content:
            from .formatting import single_line

            thinking = single_line(content, max_len=4000)
            if thinking:
                self._state.log_thinking(self._run_index, thinking)

        # Log tool calls
        for call in tool_calls or []:
            name = call.get("name", "?")
            args = call.get("args", {})
            if name == "exec_command" and "command" in args:
                self._last_cmd = args["command"]
                self._state.log_tool_call(self._run_index, "exec", str(args["command"]))
            elif name == "test_credentials":
                self._state.log_tool_call(
                    self._run_index,
                    "test",
                    f"{args.get('user')}:{args.get('password')}",
                )

    def tool_result(
        self, name: str, result: dict[str, Any], duration_ms: int | None = None
    ) -> None:
        """Log tool result to state."""
        from .formatting import single_line

        got_root = result.get("got_root", False)
        output = result.get("output", "")
        exit_code = result.get("exit_code")
        command = result.get("command") if name == "exec_command" else None

        output_text = single_line(output, max_len=4000) if output else "(no output)"

        # For test_credentials, we synthesize a command so state can track it
        if name == "test_credentials":
            user = result.get("user", "")
            password = result.get("password", "")
            command = f"test: {user}:{password}"

        self._state.log_tool_result(
            self._run_index, output_text, got_root, exit_code, command
        )

    def nudge(self, message: str) -> None:
        """Log nudge to state."""
        self._state.log_thinking(self._run_index, f"[nudge] {message[:50]}")

    def info(self, message: str) -> None:
        """Log runtime info message to state."""
        self._state.log_info(self._run_index, message)

    def session_end(
        self,
        success: bool,
        turns: int,
        tokens: int = 0,
        cost: float = 0.0,
        *,
        rejected: bool = False,
        message: str | None = None,
    ) -> None:
        """Session end (handled by runner state, no-op here)."""
        pass
