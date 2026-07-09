"""Shared TUI components: formatting, colors, and helpers."""

import re

from rich.text import Text

# Worker colors for visual distinction (supports 64+ workers via cycling)
WORKER_COLORS = [
    "cyan",
    "magenta",
    "yellow",
    "green",
    "blue",
    "red",
    "bright_cyan",
    "bright_magenta",
    "bright_yellow",
    "bright_green",
    "bright_blue",
    "bright_red",
    "bright_white",
    "white",
    "color(39)",
    "color(208)",
    "color(141)",
    "color(84)",
]


def get_worker_color(worker_id: int) -> str:
    """Get color for a worker ID (cycles through palette)."""
    return WORKER_COLORS[worker_id % len(WORKER_COLORS)]


def truncate(text: str, max_len: int = 80) -> str:
    """Truncate text with ellipsis (end-truncate)."""
    text = single_line(text)
    if len(text) > max_len:
        return text[: max_len - 1] + "…"
    return text


def single_line(text: str, max_len: int | None = None) -> str:
    """Collapse text to single line, optionally truncating."""
    text = text.replace("\n", " ").replace("\r", " ").strip()
    while "  " in text:
        text = text.replace("  ", " ")
    if max_len is not None and len(text) > max_len:
        return text[: max_len - 1] + "…"
    return text


def plain_log_message(text: str, max_len: int | None = None) -> str:
    """Render Rich markup to plain text, then collapse it to one line."""
    try:
        text = Text.from_markup(text).plain
    except Exception:
        text = re.sub(r"\[/?[a-zA-Z][^\]]*\]", "", text)
    return single_line(text, max_len=max_len)


def truncate_lines(text: str, max_lines: int = 15) -> tuple[str, int]:
    """Truncate text to max_lines. Returns (text, hidden_count)."""
    lines = text.split("\n")
    if len(lines) <= max_lines:
        return text, 0
    return "\n".join(lines[:max_lines]), len(lines) - max_lines


def format_duration(seconds: float) -> str:
    """Format duration as mm:ss or ss.s."""
    if seconds >= 60:
        mins, secs = divmod(int(seconds), 60)
        return f"{mins}:{secs:02d}"
    return f"{seconds:.1f}s"


def format_elapsed(seconds: float) -> str:
    """Format elapsed time as HH:MM:SS or MM:SS if < 1 hour."""
    if seconds >= 3600:
        hrs, remainder = divmod(int(seconds), 3600)
        mins, secs = divmod(remainder, 60)
        return f"{hrs}:{mins:02d}:{secs:02d}"
    mins, secs = divmod(int(seconds), 60)
    return f"{mins:02d}:{secs:02d}"


def format_tokens(tokens: int) -> str:
    """Format token count compactly (e.g., 1.2k, 15k, 1.2M)."""
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.1f}M"
    if tokens >= 10_000:
        return f"{tokens // 1000}k"
    if tokens >= 1_000:
        return f"{tokens / 1000:.1f}k"
    return str(tokens)
