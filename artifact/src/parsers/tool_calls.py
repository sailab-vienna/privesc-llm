from typing import List, Literal, Tuple, cast, get_args

from src.parsers import qwen3
from src.parsers.qwen3 import ToolCall, ToolCallParseError

ToolCallParser = Literal["hermes"]
SUPPORTED_TOOL_CALL_PARSERS = get_args(ToolCallParser)


def validate_tool_call_parser(parser: str) -> ToolCallParser:
    if parser in SUPPORTED_TOOL_CALL_PARSERS:
        return cast(ToolCallParser, parser)
    raise ValueError(
        f"parser must be one of: {', '.join(SUPPORTED_TOOL_CALL_PARSERS)}"
    )


def parse_tool_calls_detailed(
    text: str,
    *,
    parser: ToolCallParser = "hermes",
) -> Tuple[str, List[ToolCall], List[ToolCallParseError]]:
    if parser == "hermes":
        return qwen3.parse_tool_calls_detailed(text)
    raise AssertionError(f"Unhandled tool call parser: {parser}")
