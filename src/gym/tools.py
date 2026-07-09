"""PrivEsc tools - single source of truth for tool definitions.

This module provides:
1. Tool creation for runtime (make_privesc_tools) - bound to a scenario
2. Tool schema extraction (get_tool_schemas) - for prompts/SFT/RL

Consumers:
- LangGraph agent (runtime tool execution)
- Unsloth SFT (apply_chat_template with tools=)
- XML tool-call prompt helpers

The tool schemas are auto-generated from Pydantic models,
ensuring consistency across all pipelines.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict
from typing import Any

from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_function
from pydantic import BaseModel, Field

from src.gym.scenario import PrivEscScenario, ToolResult
from src.parsers.qwen3 import summarize_tool_call_parse_errors
from src.parsers.tool_calls import parse_tool_calls_detailed

log = logging.getLogger("privesc_tools")


def _tool_json(result: ToolResult) -> str:
    """Serialize ToolResult dataclass to a compact JSON string for tool outputs."""
    return json.dumps(asdict(result), ensure_ascii=False)


# =============================================================================
# Tool schemas (Pydantic models as single source of truth)
# =============================================================================


class ExecCommandInput(BaseModel):
    """Input for exec_command tool."""

    command: str = Field(description="The Bash command to execute.")


class TestCredentialsInput(BaseModel):
    """Input for test_credentials tool."""

    user: str = Field(description="The username to try.")
    password: str = Field(description="The password to try.")


# Tool metadata (descriptions)
TOOL_METADATA = {
    "exec_command": {
        "description": "Run a shell command as the current user. Use for info gathering or exploits. Returns output and got_root.",
        "args_schema": ExecCommandInput,
    },
    "test_credentials": {
        "description": "Try logging in with a username and password. Call when you find credentials. Returns success and got_root.",
        "args_schema": TestCredentialsInput,
    },
}


def get_tool_schemas() -> list[dict[str, Any]]:
    """Get OpenAI-format tool schemas from tool metadata.

    This is the single source of truth for tool definitions used by:
    - LangGraph agents (runtime)
    - SFT data rendering
    - Tool-call prompt helpers

    Returns:
        List of tool schemas in OpenAI function calling format
    """
    schemas = []
    for name, meta in TOOL_METADATA.items():
        tool = StructuredTool.from_function(
            func=lambda: None,
            name=name,
            description=meta["description"],  # type: ignore[arg-type]
            args_schema=meta["args_schema"],  # type: ignore[arg-type]
        )
        schemas.append(convert_to_openai_function(tool))
    return schemas


def get_tools_for_chat_template() -> list[dict[str, Any]]:
    """Get tools in format expected by HuggingFace chat templates.

    This wraps each tool schema in {"type": "function", "function": schema}
    format used by tokenizer.apply_chat_template(..., tools=tools).
    """
    schemas = get_tool_schemas()
    return [{"type": "function", "function": schema} for schema in schemas]


# =============================================================================
# Runtime tool creation (bound to scenario)
# =============================================================================


def make_privesc_tools(scenario: PrivEscScenario) -> list[StructuredTool]:
    """Create LangChain tools for the PrivEsc scenario.

    These tools are bound to a specific scenario and can be used with
    LangGraph agents for runtime execution. Uses scenario's logger for output.

    Uses TOOL_METADATA for descriptions/schemas to ensure consistency.
    """

    async def exec_command(command: str) -> str:
        log.debug("exec_command: %s", command[:80])
        start = time.perf_counter()
        result = await scenario.exec_command(command)
        duration_ms = int((time.perf_counter() - start) * 1000)
        log.debug(
            "exec_command result: exit=%d got_root=%s",
            result.exit_code,
            result.got_root,
        )
        if scenario.logger:
            scenario.logger.tool_result("exec_command", asdict(result), duration_ms)
        return _tool_json(result)

    async def test_credentials(user: str, password: str) -> str:
        log.debug("test_credentials: user=%s", user)
        start = time.perf_counter()
        result = await scenario.test_credentials(user, password)
        duration_ms = int((time.perf_counter() - start) * 1000)
        log.debug(
            "test_credentials result: success=%s got_root=%s",
            result.success,
            result.got_root,
        )
        if scenario.logger:
            scenario.logger.tool_result("test_credentials", asdict(result), duration_ms)
        return _tool_json(result)

    # Map function names to implementations
    implementations = {
        "exec_command": exec_command,
        "test_credentials": test_credentials,
    }

    # Create tools using shared metadata
    tools = []
    for name, meta in TOOL_METADATA.items():
        tool = StructuredTool.from_function(
            coroutine=implementations[name],
            name=name,
            description=meta["description"],  # type: ignore[arg-type]
            args_schema=meta["args_schema"],  # type: ignore[arg-type]
        )
        tools.append(tool)

    return tools


# =============================================================================
# XML tool call parsing
# =============================================================================


def parse_xml_tool_calls(message):
    """Parse <tool_call> XML from message content into tool_calls field."""
    if message.tool_calls:
        return message

    content = message.content
    if not isinstance(content, str) or "<tool_call>" not in content:
        return message

    _, calls, errors = parse_tool_calls_detailed(content, parser="hermes")

    if errors:
        log.warning("%s", summarize_tool_call_parse_errors(errors))
        details = [
            {
                "kind": error.kind,
                "message": error.message,
                "raw": error.raw,
            }
            for error in errors
        ]
        extra = getattr(message, "additional_kwargs", None)
        if not isinstance(extra, dict):
            extra = {}
            message.additional_kwargs = extra
        extra["invalid_tool_call_errors"] = details

    if calls:
        log.debug("Parsed %d XML tool calls", len(calls))
        message.tool_calls = [
            {
                "name": c.name,
                "args": json.loads(c.arguments),
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "tool_call",
            }
            for c in calls
        ]

    return message
