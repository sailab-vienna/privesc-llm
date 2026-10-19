"""Prompt utilities for PrivEsc agents.

This module provides:
1. System prompt rendering (render_system_prompt) - Jinja template based
2. Initial message construction (initial_messages) - for LangGraph
3. HuggingFace-style tool instruction injection (append_hf_tool_instructions)
"""

import json
import os
from typing import Any

from jinja2 import Environment, FileSystemLoader
from langchain_core.messages import HumanMessage, SystemMessage
from omegaconf import OmegaConf

from src.config import AppConfig, effective_max_assistant_turns
from src.gym.tools import get_tools_for_chat_template
from src.utils.solution import load_solution_data

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")


def _to_container(obj: Any) -> Any:
    """Convert OmegaConf objects to plain Python containers."""
    return (
        OmegaConf.to_container(obj, resolve=True) if OmegaConf.is_config(obj) else obj
    )


def _resolve_template_vars(raw_vars: Any) -> dict[str, Any]:
    raw_vars = _to_container(raw_vars)
    return (
        {str(k): v for k, v in raw_vars.items()} if isinstance(raw_vars, dict) else {}
    )


def _tojson_pretty(obj: Any) -> str:
    """JSON serialize with pretty printing and no ASCII escaping."""
    return json.dumps(obj, indent=2, ensure_ascii=False)


def render_system_prompt(config: AppConfig) -> str:
    """Render the system prompt using Jinja templates."""
    env = Environment(loader=FileSystemLoader(PROMPTS_DIR))
    env.filters["tojson_pretty"] = _tojson_pretty
    template = env.get_template(config.prompts.system_template)
    template_vars = _resolve_template_vars(config.prompts.template_vars)
    template_vars["max_turns"] = effective_max_assistant_turns(config)
    if (
        config.runner.mode == "trace_collection"
        and template_vars.get("include_solution_guidance") is True
    ):
        # Use solution from config (procedural) or fall back to YAML lookup (static)
        solution = config.scenario.solution or load_solution_data(
            config.scenario.name, __file__
        )
        template_vars["solution"] = _to_container(solution)
    return template.render(**template_vars)


def render_system_prompt_from_template(
    system_template: str, template_vars: dict[str, Any]
) -> str:
    """Render a system prompt from a template path and explicit vars.

    This is used by RL training loops where scenario-dependent variables (e.g.,
    container credentials) must be injected per-sample.
    """

    env = Environment(loader=FileSystemLoader(PROMPTS_DIR))
    env.filters["tojson_pretty"] = _tojson_pretty
    template = env.get_template(system_template)
    return template.render(**_resolve_template_vars(template_vars))


_RL_DYNAMIC_PROMPT_KEYS = {"user", "password", "max_turns", "term_cols", "term_rows"}


def base_template_vars_for_rl(template_vars: dict[str, Any]) -> dict[str, Any]:
    """Return template vars safe to reuse across RL samples.

    RL datasets inject scenario-specific values per-sample (e.g., credentials).
    This helper strips known dynamic keys so global config defaults can't leak
    into procedural training prompts.
    """

    return {k: v for k, v in template_vars.items() if k not in _RL_DYNAMIC_PROMPT_KEYS}


def initial_messages(cfg: AppConfig) -> list:
    """Return entry-point LangChain messages for the agent."""
    return [
        SystemMessage(content=render_system_prompt(cfg)),
        HumanMessage(content=cfg.prompts.start_instruction),
    ]


# =============================================================================
# HuggingFace chat-template tool instructions (single source of truth)
# =============================================================================


def get_hf_tool_instructions() -> str:
    """Generate tool instructions matching HuggingFace chat template format.

    This produces the EXACT format that Qwen's chat template uses when
    tools are passed to apply_chat_template(..., tools=tools).

    Format matches: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
    """

    tools = get_tools_for_chat_template()

    # Build tool JSON strings (compact, one per line)
    tool_jsons = "\n".join(json.dumps(t) for t in tools)

    # Match HuggingFace chat template format exactly
    return f"""# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_jsons}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""


def append_hf_tool_instructions(base_system_prompt: str) -> str:
    """Append tool call instructions matching the Qwen HuggingFace chat template."""

    return base_system_prompt + "\n\n" + get_hf_tool_instructions()
