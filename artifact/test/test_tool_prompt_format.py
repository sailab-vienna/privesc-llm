"""Tests for HuggingFace tool-call prompt formatting."""

import ast
from pathlib import Path
from typing import Callable, cast

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "exec_command",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute.",
                    }
                },
                "required": ["command"],
            },
        },
    }
]


def _load_get_hf_tool_instructions():
    path = Path("src/gym/prompts.py")
    source = path.read_text(encoding="utf-8")
    module = ast.parse(source)
    selected: list[ast.stmt] = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_hf_tool_instructions"
    ]
    globals_ns = {
        "json": __import__("json"),
        "get_tools_for_chat_template": lambda: TOOLS,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), globals_ns)
    return cast(Callable[[], str], globals_ns["get_hf_tool_instructions"])


def test_hf_tool_instructions_use_arguments_key() -> None:
    instructions = _load_get_hf_tool_instructions()()

    assert '"name": <function-name>' in instructions
    assert '"arguments": <args-json-object>' in instructions
    assert '"args": <args-json-object>' not in instructions
