"""Test chat-template tool instructions for XML tool-call models."""

import ast
from pathlib import Path
from typing import Callable, cast

import pytest

pytest.importorskip("transformers")

from transformers import AutoTokenizer  # noqa: E402

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
BASE_SYSTEM = "You are a penetration testing assistant."
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


def _load_prompt_functions():
    path = Path("src/gym/prompts.py")
    source = path.read_text(encoding="utf-8")
    module = ast.parse(source)
    selected: list[ast.stmt] = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"get_hf_tool_instructions", "append_hf_tool_instructions"}
    ]
    globals_ns = {
        "json": __import__("json"),
        "get_tools_for_chat_template": lambda: TOOLS,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), globals_ns)
    return cast(Callable[[str], str], globals_ns["append_hf_tool_instructions"])


@pytest.fixture(scope="session")
def qwen_tokenizer():
    try:
        return AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    except Exception:
        pytest.skip("Cannot load Qwen tokenizer")


@pytest.fixture(scope="session")
def tools():
    return TOOLS


def test_system_prompt_with_tools_is_equivalent(qwen_tokenizer, tools):
    user_message = "Begin your assessment."
    append_hf_tool_instructions = _load_prompt_functions()

    xml_tool_system = append_hf_tool_instructions(BASE_SYSTEM)
    xml_tool_rendered = qwen_tokenizer.apply_chat_template(
        [
            {"role": "system", "content": xml_tool_system},
            {"role": "user", "content": user_message},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    hf_rendered = qwen_tokenizer.apply_chat_template(
        [
            {"role": "system", "content": BASE_SYSTEM},
            {"role": "user", "content": user_message},
        ],
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )

    assert xml_tool_rendered == hf_rendered
