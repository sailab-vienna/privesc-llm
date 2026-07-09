from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

import jinja2
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf

from src.config import (
    AppConfig,
    plain_config_dict,
)
from src.dataset.privesc.trace_utils import (
    sft_assistant_content,
    text_content,
    trace_messages,
)
from src.gym.context import TOKENIZER_MODEL, get_num_tokens_from_messages


def _project_root() -> Path:
    try:
        return Path(get_original_cwd())
    except ValueError:
        return Path(__file__).resolve().parents[3]


def _load_default_start_instruction() -> str:
    prompt_path = _project_root() / "conf" / "prompts" / "default.yaml"
    data = OmegaConf.load(prompt_path)
    value = OmegaConf.select(data, "start_instruction")
    return str(value) if value is not None else ""


def _load_deployment_system_template() -> str:
    prompt_path = _project_root() / "conf" / "prompts" / "evaluation.yaml"
    data = OmegaConf.load(prompt_path)
    value = OmegaConf.select(data, "system_template")
    if value is None:
        raise ValueError("Missing system_template in conf/prompts/evaluation.yaml")
    return str(value)


def _prompt_base_vars(cfg: AppConfig) -> dict[str, Any]:
    prompts_cfg = getattr(cfg, "prompts", None)
    return plain_config_dict(getattr(prompts_cfg, "template_vars", None))


def _prompt_vars_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    prompt_vars = metadata.get("prompt_vars")
    if not isinstance(prompt_vars, dict) or not prompt_vars:
        raise ValueError("Missing metadata.prompt_vars for trace")
    return prompt_vars


def _render_system_prompt(
    template: jinja2.Template, base_vars: dict[str, Any], metadata: dict[str, Any]
) -> str:
    render_vars = base_vars.copy()
    render_vars.update(_prompt_vars_from_metadata(metadata))
    return template.render(**render_vars)


def sft_training_messages(trace_data: dict[str, Any]) -> list[dict[str, Any]]:
    messages = []
    for message in trace_messages(trace_data):
        item = message.copy()
        if item.get("role") == "assistant":
            item["content"] = sft_training_content(item)
        messages.append(item)
    return messages


def sft_training_content(message: dict[str, Any]) -> Any:
    content = message.get("content", "")
    if message.get("role") == "assistant":
        return sft_assistant_content(content).strip()
    return content


def sft_token_count(
    trace_data: dict[str, Any], *, tools: Sequence[Any] | None = None
) -> int:
    if tools is None:
        trace_tools = trace_data.get("tools")
        tools = trace_tools if isinstance(trace_tools, list) else None
    return get_num_tokens_from_messages(
        sft_training_messages(trace_data),
        tools=tools,
    )


def record_sft_token_count(
    trace_data: dict[str, Any], *, tools: Sequence[Any] | None = None
) -> None:
    trace_data["sft_num_tokens"] = sft_token_count(trace_data, tools=tools)
    trace_data["sft_tokenizer_model"] = TOKENIZER_MODEL
    metadata = trace_data.get("metadata")
    if isinstance(metadata, dict):
        metadata["sft_tokenizer_model"] = TOKENIZER_MODEL


class SFTPromptNormalizer:
    def __init__(self, cfg: AppConfig):
        prompt_dir = _project_root() / "src" / "prompts"
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(prompt_dir)))
        self._deployment_template = env.get_template(_load_deployment_system_template())
        self._base_prompt_vars = _prompt_base_vars(cfg)
        self._user_prompt = _load_default_start_instruction()

    def normalize_message_preamble(
        self, messages: list[dict[str, Any]], metadata: dict[str, Any]
    ) -> None:
        if (
            len(messages) < 2
            or messages[0].get("role") != "system"
            or messages[1].get("role") != "user"
        ):
            raise ValueError("Expected message preamble [system, user]")

        messages[0]["content"] = _render_system_prompt(
            self._deployment_template, self._base_prompt_vars, metadata
        )
        messages[1]["content"] = self._user_prompt

    def normalize_trace_prompts(self, trace_data: dict[str, Any]) -> dict[str, Any]:
        trace_data = deepcopy(trace_data)
        if trace_data.get("mode") != "trace_collection":
            return trace_data

        messages = trace_messages(trace_data)
        if not messages:
            return trace_data

        trace_data["_source_trace_had_solution_block"] = any(
            msg.get("role") == "system"
            and "SECRET SOLUTION DATA" in text_content(msg.get("content", ""))
            for msg in messages
        )
        metadata = trace_data.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("Missing trace metadata")
        self.normalize_message_preamble(messages, metadata)
        return trace_data
