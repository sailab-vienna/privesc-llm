import json
import logging
from typing import Any, cast

from pydantic import BaseModel, ValidationError

from src.config import SFTQualityFilterConfig
from src.dataset.privesc.trace_utils import (
    stringify_content,
    text_content,
    trace_messages,
)
from src.gym.tools import TOOL_METADATA
from src.parsers.qwen3 import parse_tool_calls_detailed

log = logging.getLogger("privesc_sft")

# Keep backward-compatible import path for tests/callers while using the
# single config schema defined in src.config.
QualityConfig = SFTQualityFilterConfig
TOOL_ARG_SCHEMAS: dict[str, type[BaseModel]] = {
    str(name): cast(type[BaseModel], meta["args_schema"])
    for name, meta in TOOL_METADATA.items()
}


def matched_keywords(text: str, keywords: list[str] | tuple[str, ...]) -> list[str]:
    if not keywords:
        return []
    lowered = text.lower()
    return [keyword for keyword in keywords if keyword.lower() in lowered]


class TraceQualityFilter:
    """Encapsulates logic for filtering traces based on quality criteria."""

    def __init__(self, config: QualityConfig):
        self.config = config

    def check_trace_live_prefix(
        self, trace_data: dict[str, Any]
    ) -> tuple[bool, dict[str, Any]]:
        messages = trace_messages(trace_data)
        return self._check_trace(
            trace_data,
            messages,
            require_success=False,
            enforce_min_turns=False,
        )

    def check_trace(self, trace_data: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        return self._check_trace(trace_data, trace_messages(trace_data))

    def _check_trace(
        self,
        trace_data: dict[str, Any],
        messages: list[dict[str, Any]],
        *,
        require_success: bool = True,
        enforce_min_turns: bool = True,
    ) -> tuple[bool, dict[str, Any]]:
        turns = trace_data.get("turns", 0)
        token_count = self._trace_token_count(trace_data)
        success = trace_data.get("success", False)

        assistant_turns = self._count_assistant_turns(messages)
        empty_turns = self._count_empty_reasoning_turns(messages)
        tool_call_free_turns = self._count_tool_call_free_turns(messages)
        leakage = self._check_leakage(messages)
        nudges = self._count_nudges(messages)
        has_solution_block = self._check_solution_block(trace_data, messages)
        tool_call_schema_errors = (
            self._check_tool_call_schema(messages)
            if self.config.reject_on_invalid_tool_call_schema
            else []
        )
        container_misconfig = (
            self._check_container_misconfig(messages)
            if self.config.reject_on_container_misconfig
            else []
        )
        html_entities = (
            self._check_html_entities(messages)
            if self.config.reject_on_html_entities
            else []
        )

        reasons: list[str] = []
        passed = True

        if tool_call_schema_errors:
            passed = False
            reasons.append(
                f"invalid_tool_call_schema ({'; '.join(tool_call_schema_errors)})"
            )

        if container_misconfig:
            passed = False
            reasons.append(f"container_misconfig ({', '.join(container_misconfig)})")

        if html_entities:
            passed = False
            reasons.append(f"html_entities ({', '.join(html_entities)})")

        if require_success and self.config.reject_on_failure and not success:
            passed = False
            reasons.append("not_successful")

        if turns > self.config.max_turns:
            passed = False
            reasons.append(f"too_many_turns ({turns} > {self.config.max_turns})")

        if enforce_min_turns and turns < self.config.min_turns:
            passed = False
            reasons.append(f"too_few_turns ({turns} < {self.config.min_turns})")

        if self.config.max_tokens and token_count > self.config.max_tokens:
            passed = False
            reasons.append(
                f"too_many_tokens ({token_count} > {self.config.max_tokens})"
            )

        if self.config.reject_on_empty_reasoning and empty_turns > 0:
            passed = False
            reasons.append(f"too_many_empty_reasoning ({empty_turns} > 0)")

        if self.config.reject_tool_call_free_turns and tool_call_free_turns > 0:
            passed = False
            reasons.append(f"tool_call_free_turn ({tool_call_free_turns} > 0)")

        if nudges > self.config.max_nudges:
            passed = False
            reasons.append(f"too_many_nudges ({nudges} > {self.config.max_nudges})")

        if self.config.verify_solution_in_prompt and not has_solution_block:
            passed = False
            reasons.append("missing_solution_data_in_prompt")

        if self.config.reject_on_secret_solution_leakage and leakage:
            passed = False
            reasons.append(f"secret_solution_leakage ({', '.join(leakage)})")

        if self.config.warn_on_secret_solution_leakage and leakage:
            scenario = trace_data.get("scenario", "unknown")
            log.warning(
                "Potential hidden solution leakage in trace %s: %s",
                scenario,
                leakage,
            )

        metrics = {
            "passed": passed,
            "reasons": reasons,
            "turns": turns,
            "sft_num_tokens": token_count,
            "assistant_turns": assistant_turns,
            "empty_reasoning_turns": empty_turns,
            "tool_call_free_turns": tool_call_free_turns,
            "secret_solution_leakage_keywords": leakage,
            "has_secret_solution_leakage": len(leakage) > 0,
            "nudges": nudges,
            "has_solution_block": has_solution_block,
            "tool_call_schema_errors": tool_call_schema_errors,
            "has_invalid_tool_call_schema": len(tool_call_schema_errors) > 0,
        }
        return passed, metrics

    def _trace_token_count(self, trace_data: dict[str, Any]) -> int:
        token_count = trace_data.get("sft_num_tokens")
        if isinstance(token_count, int):
            return token_count
        token_count = trace_data.get("total_tokens", 0)
        return token_count if isinstance(token_count, int) else 0

    def _count_assistant_turns(self, messages: list[dict[str, Any]]) -> int:
        return sum(1 for msg in messages if msg.get("role") == "assistant")

    def _count_empty_reasoning_turns(
        self,
        messages: list[dict[str, Any]],
    ) -> int:
        count = 0
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            content = self._assistant_reasoning_text(msg)
            if len(content) < self.config.min_reasoning_length:
                count += 1
        return count

    def _assistant_reasoning_text(self, message: dict[str, Any]) -> str:
        content = text_content(message.get("content", "")).strip()
        if "<tool_call>" in content and "</tool_call>" in content:
            content, _, _ = parse_tool_calls_detailed(content)
            content = content.strip()
        return content

    def _count_tool_call_free_turns(self, messages: list[dict[str, Any]]) -> int:
        count = 0
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            tool_calls = msg.get("tool_calls")
            if not (isinstance(tool_calls, list) and tool_calls):
                count += 1
        return count

    def _check_leakage(
        self,
        messages: list[dict[str, Any]],
    ) -> list[str]:
        found = []
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            content = text_content(msg.get("content", ""))
            found.extend(
                matched_keywords(content, self.config.secret_solution_leakage_keywords)
            )
        return found

    def _count_nudges(self, messages: list[dict[str, Any]]) -> int:
        count = 0
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "") or ""
                if "No tool calls received" in content:
                    count += 1
        return count

    def _check_solution_block(
        self, trace_data: dict[str, Any], messages: list[dict[str, Any]]
    ) -> bool:
        source_flag = trace_data.get("_source_trace_had_solution_block")
        if isinstance(source_flag, bool):
            return source_flag

        for msg in messages:
            if msg.get("role") == "system":
                content = text_content(msg.get("content", ""))
                if "SECRET SOLUTION DATA" in content:
                    return True
        return False

    def _check_tool_call_schema(self, messages: list[dict[str, Any]]) -> list[str]:
        errors: list[str] = []
        for msg_idx, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue

            tool_calls = msg.get("tool_calls")
            if tool_calls is None:
                continue
            if not isinstance(tool_calls, list):
                errors.append(f"assistant[{msg_idx}].tool_calls not_list")
                continue

            for tool_idx, tool_call in enumerate(tool_calls):
                error = self._tool_call_schema_error(tool_call)
                if error is not None:
                    errors.append(
                        f"assistant[{msg_idx}].tool_calls[{tool_idx}] {error}"
                    )

        return errors

    def _tool_call_schema_error(self, tool_call: Any) -> str | None:
        if not isinstance(tool_call, dict):
            return "invalid_shape"

        name, raw_arguments = self._tool_call_name_and_arguments(tool_call)
        if not isinstance(name, str) or not name:
            return "missing_name"

        args_schema = TOOL_ARG_SCHEMAS.get(name)
        if args_schema is None:
            return f"unknown_name={name}"

        if raw_arguments is None:
            return f"{name}.arguments missing"

        arguments = raw_arguments
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return f"{name}.arguments invalid_json"

        if not isinstance(arguments, dict):
            return f"{name}.arguments not_object"

        unexpected = sorted(set(arguments) - set(args_schema.model_fields))
        if unexpected:
            return f"{name}.arguments unexpected={unexpected}"

        try:
            args_schema.model_validate(arguments)
        except ValidationError as exc:
            return self._format_tool_call_validation_error(name, exc)

        return None

    def _tool_call_name_and_arguments(
        self, tool_call: dict[str, Any]
    ) -> tuple[Any, Any]:
        function = tool_call.get("function")
        if isinstance(function, dict):
            return function.get("name"), function.get("arguments", function.get("args"))
        return tool_call.get("name"), tool_call.get("arguments", tool_call.get("args"))

    def _format_tool_call_validation_error(
        self, name: str, error: ValidationError
    ) -> str:
        details = error.errors()
        if not details:
            return f"{name} invalid_arguments"

        first = details[0]
        loc = ".".join(str(part) for part in first.get("loc", ()))
        message = str(first.get("msg", "invalid_arguments"))
        return f"{name}.{loc} {message}" if loc else f"{name} {message}"

    def _check_container_misconfig(self, messages: list[dict[str, Any]]) -> list[str]:
        found: list[str] = []
        bins = [str(b) for b in self.config.container_expected_binaries]

        for msg in messages:
            if msg.get("role") != "tool":
                continue

            text = stringify_content(msg.get("content", "")).lower()
            for b in bins:
                bb = b.lower()
                if f"{bb}: command not found" in text or f"{bb}: not found" in text:
                    if b not in found:
                        found.append(b)

        return found

    def _check_html_entities(
        self,
        messages: list[dict[str, Any]],
    ) -> list[str]:
        needles = [str(n) for n in self.config.reject_html_entities]
        if not needles:
            return []

        found: list[str] = []
        for msg in messages:
            if msg.get("role") not in {"assistant", "tool"}:
                continue

            text = stringify_content(msg.get("content", ""))
            for n in needles:
                if n in text and n not in found:
                    found.append(n)

        return found
