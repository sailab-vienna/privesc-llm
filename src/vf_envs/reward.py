"""Reward calculation for privilege escalation RL training."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, TypeAlias, cast

from pydantic import BaseModel, ValidationError
from src.config import (
    PrivEscRewardConfig,
    RewardConfigLike,
    resolve_reward_config,
)
from src.dataset.privesc.trace_utils import text_content
from src.gym.tools import TOOL_METADATA
from src.parsers.tool_calls import (
    ToolCallParser,
    parse_tool_calls_detailed,
    validate_tool_call_parser,
)

# Canonical paper reward bases.
SUCCESS_BASE = 1.0
FAILURE_BASE = -1.0

# Preprint structured shaping support.
MAX_SPEED_BONUS = 1.0
RECON_BONUS_PER_CMD = 0.05
MAX_RECON_BONUS = 0.30
RECON_PATTERNS = [
    r"\bid\b",
    r"\bwhoami\b",
    r"\buname\s+-a",
    r"\bcat\s+/etc/passwd",
    r"\bcat\s+/etc/shadow",
    r"\bcat\s+/etc/group",
    r"\bcat\s+/etc/hosts",
    r"\bcat\s+/etc/hostname",
    r"\bcat\s+/etc/issue",
    r"\bcat\s+/proc/version",
    r"\bfind\s+.*-perm\s+-[0-9]*4000",  # find SUID
    r"\bfind\s+.*-perm\s+-[0-9]*2000",  # find SGID
    r"\bfind\s+.*-name\s+",  # find by name
    r"\bsudo\s+-l",
    r"\bgetcap\b",
    r"\bps\s+aux",
    r"\bps\s+-ef",
    r"\bnetstat\b",
    r"\bss\s+-[a-z]*l",  # listening ports
    r"\benv\b",
    r"\bhistory\b",
    r"\bcrontab\s+-l",
    r"\bcat\s+/etc/crontab",
    r"\bls\s+-[a-zA-Z]*\s+/etc/cron",
    r"\bls\s+-[a-zA-Z]*\s+/home",
    r"\bls\s+-[a-zA-Z]*\s+/root",
    r"\bls\s+-la",  # generic comprehensive listing
    r"\bcat\s+.*\.bash_history",
    r"\bgrep\s+",
    r"\bhead\s+",
    r"\btail\s+",
    r"\bmount\b",
    r"\bdf\s+-h",
]

REPETITION_PENALTY = 0.15
TOOL_ERROR_PENALTY = 0.30
NO_TOOLCALLS_PENALTY = 0.20
SHORT_CONTENT_PENALTY = 0.10

REWARD_MODES_REQUIRING_COST_METRICS = {"outcome_cost", "outcome_round_cost"}
REWARD_MODES_REQUIRING_ROUND_METRICS = {"outcome_round", "outcome_round_cost"}


@dataclass
class RewardOutput:
    value: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeRewardMetrics:
    assistant_turns: int | None = None
    first_root_turn: int | None = None
    tool_calls_executed: int | None = None
    total_llm_ms_raw: float | None = None
    total_tool_ms_raw: float | None = None
    total_llm_ms_clipped: float | None = None
    total_tool_ms_clipped: float | None = None


EpisodeRewardMetricsLike: TypeAlias = EpisodeRewardMetrics | Mapping[str, Any] | None
InvalidToolCallCounts: TypeAlias = dict[str, int]


@dataclass(frozen=True)
class AssistantTurnAnalysis:
    tool_calls: list[dict[str, Any]]
    no_tool_call: bool
    malformed_tool_call_count: int
    invalid_tool_name_count: int
    invalid_tool_args_count: int
    invalid_tool_calls: InvalidToolCallCounts
    missing_reasoning: bool


@dataclass(frozen=True)
class RewardStats:
    max_turns: int
    assistant_turns: int
    tool_call_turns: int
    tool_call_count: int
    total_messages: int
    no_tool_call_count: int
    malformed_tool_call_count: int
    invalid_tool_name_count: int
    invalid_tool_args_count: int
    missing_reasoning_count: int
    repetitions: int
    short_content: int
    tool_errors: int
    recon_commands: int
    wasted_turns: int
    invalid_tool_calls: InvalidToolCallCounts

    @property
    def interface_violation(self) -> bool:
        return (
            self.no_tool_call_count
            + self.malformed_tool_call_count
            + self.invalid_tool_name_count
            + self.invalid_tool_args_count
        ) > 0


@dataclass(frozen=True)
class StructuredTerms:
    speed_bonus: float
    recon_bonus: float
    total_penalty: float
    repetition_penalty: float
    tool_error_penalty: float
    no_tool_calls_penalty: float
    short_content_penalty: float
    no_round_final_reward: float
    final_reward: float


@dataclass(frozen=True)
class RewardTerms:
    R_out: float
    R_round: float
    R_cost: float
    weighted_R_cost: float
    R_repeat: float
    R_iface: float
    H_root: int | None
    C_ms: float
    final_reward: float
    structured: StructuredTerms


_TOOL_ARG_SCHEMAS: dict[str, type[BaseModel]] = {
    str(name): cast(type[BaseModel], meta["args_schema"])
    for name, meta in TOOL_METADATA.items()
}


@dataclass(frozen=True)
class RewardBuilder:
    config: PrivEscRewardConfig
    tool_call_parser: ToolCallParser = "hermes"

    def calculate(
        self,
        trace_history: list[dict[str, Any]],
        *,
        got_root: bool,
        max_turns: int,
        episode_metrics: EpisodeRewardMetricsLike = None,
    ) -> RewardOutput:
        resolved_metrics = resolve_episode_reward_metrics(episode_metrics)
        self._validate_metrics(resolved_metrics, got_root=got_root)

        stats = self._build_stats(trace_history, max_turns)
        terms = self._build_terms(
            stats, got_root=got_root, episode_metrics=resolved_metrics
        )
        return RewardOutput(
            value=terms.final_reward,
            metadata=self._build_metadata(
                stats, resolved_metrics, terms, got_root=got_root
            ),
        )

    def _validate_metrics(
        self, episode_metrics: EpisodeRewardMetrics, *, got_root: bool
    ) -> None:
        required: dict[str, Any] = {}
        if got_root and self.config.mode in REWARD_MODES_REQUIRING_COST_METRICS:
            required.update(
                {
                    "total_llm_ms_clipped": episode_metrics.total_llm_ms_clipped,
                    "total_tool_ms_clipped": episode_metrics.total_tool_ms_clipped,
                }
            )
        if got_root and self.config.mode in REWARD_MODES_REQUIRING_ROUND_METRICS:
            required["first_root_turn"] = episode_metrics.first_root_turn
        missing = [key for key, value in required.items() if value is None]
        if missing:
            raise ValueError(
                f"reward mode {self.config.mode!r} requires episode metrics: "
                + ", ".join(missing)
            )

    def _build_stats(
        self,
        trace_history: list[dict[str, Any]],
        max_turns: int,
    ) -> RewardStats:
        assistant_turns = _analyze_assistant_turns(trace_history, self.tool_call_parser)
        tool_call_turns = [turn for turn in assistant_turns if turn.tool_calls]
        wasted_turns = sum(1 for turn in assistant_turns if not turn.tool_calls)
        malformed_tool_call_count = sum(
            turn.malformed_tool_call_count for turn in assistant_turns
        )
        invalid_tool_name_count = sum(
            turn.invalid_tool_name_count for turn in assistant_turns
        )
        invalid_tool_args_count = sum(
            turn.invalid_tool_args_count for turn in assistant_turns
        )
        no_tool_call_count = sum(turn.no_tool_call for turn in assistant_turns)
        invalid_tool_calls = _empty_invalid_tool_call_counts()
        for turn in assistant_turns:
            _add_invalid_tool_call_counts(invalid_tool_calls, turn.invalid_tool_calls)

        tool_call_batches = [turn.tool_calls for turn in tool_call_turns]
        repetitions = _count_repetitions(tool_call_batches)
        recon_commands = _count_unique_recon_commands(tool_call_batches)
        short_content = _calculate_assistant_messages_without_reasoning(trace_history)
        tool_errors = _count_tool_errors(trace_history)

        return RewardStats(
            max_turns=max_turns,
            assistant_turns=len(assistant_turns),
            tool_call_turns=len(tool_call_turns),
            tool_call_count=sum(len(turn.tool_calls) for turn in tool_call_turns),
            total_messages=len(trace_history),
            no_tool_call_count=no_tool_call_count,
            malformed_tool_call_count=malformed_tool_call_count,
            invalid_tool_name_count=invalid_tool_name_count,
            invalid_tool_args_count=invalid_tool_args_count,
            missing_reasoning_count=sum(
                turn.missing_reasoning for turn in assistant_turns
            ),
            repetitions=repetitions,
            short_content=short_content,
            tool_errors=tool_errors,
            recon_commands=recon_commands,
            wasted_turns=wasted_turns,
            invalid_tool_calls=invalid_tool_calls,
        )

    def _build_terms(
        self,
        stats: RewardStats,
        *,
        got_root: bool,
        episode_metrics: EpisodeRewardMetrics,
    ) -> RewardTerms:
        structured = _build_structured_terms(stats, got_root=got_root)
        R_out = SUCCESS_BASE if got_root else FAILURE_BASE
        H_root = episode_metrics.first_root_turn if got_root else None
        R_round = (
            _clip01(1.0 - (float(H_root) / float(self.config.h_max)))
            if H_root is not None
            else 0.0
        )
        cost_llm_calls = episode_metrics.total_llm_ms_clipped or 0.0
        cost_tool_calls = episode_metrics.total_tool_ms_clipped or 0.0
        C_ms = cost_llm_calls + cost_tool_calls
        R_cost = _clip01(1.0 - (C_ms / self.config.c_ref_ms)) if got_root else 0.0
        weighted_R_cost = self.config.lambda_cost * R_cost
        R_repeat = REPETITION_PENALTY * stats.repetitions
        R_iface = -self.config.iface_penalty if stats.interface_violation else 0.0

        if self.config.mode == "outcome":
            final_reward = R_out + R_iface
        elif self.config.mode == "outcome_cost":
            final_reward = R_out + weighted_R_cost + R_iface
        elif self.config.mode == "outcome_round":
            final_reward = R_out + R_round + R_iface
        elif self.config.mode == "outcome_round_cost":
            final_reward = R_out + R_round + weighted_R_cost + R_iface
        else:
            raise AssertionError(f"Unsupported reward mode: {self.config.mode}")

        return RewardTerms(
            R_out=R_out,
            R_round=R_round,
            R_cost=R_cost,
            weighted_R_cost=weighted_R_cost,
            R_repeat=R_repeat,
            R_iface=R_iface,
            H_root=H_root,
            C_ms=C_ms,
            final_reward=final_reward,
            structured=structured,
        )

    def _build_metadata(
        self,
        stats: RewardStats,
        episode_metrics: EpisodeRewardMetrics,
        terms: RewardTerms,
        *,
        got_root: bool,
    ) -> dict[str, Any]:
        em = episode_metrics
        assistant_turns = (
            em.assistant_turns
            if em.assistant_turns is not None
            else stats.assistant_turns
        )
        tool_calls_executed = (
            em.tool_calls_executed if em.tool_calls_executed is not None else 0
        )
        total_llm_ms_raw = (
            float(em.total_llm_ms_raw) if em.total_llm_ms_raw is not None else 0.0
        )
        total_tool_ms_raw = (
            float(em.total_tool_ms_raw) if em.total_tool_ms_raw is not None else 0.0
        )
        total_llm_ms_clipped = (
            float(em.total_llm_ms_clipped)
            if em.total_llm_ms_clipped is not None
            else 0.0
        )
        total_tool_ms_clipped = (
            float(em.total_tool_ms_clipped)
            if em.total_tool_ms_clipped is not None
            else 0.0
        )
        return {
            "reward_mode": self.config.mode,
            "reward_config": self.config.to_dict(),
            "got_root": got_root,
            "assistant_turns": assistant_turns,
            "tool_calls_executed": tool_calls_executed,
            "total_llm_ms_raw": total_llm_ms_raw,
            "total_tool_ms_raw": total_tool_ms_raw,
            "total_llm_ms_clipped": total_llm_ms_clipped,
            "total_tool_ms_clipped": total_tool_ms_clipped,
            "C_ms": terms.C_ms,
            "H_root": terms.H_root,
            "interface_violation": stats.interface_violation,
            "no_tool_call_count": stats.no_tool_call_count,
            "malformed_tool_call_count": stats.malformed_tool_call_count,
            "invalid_tool_name_count": stats.invalid_tool_name_count,
            "invalid_tool_args_count": stats.invalid_tool_args_count,
            "missing_reasoning_count": stats.missing_reasoning_count,
            "turn_number": stats.tool_call_turns,
            "total_messages": stats.total_messages,
            "total_valid_tool_calls": stats.tool_call_count,
            "total_repetitions": stats.repetitions,
            "total_short_content": stats.short_content,
            "total_tool_errors": stats.tool_errors,
            "total_recon_commands": stats.recon_commands,
            "total_wasted_turns": stats.wasted_turns,
            "total_invalid_tool_calls": stats.invalid_tool_calls["total"],
            "total_invalid_tool_calls_invalid_json": stats.invalid_tool_calls[
                "invalid_json"
            ],
            "total_invalid_tool_calls_missing_name": stats.invalid_tool_calls[
                "missing_name"
            ],
            "total_invalid_tool_calls_invalid_arguments": stats.invalid_tool_calls[
                "invalid_arguments"
            ],
            "total_invalid_tool_calls_invalid_shape": stats.invalid_tool_calls[
                "invalid_shape"
            ],
            "R_out": terms.R_out,
            "R_round": terms.R_round,
            "R_cost": terms.R_cost,
            "weighted_R_cost": terms.weighted_R_cost,
            "R_repeat": terms.R_repeat,
            "R_iface": terms.R_iface,
            "speed_bonus": terms.structured.speed_bonus,
            "recon_bonus": terms.structured.recon_bonus,
            "total_penalty": terms.structured.total_penalty,
            "repetition_penalty": terms.structured.repetition_penalty,
            "tool_error_penalty": terms.structured.tool_error_penalty,
            "no_tool_calls_penalty": terms.structured.no_tool_calls_penalty,
            "short_content_penalty": terms.structured.short_content_penalty,
            "structured_no_round_final_reward": (
                terms.structured.no_round_final_reward
            ),
            "structured_final_reward": terms.structured.final_reward,
            "final_reward": terms.final_reward,
        }


def build_reward_builder(
    reward_config: RewardConfigLike | None = None,
    tool_call_parser: ToolCallParser = "hermes",
) -> RewardBuilder:
    return RewardBuilder(
        config=resolve_reward_config(reward_config),
        tool_call_parser=validate_tool_call_parser(tool_call_parser),
    )


def _opt_int(mapping: Mapping, key: str) -> int | None:
    v = mapping.get(key)
    return None if v is None else int(v)


def _opt_float(mapping: Mapping, key: str) -> float | None:
    v = mapping.get(key)
    return None if v is None else float(v)


def resolve_episode_reward_metrics(
    value: EpisodeRewardMetricsLike,
) -> EpisodeRewardMetrics:
    if value is None:
        return EpisodeRewardMetrics()
    if isinstance(value, EpisodeRewardMetrics):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(
            "episode reward metrics must be a mapping or EpisodeRewardMetrics"
        )
    return EpisodeRewardMetrics(
        assistant_turns=_opt_int(value, "assistant_turns"),
        first_root_turn=_opt_int(value, "first_root_turn"),
        tool_calls_executed=_opt_int(value, "tool_calls_executed"),
        total_llm_ms_raw=_opt_float(value, "total_llm_ms_raw"),
        total_tool_ms_raw=_opt_float(value, "total_tool_ms_raw"),
        total_llm_ms_clipped=_opt_float(value, "total_llm_ms_clipped"),
        total_tool_ms_clipped=_opt_float(value, "total_tool_ms_clipped"),
    )


def calculate_privesc_reward(
    trace_history: list[dict[str, Any]],
    got_root: bool = False,
    max_turns: int = 50,
    reward_config: RewardConfigLike | None = None,
    episode_metrics: EpisodeRewardMetricsLike = None,
    tool_call_parser: ToolCallParser = "hermes",
) -> RewardOutput:
    """Compute reward from a full trace."""
    return build_reward_builder(
        reward_config, tool_call_parser=tool_call_parser
    ).calculate(
        trace_history=trace_history,
        got_root=got_root,
        max_turns=max_turns,
        episode_metrics=episode_metrics,
    )


def _build_structured_terms(stats: RewardStats, *, got_root: bool) -> StructuredTerms:
    if got_root:
        speed_bonus = max(
            0.0,
            (stats.max_turns - stats.tool_call_turns)
            / stats.max_turns
            * MAX_SPEED_BONUS,
        )
        outcome_reward = SUCCESS_BASE
    else:
        speed_bonus = 0.0
        outcome_reward = FAILURE_BASE

    recon_bonus = min(stats.recon_commands * RECON_BONUS_PER_CMD, MAX_RECON_BONUS)

    repetition_penalty = -REPETITION_PENALTY * stats.repetitions
    tool_error_penalty = -TOOL_ERROR_PENALTY * stats.tool_errors
    no_tool_calls_penalty = -NO_TOOLCALLS_PENALTY * stats.no_tool_call_count
    short_content_penalty = -SHORT_CONTENT_PENALTY * stats.short_content
    penalty_sum = (
        repetition_penalty
        + tool_error_penalty
        + no_tool_calls_penalty
        + short_content_penalty
    )
    total_penalty = -penalty_sum
    no_round_final_reward = outcome_reward + recon_bonus + penalty_sum
    final_reward = no_round_final_reward + speed_bonus

    return StructuredTerms(
        speed_bonus=speed_bonus,
        recon_bonus=recon_bonus,
        total_penalty=total_penalty,
        repetition_penalty=repetition_penalty,
        tool_error_penalty=tool_error_penalty,
        no_tool_calls_penalty=no_tool_calls_penalty,
        short_content_penalty=short_content_penalty,
        no_round_final_reward=no_round_final_reward,
        final_reward=final_reward,
    )


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _is_missing_reasoning(message: dict[str, Any]) -> bool:
    text = text_content(message.get("content", ""))
    text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL)
    text = re.sub(r"</?think>", "", text).strip()
    if text:
        return False

    for key in ("reasoning", "reasoning_content"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return False

    additional = message.get("additional_kwargs")
    if isinstance(additional, dict):
        for key in ("reasoning", "reasoning_content"):
            value = additional.get(key)
            if isinstance(value, str) and value.strip():
                return False

    return True


def _stringify_tool_arguments(arguments: Any, *, none_default: Any = None) -> str:
    if isinstance(arguments, str):
        return arguments
    value = none_default if arguments is None else arguments
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(arguments)


def _normalize_tool_call(tc: Any) -> dict[str, Any] | None:
    if tc is None:
        return None

    if isinstance(tc, dict):
        func = tc.get("function")
        if isinstance(func, dict) and isinstance(func.get("name"), str):
            arguments = _stringify_tool_arguments(func.get("arguments", "{}"))
            return {"function": {"name": func["name"], "arguments": arguments}}

        name = tc.get("name")
        if isinstance(name, str):
            arguments = _stringify_tool_arguments(tc.get("arguments"), none_default={})
            return {"function": {"name": name, "arguments": arguments or "{}"}}

        return None

    name = getattr(tc, "name", None)
    if isinstance(name, str):
        arguments = _stringify_tool_arguments(
            getattr(tc, "arguments", None), none_default={}
        )
        return {"function": {"name": name, "arguments": arguments or "{}"}}

    return None


def _tool_call_interface_error(tool_call: dict[str, Any]) -> str | None:
    func = tool_call.get("function")
    if not isinstance(func, dict):
        return "malformed"

    name = func.get("name")
    if not isinstance(name, str) or not name:
        return "invalid_name"
    args_schema = _TOOL_ARG_SCHEMAS.get(name)
    if args_schema is None:
        return "invalid_name"

    arguments = func.get("arguments")
    if not isinstance(arguments, str):
        return "invalid_args"
    try:
        payload = json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return "invalid_args"
    if not isinstance(payload, dict):
        return "invalid_args"

    if set(payload) - set(args_schema.model_fields):
        return "invalid_args"
    try:
        parsed = args_schema.model_validate(payload)
    except ValidationError:
        return "invalid_args"
    for arg_name in args_schema.model_fields:
        value = getattr(parsed, arg_name)
        if isinstance(value, str) and not value:
            return "invalid_args"
    return None


def _append_interface_checked_tool_call(
    tool_calls: list[dict[str, Any]],
    tool_call: dict[str, Any],
) -> tuple[int, int, int, InvalidToolCallCounts]:
    counts = _empty_invalid_tool_call_counts()
    error = _tool_call_interface_error(tool_call)
    if error != "malformed":
        tool_calls.append(tool_call)
    if error is None:
        return 0, 0, 0, counts
    if error == "malformed":
        _increment_invalid_tool_call_count(counts, "invalid_shape")
        return 1, 0, 0, counts
    if error == "invalid_name":
        _increment_invalid_tool_call_count(counts, "missing_name")
        return 0, 1, 0, counts
    _increment_invalid_tool_call_count(counts, "invalid_arguments")
    return 0, 0, 1, counts


def _analyze_assistant_turns(
    trace_history: list[dict[str, Any]],
    tool_call_parser: ToolCallParser = "hermes",
) -> list[AssistantTurnAnalysis]:
    assistant_turns: list[AssistantTurnAnalysis] = []
    for index, message in enumerate(trace_history):
        if message.get("role") != "assistant":
            continue

        tool_calls: list[dict[str, Any]] = []
        malformed_tool_call_count = 0
        invalid_tool_name_count = 0
        invalid_tool_args_count = 0
        invalid_tool_calls = _empty_invalid_tool_call_counts()
        feedback_invalid_tool_calls = (
            _invalid_tool_call_feedback_counts_for_assistant_turn(trace_history, index)
        )
        has_tool_call_surface = False
        raw_calls = message.get("tool_calls")
        if isinstance(raw_calls, list) and raw_calls:
            has_tool_call_surface = True
            for tc in raw_calls:
                normalized = _normalize_tool_call(tc)
                if normalized is None:
                    if isinstance(tc, dict):
                        invalid_tool_name_count += 1
                        _increment_invalid_tool_call_count(
                            invalid_tool_calls, "missing_name"
                        )
                    else:
                        malformed_tool_call_count += 1
                        _increment_invalid_tool_call_count(
                            invalid_tool_calls, "invalid_shape"
                        )
                    continue
                malformed, invalid_name, invalid_args, counts = (
                    _append_interface_checked_tool_call(tool_calls, normalized)
                )
                malformed_tool_call_count += malformed
                invalid_tool_name_count += invalid_name
                invalid_tool_args_count += invalid_args
                _add_invalid_tool_call_counts(invalid_tool_calls, counts)
        elif raw_calls not in (None, []):
            has_tool_call_surface = True
            malformed_tool_call_count += 1
            _increment_invalid_tool_call_count(invalid_tool_calls, "invalid_shape")

        content = message.get("content")
        if (
            (not isinstance(raw_calls, list) or not raw_calls)
            and isinstance(content, str)
            and "<tool_call>" in content
        ):
            has_tool_call_surface = True
            _, parsed_calls, parse_errors = parse_tool_calls_detailed(
                content, parser=tool_call_parser
            )
            if feedback_invalid_tool_calls["total"] == 0:
                for error in parse_errors:
                    if error.kind in {"invalid_json", "invalid_shape"}:
                        malformed_tool_call_count += 1
                    elif error.kind == "missing_name":
                        invalid_tool_name_count += 1
                    else:
                        invalid_tool_args_count += 1
                    _increment_invalid_tool_call_count(invalid_tool_calls, error.kind)
            for tool_call in parsed_calls:
                normalized = {
                    "function": {
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                    }
                }
                malformed, invalid_name, invalid_args, counts = (
                    _append_interface_checked_tool_call(tool_calls, normalized)
                )
                malformed_tool_call_count += malformed
                invalid_tool_name_count += invalid_name
                invalid_tool_args_count += invalid_args
                _add_invalid_tool_call_counts(invalid_tool_calls, counts)

        if feedback_invalid_tool_calls["total"] > 0:
            has_tool_call_surface = True
            malformed_tool_call_count += (
                feedback_invalid_tool_calls["invalid_json"]
                + feedback_invalid_tool_calls["invalid_shape"]
            )
            invalid_tool_name_count += feedback_invalid_tool_calls["missing_name"]
            invalid_tool_args_count += feedback_invalid_tool_calls["invalid_arguments"]
            _add_invalid_tool_call_counts(
                invalid_tool_calls, feedback_invalid_tool_calls
            )

        no_tool_call = not has_tool_call_surface
        assistant_turns.append(
            AssistantTurnAnalysis(
                tool_calls=tool_calls,
                no_tool_call=no_tool_call,
                malformed_tool_call_count=malformed_tool_call_count,
                invalid_tool_name_count=invalid_tool_name_count,
                invalid_tool_args_count=invalid_tool_args_count,
                invalid_tool_calls=invalid_tool_calls,
                missing_reasoning=_is_missing_reasoning(message),
            )
        )
    return assistant_turns


def _tool_call_to_string(tool_call: dict[str, Any]) -> str:
    """Convert a single tool call dict into a canonical string."""
    func = tool_call.get("function")
    if not isinstance(func, dict):
        return ""
    name = func.get("name", "")
    if not isinstance(name, str) or not name:
        return ""
    return f"{name}({_canonicalize_tool_arguments(name, func.get('arguments', '{}'))})"


def _canonicalize_tool_arguments(tool_name: str, arguments: Any) -> str:
    if not isinstance(arguments, str):
        try:
            arguments = json.dumps(arguments, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(arguments)

    try:
        payload = json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return arguments.strip()

    if tool_name == "exec_command" and isinstance(payload, dict):
        command = payload.get("command")
        if isinstance(command, str):
            payload = {**payload, "command": command.strip()}

    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        return arguments.strip()


def _count_repetitions(trajectory_actions: list[list[dict[str, Any]]]) -> int:
    """Count repeated individual tool calls across all turns."""
    seen: dict[str, int] = {}
    repetitions = 0
    for turn in trajectory_actions:
        for tool_call in turn:
            tool_call_str = _tool_call_to_string(tool_call)
            if not tool_call_str:
                continue
            if tool_call_str in seen:
                repetitions += 1
            seen[tool_call_str] = seen.get(tool_call_str, 0) + 1
    return repetitions


def _count_unique_recon_commands(tool_calls: list[list[dict[str, Any]]]) -> int:
    """Count unique recon patterns matched across all exec_command calls."""
    matched_patterns: set[str] = set()

    for turn in tool_calls:
        for tool_call in turn:
            func = tool_call.get("function", {})
            if func.get("name") != "exec_command":
                continue
            try:
                arguments = json.loads(func.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(arguments, dict):
                continue
            command = arguments.get("command", "")
            if not isinstance(command, str):
                continue
            for pattern in RECON_PATTERNS:
                if re.search(pattern, command, re.IGNORECASE):
                    matched_patterns.add(pattern)

    return len(matched_patterns)


def _invalid_tool_call_feedback_counts_for_assistant_turn(
    trace_history: list[dict[str, Any]], assistant_index: int
) -> InvalidToolCallCounts:
    counts = _empty_invalid_tool_call_counts()
    for message in trace_history[assistant_index + 1 :]:
        if message.get("role") != "tool":
            break
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, dict) and isinstance(
            payload.get("invalid_tool_call_kind"), str
        ):
            _increment_invalid_tool_call_count(
                counts, payload["invalid_tool_call_kind"]
            )
    return counts


def _calculate_assistant_messages_without_reasoning(
    trace_history: list[dict[str, Any]],
) -> int:
    """Count assistant messages with too-short content (< 50 chars)."""
    count = 0
    for message in trace_history:
        if message.get("role") != "assistant":
            continue
        text = text_content(message.get("content", ""))
        text = re.sub(r"</?think>", "", text).strip()
        if len(text) < 50:
            count += 1
    return count


def _count_tool_errors(trace_history: list[dict[str, Any]]) -> int:
    """Count tool messages that indicate an error."""
    count = 0
    for message in trace_history:
        if message.get("role") != "tool":
            continue
        text = text_content(message.get("content", ""))
        if text.startswith("Error:"):
            count += 1
            continue
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, dict) and "error" in payload:
            count += 1
    return count


def _empty_invalid_tool_call_counts() -> InvalidToolCallCounts:
    return {
        "total": 0,
        "invalid_json": 0,
        "missing_name": 0,
        "invalid_arguments": 0,
        "invalid_shape": 0,
    }


def _increment_invalid_tool_call_count(
    counts: InvalidToolCallCounts, kind: str
) -> None:
    counts["total"] += 1
    if kind in counts:
        counts[kind] += 1


def _add_invalid_tool_call_counts(
    counts: InvalidToolCallCounts, increment: InvalidToolCallCounts
) -> None:
    for key in counts:
        counts[key] += int(increment.get(key, 0))
