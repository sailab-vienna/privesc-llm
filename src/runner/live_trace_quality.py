from dataclasses import dataclass
from typing import Any

from src.config import effective_max_assistant_turns
from src.dataset.privesc.collection_stats import holdout_leakage_rejection_reasons
from src.dataset.privesc.sft_preprocessing import (
    SFTPromptNormalizer,
    sft_token_count,
)
from src.dataset.privesc.sft_quality import TraceQualityFilter
from src.gym.context import TOKENIZER_MODEL
from src.utils.langchain import convert_to_trace_messages


def build_live_trace_quality(
    cfg: Any,
    tools: list[Any],
    trace_metadata: dict[str, Any],
) -> "LiveTraceQuality":
    quality_filter = (
        TraceQualityFilter(cfg.datasets.sft.quality)
        if cfg.runner.mode == "trace_collection"
        else None
    )
    return LiveTraceQuality(
        cfg=cfg,
        tools=tools,
        prompt_normalizer=SFTPromptNormalizer(cfg),
        trace_metadata=trace_metadata,
        quality_filter=quality_filter,
    )


def _total_tokens_so_far(usage_stats: dict[str, Any]) -> int:
    total = 0
    for entry in usage_stats.get("llm_usage_by_turn", []):
        if isinstance(entry, dict):
            total += int(entry.get("total_tokens", 0))
    return total


def _trace_prompt_vars(cfg: Any) -> dict[str, Any]:
    return {
        "user": cfg.scenario.container_user,
        "password": cfg.scenario.container_password,
        "max_turns": effective_max_assistant_turns(cfg),
        "term_cols": cfg.scenario.term_cols,
        "term_rows": cfg.scenario.term_rows,
    }


def _trace_metadata_with_prompt_vars(
    cfg: Any,
    trace_metadata: dict[str, Any],
) -> dict[str, Any]:
    metadata = dict(trace_metadata)
    metadata["prompt_vars"] = _trace_prompt_vars(cfg)
    return metadata


@dataclass
class LiveTraceQuality:
    cfg: Any
    tools: list[Any]
    prompt_normalizer: SFTPromptNormalizer
    trace_metadata: dict[str, Any]
    quality_filter: TraceQualityFilter | None

    @property
    def sft_tokenizer_model(self) -> str:
        return TOKENIZER_MODEL

    def sft_num_tokens(self, messages: list[Any]) -> int:
        trace_data = {
            "mode": self.cfg.runner.mode,
            "history": convert_to_trace_messages(messages),
            "metadata": _trace_metadata_with_prompt_vars(
                self.cfg, self.trace_metadata
            ),
        }
        sft_trace = self.prompt_normalizer.normalize_trace_prompts(trace_data)
        return sft_token_count(sft_trace, tools=self.tools)

    def result_fields(self, messages: list[Any]) -> dict[str, Any]:
        return {
            "sft_num_tokens": self.sft_num_tokens(messages),
            "sft_tokenizer_model": self.sft_tokenizer_model,
        }

    def payload(
        self,
        *,
        messages: list[Any],
        timing_stats: dict[str, Any],
        usage_stats: dict[str, Any],
        assistant_turns: int | None = None,
    ) -> dict[str, Any]:
        return {
            "scenario": self.cfg.scenario.name,
            "mode": self.cfg.runner.mode,
            "success": False,
            "turns": (
                int(assistant_turns)
                if assistant_turns is not None
                else int(timing_stats.get("assistant_turns", 0))
            ),
            "total_tokens": _total_tokens_so_far(usage_stats),
            "sft_num_tokens": self.sft_num_tokens(messages),
            "sft_tokenizer_model": TOKENIZER_MODEL,
            "history": convert_to_trace_messages(messages),
            "metadata": _trace_metadata_with_prompt_vars(
                self.cfg, self.trace_metadata
            ),
        }

    def rejection_reasons(
        self,
        *,
        messages: list[Any],
        timing_stats: dict[str, Any],
        usage_stats: dict[str, Any],
        assistant_turns: int | None = None,
    ) -> list[str]:
        if self.quality_filter is None:
            return []

        trace_data = self.payload(
            messages=messages,
            timing_stats=timing_stats,
            usage_stats=usage_stats,
            assistant_turns=assistant_turns,
        )
        passed, metrics = self.quality_filter.check_trace_live_prefix(trace_data)
        reasons = list(metrics.get("reasons", []))
        if self.cfg.datasets.sft.quality.reject_on_holdout_leakage:
            reasons.extend(holdout_leakage_rejection_reasons(trace_data))
        if passed and not reasons:
            return []
        return [str(reason) for reason in reasons]
