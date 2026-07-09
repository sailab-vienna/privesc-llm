import logging
from dataclasses import asdict, dataclass
from math import ceil
from typing import Any, Sequence

from langchain_core.messages import RemoveMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import MessagesState
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from transformers import AutoTokenizer

from src.utils.langchain import convert_to_trace_messages

log = logging.getLogger("privesc_agent")
logging.getLogger("transformers").setLevel(logging.ERROR)


TOKENIZER_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
_TOKENIZER: Any | None = None


def _get_tokenizer() -> Any:
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
    return _TOKENIZER


def get_num_tokens_from_messages(
    messages: Sequence[Any], tools: Sequence[Any] | None = None
) -> int:
    tokenizer = _get_tokenizer()
    chat_messages = convert_to_trace_messages(list(messages))
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": False,
        "return_dict": True,
        "return_tensors": "np",
    }
    if tools:
        kwargs["tools"] = [convert_to_openai_tool(tool) for tool in tools]
    encoded = tokenizer.apply_chat_template(chat_messages, **kwargs)
    return int(encoded["input_ids"].shape[-1])


@dataclass(frozen=True)
class ContextPolicy:
    enabled: bool = True
    max_len: int = 32768
    reserve_output_tokens: int = 1024
    reserve_ratio: float | None = None
    agent_max_tokens: int | None = None


@dataclass
class ContextStats:
    enabled: bool
    applied: bool = False
    trim_events: int = 0
    trimmed_messages_total: int = 0
    max_messages_removed_single_trim: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_policy(cls, policy: ContextPolicy) -> "ContextStats":
        return cls(enabled=policy.enabled)

    def apply_trim(self, removed_messages: int) -> None:
        self.applied = True
        if removed_messages > 0:
            self.trim_events += 1
            self.trimmed_messages_total += removed_messages
            self.max_messages_removed_single_trim = max(
                self.max_messages_removed_single_trim,
                removed_messages,
            )


def trim_to_budget(
    messages: Sequence[Any],
    tools: Sequence[Any],
    max_len: int,
    reserve_output_tokens: int,
) -> list[Any]:
    """Trim oldest messages to keep model input below context budget."""
    budget = max(1, max_len - reserve_output_tokens)

    def token_count(msgs: Sequence[Any]) -> int:
        return get_num_tokens_from_messages(msgs, tools=tools)

    all_messages = list(messages)
    total_tokens = token_count(all_messages)
    if total_tokens <= budget:
        return all_messages

    # Preserve system prompt and initial user message.
    preserved, trimmable = [], list(all_messages)
    if trimmable and getattr(trimmable[0], "type", None) == "system":
        preserved.append(trimmable.pop(0))
    if trimmable and getattr(trimmable[0], "type", None) in ("user", "human"):
        preserved.append(trimmable.pop(0))

    # Scan from newest: keep the longest recent suffix that fits.
    keep_from = len(trimmable)
    for i in reversed(range(len(trimmable))):
        if token_count(preserved + trimmable[i:]) <= budget:
            keep_from = i
        else:
            break

    trimmed = preserved + trimmable[keep_from:]
    log.debug(
        "Trimmed context: messages %d -> %d, tokens %d -> %d (budget=%d)",
        len(all_messages),
        len(trimmed),
        total_tokens,
        token_count(trimmed),
        budget,
    )
    return trimmed


def resolve_reserve_tokens(
    configured_reserve_ratio: float | None,
    max_len: int,
    agent_params: dict[str, Any] | None,
) -> int:
    if configured_reserve_ratio is not None and 0 < configured_reserve_ratio < 1:
        return max(1, ceil(max_len * configured_reserve_ratio))
    if agent_params is not None:
        max_tokens = agent_params.get("max_tokens")
        if isinstance(max_tokens, int) and max_tokens > 0:
            return max_tokens
    return 1024


def is_active(policy: ContextPolicy, runner_mode: str) -> bool:
    return policy.enabled and runner_mode == "evaluation"


def trim_context(state: MessagesState, config: RunnableConfig) -> dict[str, Any]:
    configurable = config.get("configurable")
    if configurable is None:
        raise ValueError("Missing configurable in RunnableConfig")
    policy: ContextPolicy = configurable["context_policy"]
    stats: ContextStats = configurable["context_stats"]
    runner_mode = configurable["runner_mode"]
    if not is_active(policy, runner_mode):
        stats.applied = False
        return {}

    tools = configurable["tools"]
    trimmed_messages = trim_to_budget(
        state["messages"],
        tools,
        max_len=policy.max_len,
        reserve_output_tokens=policy.reserve_output_tokens,
    )
    removed_messages = len(state["messages"]) - len(trimmed_messages)
    stats.apply_trim(removed_messages)
    if removed_messages <= 0:
        return {}
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *trimmed_messages]}
