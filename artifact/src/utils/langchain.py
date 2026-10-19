import logging
from typing import Any, ClassVar

import openai
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages.utils import convert_to_openai_messages
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


# ChatOpenAI params that should be passed explicitly (not via model_kwargs)
_CHATOPENAI_PARAMS = (
    "temperature",
    "max_tokens",
    "top_p",
    "reasoning",
    "reasoning_effort",
)

# Provider-specific params that need to go in extra_body
_EXTRA_BODY_PARAMS = ("top_k",)

log = logging.getLogger(__name__)


def build_llm_kwargs(params: dict[str, Any] | None) -> dict[str, Any]:
    """Extract ChatOpenAI kwargs from config params.

    Known params are passed explicitly.
    Provider-specific params (top_k) and explicit extra_body entries go to extra_body.
    Remaining params go to model_kwargs.
    """
    if not params:
        return {}
    params = dict(params)  # Copy to avoid mutation
    kwargs: dict[str, Any] = {}
    for key in _CHATOPENAI_PARAMS:
        if key in params:
            kwargs[key] = params.pop(key)

    extra_body = {}
    explicit_extra_body = params.pop("extra_body", None)
    if explicit_extra_body is not None:
        if not isinstance(explicit_extra_body, dict):
            raise TypeError("agent params extra_body must be a mapping")
        extra_body.update(explicit_extra_body)
    for key in _EXTRA_BODY_PARAMS:
        if key in params:
            extra_body[key] = params.pop(key)
    if extra_body:
        kwargs["extra_body"] = extra_body
    if params:
        kwargs["model_kwargs"] = params
    return kwargs


def _is_direct_deepseek(api_base: str, model: str) -> bool:
    base = api_base.lower()
    return "api.deepseek.com" in base and model.startswith("deepseek-")


def _is_openrouter_deepseek(api_base: str, model: str) -> bool:
    base = api_base.lower()
    return "openrouter.ai" in base and model.startswith("deepseek/")


def reasoning_content_from_message(message: Any) -> str:
    additional = getattr(message, "additional_kwargs", None)
    if not isinstance(additional, dict):
        return ""
    value = additional.get("reasoning_content")
    return value if isinstance(value, str) else ""


def _copy_reasoning_content_to_payload(
    payload_messages: Any,
    source_messages: list[BaseMessage],
    payload_key: str,
) -> None:
    if not isinstance(payload_messages, list):
        return
    for payload, source in zip(payload_messages, source_messages):
        if not isinstance(payload, dict) or not isinstance(source, AIMessage):
            continue
        reasoning_content = reasoning_content_from_message(source)
        if reasoning_content:
            payload[payload_key] = reasoning_content


def _reasoning_block(reasoning_content: str) -> dict[str, Any]:
    return {
        "type": "reasoning",
        "content": [
            {
                "type": "reasoning_text",
                "text": reasoning_content,
            }
        ],
    }


def _content_with_reasoning(message: AIMessage) -> Any:
    reasoning_content = reasoning_content_from_message(message)
    if not reasoning_content:
        return message.content

    blocks: list[dict[str, Any]] = [_reasoning_block(reasoning_content)]
    content = message.content
    if isinstance(content, str):
        if content:
            blocks.append({"type": "text", "text": content})
        return blocks
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") != "function_call":
                blocks.append(part)
    return blocks


def convert_to_trace_messages(messages: Any) -> list[dict[str, Any]]:
    converted = convert_to_openai_messages(messages)
    if not isinstance(messages, list) or not isinstance(converted, list):
        return converted if isinstance(converted, list) else []

    for item, source in zip(converted, messages):
        if not isinstance(item, dict) or not isinstance(source, AIMessage):
            continue
        if item.get("role") == "assistant":
            item["content"] = _content_with_reasoning(source)
    return converted


class ReasoningContentChatOpenAI(ChatOpenAI):
    reasoning_payload_key: ClassVar[str] = "reasoning_content"
    response_reasoning_key: ClassVar[str] = "reasoning_content"

    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        messages = self._convert_input(input_).to_messages()
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        _copy_reasoning_content_to_payload(
            payload.get("messages"),
            messages,
            self.reasoning_payload_key,
        )
        return payload

    def _create_chat_result(
        self,
        response: dict | openai.BaseModel,
        generation_info: dict | None = None,
    ) -> Any:
        response_dict = (
            response if isinstance(response, dict) else response.model_dump()
        )
        result = super()._create_chat_result(response, generation_info)
        choices = response_dict.get("choices")
        if not isinstance(choices, list):
            return result
        for generation, choice in zip(result.generations, choices):
            raw_message = choice.get("message") if isinstance(choice, dict) else None
            if not isinstance(raw_message, dict):
                continue
            reasoning_content = raw_message.get(self.response_reasoning_key)
            if isinstance(reasoning_content, str) and reasoning_content:
                generation.message.additional_kwargs["reasoning_content"] = (
                    reasoning_content
                )
        return result


class OpenRouterReasoningChatOpenAI(ReasoningContentChatOpenAI):
    reasoning_payload_key: ClassVar[str] = "reasoning"
    response_reasoning_key: ClassVar[str] = "reasoning"


def build_chat_openai(
    *,
    api_base: str,
    api_key: str,
    model: str,
    **kwargs: Any,
) -> ChatOpenAI:
    if _is_direct_deepseek(api_base, model):
        chat_cls = ReasoningContentChatOpenAI
    elif _is_openrouter_deepseek(api_base, model):
        chat_cls = OpenRouterReasoningChatOpenAI
    else:
        chat_cls = ChatOpenAI
    return chat_cls(
        base_url=api_base,  # type: ignore[call-arg]
        api_key=SecretStr(api_key),  # type: ignore[call-arg]
        model=model,  # type: ignore[call-arg]
        **kwargs,
    )
