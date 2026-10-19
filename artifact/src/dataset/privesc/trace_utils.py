import json
from typing import Any


def trace_messages(trace_data: dict[str, Any]) -> list[dict[str, Any]]:
    history = trace_data.get("history")
    if isinstance(history, list):
        return history
    raise ValueError("Trace payload must contain a list-valued 'history' field")


def text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type in {"text", "output_text", "reasoning_text"}:
                text = part.get("text")
                if isinstance(text, str) and text:
                    chunks.append(text)
            elif part_type == "reasoning":
                text = text_content(part.get("content"))
                if text:
                    chunks.append(text)
        return "\n\n".join(chunk.strip() for chunk in chunks if chunk.strip())
    return ""


def sft_assistant_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    reasoning_chunks: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type == "reasoning":
            text = text_content(part.get("content"))
        elif part_type == "reasoning_text":
            text = part.get("text") if isinstance(part.get("text"), str) else ""
        else:
            continue
        if text:
            reasoning_chunks.append(text)

    if reasoning_chunks:
        return "\n\n".join(chunk.strip() for chunk in reasoning_chunks if chunk.strip())
    return text_content(content)


def stringify_content(content: Any) -> str:
    text = text_content(content).strip()
    if text:
        return text
    if isinstance(content, (dict, list)):
        return json.dumps(content, indent=2, ensure_ascii=False)
    return str(content).strip()
