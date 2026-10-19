from typing import Any, Mapping


def trace_metadata(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = payload.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def trace_scenario_name(payload: Mapping[str, Any]) -> str | None:
    metadata = trace_metadata(payload)
    value = payload.get("scenario")
    if payload.get("status") == "error":
        source_scenario = metadata.get("source_scenario")
        if isinstance(source_scenario, str) and source_scenario:
            value = source_scenario
    return value if isinstance(value, str) and value else None


def trace_generator_name(payload: Mapping[str, Any]) -> str | None:
    value = trace_metadata(payload).get("generator_name") or payload.get("scenario")
    return value if isinstance(value, str) and value else None


def trace_item_run_ordinal(payload: Mapping[str, Any]) -> int | None:
    value = trace_metadata(payload).get("item_run_ordinal")
    return value if isinstance(value, int) else None
