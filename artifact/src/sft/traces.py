import json
import re
from pathlib import Path


def normalize_scenario_name(name: str) -> str:
    return re.sub(r"^\d{2}_", "", name)


def scenario_trace_paths(
    dataset_root: Path, leave_out_scenario: str | None
) -> list[Path]:
    if not dataset_root.exists():
        return []

    leave_out_norm = (
        normalize_scenario_name(leave_out_scenario) if leave_out_scenario else None
    )

    trace_paths: list[Path] = []
    for d in sorted(dataset_root.iterdir()):
        if not d.is_dir():
            continue
        if leave_out_norm and normalize_scenario_name(d.name) == leave_out_norm:
            continue
        p = d / "traces.jsonl"
        if p.exists():
            trace_paths.append(p)
    return trace_paths


def load_jsonl_conversations(trace_paths: list[Path]) -> list[dict]:
    conversations: list[dict] = []
    for p in trace_paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                if "messages" not in data:
                    raise ValueError(
                        f"{p} contains a row without 'messages'; assemble raw traces before training"
                    )
                conversations.append(data)
    return conversations
