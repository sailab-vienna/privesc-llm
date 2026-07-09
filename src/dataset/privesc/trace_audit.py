from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Collection
from collections import Counter
from functools import lru_cache
from typing import Any

from src.dataset.privesc.trace_utils import text_content, trace_messages
from src.generators.holdout_manifest import (
    ACTION_HARD_HOLDOUT_CATEGORIES,
    HOLDOUT_HARD_CLASSES_BY_CATEGORY,
    SOFT_HOLDOUT_EXPLOIT_FRAGMENTS,
    SOFT_HOLDOUT_FILENAMES,
    benchmark_atoms,
    benchmark_credentials,
    benchmark_exploit_fragments,
    is_hard_holdout_credential_pair,
    is_hard_holdout_credential_pair_rule,
    matching_holdout_credential_pair_rules,
    matching_holdout_rules,
)

GENERIC_BENCHMARK_COMMAND_SKELETONS = frozenset(
    {"cat <path>", "cat admin_on_vacation.txt", "head -n <num> <path>"}
)
SPECIFIC_BENCHMARK_COMMAND_MARKERS = (
    " -exec ",
    "checkpoint-action",
    "chroot",
    "docker run",
    "expect -c",
    "nopasswd",
    "sshpass",
    "sudo ",
    "su root",
    "chmod u+s",
)
SOLUTION_METADATA_HOLDOUT_FIELDS = frozenset(
    {
        "artifact_path",
        "backup_dir",
        "binary_path",
        "exploit_cmds",
        "filename",
        "history_path",
        "history_template",
        "intended_password",
        "intended_reuse_user",
        "job_path",
        "key_name",
        "key_path",
        "password_file",
        "read_binary_path",
        "root_password",
        "script_path",
        "secret_filename",
        "secret_path",
        "secret_path_template",
        "source_image",
        "sudo_l_paths",
        "sudoers_command",
        "sudoers_glob",
        "sudoers_rule",
        "target_password",
    }
)
BINARY_METADATA_HOLDOUT_FIELDS = frozenset(
    {
        "binary_name",
        "binary_path",
        "exploit_cmds",
        "read_binary_path",
        "sudoers_command",
        "sudoers_glob",
    }
)


def audit_holdout_leakage(payload: dict[str, Any], generator: str) -> dict[str, Any]:
    solution_blob = _solution_audit_blob(payload)
    atom_matches, atom_counts, atom_counts_by_scenario, leakage_classes = (
        _benchmark_atom_audit(payload, solution_blob)
    )
    hard_classes, hard_leakage = _exact_holdout_surface_audit(
        payload, generator, solution_blob
    )
    leakage_classes.update(hard_leakage)

    credential_hard, credential_leakage = _credential_holdout_audit(payload)
    hard_classes.update(credential_hard)
    leakage_classes.update(credential_leakage)

    skeleton_overlaps = _skeleton_overlaps(payload)
    skeleton_counts = Counter(o["benchmark_skeleton_hash"] for o in skeleton_overlaps)
    return {
        "hard_rejection_classes": sorted(hard_classes),
        "leakage_classes": sorted(leakage_classes),
        "benchmark_atom_matches": atom_matches,
        "benchmark_atom_counts": dict(sorted(atom_counts.items())),
        "benchmark_atom_counts_by_scenario": dict(
            sorted(atom_counts_by_scenario.items())
        ),
        "command_skeleton_overlaps": skeleton_overlaps,
        "command_skeleton_overlap_counts": dict(sorted(skeleton_counts.items())),
    }


def command_skeleton(command: str) -> str:
    text = _clean(command).lower()
    text = re.sub(r"""(["']).*?\1""", "<str>", text)
    text = re.sub(r"/[^\s'\";|&]+", "<path>", text)
    text = re.sub(r"\b[0-9a-f]{8,}\b", "<token>", text)
    text = re.sub(r"\b\d+\b", "<num>", text)
    return _clean(text)


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_json(value: Any) -> str:
    return hash_text(json.dumps(value, sort_keys=True, default=str))


def clean_atom(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    atom = value.strip()
    if len(atom) < 4 or atom.lower() in {"root", "user", "sudo", "bash", "none"}:
        return None
    return atom


def _benchmark_atom_audit(
    payload: dict[str, Any], solution_blob: str
) -> tuple[list[dict[str, Any]], Counter[str], Counter[str], set[str]]:
    atom_matches = [
        *_benchmark_atom_matches(
            _audit_blob(payload), exclude_categories={"binary_names"}
        ),
        *_benchmark_atom_matches(solution_blob, categories={"binary_names"}),
    ]
    atom_counts = Counter[str]()
    atom_counts_by_scenario = Counter[str]()
    leakage_classes: set[str] = set()
    for match in atom_matches:
        category = str(match["category"])
        atom_counts[category] += int(match["count"])
        atom_counts_by_scenario[f"{match['scenario']}:{category}"] += int(
            match["count"]
        )
        leakage_classes.add(f"benchmark_{category}")
    return atom_matches, atom_counts, atom_counts_by_scenario, leakage_classes


def _exact_holdout_surface_audit(
    payload: dict[str, Any], generator: str, solution_blob: str
) -> tuple[set[str], set[str]]:
    hard_classes: set[str] = set()
    leakage_classes: set[str] = set()
    hard_matches = [
        *_benchmark_atom_matches(
            solution_blob,
            categories=set(HOLDOUT_HARD_CLASSES_BY_CATEGORY) - {"binary_names"},
            hard_surface=True,
        ),
        *_benchmark_atom_matches(
            _action_audit_blob(payload),
            categories=set(ACTION_HARD_HOLDOUT_CATEGORIES),
            hard_surface=True,
        ),
        *_benchmark_atom_matches(
            _binary_solution_audit_blob(payload),
            categories={"binary_names"},
            hard_surface=True,
        ),
    ]
    for match in hard_matches:
        category = str(match["category"])
        hard_class = HOLDOUT_HARD_CLASSES_BY_CATEGORY.get(category)
        if hard_class:
            hard_classes.add(hard_class)
            leakage_classes.add(f"benchmark_{category}")

    for rule in matching_holdout_rules(generator, solution_blob):
        leakage_classes.add(f"benchmark_{rule.category}")
        hard_class = HOLDOUT_HARD_CLASSES_BY_CATEGORY.get(rule.category)
        if hard_class:
            hard_classes.add(hard_class)
    return hard_classes, leakage_classes


def _credential_holdout_audit(payload: dict[str, Any]) -> tuple[set[str], set[str]]:
    hard_classes: set[str] = set()
    leakage_classes: set[str] = set()
    metadata = _metadata(payload)
    for user, password in _credential_pairs(payload):
        if f"{user}:{password}" in benchmark_credentials():
            leakage_classes.add("benchmark_credential_pair")
            if is_hard_holdout_credential_pair(user, password):
                hard_classes.add("exact_benchmark_credential")
    for user, password in _metadata_credential_pairs(metadata):
        if f"{user}:{password}" in benchmark_credentials():
            hard_classes.add("exact_benchmark_credential")
            leakage_classes.add("benchmark_credential_pair")
    for rule in matching_holdout_credential_pair_rules(_assistant_text_blob(payload)):
        leakage_classes.add("benchmark_credential_pair")
        if is_hard_holdout_credential_pair_rule(rule):
            hard_classes.add("exact_benchmark_credential")
    return hard_classes, leakage_classes


def _audit_blob(payload: dict[str, Any]) -> str:
    return json.dumps(
        {
            "metadata": _metadata(payload),
            "actions": _actions(payload),
            "observations": _tool_observations(payload),
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def _action_audit_blob(payload: dict[str, Any]) -> str:
    return json.dumps(
        {"actions": _actions(payload)}, ensure_ascii=False, sort_keys=True
    )


def _assistant_text_blob(payload: dict[str, Any]) -> str:
    return "\n".join(
        step["content"]
        for step in assistant_steps(trace_messages(payload))
        if step["content"]
    )


def _solution_audit_blob(payload: dict[str, Any]) -> str:
    return _metadata_blob(_metadata(payload), SOLUTION_METADATA_HOLDOUT_FIELDS)


def _binary_solution_audit_blob(payload: dict[str, Any]) -> str:
    return _metadata_blob(_metadata(payload), BINARY_METADATA_HOLDOUT_FIELDS)


def _metadata_blob(metadata: dict[str, Any], keys: Collection[str]) -> str:
    return json.dumps(
        {
            key: metadata[key]
            for key in keys
            if metadata.get(key) not in (None, "", [], {})
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def _tool_observations(payload: dict[str, Any]) -> list[dict[str, Any]]:
    fields = ("command", "output", "message", "user", "password")
    return [
        {key: result["payload"][key] for key in fields if key in result["payload"]}
        for result in tool_results(trace_messages(payload))
    ]


def _benchmark_atom_matches(
    blob: str,
    *,
    categories: set[str] | None = None,
    exclude_categories: set[str] | None = None,
    hard_surface: bool = False,
) -> list[dict[str, Any]]:
    matches = []
    for atom in benchmark_atoms():
        category = atom["category"]
        if categories is not None and category not in categories:
            continue
        if exclude_categories is not None and category in exclude_categories:
            continue
        if (
            hard_surface
            and category == "filenames"
            and atom["value"] in SOFT_HOLDOUT_FILENAMES
        ):
            continue
        if (
            hard_surface
            and category == "exploit_fragments"
            and atom["value"] in SOFT_HOLDOUT_EXPLOIT_FRAGMENTS
        ):
            continue
        count = _atom_count(blob, atom["value"], category)
        if count:
            matches.append(
                {
                    "scenario": atom["scenario"],
                    "category": category,
                    "atom_hash": atom["hash"],
                    "count": count,
                }
            )
    return matches


def _atom_count(blob: str, value: str, category: str) -> int:
    if category == "binary_names":
        return len(re.findall(rf"(?<![\w./-]){re.escape(value)}(?![\w./-])", blob))
    if category in {
        "service_names",
        "usernames",
        "password_strings",
        "filenames",
        "ssh_key_names",
    }:
        return len(re.findall(rf"(?<![\w./-]){re.escape(value)}(?![\w./-])", blob))
    return blob.count(value)


def _credential_pairs(payload: dict[str, Any]) -> set[tuple[str, str]]:
    pairs = set()
    for step in assistant_steps(trace_messages(payload)):
        pairs.update(step["credential_pairs"])
    for result in tool_results(trace_messages(payload)):
        result_payload = result["payload"]
        user, password = result_payload.get("user"), result_payload.get("password")
        if isinstance(user, str) and isinstance(password, str):
            pairs.add((user, password))
    return pairs


def _metadata_targets_root(metadata: dict[str, Any]) -> bool:
    return any(
        metadata.get(key) == "root"
        for key in (
            "target_user",
            "target_role",
            "intended_target_user",
            "intended_target_role",
            "intended_reuse_user",
            "intended_reuse_role",
        )
    )


def _metadata_credential_pairs(value: Any) -> set[tuple[str, str]]:
    pairs = set()
    if isinstance(value, dict):
        if value.get("root_password") == "root":
            pairs.add(("root", "root"))
        user = next(
            (
                str(value[key])
                for key in (
                    "user",
                    "username",
                    "target_user",
                    "intended_target_user",
                    "intended_reuse_user",
                )
                if isinstance(value.get(key), str)
            ),
            None,
        )
        password = next(
            (
                str(value[key])
                for key in (
                    "password",
                    "target_password",
                    "intended_password",
                    "root_password",
                )
                if isinstance(value.get(key), str)
            ),
            None,
        )
        if user is not None and password is not None:
            pairs.add((user, password))
        if password == "root" and _metadata_targets_root(value):
            pairs.add(("root", "root"))
        if value.get(
            "password_equals_target_username"
        ) is True and _metadata_targets_root(value):
            pairs.add(("root", "root"))
        for child in value.values():
            pairs.update(_metadata_credential_pairs(child))
    elif isinstance(value, list):
        for child in value:
            pairs.update(_metadata_credential_pairs(child))
    return pairs


def _skeleton_overlaps(payload: dict[str, Any]) -> list[dict[str, Any]]:
    overlaps = []
    for action in _actions(payload):
        skeleton = command_skeleton(action["text"])
        for benchmark in _benchmark_skeletons():
            if _skeletons_overlap(skeleton, benchmark["skeleton"]):
                overlaps.append(
                    {
                        "round": action["round"],
                        "surface": action["surface"],
                        "command_hash": hash_text(action["text"]),
                        "command_skeleton_hash": hash_text(skeleton),
                        "benchmark_scenario": benchmark["scenario"],
                        "benchmark_fragment_hash": benchmark["fragment_hash"],
                        "benchmark_skeleton_hash": benchmark["skeleton_hash"],
                    }
                )
    return overlaps


@lru_cache(maxsize=1)
def _benchmark_skeletons() -> tuple[dict[str, str], ...]:
    rows = []
    seen = set()
    for scenario, fragment in benchmark_exploit_fragments():
        skeleton = command_skeleton(fragment)
        if (
            not _is_specific_benchmark_skeleton(skeleton)
            or (scenario, skeleton) in seen
        ):
            continue
        seen.add((scenario, skeleton))
        rows.append(
            {
                "scenario": scenario,
                "skeleton": skeleton,
                "fragment_hash": hash_text(fragment),
                "skeleton_hash": hash_text(skeleton),
            }
        )
    return tuple(rows)


def _is_specific_benchmark_skeleton(skeleton: str) -> bool:
    if skeleton in GENERIC_BENCHMARK_COMMAND_SKELETONS:
        return False
    return any(marker in skeleton for marker in SPECIFIC_BENCHMARK_COMMAND_MARKERS)


def _actions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {"round": step["round"], "surface": "assistant_action", "text": action}
        for step in assistant_steps(trace_messages(payload))
        for action in step["actions"]
    ]
    rows.extend(
        {"round": result["round"], "surface": "tool_payload", "text": signature}
        for result in tool_results(trace_messages(payload))
        for signature in [tool_result_signature(result)]
        if signature
    )
    return rows


def _skeletons_overlap(left: str, right: str) -> bool:
    if len(left.split()) < 2 or len(right.split()) < 2:
        return False
    if left == right:
        return True
    return min(len(left.split()), len(right.split())) >= 4 and (
        left in right or right in left
    )


def assistant_steps(messages: list[Any]) -> list[dict[str, Any]]:
    assistant_round = 0
    steps: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        assistant_round += 1
        actions = [_tool_call_signature(call) for call in _tool_calls(message)]
        steps.append(
            {
                "round": assistant_round,
                "content": text_content(message.get("content", "")),
                "actions": [action for action in actions if action],
                "credential_pairs": _tool_call_credential_pairs(message),
            }
        )
    return steps


def tool_results(messages: list[Any]) -> list[dict[str, Any]]:
    call_rounds: dict[str, int] = {}
    assistant_round = 0
    results: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            assistant_round += 1
            for call in _tool_calls(message):
                call_id = str(call.get("id", ""))
                if call_id:
                    call_rounds[call_id] = assistant_round
        elif message.get("role") == "tool":
            call_id = str(message.get("tool_call_id", ""))
            results.append(
                {
                    "round": call_rounds.get(call_id, assistant_round),
                    "payload": json_obj(message.get("content")),
                }
            )
    return results


def _tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    calls = message.get("tool_calls")
    return (
        [call for call in calls if isinstance(call, dict)]
        if isinstance(calls, list)
        else []
    )


def _tool_call_signature(call: dict[str, Any]) -> str | None:
    name, args = _tool_call_name_args(call)
    if name == "exec_command" and isinstance(args.get("command"), str):
        return _clean(args["command"])
    if name == "test_credentials":
        user, password = args.get("user"), args.get("password")
        if isinstance(user, str) and isinstance(password, str):
            return f"test_credentials:{user}:{password}"
    return None


def tool_result_signature(result: dict[str, Any]) -> str | None:
    payload = result["payload"]
    command = payload.get("command")
    if isinstance(command, str) and command.strip():
        return _clean(command)
    user, password = payload.get("user"), payload.get("password")
    if isinstance(user, str) and isinstance(password, str):
        return f"test_credentials:{user}:{password}"
    return None


def _tool_call_name_args(call: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    function = call.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        raw_args = function.get("arguments", function.get("args"))
    else:
        name = call.get("name")
        raw_args = call.get("arguments", call.get("args"))
    return (name if isinstance(name, str) else None), json_obj(raw_args)


def _tool_call_credential_pairs(message: dict[str, Any]) -> set[tuple[str, str]]:
    pairs = set()
    for call in _tool_calls(message):
        name, args = _tool_call_name_args(call)
        user, password = args.get("user"), args.get("password")
        if (
            name == "test_credentials"
            and isinstance(user, str)
            and isinstance(password, str)
        ):
            pairs.add((user, password))
    return pairs


def json_obj(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _metadata(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _clean(text: str) -> str:
    return " ".join(text.split())
