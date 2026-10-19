"""Schedule building and result I/O for runner."""

import json
import os
import random
from collections.abc import Iterator
from dataclasses import asdict
from datetime import datetime
import tempfile
from typing import Any, cast

from omegaconf import OmegaConf, SCMode

from src.config import AppConfig, ScenarioConfig, resolve_reward_config
from src.dataset.privesc.collection_stats import SFTAssemblyTargetChecker
from src.paths import EVALUATION_BASE, TRACE_COLLECTION_BASE, traces_dir
from src.runner.result_policy import (
    is_benchmark_eligible_payload,
    is_trace_collection_resume_payload,
)
from src.scenarios import (
    ScenarioInstance,
    ScenarioSource,
    ProceduralScenarioSource,
    StaticScenarioSource,
)
from src.tui import console


def output_dir(cfg: AppConfig) -> str:
    base = cfg.runner.output_dir or (
        TRACE_COLLECTION_BASE
        if cfg.runner.mode == "trace_collection"
        else EVALUATION_BASE
    )
    return traces_dir(base, cfg.agent.model)


def count_existing(output_dir: str, prefix: str) -> int:
    if not os.path.isdir(output_dir):
        return 0
    count = 0
    for _filename, payload in _iter_result_payloads(output_dir, prefix=prefix):
        if is_benchmark_eligible_payload(payload):
            count += 1
    return count


def save_result(
    cfg: AppConfig,
    payload: dict[str, Any],
    filename_prefix: str | None = None,
) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    d = output_dir(cfg)
    os.makedirs(d, exist_ok=True)
    stem = filename_prefix or str(payload["scenario"])
    # Use atomic creation so concurrent workers cannot silently overwrite traces.
    fd, path = tempfile.mkstemp(prefix=f"{stem}_{ts}_", suffix=".json", dir=d)
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def update_cfg_with_instance(cfg: AppConfig, instance: ScenarioInstance) -> Any:
    reward_dict = cast(
        dict[str, Any] | None,
        OmegaConf.to_container(
            cfg.reward, resolve=True, structured_config_mode=SCMode.DICT
        ),
    )
    cfg_dict = cast(
        dict[str, Any],
        OmegaConf.to_container(cfg, resolve=False, structured_config_mode=SCMode.DICT),
    )
    cfg_dict.pop("experiment", None)
    cfg_dict.pop("reward", None)

    scenario_dict = asdict(instance.config)
    configured_backend = getattr(cfg.scenario, "backend", "")
    if configured_backend:
        configured_backend = str(configured_backend).strip()
        if configured_backend and configured_backend != ScenarioConfig().backend:
            scenario_dict["backend"] = configured_backend
    backend_override = os.getenv("PRIVESC_SCENARIO_BACKEND", "").strip()
    if backend_override:
        scenario_dict["backend"] = backend_override
    if instance.solution:
        scenario_dict["solution"] = instance.solution
    cfg_dict["scenario"] = scenario_dict

    merged = OmegaConf.merge(OmegaConf.structured(AppConfig()), cfg_dict)
    if reward_dict is not None:
        merged.reward = resolve_reward_config(reward_dict)
    return merged


def _trace_slot(metadata: dict[str, Any]) -> tuple[str, int, int] | None:
    generator_name = metadata.get("generator_name")
    ordinal = metadata.get("item_run_ordinal")
    seed = metadata.get("seed")
    if not isinstance(generator_name, str) or not generator_name:
        return None
    if not isinstance(ordinal, int) or not isinstance(seed, int):
        return None
    return generator_name, ordinal, seed


def _metadata_sample(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metadata.items()
        if key
        in {
            "category",
            "binary_name",
            "binary_path",
            "grant_subject",
            "sudoers_file",
            "backup_dir",
            "backup_dir_name",
            "job_name",
            "job_path",
            "script_name",
            "script_path",
            "script_mode",
            "artifact_mode",
            "template_kind",
            "template_filename",
            "filename",
            "placement_location",
            "history_filename",
            "history_template",
            "key_type",
            "key_name",
            "ssh_dir",
            "ssh_dir_kind",
            "ssh_target",
            "reuse_pattern",
            "weak_password_pattern",
            "password_source",
            "wait_seconds",
            "exploit_script_name",
        }
    }


def _iter_result_payloads(out_dir: str, prefix: str | None = None):
    for filename in os.listdir(out_dir):
        if not filename.endswith(".json") or (prefix and not filename.startswith(prefix)):
            continue

        path = os.path.join(out_dir, filename)
        try:
            with open(path, "r") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        yield filename, payload


def _iter_matching_result_payloads(
    out_dir: str,
    items: list[str],
) -> Iterator[tuple[str, str, dict[str, Any]]]:
    item_prefixes = [(f"{item}_", item) for item in items]
    for filename, payload in _iter_result_payloads(out_dir):
        file_item = next(
            (item for prefix, item in item_prefixes if filename.startswith(prefix)), None
        )
        if file_item is None:
            continue
        yield file_item, filename, payload


def _count_by_existing_trace_keys(
    cfg: AppConfig,
    out_dir: str,
    items: list[str],
) -> tuple[dict[str, int], set[tuple[str, int]]]:
    counts: dict[str, int] = {item: 0 for item in items}
    keys: set[tuple[str, int]] = set()
    attempts: dict[tuple[str, int], list[dict[str, Any]]] = {}
    if not os.path.isdir(out_dir):
        return counts, keys

    checker = SFTAssemblyTargetChecker(cfg)
    runs_per_item = int(cfg.runner.runs_per_item)
    for _file_item, filename, payload in _iter_matching_result_payloads(out_dir, items):
        if not is_trace_collection_resume_payload(payload):
            continue

        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            continue
        slot = _trace_slot(metadata)
        if slot is None:
            continue
        generator_name, ordinal, seed = slot
        if generator_name not in counts or ordinal < 0 or ordinal >= runs_per_item:
            continue

        attempt_key = (generator_name, ordinal)
        attempts.setdefault(attempt_key, []).append(
            {
                "file": filename,
                "seed": seed,
                "success": payload.get("success"),
                "status": payload.get("status"),
                "error": payload.get("error"),
                "failure_reason": payload.get("failure_reason"),
                "metadata": _metadata_sample(metadata),
            }
        )
        if not isinstance(payload.get("history"), list):
            continue

        if checker.counts(payload):
            if attempt_key not in keys:
                keys.add(attempt_key)
                counts[generator_name] += 1

    retry_limit = int(getattr(cfg.runner, "max_retries_per_run", 1)) + 1
    exhausted = [
        (key, values)
        for key, values in attempts.items()
        if key not in keys and len(values) >= retry_limit
    ]
    if exhausted:
        details = []
        for (generator_name, ordinal), values in sorted(exhausted)[:5]:
            seed = values[-1].get("seed")
            details.append(
                f"{generator_name}[ordinal={ordinal}, seed={seed}, attempts={len(values)}, "
                f"metadata={values[-1].get('metadata')}]"
            )
        raise RuntimeError(
            "Trace collection exhausted retry budget without an SFT-usable trace for "
            + "; ".join(details)
        )

    return counts, keys


def _count_by_existing_static_ordinals(
    out_dir: str,
    items: list[str],
    runs_per_item: int,
) -> tuple[dict[str, int], set[tuple[str, int]]]:
    counts: dict[str, int] = {item: 0 for item in items}
    fallback_counts: dict[str, int] = {item: 0 for item in items}
    keys: set[tuple[str, int]] = set()
    if not os.path.isdir(out_dir):
        return counts, keys

    for file_item, _filename, payload in _iter_matching_result_payloads(out_dir, items):
        if not is_benchmark_eligible_payload(payload):
            continue

        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            fallback_counts[file_item] += 1
            continue

        ordinal = metadata.get("item_run_ordinal")
        if ordinal is None or not isinstance(ordinal, int):
            fallback_counts[file_item] += 1
            continue
        if ordinal < 0 or ordinal >= runs_per_item:
            continue

        key = (file_item, ordinal)
        if key in keys:
            continue
        keys.add(key)
        counts[file_item] += 1

    keyed_items = {item for item, _ordinal in keys}
    combined_counts = {
        item: counts[item] if item in keyed_items else fallback_counts[item]
        for item in items
    }
    return combined_counts, keys


def _procedural_seed(source_cfg: Any, run_index: int) -> int:
    base_seed = int(source_cfg.seed)
    if bool(source_cfg.random_seed):
        rng = random.Random(base_seed + run_index)
        return int(rng.randrange(2**31))
    return base_seed + run_index


def _use_procedural_seed_dedupe(cfg: AppConfig, source_cfg: Any) -> bool:
    return source_cfg.type == "procedural" and cfg.runner.mode == "trace_collection"


def _existing_counts_and_keys(
    cfg: AppConfig,
    source_cfg: Any,
    out_dir: str,
    items: list[str],
) -> tuple[list[int], set[tuple[str, int]]]:
    if source_cfg.type == "static":
        static_counts, existing_keys = _count_by_existing_static_ordinals(
            out_dir, items, int(cfg.runner.runs_per_item)
        )
        return [static_counts[item] for item in items], existing_keys

    if not _use_procedural_seed_dedupe(cfg, source_cfg):
        return [count_existing(out_dir, f"{item}_") for item in items], set()

    proc_counts, existing_keys = _count_by_existing_trace_keys(cfg, out_dir, items)
    return [proc_counts[item] for item in items], existing_keys


def _next_run_index_for_item(
    *,
    idx: int,
    item: str,
    num_items: int,
    next_run_nums: list[int],
    use_key_dedupe: bool,
    existing_keys: set[tuple[str, int]],
    planned_keys: set[tuple[str, int]],
) -> int:
    if not use_key_dedupe:
        run_index = next_run_nums[idx] * num_items + idx
        next_run_nums[idx] += 1
        return run_index

    while True:
        run_num = next_run_nums[idx]
        run_index = run_num * num_items + idx
        next_run_nums[idx] += 1
        key = (item, run_num)
        if key in existing_keys or key in planned_keys:
            continue
        planned_keys.add(key)
        return run_index


def build_schedule(
    cfg: AppConfig,
    source: ScenarioSource,
) -> tuple[list[int], str]:
    """Build schedule of run indices based on source type and existing results."""
    source_cfg = cfg.runner.source
    out_dir = output_dir(cfg)
    max_runs = cfg.runner.max_runs
    runs_per_item = cfg.runner.runs_per_item

    if source_cfg.type == "static":
        static_source = cast(StaticScenarioSource, source)
        items = static_source.scenarios
        item_type = "scenario"
    else:
        proc_source = cast(ProceduralScenarioSource, source)
        items = proc_source.generators
        item_type = "generator"

    num_items = len(items)

    target_total = runs_per_item * num_items
    if max_runs is not None:
        if max_runs < num_items:
            console.print(
                f"[yellow]Warning: max_runs ({max_runs}) < {item_type}s ({num_items}), "
                "cap will truncate the schedule[/yellow]"
            )
        target_total = min(target_total, max_runs)

    existing_counts, existing_keys = _existing_counts_and_keys(
        cfg, source_cfg, out_dir, items
    )
    keyed_items = {item for item, _ordinal in existing_keys}
    use_procedural_key_dedupe = _use_procedural_seed_dedupe(cfg, source_cfg)

    needed_counts = [max(0, runs_per_item - existing) for existing in existing_counts]
    next_run_nums = [
        0 if use_procedural_key_dedupe or item in keyed_items else existing_counts[idx]
        for idx, item in enumerate(items)
    ]
    schedule: list[int] = []
    planned_keys: set[tuple[str, int]] = set()

    if any(needed_counts):
        while len(schedule) < target_total and any(needed_counts):
            for idx in range(num_items):
                if len(schedule) >= target_total:
                    break
                if needed_counts[idx] <= 0:
                    continue

                run_index = _next_run_index_for_item(
                    idx=idx,
                    item=items[idx],
                    num_items=num_items,
                    next_run_nums=next_run_nums,
                    use_key_dedupe=(
                        use_procedural_key_dedupe or items[idx] in keyed_items
                    ),
                    existing_keys=existing_keys,
                    planned_keys=planned_keys,
                )
                schedule.append(run_index)

                needed_counts[idx] -= 1

    if not schedule:
        return [], f"All {num_items} {item_type}s have {runs_per_item} runs"

    source_label = "Static" if source_cfg.type == "static" else "Procedural"
    cap_desc = f" · cap {max_runs}" if max_runs is not None else ""
    desc = (
        f"{source_label} · {num_items} {item_type}s · "
        f"{runs_per_item} each{cap_desc} · {len(schedule)} runs"
    )
    return schedule, desc
