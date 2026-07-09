from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from rich import get_console
from rich.table import Table

from src.config import SFTQualityFilterConfig
from src.dataset.privesc.sft_preprocessing import (
    SFTPromptNormalizer,
    record_sft_token_count,
)
from src.dataset.privesc.sft_quality import TraceQualityFilter
from src.dataset.privesc.trace_utils import text_content, trace_messages
from src.dataset.privesc.trace_audit import (
    assistant_steps as trace_assistant_steps,
    audit_holdout_leakage,
    clean_atom,
    command_skeleton,
    hash_json,
    hash_text,
    json_obj as trace_json_obj,
    tool_result_signature as trace_tool_result_signature,
    tool_results as trace_tool_results,
)

console = get_console()

SOLUTION_SIGNATURE_KEYS = (
    "generator_name",
    "category",
    "binary_name",
    "binary_path",
    "required_caps",
    "setcap_spec",
    "read_binary_path",
    "sudoers_glob",
    "sudoers_command",
    "secret_path_template",
    "secret_filename",
    "service_name",
    "job_path",
    "backup_dir",
    "script_path",
    "artifact_mode",
    "artifact_path",
    "history_template",
    "key_path",
    "key_type",
    "ssh_target",
    "group_name",
    "source_image",
    "reuse_pattern",
    "intended_reuse_role",
    "intended_reuse_user",
    "root_reuses_user_password",
    "password_equals_target_username",
    "exploit_script_name",
    "wait_seconds",
)
SOLUTION_ATOM_KEYS = (
    "binary_path",
    "binary_name",
    "sudoers_command",
    "sudoers_glob",
    "secret_path",
    "secret_filename",
    "key_path",
    "key_name",
    "backup_dir",
    "job_path",
    "script_path",
    "artifact_path",
    "history_path",
    "group_name",
    "container_name",
    "local_image",
    "intended_reuse_user",
    "filename",
    "password_file",
    "root_password",
    "target_password",
    "intended_password",
    "sudoers_rule",
    "read_binary_path",
    "source_image",
)
SOLUTION_ATOM_LIST_KEYS = ("sudo_l_paths", "log_paths", "exploit_cmds")
EVIDENCE_KEYS = (
    "binary_path",
    "binary_name",
    "sudoers_command",
    "sudoers_glob",
    "secret_path",
    "secret_filename",
    "key_path",
    "key_name",
    "backup_dir",
    "job_path",
    "script_path",
    "artifact_path",
    "history_path",
    "group_name",
    "container_name",
    "local_image",
    "intended_reuse_user",
)
LIST_EVIDENCE_KEYS = ("sudo_l_paths", "log_paths")
PATH_EVIDENCE_KEYS = {
    "artifact_path",
    "backup_dir",
    "binary_path",
    "job_path",
    "key_path",
    "password_file",
    "script_path",
    "secret_path",
}
ORACLE_HARD_ATOM_KEYS = {
    "exploit_cmds",
    "intended_password",
    "root_password",
    "target_password",
}
INVALID_COMMAND_MARKERS = (
    "command not found",
    "executable file not found",
    "not found in $path",
    "unknown option",
    "invalid option",
)
MIN_REASONING_LENGTH = 60
GENERATOR_FAMILIES = {
    "password_file": "credential",
    "password_history": "credential",
    "password_reuse": "credential",
    "weak_password": "credential",
    "sudo_gtfobins": "sudo",
    "suid_gtfobins": "suid",
    "capabilities_gtfobins": "suid",
    "cron_wildcard": "cron",
    "cron_writable_script": "cron",
    "ssh_key_reuse": "ssh",
}
COUNT_FIELDS = (
    "raw",
    "usable",
    "sft_usable",
    "sft_rejected",
    "benchmark_eligible",
    "teacher_solved",
    "teacher_failed",
    "errors",
    "audit_rejected",
    "live_rejected",
    "leakage_rejected",
    "holdout_leakage_rejected",
    "other_live_rejected",
    "repeated_command_traces",
    "invalid_command_traces",
    "near_empty_reasoning_tool_call_turns",
    "assistant_turns_without_tools",
    "post_root_continuation_turns",
    "near_empty_reasoning_tool_call_traces",
    "assistant_turns_without_tools_traces",
    "post_root_continuation_traces",
    "credential_attempt_traces",
    "parallel_credential_batch_traces",
)
RATE_FIELDS = {
    "teacher_solve_rate": "teacher_solved",
    "usable_rate": "usable",
    "sft_usable_rate": "sft_usable",
    "sft_rejection_rate": "sft_rejected",
    "benchmark_eligible_rate": "benchmark_eligible",
    "audit_rejection_rate": "audit_rejected",
    "live_rejection_rate": "live_rejected",
    "leakage_rejection_rate": "leakage_rejected",
    "holdout_leakage_rejection_rate": "holdout_leakage_rejected",
    "other_live_rejection_rate": "other_live_rejected",
    "repeated_command_trace_rate": "repeated_command_traces",
    "invalid_command_trace_rate": "invalid_command_traces",
    "near_empty_reasoning_tool_call_trace_rate": "near_empty_reasoning_tool_call_traces",
    "assistant_turns_without_tools_trace_rate": "assistant_turns_without_tools_traces",
    "post_root_continuation_trace_rate": "post_root_continuation_traces",
    "credential_attempt_trace_rate": "credential_attempt_traces",
    "parallel_credential_batch_trace_rate": "parallel_credential_batch_traces",
}
MEAN_FIELDS = {
    "avg_turns": "turns",
    "avg_tool_calls": "tool_calls",
    "avg_credential_attempts": "credential_attempts",
}
MEDIAN_FIELDS = {
    "median_evidence_round": "evidence_rounds",
    "median_exploit_round": "exploit_rounds",
    "median_root_round": "root_rounds",
}
NULLABLE_MEAN_FIELDS = {
    "avg_evidence_to_exploit_latency": "evidence_to_exploit_latencies",
    "avg_exploit_to_root_latency": "exploit_to_root_latencies",
    "avg_evidence_to_root_latency": "evidence_to_root_latencies",
}
LIST_FIELDS = (
    *MEAN_FIELDS.values(),
    *MEDIAN_FIELDS.values(),
    *NULLABLE_MEAN_FIELDS.values(),
)
SET_FIELDS = (
    "task_ids",
    "seeds",
    "solution_signature_hashes",
)
COUNTER_FIELDS = (
    "task_trace_counts",
    "leakage_class_counts",
    "structured_rejection_classes",
    "family_behavior_counts",
)
METADATA_HISTOGRAM_FIELDS = (
    "generator_name",
    "category",
    "binary_name",
    "binary_path",
    "grant_subject",
    "sudoers_file",
    "decoys_enabled",
    "decoy_count",
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
    "raw_template_filename",
    "filename",
    "placement_location",
    "history_filename",
    "history_prefix_noise_count",
    "history_suffix_noise_count",
    "history_template",
    "key_type",
    "keygen_args",
    "key_name",
    "ssh_dir",
    "ssh_dir_kind",
    "ssh_target",
    "reuse_pattern",
    "weak_password_pattern",
    "password_source",
    "intended_target_role",
    "intended_target_user",
    "intended_reuse_role",
    "intended_reuse_user",
    "root_reuses_user_password",
    "wait_seconds",
    "exploit_script_name",
)
METADATA_HISTOGRAM_BUCKETS = ("raw", "teacher_solved", "sft_usable")


def _new_bucket() -> dict[str, Any]:
    bucket: dict[str, Any] = {key: 0 for key in COUNT_FIELDS}
    bucket.update({key: [] for key in LIST_FIELDS})
    bucket.update({key: set() for key in SET_FIELDS})
    bucket.update({key: Counter() for key in COUNTER_FIELDS})
    bucket["metadata_histograms"] = {
        name: {field: Counter() for field in METADATA_HISTOGRAM_FIELDS}
        for name in METADATA_HISTOGRAM_BUCKETS
    }
    return bucket


class SFTAssemblyTargetChecker:
    def __init__(self, cfg: Any):
        quality_dict = OmegaConf.to_container(cfg.datasets.sft.quality, resolve=True)
        if not isinstance(quality_dict, dict):
            raise ValueError("datasets.sft.quality must be a mapping")
        self.quality_filter = TraceQualityFilter(
            SFTQualityFilterConfig(**{str(k): v for k, v in quality_dict.items()})
        )
        self.cfg = cfg
        self.prompt_normalizer: SFTPromptNormalizer | None = None

    def check(self, payload: dict[str, Any]) -> dict[str, Any]:
        messages = trace_messages(payload)
        if not messages:
            return {"sft_usable": False, "sft_rejection_reasons": ["missing_history"]}
        if self.prompt_normalizer is None:
            self.prompt_normalizer = SFTPromptNormalizer(self.cfg)
        trace_data = self.prompt_normalizer.normalize_trace_prompts(payload)
        record_sft_token_count(trace_data)
        passed, metrics = self.quality_filter.check_trace(trace_data)
        return {
            "sft_usable": passed,
            "sft_rejection_reasons": list(metrics["reasons"]),
        }

    def counts(self, payload: dict[str, Any]) -> bool:
        audit = audit_trace_payload(payload)
        holdout_ok = (
            audit["audit_usable"]
            or not self.quality_filter.config.reject_on_holdout_leakage
        )
        return bool(holdout_ok and self.check(payload)["sft_usable"])


def _record_bucket(
    bucket: dict[str, Any],
    payload: dict[str, Any],
    metrics: dict[str, Any],
    audit: dict[str, Any],
    sft_quality: dict[str, Any],
    rejection_reasons: Counter[str] | None = None,
) -> None:
    bucket["raw"] += 1
    if payload.get("success") is True:
        bucket["teacher_solved"] += 1
    elif payload.get("status") == "error" or payload.get("error"):
        bucket["errors"] += 1
    else:
        bucket["teacher_failed"] += 1

    if audit["base_usable"] and not audit["hard_rejection_classes"]:
        bucket["usable"] += 1
    elif audit["base_usable"]:
        bucket["audit_rejected"] += 1
        if audit["leakage_classes"]:
            bucket["leakage_rejected"] += 1
            bucket["holdout_leakage_rejected"] += 1

    if sft_quality["sft_usable"]:
        bucket["sft_usable"] += 1
    elif audit["audit_usable"]:
        bucket["sft_rejected"] += 1
        if rejection_reasons is not None:
            rejection_reasons.update(sft_quality["sft_rejection_reasons"])

    metadata = _metadata(payload)
    _record_metadata_histograms(bucket, metadata, payload, sft_quality)
    bucket["benchmark_eligible"] += int(metadata.get("benchmark_eligible") is True)
    _record_metric_lists(bucket, metrics)
    bucket["repeated_command_traces"] += int(metrics["has_repeated_command"])
    bucket["invalid_command_traces"] += int(metrics["has_invalid_command"])
    bucket["near_empty_reasoning_tool_call_turns"] += int(
        metrics["near_empty_reasoning_tool_call_turns"]
    )
    bucket["assistant_turns_without_tools"] += int(
        metrics["assistant_turns_without_tools"]
    )
    bucket["post_root_continuation_turns"] += int(
        metrics["post_root_continuation_turns"]
    )
    bucket["near_empty_reasoning_tool_call_traces"] += int(
        metrics["near_empty_reasoning_tool_call_turns"] > 0
    )
    bucket["assistant_turns_without_tools_traces"] += int(
        metrics["assistant_turns_without_tools"] > 0
    )
    bucket["post_root_continuation_traces"] += int(
        metrics["post_root_continuation_turns"] > 0
    )
    bucket["credential_attempt_traces"] += int(metrics["credential_attempts"] > 0)
    bucket["parallel_credential_batch_traces"] += int(
        metrics["has_parallel_credential_batch"]
    )
    _record_sets_and_counters(bucket, metrics, audit, metadata)
    _record_live_rejections(bucket, metadata, rejection_reasons)


def _record_metric_lists(bucket: dict[str, Any], metrics: dict[str, Any]) -> None:
    bucket["turns"].append(int(metrics["turns"]))
    bucket["tool_calls"].append(int(metrics["tool_calls"]))
    bucket["credential_attempts"].append(int(metrics["credential_attempts"]))
    for field, metric in (
        ("evidence_rounds", "evidence_round"),
        ("exploit_rounds", "exploit_round"),
        ("root_rounds", "root_round"),
        ("evidence_to_exploit_latencies", "evidence_to_exploit_latency"),
        ("exploit_to_root_latencies", "exploit_to_root_latency"),
        ("evidence_to_root_latencies", "evidence_to_root_latency"),
    ):
        value = metrics[metric]
        if value is not None:
            bucket[field].append(int(value))


def _record_sets_and_counters(
    bucket: dict[str, Any],
    metrics: dict[str, Any],
    audit: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    for metric, field in (
        ("task_id", "task_ids"),
        ("seed", "seeds"),
        ("solution_signature_hash", "solution_signature_hashes"),
    ):
        _add(metrics[metric], bucket[field])
    if metrics["task_id"] is not None:
        bucket["task_trace_counts"][str(metrics["task_id"])] += 1
    bucket["leakage_class_counts"].update(audit["leakage_classes"])
    bucket["structured_rejection_classes"].update(audit["structured_rejection_classes"])
    bucket["family_behavior_counts"].update(metrics["family_behavior_flags"])


def _record_live_rejections(
    bucket: dict[str, Any],
    metadata: dict[str, Any],
    rejection_reasons: Counter[str] | None,
) -> None:
    if metadata.get("pre_repair_rejected") is not True:
        return

    bucket["live_rejected"] += 1
    reasons = metadata.get("pre_repair_rejection_reasons")
    for raw_reason in reasons if isinstance(reasons, list) else []:
        reason = str(raw_reason)
        category = reason.split(" (", 1)[0]
        if category == "static_eval_adjacent_artifacts":
            category = "holdout_leakage"
        if category == "solution_leakage":
            category = "secret_solution_leakage"
        if rejection_reasons is not None:
            rejection_reasons[reason] += 1
        bucket["structured_rejection_classes"][category] += 1
        if category == "secret_solution_leakage":
            bucket["leakage_rejected"] += 1
        elif category == "holdout_leakage" or category.startswith("holdout_leakage:"):
            bucket["leakage_rejected"] += 1
            bucket["holdout_leakage_rejected"] += 1
        else:
            bucket["other_live_rejected"] += 1


def _record_metadata_histograms(
    bucket: dict[str, Any],
    metadata: dict[str, Any],
    payload: dict[str, Any],
    sft_quality: dict[str, Any],
) -> None:
    _update_metadata_histogram_bucket(bucket, "raw", metadata)
    if payload.get("success") is True:
        _update_metadata_histogram_bucket(bucket, "teacher_solved", metadata)
    if sft_quality["sft_usable"]:
        _update_metadata_histogram_bucket(bucket, "sft_usable", metadata)


def _update_metadata_histogram_bucket(
    bucket: dict[str, Any], name: str, metadata: dict[str, Any]
) -> None:
    histograms = bucket["metadata_histograms"][name]
    for field in METADATA_HISTOGRAM_FIELDS:
        if field not in metadata:
            continue
        value = metadata[field]
        if not _is_scalar_metadata_value(value):
            continue
        histograms[field][_metadata_histogram_key(value)] += 1


def selected_metadata_histograms(
    metadata_values: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    histograms = {field: Counter() for field in METADATA_HISTOGRAM_FIELDS}
    for metadata in metadata_values:
        for field in METADATA_HISTOGRAM_FIELDS:
            if field not in metadata:
                continue
            value = metadata[field]
            if _is_scalar_metadata_value(value):
                histograms[field][_metadata_histogram_key(value)] += 1
    return _counter_map_summary(histograms)


def _metadata_histogram_summary(
    buckets: dict[str, dict[str, Counter[str]]],
) -> dict[str, dict[str, dict[str, int]]]:
    return {name: _counter_map_summary(histograms) for name, histograms in buckets.items()}


def _counter_map_summary(
    histograms: dict[str, Counter[str]],
) -> dict[str, dict[str, int]]:
    return {
        field: dict(sorted(counter.items()))
        for field, counter in histograms.items()
        if counter
    }


def _is_scalar_metadata_value(value: Any) -> bool:
    return value is None or isinstance(value, str | int | float | bool)


def _metadata_histogram_key(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _bucket_summary(bucket: dict[str, Any]) -> dict[str, Any]:
    raw = int(bucket["raw"])
    summary = {key: bucket[key] for key in COUNT_FIELDS}
    summary.update(
        {name: _rate(bucket[count], raw) for name, count in RATE_FIELDS.items()}
    )
    summary.update({name: _mean(bucket[field]) for name, field in MEAN_FIELDS.items()})
    summary.update(
        {name: _median(bucket[field]) for name, field in MEDIAN_FIELDS.items()}
    )
    summary.update(
        {
            name: _nullable_mean(bucket[field])
            for name, field in NULLABLE_MEAN_FIELDS.items()
        }
    )
    summary.update(
        {
            "unique_task_ids": len(bucket["task_ids"]),
            "unique_seeds": len(bucket["seeds"]),
            "unique_solution_signature_hashes": len(
                bucket["solution_signature_hashes"]
            ),
            "leakage_class_counts": dict(
                sorted(bucket["leakage_class_counts"].items())
            ),
            "structured_rejection_classes": dict(
                sorted(bucket["structured_rejection_classes"].items())
            ),
            "family_behavior_counts": dict(
                sorted(bucket["family_behavior_counts"].items())
            ),
            "traces_per_task_mean": _mean(list(bucket["task_trace_counts"].values())),
            "traces_per_task_max": max(bucket["task_trace_counts"].values(), default=0),
            "metadata_histograms": _metadata_histogram_summary(
                bucket["metadata_histograms"]
            ),
        }
    )
    return summary


def summarize_trace_files(
    trace_files: list[Path], *, generators: list[str] | None = None
) -> dict[str, Any]:
    summary, _, _ = build_collection_audit(trace_files, generators=generators)
    return summary


def build_collection_audit(
    trace_files: list[Path],
    *,
    generators: list[str] | None = None,
    sft_checker: SFTAssemblyTargetChecker | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    configured_generators = [str(name) for name in generators or []]
    totals = _new_bucket()
    by_generator = {name: _new_bucket() for name in configured_generators}
    by_family = {_family_name(name): _new_bucket() for name in configured_generators}
    rejection_reasons: Counter[str] = Counter()
    atom_counts: Counter[str] = Counter()
    atom_counts_by_scenario: Counter[str] = Counter()
    skeleton_counts: Counter[str] = Counter()
    task_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    provenance = {
        "prompt_hashes": set(),
        "config_hashes": set(),
        "git_commits": set(),
        "models": set(),
        "sft_tokenizer_models": set(),
    }

    for trace_file in trace_files:
        payload = json.loads(trace_file.read_text(encoding="utf-8"))
        generator = _generator_name(payload, trace_file)
        family = _family_name(generator)
        metrics = _trace_metrics(payload)
        audit = _trace_audit(payload, generator, metrics)
        quality = (
            sft_checker.check(payload)
            if sft_checker is not None
            else {"sft_usable": audit["audit_usable"], "sft_rejection_reasons": []}
        )
        holdout_ok = audit["audit_usable"] or (
            sft_checker is not None
            and not sft_checker.quality_filter.config.reject_on_holdout_leakage
        )
        sft_quality = {
            "sft_usable": bool(holdout_ok and quality["sft_usable"]),
            "sft_rejection_reasons": quality["sft_rejection_reasons"],
        }
        metadata = _metadata(payload)

        _record_bucket(
            by_generator.setdefault(generator, _new_bucket()),
            payload,
            metrics,
            audit,
            sft_quality,
            rejection_reasons,
        )
        _record_bucket(
            by_family.setdefault(family, _new_bucket()),
            payload,
            metrics,
            audit,
            sft_quality,
        )
        _record_bucket(totals, payload, metrics, audit, sft_quality)
        _update_counter(atom_counts, audit["benchmark_atom_counts"])
        _update_counter(
            atom_counts_by_scenario, audit["benchmark_atom_counts_by_scenario"]
        )
        _update_counter(skeleton_counts, audit["command_skeleton_overlap_counts"])
        _record_provenance(provenance, payload, metadata, audit)
        task_rows.append(
            _task_signature_row(
                trace_file, generator, family, payload, metrics, audit, sft_quality
            )
        )
        leakage_rows.append(
            _trace_leakage_row(
                trace_file, generator, family, payload, metrics, audit, sft_quality
            )
        )

    total = _bucket_summary(totals)
    return (
        {
            "total": total,
            "by_generator": {
                name: _bucket_summary(bucket)
                for name, bucket in sorted(by_generator.items())
            },
            "by_family": {
                name: _bucket_summary(bucket)
                for name, bucket in sorted(by_family.items())
            },
            "configured_generators": configured_generators,
            "rejection_reasons": dict(sorted(rejection_reasons.items())),
            "structured_rejection_classes": total["structured_rejection_classes"],
            "leakage_class_counts": total["leakage_class_counts"],
            "benchmark_atom_counts": dict(sorted(atom_counts.items())),
            "benchmark_atom_counts_by_scenario": dict(
                sorted(atom_counts_by_scenario.items())
            ),
            "command_payload_skeleton_overlap_counts": dict(
                sorted(skeleton_counts.items())
            ),
            "provenance": {key: sorted(values) for key, values in provenance.items()},
        },
        task_rows,
        leakage_rows,
    )


def write_collection_summary(cfg: Any) -> dict[str, Any]:
    from hydra.utils import get_original_cwd

    from src.config import sft_teacher_model
    from src.paths import resolve_traces_dir

    teacher_model = sft_teacher_model(cfg)
    base_dir = Path(get_original_cwd())
    source_dir = str(getattr(cfg.datasets.sft, "source_dir", "training"))
    trace_root = getattr(cfg.datasets.sft, "trace_root", None)
    traces_dir = resolve_traces_dir(
        base_dir, teacher_model, source_dir, trace_root=trace_root
    )
    summary, task_rows, leakage_rows = build_collection_audit(
        sorted(traces_dir.glob("*.json")) if traces_dir else [],
        generators=_configured_generators(cfg),
        sft_checker=SFTAssemblyTargetChecker(cfg),
    )
    summary.update(
        {
            "teacher_model": teacher_model,
            "source_dir": source_dir,
            "target_per_generator": int(getattr(cfg.runner, "runs_per_item", 0) or 0),
            "traces_dir": str(traces_dir) if traces_dir else None,
        }
    )

    stats_dir = _stats_dir(base_dir, source_dir, trace_root)
    stats_dir.mkdir(parents=True, exist_ok=True)
    output_path = stats_dir / "collection_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_jsonl(stats_dir / "task_signatures.jsonl", task_rows)
    _write_jsonl(stats_dir / "trace_leakage.jsonl", leakage_rows)
    print_summary(summary, output_path)
    return summary


def print_summary(summary: dict[str, Any], output_path: Path) -> None:
    total = summary["total"]
    console.print(
        "[bold green]Collection Summary[/] "
        f"teacher_solve_rate={_percent(total['teacher_solve_rate'])} "
        f"sft_usable={total['sft_usable']}/{total['raw']} "
        f"audit_usable={total['usable']}/{total['raw']} "
        f"leakage_rejects={total['leakage_rejected']} "
        f"audit_rejects={total['audit_rejected']} "
        f"avg_turns={total['avg_turns']:.1f} "
        f"avg_tool_calls={total['avg_tool_calls']:.1f} "
        f"median_root_round={_number(total['median_root_round'])}"
    )

    table = Table(
        "Generator",
        "SFT",
        "Audit",
        "Solved",
        "Solve",
        "Leak",
        "SFTRej",
        "AuditRej",
        "LiveRej",
    )
    for generator, stats in summary["by_generator"].items():
        table.add_row(
            generator,
            str(stats["sft_usable"]),
            str(stats["usable"]),
            f"{stats['teacher_solved']}/{stats['raw']}",
            _percent(stats["teacher_solve_rate"]),
            str(stats["leakage_rejected"]),
            str(stats["sft_rejected"]),
            str(stats["audit_rejected"]),
            str(stats["live_rejected"]),
        )
    console.print(table)
    console.print(f"[dim]Wrote {output_path}[/]")


def missing_total(
    summary: dict[str, Any],
    *,
    generators: list[str],
    target_per_generator: int,
    usable_field: str = "sft_usable",
) -> int:
    by_generator = summary.get("by_generator", {})
    return sum(
        max(
            0,
            target_per_generator
            - _usable_count(_generator_stats(by_generator, g), usable_field),
        )
        for g in generators
    )


def format_target_status(
    summary: dict[str, Any],
    *,
    generators: list[str],
    target_per_generator: int,
    usable_field: str = "sft_usable",
) -> str:
    by_generator = summary.get("by_generator", {})
    total = summary.get("total", {})
    expected = target_per_generator * len(generators)
    missing = 0
    lines = [
        "",
        f"Trace status (expected usable: {expected})",
        (
            f"  {'':2} {'Generator':25} {'Usable':>7} {'Target':>7} "
            f"{'Missing':>7} {'Solve':>8} {'Leak':>6} {'Audit':>6} "
            f"{'SFTRej':>7} {'LiveRej':>7} {'Raw':>7}"
        ),
    ]
    for generator in generators:
        stats = _generator_stats(by_generator, generator)
        usable = _usable_count(stats, usable_field)
        gen_missing = max(0, target_per_generator - usable)
        missing += gen_missing
        mark = "ok" if usable >= target_per_generator else "no"
        lines.append(
            f"  {mark:2} {generator:25} {usable:7d} {target_per_generator:7d} "
            f"{gen_missing:7d} {100 * float(stats.get('teacher_solve_rate', 0.0)):7.1f}% "
            f"{int(stats.get('leakage_rejected', 0)):6d} "
            f"{int(stats.get('audit_rejected', 0)):6d} "
            f"{int(stats.get('sft_rejected', 0)):7d} "
            f"{int(stats.get('live_rejected', 0)):7d} "
            f"{int(stats.get('raw', 0)):7d}"
        )
    lines.extend(
        [
            "",
            (
                "Totals: "
                f"usable={_usable_count(total, usable_field)}/{expected}, "
                f"raw={int(total.get('raw', 0))}, "
                f"teacher_solve_rate={100 * float(total.get('teacher_solve_rate', 0.0)):.1f}%, "
                f"leakage_rejects={int(total.get('leakage_rejected', 0))}, "
                f"audit_rejects={int(total.get('audit_rejected', 0))}, "
                f"live_rejects={int(total.get('live_rejected', 0))}, "
                f"missing={missing}"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def summary_generators(summary: dict[str, Any]) -> list[str]:
    configured = summary.get("configured_generators", [])
    if isinstance(configured, list) and configured:
        return [str(name) for name in configured]
    by_generator = summary.get("by_generator", {})
    return (
        sorted(str(name) for name in by_generator)
        if isinstance(by_generator, dict)
        else []
    )


def summary_target_per_generator(summary: dict[str, Any]) -> int | None:
    value = summary.get("target_per_generator")
    if isinstance(value, int | float):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def audit_trace_payload(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = _metadata(payload)
    generator = str(metadata.get("generator_name") or payload.get("scenario") or "")
    return _trace_audit(payload, generator, _trace_metrics(payload))


def holdout_leakage_rejection_reasons(payload: dict[str, Any]) -> list[str]:
    return [
        f"holdout_leakage:{class_name}"
        for class_name in audit_trace_payload(payload)["hard_rejection_classes"]
    ]


def counts_toward_audited_trace_collection_target(payload: dict[str, Any]) -> bool:
    return bool(audit_trace_payload(payload)["audit_usable"])


def _trace_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    messages = trace_messages(payload)
    tool_results = trace_tool_results(messages)
    commands = [c for c in (trace_tool_result_signature(r) for r in tool_results) if c]
    root_round = _first_round(
        tool_results, lambda r: r["payload"].get("got_root") is True
    )
    evidence_round = _evidence_round(tool_results, _evidence_markers(payload))
    exploit_round = _exploit_round(
        tool_results, evidence_round, _solution_command_skeletons(payload)
    )
    behavior = _behavior_metrics(messages, commands)
    turn_quality = _turn_quality_metrics(messages)
    signature = solution_signature(payload)
    return {
        "turns": int(payload.get("turns", 0) or 0),
        "tool_calls": _count_tool_calls(messages),
        "evidence_round": evidence_round,
        "exploit_round": exploit_round,
        "root_round": root_round,
        "evidence_to_exploit_latency": _latency(evidence_round, exploit_round),
        "exploit_to_root_latency": _latency(exploit_round, root_round),
        "evidence_to_root_latency": _latency(evidence_round, root_round),
        "has_repeated_command": len(commands) != len(set(commands)),
        "has_invalid_command": any(_invalid_result(r) for r in tool_results),
        **behavior,
        **turn_quality,
        "task_id": _task_id(payload),
        "seed": _seed(payload),
        "solution_signature": signature,
        "solution_signature_hash": _hash(signature) if signature else None,
    }


def solution_signature(payload: dict[str, Any]) -> str | None:
    metadata = _metadata(payload)
    fields = {
        key: metadata[key]
        for key in SOLUTION_SIGNATURE_KEYS
        if metadata.get(key) not in (None, "", [], {})
    }
    if not fields and payload.get("scenario") is not None:
        fields["scenario"] = payload["scenario"]
    return json.dumps(fields, sort_keys=True, default=str) if fields else None


def solution_atom_refs(payload: dict[str, Any]) -> list[dict[str, str]]:
    metadata = _metadata(payload)
    refs: list[dict[str, str]] = []
    for key in SOLUTION_ATOM_KEYS:
        atom = clean_atom(metadata.get(key))
        if atom:
            refs.append({"key": key, "value": atom, "hash": hash_text(atom)})
    for key in SOLUTION_ATOM_LIST_KEYS:
        value = metadata.get(key)
        if isinstance(value, list):
            refs.extend(
                {"key": key, "value": atom, "hash": hash_text(atom)}
                for item in value
                for atom in [clean_atom(item)]
                if atom
            )
    return refs


def _invalid_result(result: dict[str, Any]) -> bool:
    payload = result["payload"]
    output = str(payload.get("output") or "").lower()
    return payload.get("exit_code") == 127 or any(
        m in output for m in INVALID_COMMAND_MARKERS
    )


def _behavior_metrics(messages: list[Any], commands: list[str]) -> dict[str, Any]:
    steps = trace_assistant_steps(messages)
    credential_pairs = {pair for step in steps for pair in step["credential_pairs"]}
    lowered = [command.lower() for command in commands]
    flags = set()
    if credential_pairs:
        flags.add("credential_attempt")
    if any(len(step["credential_pairs"]) > 1 for step in steps):
        flags.add("parallel_credential_batch")
    flag_checks = {
        "sudo_probe": lambda c: "sudo -l" in c,
        "sudo_command": lambda c: c.startswith("sudo "),
        "suid_probe": lambda c: "-perm" in c and "4000" in c,
        "capability_probe": lambda c: "getcap" in c,
        "cron_probe": lambda c: "/etc/cron" in c or "crontab" in c,
        "cron_wait": lambda c: "sleep" in c,
        "ssh_probe": lambda c: ".ssh" in c or c.startswith("ssh "),
        "container_runtime": lambda c: "docker" in c,
        "container_host_mount": lambda c: "chroot /mnt" in c or "-v /:/mnt" in c,
    }
    for flag, check in flag_checks.items():
        if any(check(command) for command in lowered):
            flags.add(flag)
    return {
        "credential_attempts": len(credential_pairs),
        "has_parallel_credential_batch": "parallel_credential_batch" in flags,
        "family_behavior_flags": sorted(flags),
    }


def _turn_quality_metrics(messages: list[Any]) -> dict[str, int]:
    near_empty_reasoning = 0
    no_tools = 0
    post_root = 0
    root_seen = False

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if root_seen and role == "assistant":
            post_root += 1
        if role == "assistant":
            tool_calls = message.get("tool_calls")
            has_tools = isinstance(tool_calls, list) and bool(tool_calls)
            reasoning = text_content(message.get("content", "")).strip()
            if has_tools and len(reasoning) < MIN_REASONING_LENGTH:
                near_empty_reasoning += 1
            elif not has_tools:
                no_tools += 1
        elif (
            role == "tool"
            and trace_json_obj(message.get("content")).get("got_root") is True
        ):
            root_seen = True

    return {
        "near_empty_reasoning_tool_call_turns": near_empty_reasoning,
        "assistant_turns_without_tools": no_tools,
        "post_root_continuation_turns": post_root,
    }


def _count_tool_calls(messages: list[Any]) -> int:
    total = 0
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            total += sum(1 for call in tool_calls if isinstance(call, dict))
    return total


def _first_round(tool_results: list[dict[str, Any]], predicate: Any) -> int | None:
    rounds = [int(result["round"]) for result in tool_results if predicate(result)]
    return min(rounds) if rounds else None


def _evidence_round(
    tool_results: list[dict[str, Any]], markers: set[str]
) -> int | None:
    if not markers:
        return None
    for result in tool_results:
        payload = result["payload"]
        if payload.get("got_root") is True:
            continue
        observed = " ".join(str(payload.get(k) or "") for k in ("output", "message"))
        if any(marker.lower() in observed.lower() for marker in markers):
            return int(result["round"])
    return None


def _exploit_round(
    tool_results: list[dict[str, Any]],
    evidence_round: int | None,
    solution_skeletons: set[str],
) -> int | None:
    for result in tool_results:
        round_index = int(result["round"])
        if evidence_round is not None and round_index < evidence_round:
            continue
        signature = trace_tool_result_signature(result)
        if result["payload"].get("got_root") is True or (
            signature and signature.startswith("test_credentials:")
        ):
            return round_index
        if signature and command_skeleton(signature) in solution_skeletons:
            return round_index
    return None


def _latency(start: int | None, end: int | None) -> int | None:
    return None if start is None or end is None else max(0, end - start)


def _evidence_markers(payload: dict[str, Any]) -> set[str]:
    metadata = _metadata(payload)
    markers = set()
    for key in EVIDENCE_KEYS:
        markers.update(
            _evidence_marker_values(
                metadata.get(key), path_aliases=key in PATH_EVIDENCE_KEYS
            )
        )
    for key in LIST_EVIDENCE_KEYS:
        value = metadata.get(key)
        if isinstance(value, list):
            for item in value:
                markers.update(_evidence_marker_values(item, path_aliases=True))
    for ref in solution_atom_refs(payload):
        if ref["key"] != "exploit_cmds":
            markers.update(
                _evidence_marker_values(
                    ref["value"], path_aliases=ref["key"] in PATH_EVIDENCE_KEYS
                )
            )
    return {marker for marker in markers if marker}


def _evidence_marker_values(value: Any, *, path_aliases: bool = False) -> set[str]:
    atom = clean_atom(value)
    if not atom:
        return set()
    markers = {atom}
    if path_aliases and "/" in atom:
        basename = clean_atom(atom.rsplit("/", 1)[-1])
        if basename and _useful_path_alias(basename):
            markers.add(basename)
    return markers


def _useful_path_alias(value: str) -> bool:
    return value.startswith(".") or "." in value or len(value) >= 6


def _solution_command_skeletons(payload: dict[str, Any]) -> set[str]:
    commands = _metadata(payload).get("exploit_cmds")
    if not isinstance(commands, list):
        return set()
    return {
        command_skeleton(command) for command in commands if isinstance(command, str)
    }


def _trace_audit(
    payload: dict[str, Any], generator: str, metrics: dict[str, Any]
) -> dict[str, Any]:
    metadata = _metadata(payload)
    holdout = audit_holdout_leakage(payload, generator)

    oracle_matches = _oracle_before_evidence(
        payload,
        generator=generator,
        metadata=metadata,
        cutoff=metrics["evidence_round"] or metrics["root_round"],
    )

    base_usable = _base_usable(payload)
    hard_classes = holdout["hard_rejection_classes"]
    return {
        "base_usable": base_usable,
        "audit_usable": base_usable and not hard_classes,
        "hard_rejection_classes": hard_classes,
        "structured_rejection_classes": hard_classes,
        "leakage_classes": holdout["leakage_classes"],
        "benchmark_atom_matches": holdout["benchmark_atom_matches"],
        "benchmark_atom_counts": holdout["benchmark_atom_counts"],
        "benchmark_atom_counts_by_scenario": holdout[
            "benchmark_atom_counts_by_scenario"
        ],
        "command_skeleton_overlaps": holdout["command_skeleton_overlaps"],
        "command_skeleton_overlap_counts": holdout["command_skeleton_overlap_counts"],
        "oracle_before_evidence": oracle_matches,
        "prompt_hashes": _prompt_hashes(payload),
        "config_hash": _config_hash(payload),
    }


def _oracle_before_evidence(
    payload: dict[str, Any],
    *,
    generator: str,
    metadata: dict[str, Any],
    cutoff: int | None,
) -> list[dict[str, Any]]:
    if cutoff is None:
        return []
    matches = []
    refs = solution_atom_refs(payload)
    for step in trace_assistant_steps(trace_messages(payload)):
        if int(step["round"]) > cutoff:
            continue
        for surface, text in [
            ("reasoning", step["content"]),
            *[("action", a) for a in step["actions"]],
        ]:
            lowered = text.lower()
            for ref in refs:
                if ref["key"] not in ORACLE_HARD_ATOM_KEYS:
                    continue
                if ref["value"].lower() not in lowered:
                    continue
                if _allowed_known_guess(generator, ref, surface, text, metadata):
                    continue
                matches.append(
                    {
                        "round": step["round"],
                        "surface": surface,
                        "atom_key": ref["key"],
                        "atom_hash": ref["hash"],
                    }
                )
    return matches


def _allowed_known_guess(
    generator: str,
    atom_ref: dict[str, str],
    surface: str,
    text: str,
    metadata: dict[str, Any],
) -> bool:
    if atom_ref["key"] not in {"root_password", "target_password", "intended_password"}:
        return False
    if generator == "weak_password":
        return _allowed_weak_password_guess(atom_ref, surface, text, metadata)
    if generator == "password_reuse" and atom_ref["value"] in _known_login_passwords(
        metadata
    ):
        return surface == "reasoning" or text.startswith("test_credentials:")
    return False


def _allowed_weak_password_guess(
    atom_ref: dict[str, str], surface: str, text: str, metadata: dict[str, Any]
) -> bool:
    if "common_password" not in _credential_password_sources(
        metadata, atom_ref["value"]
    ):
        return False
    return surface == "reasoning" or text.startswith("test_credentials:")


def _credential_password_sources(metadata: dict[str, Any], password: str) -> set[str]:
    sources = {
        str(candidate.get("password_source"))
        for candidate in _credential_candidates(metadata)
        if candidate.get("password") == password and candidate.get("gives_root") is True
    }
    if metadata.get("root_password") == password and metadata.get("password_source"):
        sources.add(str(metadata["password_source"]))
    return sources


def _credential_candidates(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = metadata.get("credential_candidates")
    if not isinstance(candidates, list):
        return []
    return [candidate for candidate in candidates if isinstance(candidate, dict)]


def _known_login_passwords(metadata: dict[str, Any]) -> set[str]:
    return {
        str(metadata[key])
        for key in ("user_password", "container_password")
        if isinstance(metadata.get(key), str) and metadata[key]
    }


def _task_signature_row(
    trace_file: Path,
    generator: str,
    family: str,
    payload: dict[str, Any],
    metrics: dict[str, Any],
    audit: dict[str, Any],
    sft_quality: dict[str, Any],
) -> dict[str, Any]:
    metadata = _metadata(payload)
    return {
        "trace_file": str(trace_file),
        "generator": generator,
        "family": family,
        "task_id": metrics["task_id"],
        "seed": metrics["seed"],
        "scenario": payload.get("scenario"),
        "model": payload.get("model"),
        "git_commit": metadata.get("git_commit"),
        "prompt_hashes": sorted(audit["prompt_hashes"]),
        "config_hash": audit["config_hash"],
        "expected_solution_signature_hash": metrics["solution_signature_hash"],
        "expected_solution_signature": metrics["solution_signature"],
        "success": payload.get("success") is True,
        "base_usable": audit["base_usable"],
        "audit_usable": audit["audit_usable"],
        "sft_usable": sft_quality["sft_usable"],
        "sft_rejection_reasons": sft_quality["sft_rejection_reasons"],
    }


def _trace_leakage_row(
    trace_file: Path,
    generator: str,
    family: str,
    payload: dict[str, Any],
    metrics: dict[str, Any],
    audit: dict[str, Any],
    sft_quality: dict[str, Any],
) -> dict[str, Any]:
    return {
        "trace_file": str(trace_file),
        "generator": generator,
        "family": family,
        "task_id": metrics["task_id"],
        "seed": metrics["seed"],
        "success": payload.get("success") is True,
        "base_usable": audit["base_usable"],
        "audit_usable": audit["audit_usable"],
        "sft_usable": sft_quality["sft_usable"],
        "sft_rejection_reasons": sft_quality["sft_rejection_reasons"],
        "hard_rejection_classes": audit["hard_rejection_classes"],
        "leakage_classes": audit["leakage_classes"],
        "benchmark_atom_counts": audit["benchmark_atom_counts"],
        "benchmark_atom_counts_by_scenario": audit["benchmark_atom_counts_by_scenario"],
        "benchmark_atom_matches": audit["benchmark_atom_matches"],
        "command_skeleton_overlaps": audit["command_skeleton_overlaps"],
        "oracle_before_evidence": audit["oracle_before_evidence"],
        "evidence_round": metrics["evidence_round"],
        "exploit_round": metrics["exploit_round"],
        "root_round": metrics["root_round"],
        "evidence_to_exploit_latency": metrics["evidence_to_exploit_latency"],
        "exploit_to_root_latency": metrics["exploit_to_root_latency"],
        "evidence_to_root_latency": metrics["evidence_to_root_latency"],
    }


def _record_provenance(
    provenance: dict[str, set[str]],
    payload: dict[str, Any],
    metadata: dict[str, Any],
    audit: dict[str, Any],
) -> None:
    provenance["prompt_hashes"].update(audit["prompt_hashes"])
    provenance["config_hashes"].add(audit["config_hash"])
    _add(metadata.get("git_commit"), provenance["git_commits"])
    _add(payload.get("model"), provenance["models"])
    _add(payload.get("sft_tokenizer_model"), provenance["sft_tokenizer_models"])
    _add(metadata.get("sft_tokenizer_model"), provenance["sft_tokenizer_models"])


def _prompt_hashes(payload: dict[str, Any]) -> set[str]:
    return {
        _hash(content)
        for message in trace_messages(payload)
        if isinstance(message, dict) and message.get("role") == "system"
        for content in [text_content(message.get("content", ""))]
        if content
    }


def _config_hash(payload: dict[str, Any]) -> str:
    metadata = _metadata(payload)
    return _hash_json(
        {
            "model": payload.get("model"),
            "mode": payload.get("mode"),
            "scenario": payload.get("scenario"),
            "generator_name": metadata.get("generator_name"),
            "category": metadata.get("category"),
            "source_type": metadata.get("source_type"),
            "scenario_backend": metadata.get("scenario_backend"),
            "agent_api_base": metadata.get("agent_api_base"),
            "prompt_vars": metadata.get("prompt_vars"),
            "sft_tokenizer_model": payload.get("sft_tokenizer_model")
            or metadata.get("sft_tokenizer_model"),
        }
    )


def _task_id(payload: dict[str, Any]) -> str | None:
    metadata = _metadata(payload)
    explicit = metadata.get("task_id") or metadata.get("source_scenario")
    if explicit is not None:
        return str(explicit)
    generator = metadata.get("generator_name") or payload.get("scenario")
    seed = metadata.get("seed")
    if generator is not None and seed is not None:
        return f"{generator}:{seed}"
    return str(payload["scenario"]) if payload.get("scenario") is not None else None


def _seed(payload: dict[str, Any]) -> str | None:
    seed = _metadata(payload).get("seed")
    return str(seed) if seed is not None else None


def _metadata(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _base_usable(payload: dict[str, Any]) -> bool:
    from src.runner.result_policy import counts_toward_trace_collection_target

    return counts_toward_trace_collection_target(payload)


def _mean(values: list[int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _nullable_mean(values: list[int]) -> float | None:
    return _mean(values) if values else None


def _median(values: list[int]) -> float | int | None:
    return statistics.median(values) if values else None


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _update_counter(counter: Counter[str], values: dict[str, Any]) -> None:
    for key, count in values.items():
        counter[str(key)] += int(count)


def _add(value: Any, values: set[str]) -> None:
    if isinstance(value, str) and value:
        values.add(value)


def _hash(value: str) -> str:
    return hash_text(value)


def _hash_json(value: Any) -> str:
    return hash_json(value)


def _family_name(generator: str) -> str:
    return GENERATOR_FAMILIES.get(generator, "other")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str))
            handle.write("\n")


def _generator_name(payload: dict[str, Any], trace_file: Path) -> str:
    generator = _metadata(payload).get("generator_name")
    if isinstance(generator, str) and generator:
        return generator
    return trace_file.name.split("_202", 1)[0]


def _configured_generators(cfg: Any) -> list[str]:
    source = getattr(cfg.runner, "source", None)
    generators = getattr(source, "generators", None)
    return [str(name) for name in generators] if generators else []


def _stats_dir(base_dir: Path, source_dir: str, trace_root: str | None) -> Path:
    from src.paths import TRACE_COLLECTION_BASE

    root = Path(trace_root) if trace_root else Path(TRACE_COLLECTION_BASE)
    return (root if root.is_absolute() else base_dir / root) / source_dir / "stats"


def _percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def _number(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.1f}"


def _generator_stats(by_generator: Any, generator: str) -> dict[str, Any]:
    if not isinstance(by_generator, dict):
        return {}
    stats = by_generator.get(generator, {})
    return stats if isinstance(stats, dict) else {}


def _usable_count(stats: dict[str, Any], field: str) -> int:
    value = stats.get(field, stats.get("usable", 0))
    if isinstance(value, int | float):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return 0


def _read_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"total": {}, "by_generator": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {"total": {}, "by_generator": {}}


def _run_status_cli(argv: list[str]) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Read trace collection summary.")
    parser.add_argument("action", choices=["missing", "status", "generators", "target"])
    parser.add_argument("summary_path", type=Path)
    parser.add_argument("--target", type=int)
    parser.add_argument("--generator", action="append", default=[])
    parser.add_argument("--usable-field", default="sft_usable")
    args = parser.parse_args(argv)

    summary = _read_summary(args.summary_path)
    generators = [str(generator) for generator in args.generator] or summary_generators(
        summary
    )
    if args.action == "generators":
        print("\n".join(generators))
        return
    if args.action == "target":
        target = summary_target_per_generator(summary)
        if target is None:
            parser.error("summary does not contain target_per_generator")
        print(target)
        return
    target = (
        args.target
        if args.target is not None
        else summary_target_per_generator(summary)
    )
    if target is None:
        parser.error("--target is required for missing/status")
    if args.action == "missing":
        print(
            missing_total(
                summary,
                generators=generators,
                target_per_generator=target,
                usable_field=args.usable_field,
            )
        )
    else:
        print(
            format_target_status(
                summary,
                generators=generators,
                target_per_generator=target,
                usable_field=args.usable_field,
            )
        )


def _run_hydra_cli() -> None:
    import hydra

    from src.config import register_with_hydra

    @hydra.main(version_base=None, config_path="../../../conf", config_name="config")
    def main(cfg: Any) -> None:
        write_collection_summary(cfg)

    register_with_hydra()
    main()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in {
        "missing",
        "status",
        "generators",
        "target",
    }:
        _run_status_cli(sys.argv[1:])
    else:
        _run_hydra_cli()
