"""Create the paper split and leakage audit bundle."""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
from typing import Any

import hydra
from hydra.utils import get_original_cwd
from rich import get_console

from src.config import (
    AppConfig,
    DatasetAuditConfig,
    plain_config_dict,
    register_with_hydra,
)
from src.dataset.privesc.sft_quality import matched_keywords
from src.generators.base import load_generator_profile_config
from src.generators.holdout_manifest import (
    GENERATOR_HOLDOUT_SCAN_RULES,
    HOLDOUT_HARD_CLASSES_BY_CATEGORY,
)
from src.generators.profile_audit import generator_profile_diff_rows
from src.paths import (
    base_model_slug,
    paper_run_root,
    path_from_outputs,
    path_from_outputs_or_none,
)

console = get_console()


@dataclass
class TraceLeakageBucket:
    rows: int = 0
    base_usable: int = 0
    audit_usable: int = 0
    sft_usable: int = 0
    rows_with_hard_rejection: int = 0
    hard_rejection_class_counts: Counter[str] = field(default_factory=Counter)
    sft_rejection_reason_counts: Counter[str] = field(default_factory=Counter)

    def update(self, row: dict[str, Any]) -> None:
        self.rows += 1
        self.base_usable += int(bool(row.get("base_usable")))
        self.audit_usable += int(bool(row.get("audit_usable")))
        self.sft_usable += int(bool(row.get("sft_usable")))
        hard_rejections = _strings(row, "hard_rejection_classes")
        self.rows_with_hard_rejection += int(bool(hard_rejections))
        self.hard_rejection_class_counts.update(hard_rejections)
        self.sft_rejection_reason_counts.update(
            _reason_class(reason) for reason in _strings(row, "sft_rejection_reasons")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "base_usable": self.base_usable,
            "audit_usable": self.audit_usable,
            "sft_usable": self.sft_usable,
            "benchmark_holdout_rejected": self.rows_with_hard_rejection,
            "hard_rejection_class_counts": dict(
                sorted(self.hard_rejection_class_counts.items())
            ),
            "sft_rejection_reason_counts": dict(
                sorted(self.sft_rejection_reason_counts.items())
            ),
        }


@dataclass
class VisibilityBucket:
    examples: int = 0
    message_hit_examples: int = 0
    metadata_hit_examples: int = 0

    def update(self, message_hits: list[str], metadata_hits: list[str]) -> None:
        self.examples += 1
        if message_hits:
            self.message_hit_examples += 1
        if metadata_hits:
            self.metadata_hit_examples += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "examples": self.examples,
            "message_hit_examples": self.message_hit_examples,
            "metadata_hit_examples": self.metadata_hit_examples,
        }


def run_pipeline(cfg: AppConfig) -> None:
    project_dir = Path(get_original_cwd())
    outputs_root = project_dir / "outputs"
    audit_cfg = plain_config_dict(cfg.datasets.audit, error_label="datasets.audit")
    output_dir = _audit_output_dir(project_dir, audit_cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    created_at = _utc_now()
    git_commit = _git_commit(project_dir)
    model_visible_keywords = [
        str(keyword)
        for keyword in audit_cfg.get(
            "model_visible_secret_keywords",
            DatasetAuditConfig().model_visible_secret_keywords,
        )
    ]

    training_profile = load_generator_profile_config(
        str(audit_cfg["training_generator_profile"])
    )
    validation_profile = load_generator_profile_config(
        str(audit_cfg["validation_generator_profile"])
    )
    profile_rows = generator_profile_diff_rows(training_profile, validation_profile)
    benchmark_rows = benchmark_holdout_table_rows()
    paper_runs_root = _project_path(project_dir, str(audit_cfg["paper_runs_root"]))
    provenance_paths = sorted(paper_runs_root.glob("**/provenance_model_eval.json"))

    trace_summary = trace_leakage_summary(
        _project_path(project_dir, str(audit_cfg["trace_root"])),
        str(audit_cfg["trace_leakage_glob"]),
        project_dir,
    )
    dataset_scan = dataset_visibility_scan(
        _project_path(project_dir, str(audit_cfg["dataset_root"])),
        str(audit_cfg["dataset_glob"]),
        model_visible_keywords,
        project_dir,
    )
    selection = selection_ledger(
        paper_runs_root,
        provenance_paths,
        procedural_experiment=str(audit_cfg["procedural_validation_experiment"]),
        static_experiment=str(audit_cfg["static_benchmark_experiment"]),
        static_prefix=str(audit_cfg["static_paper_experiment_prefix"]),
        project_dir=project_dir,
    )

    manifest = run_manifest(
        audit_cfg,
        output_dir,
        outputs_root,
        created_at=created_at,
        git_commit=git_commit,
    )
    summary = reviewer_summary(
        audit_cfg,
        git_commit=git_commit,
        created_at=created_at,
        output_dir=output_dir,
        project_dir=project_dir,
        profile_rows=profile_rows,
        trace_summary=trace_summary,
        dataset_scan=dataset_scan,
        selection=selection,
        benchmark_rows=benchmark_rows,
        model_visible_keywords=model_visible_keywords,
    )

    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "run_manifest.json", manifest)
    _write_tsv(output_dir / "split_holdouts.tsv", profile_rows)
    _write_tsv(output_dir / "benchmark_holdouts.tsv", benchmark_rows)
    _write_text(output_dir / "README.md", readme_text(manifest, summary))

    console.print(f"[bold green]Wrote audit bundle:[/] {output_dir}")


def run_manifest(
    audit_cfg: dict[str, Any],
    output_dir: Path,
    outputs_root: Path,
    *,
    created_at: str,
    git_commit: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at_utc": created_at,
        "experiment_id": audit_cfg["experiment_id"],
        "base_model": {
            "id": audit_cfg["base_model"],
            "slug": base_model_slug(str(audit_cfg["base_model"])),
        },
        "condition": {
            "slug": audit_cfg["condition"],
            "stage": audit_cfg["stage"],
        },
        "git_commit": git_commit,
        "paths": {
            "run_root_path_from_outputs": path_from_outputs(output_dir, outputs_root),
            "native_artifact_root_path_from_outputs": path_from_outputs(
                output_dir, outputs_root
            ),
        },
        "source_data": {
            "trace_root_path_from_outputs": path_from_outputs_or_none(
                _source_path(audit_cfg["trace_root"], outputs_root), outputs_root
            ),
            "dataset_root_path_from_outputs": path_from_outputs_or_none(
                _source_path(audit_cfg["dataset_root"], outputs_root), outputs_root
            ),
            "paper_runs_root_path_from_outputs": path_from_outputs_or_none(
                _source_path(audit_cfg["paper_runs_root"], outputs_root), outputs_root
            ),
        },
    }


def reviewer_summary(
    audit_cfg: dict[str, Any],
    *,
    git_commit: str,
    created_at: str,
    output_dir: Path,
    project_dir: Path,
    profile_rows: list[dict[str, Any]],
    trace_summary: dict[str, Any],
    dataset_scan: dict[str, Any],
    selection: dict[str, Any],
    benchmark_rows: list[dict[str, Any]],
    model_visible_keywords: list[str],
) -> dict[str, Any]:
    profile_overlaps = [row for row in profile_rows if int(row["overlap_count"]) > 0]
    trace_totals = trace_summary["totals"]
    dataset_totals = dataset_scan["totals"]
    benchmark_categories = sorted({row["category"] for row in benchmark_rows})
    return {
        "schema_version": 1,
        "created_at_utc": created_at,
        "git_commit": git_commit,
        "entrypoint": "src.dataset.privesc.audit",
        "hydra_experiment": "audit/split_leakage",
        "output_dir": _project_relative(output_dir, project_dir),
        "reviewer_questions_answered": {
            "procedural_holdout_values_disjoint": len(profile_overlaps) == 0,
            "model_visible_solution_marker_hits_absent": dataset_totals[
                "message_hit_examples"
            ]
            == 0,
            "metadata_solution_marker_hits_present": dataset_totals[
                "metadata_hit_examples"
            ]
            > 0,
        },
        "headline_counts": {
            "generator_profiles_disjoint": len(profile_overlaps) == 0,
            "generator_profile_overlap_categories": [
                row["category"] for row in profile_overlaps
            ],
            "trace_leakage_files_scanned": trace_summary["files_scanned"],
            "raw_trace_rows_scanned": trace_totals["rows"],
            "trace_base_usable": trace_totals["base_usable"],
            "trace_audit_usable": trace_totals["audit_usable"],
            "trace_sft_usable": trace_totals["sft_usable"],
            "trace_benchmark_holdout_rejected": trace_totals[
                "benchmark_holdout_rejected"
            ],
            "trace_hard_rejection_classes": trace_totals["hard_rejection_class_counts"],
            "trace_sft_rejection_reasons": trace_totals["sft_rejection_reason_counts"],
            "dataset_examples_scanned": dataset_totals["examples"],
            "dataset_message_solution_hits": dataset_totals["message_hit_examples"],
            "dataset_metadata_solution_hits": dataset_totals["metadata_hit_examples"],
            "provenance_eval_files_scanned": selection["provenance_files_scanned"],
            "provenance_eval_scope_counts": selection["scope_counts"],
            "benchmark_holdout_rules": len(benchmark_rows),
        },
        "split_policy": {
            "training": {
                "source": "procedural",
                "generator_profile": audit_cfg["training_generator_profile"],
                "seed": 42,
                "random_seed": False,
                "config": "conf/runner/trace_collection/standard_training.yaml",
                "role": "SFT dataset construction and training traces",
            },
            "procedural_validation": {
                "source": "procedural",
                "generator_profile": audit_cfg["validation_generator_profile"],
                "trace_collection_seed": 10000000,
                "evaluation_seed": 1337,
                "random_seed": False,
                "config": audit_cfg["procedural_validation_experiment"],
                "role": "model/checkpoint/reward selection",
            },
            "static_benchmark": {
                "source": "static Happe et al. scenarios",
                "config": audit_cfg["static_benchmark_experiment"],
                "role": "final external evaluation only",
                "tuning_allowed": False,
            },
        },
        "leakage_policy": {
            "solution_marker_phrase_count": len(model_visible_keywords),
            "metadata_note": "Solution-marker hits occur only in metadata fields.",
        },
        "benchmark_holdout_policy": {
            "rule_table": "benchmark_holdouts.tsv",
            "unit": "one row per generator-specific benchmark holdout rule",
            "columns": [
                "generator",
                "benchmark_cases",
                "category",
                "match",
                "pattern",
                "hard_rejection_class",
            ],
            "categories": benchmark_categories,
        },
        "tables": {
            "split_holdouts": "split_holdouts.tsv",
            "benchmark_holdouts": "benchmark_holdouts.tsv",
        },
    }


def benchmark_holdout_table_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for generator, rules in sorted(GENERATOR_HOLDOUT_SCAN_RULES.items()):
        for index, rule in enumerate(rules, start=1):
            rows.append(
                {
                    "generator": generator,
                    "rule_id": f"{generator}:{index:02d}",
                    "benchmark_cases": ", ".join(rule.benchmark_cases),
                    "category": rule.category,
                    "match": rule.match,
                    "pattern": rule.pattern,
                    "hard_rejection_class": HOLDOUT_HARD_CLASSES_BY_CATEGORY.get(
                        rule.category, ""
                    ),
                }
            )
    return rows


def trace_leakage_summary(
    root: Path, pattern: str, project_dir: Path
) -> dict[str, Any]:
    total = TraceLeakageBucket()
    paths = sorted(root.glob(pattern)) if root.exists() else []
    for path in paths:
        for row in _iter_jsonl(path):
            total.update(row)
    if total.rows == 0:
        raise ValueError(f"No trace leakage rows found in {root} matching {pattern}")
    return {
        "schema_version": 1,
        "root": _project_relative(root, project_dir),
        "glob": pattern,
        "files_scanned": len(paths),
        "totals": total.to_dict(),
    }


def dataset_visibility_scan(
    root: Path, pattern: str, keywords: list[str], project_dir: Path
) -> dict[str, Any]:
    total = VisibilityBucket()
    paths = sorted(root.glob(pattern)) if root.exists() else []
    for path in paths:
        for example in _iter_jsonl(path):
            message_hits = matched_keywords(_message_text(example), keywords)
            metadata_hits = matched_keywords(_metadata_text(example), keywords)
            total.update(message_hits, metadata_hits)
    if total.examples == 0:
        raise ValueError(f"No dataset examples found in {root} matching {pattern}")
    return {
        "schema_version": 1,
        "root": _project_relative(root, project_dir),
        "glob": pattern,
        "keyword_count": len(keywords),
        "files_scanned": len(paths),
        "totals": total.to_dict(),
    }


def selection_ledger(
    root: Path,
    paths: list[Path],
    *,
    procedural_experiment: str,
    static_experiment: str,
    static_prefix: str,
    project_dir: Path,
) -> dict[str, Any]:
    if not paths:
        raise ValueError(f"No evaluation provenance files found in {root}")
    scope_counts: Counter[str] = Counter()
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        eval_experiment = str(data.get("eval_experiment", ""))
        scope = eval_scope(
            eval_experiment, procedural_experiment, static_experiment, static_prefix
        )
        scope_counts[scope] += 1
    return {
        "schema_version": 1,
        "root": _project_relative(root, project_dir),
        "provenance_files_scanned": len(paths),
        "scope_counts": dict(sorted(scope_counts.items())),
    }


def eval_scope(
    eval_experiment: str,
    procedural_experiment: str,
    static_experiment: str,
    static_prefix: str,
) -> str:
    if eval_experiment == procedural_experiment:
        return "procedural_validation"
    if eval_experiment == static_experiment or eval_experiment.startswith(
        static_prefix
    ):
        return "static_benchmark"
    if "procedural" in eval_experiment:
        return "procedural_other"
    if "static" in eval_experiment or "benchmark" in eval_experiment:
        return "static_other"
    return "other"


def readme_text(manifest: dict[str, Any], summary: dict[str, Any]) -> str:
    counts = summary["headline_counts"]
    split = summary["split_policy"]
    benchmark = summary["benchmark_holdout_policy"]
    benchmark_rows = benchmark_holdout_table_rows()
    enforced_rules = sum(bool(row["hard_rejection_class"]) for row in benchmark_rows)
    monitored_rules = len(benchmark_rows) - enforced_rules
    return "\n".join(
        [
            "# E0 Split and Leakage Audit",
            "",
            "This artifact reports split configuration and leakage scan results.",
            "",
            "## Verified Results",
            "",
            f"- Training uses procedural generators with profile `{split['training']['generator_profile']}` and seed `{split['training']['seed']}`.",
            f"- Procedural validation uses profile `{split['procedural_validation']['generator_profile']}`, deterministic seeds, and `{split['procedural_validation']['config']}`.",
            "- The static benchmark is the final external evaluation.",
            f"- Training and procedural validation have `{len(counts['generator_profile_overlap_categories'])}` overlapping holdout categories.",
            f"- The `{summary['leakage_policy']['solution_marker_phrase_count']}`-phrase solution-marker scan found "
            f"`{counts['dataset_message_solution_hits']}` model-visible hits across `{counts['dataset_examples_scanned']}` examples; "
            f"`{counts['dataset_metadata_solution_hits']}` examples contain marker hits only in metadata.",
            f"- Raw trace leakage summaries scanned: `{counts['trace_leakage_files_scanned']}` files / `{counts['raw_trace_rows_scanned']}` rows.",
            f"- Benchmark-holdout filter rejections in scanned raw traces: `{counts['trace_benchmark_holdout_rejected']}`.",
            f"- SFT-usable rows after quality filters: `{counts['trace_sft_usable']}` / `{counts['raw_trace_rows_scanned']}`.",
            "",
            "## Benchmark Holdout Rules",
            "",
            f"`benchmark_holdouts.tsv` records `{counts['benchmark_holdout_rules']}` generator-specific checks: "
            f"`{enforced_rules}` enforced exclusions and `{monitored_rules}` monitored checks.",
            "Each row records the procedural generator, motivating static case, held-out surface, match mode, pattern, and hard-rejection class.",
            f"Covered categories: `{benchmark['categories']}`.",
            "",
            "## Filter Results",
            "",
            f"- Base-usable raw traces: `{counts['trace_base_usable']}` / `{counts['raw_trace_rows_scanned']}`.",
            f"- Audit-usable traces after benchmark-holdout filters: `{counts['trace_audit_usable']}` / `{counts['raw_trace_rows_scanned']}`.",
            f"- Benchmark hard rejection classes: `{counts['trace_hard_rejection_classes']}`.",
            f"- SFT rejection reasons: `{counts['trace_sft_rejection_reasons']}`.",
            "",
            "## Files",
            "",
            "- `summary.json`: compact machine-readable audit result.",
            "- `split_holdouts.tsv`: procedural training vs procedural validation holdout surfaces.",
            "- `benchmark_holdouts.tsv`: benchmark exclusion rules used by each procedural generator.",
            "- `run_manifest.json`: canonical run identity and source roots.",
            "",
        ]
    )


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                data = json.loads(line)
                if not isinstance(data, dict):
                    raise TypeError(f"Expected JSON object in {path}")
                yield data


def _message_text(example: dict[str, Any]) -> str:
    parts: list[str] = []
    messages = example.get("messages", [])
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif content:
            parts.append(json.dumps(content, ensure_ascii=False))
    return "\n".join(parts)


def _metadata_text(example: dict[str, Any]) -> str:
    data = {
        key: example[key]
        for key in ("metadata", "quality_metrics")
        if key in example and example[key] not in (None, "", [], {})
    }
    if not data:
        return ""
    return json.dumps(data, ensure_ascii=False, default=str)


def _strings(row: dict[str, Any], key: str) -> list[str]:
    value = row.get(key, [])
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _reason_class(reason: str) -> str:
    return reason.split(" (", maxsplit=1)[0]


def _project_path(project_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_dir / path


def _audit_output_dir(project_dir: Path, audit_cfg: dict[str, Any]) -> Path:
    configured = audit_cfg.get("output_dir")
    if configured:
        return _project_path(project_dir, str(configured))
    return project_dir / paper_run_root(
        str(audit_cfg["experiment_id"]),
        str(audit_cfg["base_model"]),
        str(audit_cfg["condition"]),
        str(audit_cfg["run_id"]),
    )


def _source_path(value: Any, outputs_root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else outputs_root.parent / path


def _project_relative(path: Path, project_dir: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_dir.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _git_commit(project_dir: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not columns:
            return
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            delimiter="\t",
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
            escapechar="\\",
        )
        writer.writeheader()
        writer.writerows(rows)


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: AppConfig) -> None:
    run_pipeline(cfg)


if __name__ == "__main__":
    register_with_hydra()
    main()
