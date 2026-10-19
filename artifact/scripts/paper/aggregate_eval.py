#!/usr/bin/env python3
"""Run paper analysis over multiple existing eval trace directories.

The script creates a temporary ``traces/<model>`` view copied from source trace
directories, runs ``src.evaluation.analyze``, then copies only derived ``stats/``
and ``plots/`` outputs into the requested aggregate artifact root.
For canonical paper outputs, put ``--output-dir`` under
``outputs/runs/paper/.../eval/<split>/aggregate/<name>`` and use
``--manifest-root`` for the enclosing paper run root.
"""

import argparse
import json
import random
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.paper.write_run_manifest import build_run_manifest_payload
from src.paths import assert_no_path_overlap, path_from_outputs_or_none
from src.evaluation.analysis.trace_payload import (
    trace_generator_name,
    trace_item_run_ordinal,
    trace_scenario_name,
)

DEFAULT_EXPERIMENT = "eval/paper_static_qwen3_4b_base"
DEFAULT_EXPERIMENT_ID = "04_static_benchmark"
DEFAULT_BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DownsampleGroup = Literal["scenario", "generator"]


@dataclass(frozen=True)
class TraceCandidate:
    path: Path
    group: str
    item_run_ordinal: int | None


@dataclass(frozen=True)
class DownsampleSummary:
    model_tag: str
    source_trace_dir: Path
    group_by: DownsampleGroup
    runs_per_group: int
    random_seed: int | None
    selected_total: int
    skipped_ineligible: int
    available_counts: dict[str, int]
    selected_counts: dict[str, int]


def read_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else None


def positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def trace_group_key(payload: dict, group_by: DownsampleGroup, path: Path) -> str:
    if group_by == "generator":
        value = trace_generator_name(payload)
    else:
        value = trace_scenario_name(payload)
    if value is None:
        raise ValueError(f"trace missing {group_by} grouping key: {path}")
    return value


def link_protocol_trace_view(
    *,
    model_tag: str,
    source_trace_dir: Path,
    target_trace_dir: Path,
    group_by: DownsampleGroup,
    runs_per_group: int,
    rng: random.Random | None,
    random_seed: int | None,
) -> DownsampleSummary:
    from src.runner.result_policy import is_benchmark_eligible_payload

    candidates: list[TraceCandidate] = []
    skipped_ineligible = 0
    for trace_file in sorted(source_trace_dir.glob("*.json")):
        payload = read_json_if_exists(trace_file)
        if payload is None:
            raise ValueError(f"trace file must contain a JSON object: {trace_file}")
        if not is_benchmark_eligible_payload(payload):
            skipped_ineligible += 1
            continue
        candidates.append(
            TraceCandidate(
                path=trace_file,
                group=trace_group_key(payload, group_by, trace_file),
                item_run_ordinal=trace_item_run_ordinal(payload),
            )
        )

    grouped: dict[str, list[TraceCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.group, []).append(candidate)

    target_trace_dir.mkdir(parents=True, exist_ok=True)
    available_counts: dict[str, int] = {}
    selected_counts: dict[str, int] = {}
    selected_total = 0
    for group in sorted(grouped):
        group_candidates = sorted(
            grouped[group],
            key=lambda candidate: (
                candidate.item_run_ordinal is None,
                (
                    candidate.item_run_ordinal
                    if candidate.item_run_ordinal is not None
                    else 0
                ),
                candidate.path.name,
            ),
        )
        if rng is None:
            selected = group_candidates[:runs_per_group]
        else:
            selected = rng.sample(
                group_candidates,
                min(runs_per_group, len(group_candidates)),
            )
        available_counts[group] = len(group_candidates)
        selected_counts[group] = len(selected)
        selected_total += len(selected)
        for candidate in selected:
            shutil.copy2(candidate.path, target_trace_dir / candidate.path.name)

    shortfalls = {
        group: count
        for group, count in selected_counts.items()
        if count < runs_per_group
    }
    if shortfalls:
        print(
            "[WARN] Protocol trace view has groups below requested runs per "
            f"{group_by}: {shortfalls}"
        )
    if skipped_ineligible:
        print(
            f"[INFO] Skipped {skipped_ineligible} ineligible trace(s) while "
            f"building protocol trace view for {model_tag}"
        )

    return DownsampleSummary(
        model_tag=model_tag,
        source_trace_dir=source_trace_dir,
        group_by=group_by,
        runs_per_group=runs_per_group,
        random_seed=random_seed,
        selected_total=selected_total,
        skipped_ineligible=skipped_ineligible,
        available_counts=available_counts,
        selected_counts=selected_counts,
    )


def parse_eval_root(raw: str) -> tuple[str, Path, Path]:
    eval_root = Path(raw).expanduser().resolve(strict=True)
    traces_root = eval_root / "traces"
    if traces_root.is_dir():
        trace_dirs = [path for path in traces_root.iterdir() if path.is_dir()]
        if len(trace_dirs) != 1:
            raise ValueError(
                "eval root must contain exactly one traces/<model> directory: "
                f"{eval_root}"
            )
        return trace_dirs[0].name, trace_dirs[0], eval_root
    if any(eval_root.glob("*.json")):
        return eval_root.name, eval_root, eval_root
    raise ValueError(
        f"eval root contains neither traces/<model> nor trace JSON files: {eval_root}"
    )


def infer_common_value(name: str, values: list[str | None], default: str) -> str:
    present_values = sorted({value for value in values if value})
    if not present_values:
        return default
    if len(present_values) > 1:
        raise ValueError(
            f"eval roots disagree on {name}; found: {', '.join(present_values)}"
        )
    return present_values[0]


def infer_eval_metadata(
    eval_roots: list[Path],
    *,
    infer_experiment: bool = True,
    infer_experiment_id: bool = True,
    infer_base_model: bool = True,
) -> tuple[str, str, str]:
    experiments: list[str | None] = []
    experiment_ids: list[str | None] = []
    base_models: list[str | None] = []
    for eval_root in eval_roots:
        provenance = read_json_if_exists(eval_root / "provenance_model_eval.json")
        run_manifest = read_json_if_exists(eval_root / "run_manifest.json")
        if infer_experiment:
            experiments.append(
                str(provenance.get("eval_experiment"))
                if isinstance(provenance, dict) and provenance.get("eval_experiment")
                else None
            )
        if infer_experiment_id:
            experiment_ids.append(
                str(run_manifest.get("experiment_id"))
                if isinstance(run_manifest, dict) and run_manifest.get("experiment_id")
                else None
            )
        if infer_base_model:
            base_model = (
                run_manifest.get("base_model")
                if isinstance(run_manifest, dict)
                else None
            )
            if isinstance(base_model, dict) and base_model.get("id"):
                base_models.append(str(base_model["id"]))
            elif isinstance(provenance, dict) and provenance.get("model_ref"):
                base_models.append(str(provenance["model_ref"]))
            else:
                base_models.append(None)
    return (
        infer_common_value("experiment", experiments, DEFAULT_EXPERIMENT)
        if infer_experiment
        else DEFAULT_EXPERIMENT,
        infer_common_value("experiment_id", experiment_ids, DEFAULT_EXPERIMENT_ID)
        if infer_experiment_id
        else DEFAULT_EXPERIMENT_ID,
        infer_common_value("base_model", base_models, DEFAULT_BASE_MODEL)
        if infer_base_model
        else DEFAULT_BASE_MODEL,
    )


def copy_output_dir(source: Path, target: Path) -> None:
    assert_no_path_overlap(source, target)
    parent = target.parent
    with tempfile.TemporaryDirectory(prefix=f".{target.name}.", dir=parent) as tmp:
        tmp_target = Path(tmp) / target.name
        shutil.copytree(source, tmp_target)
        if target.exists():
            shutil.rmtree(target)
        tmp_target.rename(target)


def validate_canonical_aggregate_output(manifest_root: Path, output_dir: Path) -> None:
    try:
        relative = output_dir.relative_to(manifest_root)
    except ValueError as exc:
        raise ValueError(
            "--output-dir must be under --manifest-root when --manifest-root is set"
        ) from exc

    if (
        len(relative.parts) < 4
        or relative.parts[0] != "eval"
        or relative.parts[2] != "aggregate"
    ):
        raise ValueError(
            "--output-dir must be under --manifest-root/eval/<split>/aggregate/<name> "
            "when --manifest-root is set"
        )


def write_manifests(
    *,
    manifest_root: Path,
    output_dir: Path,
    project_dir: Path,
    experiment_id: str,
    experiment: str,
    base_model: str,
    condition: str,
    sources: list[tuple[str, Path]],
    command: list[str],
    downsample_summaries: list[DownsampleSummary] | None = None,
) -> None:
    created_at = datetime.now(UTC).isoformat()
    source_payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": created_at,
        "experiment": experiment,
        "sources": [
            {
                "model_tag": model_tag,
                "trace_dir": str(trace_dir),
                "trace_dir_path_from_outputs": path_from_outputs_or_none(trace_dir),
            }
            for model_tag, trace_dir in sources
        ],
        "analysis_command": command,
    }
    if downsample_summaries:
        source_payload["downsample"] = [
            {
                "model_tag": summary.model_tag,
                "source_trace_dir": str(summary.source_trace_dir),
                "source_trace_dir_path_from_outputs": path_from_outputs_or_none(
                    summary.source_trace_dir
                ),
                "group_by": summary.group_by,
                "runs_per_group": summary.runs_per_group,
                "random_seed": summary.random_seed,
                "selected_total": summary.selected_total,
                "skipped_ineligible": summary.skipped_ineligible,
                "available_counts": summary.available_counts,
                "selected_counts": summary.selected_counts,
            }
            for summary in downsample_summaries
        ]
    manifest_root.mkdir(parents=True, exist_ok=True)
    (manifest_root / "source_manifest.json").write_text(
        json.dumps(source_payload, indent=2) + "\n", encoding="utf-8"
    )

    lineage_extra: dict[str, Any] = {
        "source_trace_dirs_path_from_outputs": [
            path_from_outputs_or_none(trace_dir) for _model_tag, trace_dir in sources
        ],
    }
    if downsample_summaries:
        lineage_extra["downsample"] = [
            {
                "model_tag": summary.model_tag,
                "group_by": summary.group_by,
                "runs_per_group": summary.runs_per_group,
                "random_seed": summary.random_seed,
                "selected_total": summary.selected_total,
            }
            for summary in downsample_summaries
        ]

    run_payload = build_run_manifest_payload(
        project_dir=project_dir,
        experiment_id=experiment_id,
        base_model=base_model,
        condition=condition,
        stage="analysis",
        run_root=manifest_root,
        native_artifact_root=output_dir,
        lineage_extra=lineage_extra,
    )
    run_payload["created_at_utc"] = created_at
    run_manifest_text = json.dumps(run_payload, indent=2) + "\n"
    (manifest_root / "run_manifest.json").write_text(
        run_manifest_text, encoding="utf-8"
    )
    if output_dir != manifest_root:
        (output_dir / "run_manifest.json").write_text(
            run_manifest_text, encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Aggregate artifact root. Canonical paper outputs should use "
            "outputs/runs/paper/.../eval/<split>/aggregate/<name>."
        ),
    )
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=None,
        help=(
            "Optional canonical paper run root for run_manifest.json and "
            "source_manifest.json. When set, --output-dir must be under "
            "<manifest-root>/eval/<split>/aggregate/<name>. Manifests are not "
            "written when omitted."
        ),
    )
    parser.add_argument(
        "--condition",
        default=None,
        help=(
            "Condition slug for the run manifest. Defaults to the manifest root "
            "parent directory name when --manifest-root is set, otherwise the "
            "output directory name."
        ),
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="Override inferred analysis experiment for mixed-source aggregates.",
    )
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Override inferred paper experiment id for mixed-source aggregates.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Override inferred base model id for mixed-source aggregates.",
    )
    parser.add_argument(
        "--eval-root",
        action="append",
        required=True,
        help=(
            "Eval root containing exactly one traces/<model> directory, or a "
            "directory containing trace JSON files directly."
        ),
    )
    downsample_group = parser.add_mutually_exclusive_group()
    downsample_group.add_argument(
        "--runs-per-scenario",
        type=positive_int,
        default=None,
        help=(
            "Build a temporary analysis view with at most this many eligible "
            "runs per static scenario for each model."
        ),
    )
    downsample_group.add_argument(
        "--runs-per-generator",
        type=positive_int,
        default=None,
        help=(
            "Build a temporary analysis view with at most this many eligible "
            "runs per procedural generator for each model."
        ),
    )
    parser.add_argument(
        "--allow-incomplete-runs",
        action="store_true",
        help="Pass through to src.evaluation.analyze.",
    )
    parser.add_argument(
        "--downsample-random-seed",
        type=int,
        default=None,
        help=(
            "Use one seeded RNG across all sources to sample runs across "
            "sorted downsample groups. Without this, groups use the lowest "
            "item_run_ordinal traces."
        ),
    )
    parser.add_argument(
        "--expected-runs-per-scenario",
        type=int,
        default=None,
        help="Pass through to src.evaluation.analyze.",
    )
    parser.add_argument(
        "--show-titles",
        action="store_true",
        help="Pass through to src.evaluation.analyze.",
    )
    args = parser.parse_args()

    project_dir = Path.cwd().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    manifest_root = (
        args.manifest_root.expanduser().resolve(strict=False)
        if args.manifest_root is not None
        else None
    )
    if manifest_root is not None:
        validate_canonical_aggregate_output(manifest_root, output_dir)
    eval_sources = [parse_eval_root(raw) for raw in args.eval_root]
    eval_roots = [eval_root for _model_tag, _trace_dir, eval_root in eval_sources]
    sources = [
        (model_tag, trace_dir) for model_tag, trace_dir, _eval_root in eval_sources
    ]
    condition = args.condition or (
        manifest_root.parent.name if manifest_root is not None else output_dir.name
    )
    if args.experiment and args.experiment_id and args.base_model:
        experiment = str(args.experiment)
        experiment_id = str(args.experiment_id)
        base_model = str(args.base_model)
    else:
        inferred_experiment, inferred_experiment_id, inferred_base_model = (
            infer_eval_metadata(
                eval_roots,
                infer_experiment=args.experiment is None,
                infer_experiment_id=args.experiment_id is None,
                infer_base_model=args.base_model is None,
            )
        )
        experiment = args.experiment or inferred_experiment
        experiment_id = args.experiment_id or inferred_experiment_id
        base_model = args.base_model or inferred_base_model
    downsample_by: DownsampleGroup | None = None
    downsample_runs: int | None = None
    if args.runs_per_scenario is not None:
        downsample_by = "scenario"
        downsample_runs = args.runs_per_scenario
    elif args.runs_per_generator is not None:
        downsample_by = "generator"
        downsample_runs = args.runs_per_generator
    expected_runs_per_scenario = args.expected_runs_per_scenario
    if downsample_runs is not None and expected_runs_per_scenario is None:
        expected_runs_per_scenario = downsample_runs

    output_dir.mkdir(parents=True, exist_ok=True)
    downsample_summaries: list[DownsampleSummary] = []
    downsample_rng = (
        random.Random(args.downsample_random_seed)
        if args.downsample_random_seed is not None
        else None
    )
    with tempfile.TemporaryDirectory(prefix="privesc_aggregate_analysis_") as tmp:
        tmp_dir = Path(tmp)
        tmp_traces = tmp_dir / "traces"
        tmp_traces.mkdir()
        for model_tag, trace_dir in sources:
            target_trace_dir = tmp_traces / model_tag
            if downsample_by is None or downsample_runs is None:
                shutil.copytree(trace_dir, target_trace_dir)
            else:
                downsample_summaries.append(
                    link_protocol_trace_view(
                        model_tag=model_tag,
                        source_trace_dir=trace_dir,
                        target_trace_dir=target_trace_dir,
                        group_by=downsample_by,
                        runs_per_group=downsample_runs,
                        rng=downsample_rng,
                        random_seed=args.downsample_random_seed,
                    )
                )

        analyze_cmd = [
            sys.executable,
            "-m",
            "src.evaluation.analyze",
            "--base-dir",
            str(tmp_dir),
            "--experiment",
            experiment,
            "--all-models",
        ]
        if args.allow_incomplete_runs:
            analyze_cmd.append("--allow-incomplete-runs")
        if expected_runs_per_scenario is not None:
            analyze_cmd.extend(
                ["--expected-runs-per-scenario", str(expected_runs_per_scenario)]
            )
        if args.show_titles:
            analyze_cmd.append("--show-titles")
        subprocess.run(analyze_cmd, cwd=project_dir, check=True)
        copy_output_dir(tmp_dir / "stats", output_dir / "stats")
        copy_output_dir(tmp_dir / "plots", output_dir / "plots")

    if manifest_root is not None:
        write_manifests(
            manifest_root=manifest_root,
            output_dir=output_dir,
            project_dir=project_dir,
            experiment_id=experiment_id,
            experiment=experiment,
            base_model=base_model,
            condition=condition,
            sources=sources,
            command=[sys.executable, *sys.argv],
            downsample_summaries=downsample_summaries,
        )
    print(f"[INFO] Aggregate analysis written to {output_dir}")


if __name__ == "__main__":
    main()
