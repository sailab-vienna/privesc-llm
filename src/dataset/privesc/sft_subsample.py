#!/usr/bin/env python3
"""Create balanced SFT data-scaling subsets from an assembled dataset."""

import json
import shutil
import subprocess
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import get_original_cwd

from src.config import (
    AppConfig,
    SFTSubsampleConfig,
    plain_config_dict,
    register_with_hydra,
)
from src.dataset.privesc.sft_reporting import calculate_stats, write_json
from src.sft.traces import scenario_trace_paths


def expected_output_profile(n: int) -> str:
    return f"data_scaling_n{n:04d}"


def dataset_root(project_dir: Path, *, profile: str, subsample: SFTSubsampleConfig) -> Path:
    return (
        project_dir
        / "outputs"
        / "data"
        / "privesc_sft"
        / profile
        / subsample.regime
        / subsample.teacher
        / subsample.reasoning_variant
    )


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def read_jsonl(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
                if limit is not None and len(rows) == limit:
                    break
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def scenario_trace_files(training_root: Path) -> dict[str, Path]:
    if not training_root.is_dir():
        raise FileNotFoundError(f"Training dataset not found: {training_root}")

    paths = {path.parent.name: path for path in scenario_trace_paths(training_root, None)}
    if not paths:
        raise ValueError(f"No scenario traces found under {training_root}")
    return paths


def copy_metadata_files(source_split: Path, output_split: Path) -> None:
    for name in ("tools.json",):
        source = source_split / name
        if source.exists():
            output_split.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, output_split / name)


def git_commit(project_dir: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_dir,
        check=False,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    return commit or None


def validate_subset_request(subsample: SFTSubsampleConfig) -> tuple[int, str]:
    if subsample.n is None:
        raise ValueError("datasets.sft.subsample.n must be set")
    if subsample.output_profile is None:
        raise ValueError("datasets.sft.subsample.output_profile must be set")
    n = int(subsample.n)
    if n <= 0:
        raise ValueError("datasets.sft.subsample.n must be positive")
    expected = expected_output_profile(n)
    output_profile = subsample.output_profile
    if output_profile != expected:
        raise ValueError(
            "datasets.sft.subsample.output_profile must match "
            f"{expected!r} for n={n}; got {output_profile!r}"
        )
    return n, output_profile


def write_split_stats_from_rows(
    split_root: Path, rows_by_scenario: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    all_rows: list[dict[str, Any]] = []
    for scenario, rows in rows_by_scenario.items():
        write_json(str(split_root / scenario / "stats.json"), calculate_stats(rows))
        all_rows.extend(rows)
    stats = calculate_stats(all_rows)
    write_json(str(split_root / "stats.json"), stats)
    return stats


def derived_config_snapshot(
    *,
    config_snapshot: dict[str, Any] | None,
    n: int,
    output_profile: str,
    subsample: SFTSubsampleConfig,
    output_training: Path,
    output_validation: Path,
) -> dict[str, Any]:
    snapshot = deepcopy(config_snapshot) if config_snapshot is not None else {}
    datasets = snapshot.setdefault("datasets", {})
    sft = datasets.setdefault("sft", {})
    sft["profile"] = output_profile
    sft["regime"] = subsample.regime
    sft["teacher"] = subsample.teacher
    sft["reasoning_variant"] = subsample.reasoning_variant
    sft["output_dir"] = str(output_training)
    sft["validation_output_dir"] = str(output_validation)
    sft["subsample"] = {
        "n": n,
        "source_profile": subsample.source_profile,
        "output_profile": output_profile,
        "regime": subsample.regime,
        "teacher": subsample.teacher,
        "reasoning_variant": subsample.reasoning_variant,
    }
    return snapshot


def create_balanced_subset(
    *,
    project_dir: Path,
    subsample: SFTSubsampleConfig,
    config_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    n, output_profile = validate_subset_request(subsample)

    source_root = dataset_root(
        project_dir,
        profile=subsample.source_profile,
        subsample=subsample,
    )
    output_root = dataset_root(
        project_dir,
        profile=output_profile,
        subsample=subsample,
    )

    source_training = source_root / "training"
    source_validation = source_root / "validation"
    output_training = output_root / "training"
    output_validation = output_root / "validation"
    trace_files = scenario_trace_files(source_training)
    if not source_validation.is_dir():
        raise FileNotFoundError(f"Validation dataset not found: {source_validation}")
    validation_trace_files = scenario_trace_files(source_validation)
    if set(validation_trace_files) != set(trace_files):
        raise ValueError("Training and validation scenarios must match")
    validation_stats = read_json(source_validation / "stats.json")
    generator_count = len(trace_files)
    if n % generator_count != 0:
        raise ValueError(
            f"n={n} must be divisible by generator count {generator_count}"
        )
    per_generator = n // generator_count

    selected_by_generator: dict[str, list[dict[str, Any]]] = {}
    generator_counts: dict[str, int] = {}
    for scenario, trace_path in trace_files.items():
        rows = read_jsonl(trace_path, limit=per_generator)
        if len(rows) < per_generator:
            raise ValueError(
                f"{scenario} has only {len(rows)} traces; need {per_generator}"
            )
        selected_by_generator[scenario] = rows

    snapshot = derived_config_snapshot(
        config_snapshot=config_snapshot,
        n=n,
        output_profile=output_profile,
        subsample=subsample,
        output_training=output_training,
        output_validation=output_validation,
    )

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(project_dir),
        "source_dataset": str(source_root),
        "output_dataset": str(output_root),
        "source_profile": subsample.source_profile,
        "output_profile": output_profile,
        "regime": subsample.regime,
        "teacher": subsample.teacher,
        "reasoning_variant": subsample.reasoning_variant,
        "n": n,
        "generator_count": generator_count,
        "traces_per_generator": per_generator,
        "selection_rule": "first_k_rows_per_generator_in_existing_jsonl_order",
        "validation_rule": "copy_full_source_validation_split",
        "validation_num_runs": validation_stats.get("num_runs"),
    }

    output_root.mkdir(parents=True, exist_ok=False)
    try:
        for scenario, subset in selected_by_generator.items():
            write_jsonl(output_training / scenario / "traces.jsonl", subset)
            generator_counts[scenario] = len(subset)

        copy_metadata_files(source_training, output_training)
        write_json(str(output_training / "config_snapshot.json"), snapshot)
        training_stats = write_split_stats_from_rows(
            output_training, selected_by_generator
        )

        manifest["training_counts"] = generator_counts
        manifest["training_num_runs"] = training_stats.get("num_runs")
        shutil.copytree(source_validation, output_validation)
        write_json(str(output_validation / "config_snapshot.json"), snapshot)
        write_json(str(output_root / "subset_manifest.json"), manifest)
    except Exception:
        shutil.rmtree(output_root)
        raise
    return manifest


def create_subset_from_config(cfg: AppConfig, project_dir: Path | None = None) -> dict[str, Any]:
    subsample = SFTSubsampleConfig(
        **{
            **plain_config_dict(SFTSubsampleConfig()),
            **plain_config_dict(cfg.datasets.sft.subsample),
        }
    )

    config_snapshot = plain_config_dict(cfg, resolve=False)
    datasets_snapshot = config_snapshot.setdefault("datasets", {})
    sft_snapshot = datasets_snapshot.setdefault("sft", {})
    sft_snapshot["subsample"] = plain_config_dict(subsample)
    return create_balanced_subset(
        project_dir=project_dir or Path(get_original_cwd()),
        subsample=subsample,
        config_snapshot=config_snapshot,
    )


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: AppConfig) -> None:
    create_subset_from_config(cfg)


if __name__ == "__main__":
    register_with_hydra()
    main()
