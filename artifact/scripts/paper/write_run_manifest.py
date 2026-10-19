#!/usr/bin/env python3
"""Write a minimal paper run manifest."""

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.paths import base_model_slug, path_from_outputs_or_none


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


def build_run_manifest_payload(
    *,
    project_dir: Path,
    experiment_id: str,
    base_model: str,
    condition: str,
    stage: str,
    run_root: Path,
    native_artifact_root: Path,
    train_dataset: Path | None = None,
    validation_dataset: Path | None = None,
    checkpoint: Path | None = None,
    lineage_extra: dict | None = None,
) -> dict:
    derived_base_model_slug = base_model_slug(base_model)
    lineage = {
        "checkpoint_path_from_outputs": path_from_outputs_or_none(checkpoint),
    }
    if lineage_extra:
        lineage.update(lineage_extra)
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "experiment_id": experiment_id,
        "base_model": {
            "id": base_model,
            "slug": derived_base_model_slug,
        },
        "condition": {
            "slug": condition,
            "stage": stage,
        },
        "git_commit": git_commit(project_dir),
        "paths": {
            "run_root_path_from_outputs": path_from_outputs_or_none(run_root),
            "native_artifact_root_path_from_outputs": path_from_outputs_or_none(
                native_artifact_root
            ),
        },
        "source_data": {
            "train_dataset_path_from_outputs": path_from_outputs_or_none(
                train_dataset
            ),
            "validation_dataset_path_from_outputs": path_from_outputs_or_none(
                validation_dataset
            ),
        },
        "lineage": lineage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--base-model-slug")
    parser.add_argument("--condition", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--native-artifact-root", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path)
    parser.add_argument("--validation-dataset", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    derived_base_model_slug = base_model_slug(args.base_model)
    if args.base_model_slug and args.base_model_slug != derived_base_model_slug:
        raise ValueError(
            f"base model slug mismatch: got {args.base_model_slug!r}, expected {derived_base_model_slug!r}"
        )

    args.run_root.mkdir(parents=True, exist_ok=True)
    payload = build_run_manifest_payload(
        project_dir=args.project_dir,
        experiment_id=args.experiment_id,
        base_model=args.base_model,
        condition=args.condition,
        stage=args.stage,
        run_root=args.run_root,
        native_artifact_root=args.native_artifact_root,
        train_dataset=args.train_dataset,
        validation_dataset=args.validation_dataset,
        checkpoint=args.checkpoint,
    )
    (args.run_root / "run_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
