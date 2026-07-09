"""Shared path utilities for consistent directory naming across the project."""

import os
import re
from pathlib import Path

# Canonical bases for results, relative to project root.
OUTPUTS_ROOT = "outputs"
TRACE_COLLECTION_BASE = os.path.join("outputs", "traces", "trace_collection")
EVALUATION_BASE = os.path.join("outputs", "evals", "evaluation")
PAPER_RUNS_BASE = os.path.join("outputs", "runs", "paper")
PAPER_ANALYSIS_BASE = os.path.join("outputs", "analysis", "paper")
PAPER_INDEX_BASE = os.path.join("outputs", "paper")


def path_slug(value: str) -> str:
    """Return a stable, readable slug for path segments."""
    slug = value.strip().lower()
    slug = re.sub(r"[^a-z0-9._+-]+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-._")
    if not slug:
        raise ValueError(f"Cannot derive path slug from {value!r}")
    return slug


def model_dir_name(model: str) -> str:
    """Derive the filesystem-safe directory name from a model identifier.

    Strips the provider prefix (e.g. "deepseek/") and any colon-delimited
    variant suffix (e.g. ":free") so that "deepseek/deepseek-v4-flash:free"
    becomes "deepseek-v4-flash".
    """
    return model.split("/")[-1].split(":", 1)[0]


def base_model_slug(model: str) -> str:
    """Derive the paper run path slug for a pretrained base model."""
    return path_slug(model_dir_name(model))


def paper_run_root(
    experiment_id: str,
    base_model: str,
    condition: str,
    run_id: str,
) -> Path:
    """Build the canonical paper run root relative path."""
    return paper_run_root_from_slug(
        experiment_id, base_model_slug(base_model), condition, run_id
    )


def paper_run_root_from_slug(
    experiment_id: str,
    base_model_slug: str,
    condition: str,
    run_id: str,
) -> Path:
    """Build the canonical paper run root relative path from an explicit slug."""
    return Path(PAPER_RUNS_BASE) / experiment_id / base_model_slug / condition / run_id


def path_from_outputs(path: str | Path, outputs_root: str | Path = OUTPUTS_ROOT) -> str:
    """Return a POSIX path relative to outputs/ for manifests."""
    raw_path = Path(path)
    root = Path(outputs_root)
    default_outputs_root = root == Path(OUTPUTS_ROOT)
    if raw_path.is_absolute():
        raw_path = raw_path.resolve()
        root = root.resolve()
    try:
        return raw_path.relative_to(root).as_posix()
    except ValueError:
        parts = raw_path.parts
        if default_outputs_root and OUTPUTS_ROOT in parts:
            idx = parts.index(OUTPUTS_ROOT)
            return Path(*parts[idx + 1 :]).as_posix()
        raise ValueError(f"Path is not under {outputs_root}: {path}")


def path_from_outputs_or_none(
    path: str | Path | None, outputs_root: str | Path = OUTPUTS_ROOT
) -> str | None:
    """Return a path relative to outputs/, or None for missing/outside paths."""
    if path is None:
        return None
    try:
        return path_from_outputs(path, outputs_root)
    except ValueError:
        return None


def assert_no_path_overlap(source: str | Path, target: str | Path) -> None:
    """Reject identical or nested source/target paths before destructive writes."""
    source_path = Path(source).resolve()
    target_path = Path(target).resolve()
    if source_path == target_path:
        raise ValueError(f"source and target paths must differ: {source_path}")
    if source_path in target_path.parents:
        raise ValueError(f"target path must not be inside source path: {target_path}")
    if target_path in source_path.parents:
        raise ValueError(f"source path must not be inside target path: {source_path}")


def traces_dir(output_base: str, model: str) -> str:
    """Build the traces directory path for a given model under an output base.

    Used by the runner to determine where to write trace JSON files and by
    the schedule builder to count existing traces for resume support.
    """
    return os.path.join(output_base, "traces", model_dir_name(model))


def resolve_traces_dir(
    base_dir: Path,
    model: str,
    split: str,
    trace_root: str | None = None,
) -> Path | None:
    """Resolve the traces directory for a given split, returning None if missing.

    Used by the SFT dataset assembler to locate raw trace files for a
    specific training/validation split.
    """
    root = Path(trace_root) if trace_root else Path(TRACE_COLLECTION_BASE)
    if not root.is_absolute():
        root = base_dir / root
    path = root / split / "traces" / model_dir_name(model)
    return path if path.exists() else None
