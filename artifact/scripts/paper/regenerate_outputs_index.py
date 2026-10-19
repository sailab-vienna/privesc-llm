#!/usr/bin/env python3
"""Regenerate outputs/INDEX.md from paper run manifests."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SECTION_PLACEHOLDERS = {
    "00_split_and_leakage_audit": "00 - Split and leakage audit - placeholder, no runs yet.",
}

SECTION_ORDER = (
    "00_split_and_leakage_audit",
    "01_sft_hyperparam_sweep",
    "02_sft_trace_design",
    "03_reward_ladder",
    "04_static_benchmark",
    "05_prompt_sensitivity",
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def row_root_for_manifest(outputs_dir: Path, manifest_path: Path) -> Path | None:
    relative_parts = manifest_path.relative_to(outputs_dir).parts
    if (
        len(relative_parts) < 7
        or relative_parts[0] != "runs"
        or relative_parts[1] != "paper"
        or relative_parts[-1] != "run_manifest.json"
    ):
        return None
    return outputs_dir.joinpath(*relative_parts[:6])


def choose_row_manifest(row_root: Path, manifests: list[Path]) -> Path:
    root_manifest = row_root / "run_manifest.json"
    if root_manifest in manifests:
        return root_manifest
    return sorted(manifests, key=lambda path: (len(path.parts), path.as_posix()))[0]


def eval_kind(outputs_dir: Path, manifest_path: Path) -> str:
    relative_parts = manifest_path.relative_to(outputs_dir).parts
    path_text = manifest_path.as_posix()
    if "static" in relative_parts or "static_benchmark" in path_text:
        return "benchmark"
    if "procedural" in relative_parts:
        return "procedural"
    return "other"


def trace_counts(final_root: Path) -> tuple[int, int, dict[str, int]]:
    traces_root = final_root / "traces"
    valid = 0
    errors = 0
    by_scenario: Counter[str] = Counter()
    if not traces_root.is_dir():
        return valid, errors, {}

    for trace_file in traces_root.rglob("*.json"):
        name = trace_file.name
        if name.startswith("error_run_"):
            errors += 1
            continue
        valid += 1
        by_scenario[name.split("_202", 1)[0]] += 1
    return valid, errors, dict(sorted(by_scenario.items()))


def zero_eval_reason(row_root: Path) -> str:
    final_root = row_root / "eval" / "static" / "raw" / "final"
    valid, errors, by_scenario = trace_counts(final_root)
    if by_scenario:
        counts = list(by_scenario.values())
        return (
            "Static traces exist, but no eval manifest/stats were generated because "
            "valid non-error traces are incomplete for the paper protocol: "
            f"{valid} valid, {errors} error_run files, "
            f"{min(counts)}-{max(counts)} valid per scenario; requires 20 per scenario."
        )
    if final_root.is_dir():
        return "Eval directory exists, but it has no usable trace set or stats artifact to index."
    return "No eval artifact directory found under the run root."


def checkpoint_display(manifest: dict[str, Any]) -> str:
    checkpoint = manifest.get("lineage", {}).get("checkpoint_path_from_outputs")
    if not isinstance(checkpoint, str) or not checkpoint:
        return "`-`"
    parts = checkpoint.split("/")
    if len(parts) < 3:
        return f"`{checkpoint}`"
    return "`.../" + "/".join(parts[-3:]) + "`"


def git_display(manifest: dict[str, Any]) -> str:
    git_commit = manifest.get("git_commit")
    return f"`{str(git_commit)[:7]}`" if git_commit else "`-`"


def build_index(outputs_dir: Path) -> str:
    paper_dir = outputs_dir / "runs" / "paper"
    all_manifests = sorted(paper_dir.rglob("run_manifest.json"))

    manifests_by_row_root: dict[Path, list[Path]] = defaultdict(list)
    for manifest_path in all_manifests:
        row_root = row_root_for_manifest(outputs_dir, manifest_path)
        if row_root is not None:
            manifests_by_row_root[row_root].append(manifest_path)

    row_roots = sorted(manifests_by_row_root)
    row_manifest_by_root = {
        row_root: choose_row_manifest(row_root, manifests)
        for row_root, manifests in manifests_by_row_root.items()
    }
    eval_manifests = [
        path for path in all_manifests if "eval" in path.relative_to(outputs_dir).parts
    ]

    external_by_checkpoint: dict[str, list[Path]] = defaultdict(list)
    for manifest_path in eval_manifests:
        manifest = read_json(manifest_path)
        checkpoint = manifest.get("lineage", {}).get("checkpoint_path_from_outputs")
        if isinstance(checkpoint, str) and checkpoint:
            external_by_checkpoint[checkpoint].append(manifest_path)

    sections: dict[str, list[dict[str, Any]]] = defaultdict(list)
    zero_notes: list[tuple[str, str, str, str, str]] = []

    for row_root in row_roots:
        manifest_path = row_manifest_by_root[row_root]
        manifest = read_json(manifest_path)
        relative_parts = row_root.relative_to(outputs_dir).parts
        experiment, subject, condition, run_id = relative_parts[2:6]

        row_root_from_outputs = row_root.relative_to(outputs_dir).as_posix()
        checkpoint_prefixes = [f"{row_root_from_outputs}/"]
        native_root = manifest.get("paths", {}).get(
            "native_artifact_root_path_from_outputs"
        )
        if isinstance(native_root, str) and native_root:
            checkpoint_prefixes.append(f"{native_root.rstrip('/')}/")

        procedural = 0
        benchmark = 0
        counted: set[Path] = set()
        for eval_manifest in eval_manifests:
            if eval_manifest in counted:
                continue
            if not eval_manifest.is_relative_to(row_root / "eval"):
                continue
            kind = eval_kind(outputs_dir, eval_manifest)
            procedural += int(kind == "procedural")
            benchmark += int(kind == "benchmark")
            counted.add(eval_manifest)

        for checkpoint, manifests in external_by_checkpoint.items():
            if not any(checkpoint.startswith(prefix) for prefix in checkpoint_prefixes):
                continue
            for eval_manifest in manifests:
                if eval_manifest in counted:
                    continue
                kind = eval_kind(outputs_dir, eval_manifest)
                procedural += int(kind == "procedural")
                benchmark += int(kind == "benchmark")
                counted.add(eval_manifest)

        base_model = manifest.get("base_model", {})
        base_slug = base_model.get("slug") if isinstance(base_model, dict) else None
        stage = manifest.get("condition", {}).get("stage") or "-"
        created_at = str(manifest.get("created_at_utc", ""))[:10] or "-"

        if procedural == 0 and benchmark == 0:
            zero_notes.append(
                (experiment, subject, condition, run_id, zero_eval_reason(row_root))
            )

        sections[experiment].append(
            {
                "subject": base_slug or subject,
                "condition": condition,
                "run_id": run_id,
                "stage": stage,
                "created": created_at,
                "git": git_display(manifest),
                "procedural": procedural,
                "benchmark": benchmark,
                "checkpoint": checkpoint_display(manifest),
            }
        )

    lines = [
        "# outputs/ index",
        "",
        "Generated from `run_manifest.json` files under `outputs/runs/paper/`. "
        "One row per top-level run root.",
        "",
        "Layout: `runs/paper/<experiment>/<subject>/<condition>/<run_id>/`",
        "",
        "Eval counts are split by eval split: `Procedural evals` counts eval "
        "manifests under `eval/procedural/`; `Benchmark evals` counts static "
        "benchmark and prompt-sensitivity eval manifests under `eval/static/`. "
        "Counts include both evals nested under the run root and external "
        "eval/analysis manifests whose `lineage.checkpoint_path_from_outputs` "
        "points back into the run.",
        "",
    ]

    for experiment in SECTION_ORDER:
        lines.extend([f"## {experiment}", ""])
        rows = sections.get(experiment, [])
        if not rows:
            lines.extend([f"_{SECTION_PLACEHOLDERS[experiment]}_", ""])
            continue

        lines.extend(
            [
                "| Subject | Condition | Run ID | Stage | Created | Git | "
                "Procedural evals | Benchmark evals | Checkpoint lineage |",
                "|---|---|---|---|---|---|---:|---:|---|",
            ]
        )
        for row in rows:
            lines.append(
                f"| `{row['subject']}` | `{row['condition']}` | `{row['run_id']}` | "
                f"{row['stage']} | {row['created']} | {row['git']} | "
                f"{row['procedural']} | {row['benchmark']} | {row['checkpoint']} |"
            )
        lines.extend(["", f"Total: {len(rows)} run roots.", ""])

    if zero_notes:
        lines.extend(
            [
                "## Zero-eval notes",
                "",
                "| Experiment | Subject | Condition | Run ID | Reason |",
                "|---|---|---|---|---|",
            ]
        )
        for experiment, subject, condition, run_id, reason in zero_notes:
            lines.append(
                f"| `{experiment}` | `{subject}` | `{condition}` | `{run_id}` | {reason} |"
            )
        lines.append("")

    lines.extend(
        [
            "---",
            "",
            f"Totals: {len(row_roots)} run roots, {len(all_manifests)} "
            "`run_manifest.json` files.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--index-path", type=Path, default=None)
    args = parser.parse_args()

    outputs_dir = args.outputs_dir.resolve()
    index_path = args.index_path or outputs_dir / "INDEX.md"
    index_path.write_text(build_index(outputs_dir), encoding="utf-8")


if __name__ == "__main__":
    main()
