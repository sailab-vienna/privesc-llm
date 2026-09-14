#!/usr/bin/env python3
"""Download and verify the released ACSAC evaluation evidence."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from src.evaluation.analysis.constants import format_model_name

EVAL_REPO = "sailab-vienna/privesc-llm-evals"
EVAL_REVISION = "5dba67dfd4c16da78a720c16f01e93820cb3471c"
DATA_REPO = "sailab-vienna/privesc-llm-data"
DATA_REVISION = "39a9d2ff37222184fcaf2a6d9a124906e8bb49ad"

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

console = Console(
    highlight=False,
    soft_wrap=True,
    color_system=None if "NO_COLOR" in os.environ else "auto",
)
TOTAL_STAGES = 5

HEADLINE_ROOTS = (
    "headline_static_traces/claude-opus-4.7/base/20260418/eval/static/raw/final",
    "headline_static_traces/deepseek-v3.2/base/20260418/eval/static/raw/final",
    "headline_static_traces/gemma4-31b/base/20260418/eval/static/raw/final",
    "headline_static_traces/qwen3-4b-instruct-2507/base/20260519T075017Z/eval/static/raw/final",
    "headline_static_traces/qwen3-4b-instruct-2507/sft_sd2026_outcome_cost_step300/20260519T163319Z/eval/static",
    "headline_static_traces/qwen3-4b-instruct-2507/sft_unguided_phase_c_lr1p5e-4_r8_sd2026/20260517T013021Z/eval/static",
)
REBUTTAL_ROOTS = (
    "rebuttal_static_traces/qwen3-8b-thinking",
    "rebuttal_static_traces/qwen3-8b-nonthinking",
    "rebuttal_static_traces/qwen3-14b-fp8-thinking",
    "rebuttal_static_traces/qwen3-14b-fp8-nonthinking",
    "rebuttal_static_traces/llama-3.2-3b-instruct",
    "rebuttal_static_traces/deepseek-v4-flash",
)
HEADLINE_SUMMARY = "summaries/runs/paper/04_static_benchmark/comparison/headline/20260524/eval/static/aggregate/final/stats/evaluation_summary_paper.json"
REBUTTAL_SUMMARY = "summaries/runs/paper/04_static_benchmark/comparison/rebuttal/20260823/eval/static/aggregate/final/stats/evaluation_summary_paper.json"
SFT_INIT_SUMMARY = "summaries/runs/paper/01_sft_hyperparam_sweep/qwen3-4b-instruct-2507/unguided_deepseek_long_reasoning/20260515T110310Z_phase_c_lr1p5e-4_r8_ep10_sd2026/eval/procedural/stats/evaluation_summary_paper.json"


def stage(number: int, message: str) -> None:
    console.print()
    console.print(f"[bold cyan][{number}/{TOTAL_STAGES}][/bold cyan] {message}")


def fail(message: str, details: str = "") -> None:
    console.print(f"[bold red]FAIL:[/bold red] {message}")
    if details:
        console.print(details, markup=False)
    raise SystemExit(1)


def run_checked(command: list[str], project: Path, label: str) -> str:
    result = subprocess.run(
        command,
        cwd=project,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode:
        fail(f"{label} failed.", result.stdout.rstrip())
    return result.stdout.strip()


def aggregate(
    project: Path, evidence: Path, output: Path, roots: tuple[str, ...]
) -> None:
    command = [
        sys.executable,
        "scripts/paper/aggregate_eval.py",
        "--output-dir",
        str(output),
        "--expected-runs-per-scenario",
        "10",
    ]
    for root in roots:
        command.extend(("--eval-root", str(evidence / root)))
    run_checked(command, project, f"{output.name} trace reaggregation")


def assert_same_summary(actual: Path, expected: Path, label: str) -> dict:
    payload = json.loads(actual.read_text())
    try:
        pd.testing.assert_frame_equal(
            pd.json_normalize(payload).sort_index(axis=1),
            pd.json_normalize(json.loads(expected.read_text())).sort_index(axis=1),
            check_exact=False,
            rtol=1e-10,
            atol=1e-10,
        )
    except AssertionError as error:
        fail(f"{label} aggregate differs from the released summary.", str(error))
    return payload


def print_successes(label: str, summary: dict) -> None:
    console.print(f"[bold]{label}[/bold]")
    for model, stats in summary["model_stats"].items():
        runs = stats["num_runs"]
        rates = [stats[f"sr_within_rounds_{budget}"] for budget in (10, 20, 60)]
        successes = [round(rate * runs) for rate in rates]
        name = " ".join(format_model_name(model).splitlines())
        console.print(
            f"  {name}: N={runs}; r10/r20/r60="
            f"{'/'.join(map(str, successes))} "
            f"({'/'.join(f'{rate:.1%}' for rate in rates)})",
            markup=False,
        )


def verify_split_audit(data: Path) -> None:
    summary = json.loads((data / "leakage_audit/summary.json").read_text())
    counts = summary["headline_counts"]
    verified = (
        counts["generator_profiles_disjoint"] is True
        and counts["generator_profile_overlap_categories"] == []
        and counts["dataset_examples_scanned"] == 13200
        and counts["dataset_message_solution_hits"] == 0
        and counts["dataset_metadata_solution_hits"] == 2040
        and summary["leakage_policy"]["solution_marker_phrase_count"] == 14
        and counts["benchmark_holdout_rules"] == 86
        and summary["split_policy"]["static_benchmark"]["tuning_allowed"] is False
    )
    rules = pd.read_csv(data / "leakage_audit/benchmark_holdouts.tsv", sep="\t")
    rule_counts = (
        len(rules),
        int(rules["hard_rejection_class"].notna().sum()),
        int(rules["hard_rejection_class"].isna().sum()),
    )
    example_count = len(list((data / "examples").glob("*.txt")))
    if not verified or rule_counts != (86, 79, 7) or example_count != 12:
        fail("Split/leakage evidence differs from the expected audit package.")
    console.print(
        "Split audit: profiles disjoint; zero model-visible solution-marker hits "
        "across 13,200 examples.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evals-dir",
        type=Path,
        help="Use a local evaluation-evidence snapshot.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Use a local data-artifact snapshot.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ae/archived"))
    args = parser.parse_args()
    if args.evals_dir and not args.evals_dir.is_dir():
        fail(f"Evaluation evidence directory does not exist: {args.evals_dir}")
    if args.data_dir and not args.data_dir.is_dir():
        fail(f"Data artifact directory does not exist: {args.data_dir}")

    project = Path(__file__).resolve().parents[2]
    output = (project / args.output_dir).resolve()
    eval_source = (
        str(args.evals_dir.resolve())
        if args.evals_dir
        else f"{EVAL_REPO}@{EVAL_REVISION}"
    )
    data_source = (
        str(args.data_dir.resolve())
        if args.data_dir
        else f"{DATA_REPO}@{DATA_REVISION}"
    )

    console.print("\n[bold]PrivEsc-LLM ACSAC artifact verification[/bold]")
    console.print(f"  Eval: {eval_source}", markup=False)
    console.print(f"  Data: {data_source}", markup=False)
    console.print("  Protocol: 12 scenarios x 10 runs; r20 primary; r60 maximum")

    stage(1, "Load evidence")
    if args.evals_dir is None or args.data_dir is None:
        from huggingface_hub import snapshot_download

    evidence = (
        args.evals_dir.resolve(strict=True)
        if args.evals_dir
        else Path(
            snapshot_download(
                repo_id=EVAL_REPO,
                repo_type="dataset",
                revision=EVAL_REVISION,
            )
        )
    )
    data = (
        args.data_dir.resolve(strict=True)
        if args.data_dir
        else Path(
            snapshot_download(
                repo_id=DATA_REPO,
                repo_type="dataset",
                revision=DATA_REVISION,
            )
        )
    )

    stage(2, "Run metric, path, and audit-code tests")
    run_checked(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "test/test_evaluation_analyze_metrics.py",
            "test/test_paths.py",
            "test/sft/test_audit_bundle.py",
        ],
        project,
        "Focused tests",
    )
    console.print("  Metric, path, and audit-code tests verified")

    stage(3, "Reaggregate 720 headline traces at 10 runs per scenario")
    aggregate(project, evidence, output / "headline", HEADLINE_ROOTS)
    headline = assert_same_summary(
        output / "headline/stats/evaluation_summary_paper.json",
        evidence / HEADLINE_SUMMARY,
        "Headline",
    )
    print_successes("Headline results", headline)

    stage(4, "Reaggregate 720 rebuttal traces at 10 runs per scenario")
    aggregate(project, evidence, output / "rebuttal", REBUTTAL_ROOTS)
    rebuttal = assert_same_summary(
        output / "rebuttal/stats/evaluation_summary_paper.json",
        evidence / REBUTTAL_SUMMARY,
        "Rebuttal",
    )
    print_successes("Rebuttal results", rebuttal)

    stage(5, "Verify 10,600 scalar records, split audit, and cost claims")
    sft_init = json.loads((evidence / SFT_INIT_SUMMARY).read_text())
    stats = next(iter(sft_init["model_stats"].values()))
    observed = (
        stats["num_runs"],
        stats["num_successful_runs_within_primary_budget"],
        stats["success_rate_within_primary_budget"],
    )
    if observed != (300, 228, 0.76):
        fail(f"SFT initialization expected N=300 and r20=228/300; found {observed}.")

    scalar_output = run_checked(
        [sys.executable, "scripts/paper/verify_archived_scalars.py", str(evidence)],
        project,
        "Scalar, cost, and ChainReactor checks",
    )
    console.print(scalar_output, markup=False)
    verify_split_audit(data)

    console.print(f"  Outputs: {output}", markup=False)
    console.print()
    console.print(
        "[bold green]PASS:[/bold green] archived evidence matches all checked paper and rebuttal claims"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        fail(f"{type(error).__name__}: {error}")
