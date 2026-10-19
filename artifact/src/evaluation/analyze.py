import argparse
from pathlib import Path
import sys

from hydra import compose, initialize_config_dir
import pandas as pd
from hydra.core.global_hydra import GlobalHydra
from rich.console import Console
from rich.status import Status

from src.config import register_with_hydra
from src.evaluation.analysis.captions import write_figure_captions
from src.evaluation.analysis.constants import (
    PAPER_MODEL_ORDER,
    PRIMARY_ROUND_BUDGET,
    ROUND_BUDGETS as ANALYSIS_ROUND_BUDGETS,
)
from src.evaluation.analysis.metrics import (
    budgeted_success_rate_for_group as _budgeted_success_rate_for_group,
    budgeted_success_rate_with_ci_for_group as _budgeted_success_rate_with_ci_for_group,
    budgeted_success_stats_for_group as _budgeted_success_stats_for_group,
    calculate_and_save_stats,
    compute_metrics_for_group as _compute_metrics_for_group,
    expected_cost_per_successful_root_for_group as _expected_cost_per_successful_root_for_group,
    paired_mcnemar_test as _paired_mcnemar_test,
    write_results_csv,
)
from src.evaluation.analysis.parsing import (
    parse_evaluation_results_detailed,
)
from src.evaluation.analysis.plotting import save_plots_for_paper_theme
from src.evaluation.analysis.summary import (
    create_scenario_model_matrix as _create_scenario_model_matrix,
    create_summary_table as _create_summary_table,
    print_evaluation_summary,
)
from src.scenarios import build_scenario_source
from src.scenarios.procedural import ProceduralScenarioSource
from src.scenarios.static import StaticScenarioSource

console = Console()
DEFAULT_ANALYSIS_EXPERIMENT = "eval/paper_static_qwen3_4b_base"

# Public API used by tests and external callers.
budgeted_success_rate_for_group = _budgeted_success_rate_for_group
budgeted_success_rate_with_ci_for_group = _budgeted_success_rate_with_ci_for_group
budgeted_success_stats_for_group = _budgeted_success_stats_for_group
expected_cost_per_successful_root_for_group = _expected_cost_per_successful_root_for_group
paired_mcnemar_test = _paired_mcnemar_test
compute_metrics_for_group = _compute_metrics_for_group
create_summary_table = _create_summary_table
create_scenario_model_matrix = _create_scenario_model_matrix
ROUND_BUDGETS = ANALYSIS_ROUND_BUDGETS
PRIMARY_BUDGET = PRIMARY_ROUND_BUDGET


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze agent results for a given base directory."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="outputs/evals/evaluation",
        help="Base directory for results.",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help=("Analyze all parsed models without restricting to PAPER_MODEL_ORDER."),
    )
    parser.add_argument(
        "--allow-incomplete-runs",
        action="store_true",
        help=(
            "Allow analysis when model-scenario cells have fewer or more than the "
            "paper protocol run count."
        ),
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=DEFAULT_ANALYSIS_EXPERIMENT,
        help=(
            "Hydra experiment config used to run the eval. When provided, "
            "paper validation uses its runner.source entries and runner.runs_per_item. "
            f"Defaults to {DEFAULT_ANALYSIS_EXPERIMENT!r}."
        ),
    )
    parser.add_argument(
        "--expected-runs-per-scenario",
        type=int,
        default=None,
        help="Override the expected valid run count per model-scenario cell.",
    )
    parser.add_argument(
        "--show-titles",
        action="store_true",
        help="Render plot titles for slide or standalone use.",
    )
    return parser


def _expected_protocol_from_experiment(
    experiment: str | None,
) -> tuple[list[str] | None, int | None]:
    if not experiment:
        return None, None

    register_with_hydra()
    config_dir = Path(__file__).resolve().parents[2] / "conf"
    hydra_state = GlobalHydra.instance()
    if hydra_state.is_initialized():
        cfg = compose(config_name="config", overrides=[f"+experiment={experiment}"])
    else:
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            cfg = compose(config_name="config", overrides=[f"+experiment={experiment}"])

    source = build_scenario_source(cfg.runner.source, generators_cfg=cfg.generators)
    runs_per_scenario = int(cfg.runner.runs_per_item)
    if isinstance(source, StaticScenarioSource):
        return [str(scenario) for scenario in source.scenarios], runs_per_scenario
    if isinstance(source, ProceduralScenarioSource):
        return [str(generator) for generator in source.generators], runs_per_scenario
    raise ValueError(f"Unsupported scenario source for experiment {experiment!r}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    base_dir = args.base_dir
    eval_results_dir = f"{base_dir}/traces"
    stats_output_dir = f"{base_dir}/stats"
    plots_output_dir = f"{base_dir}/plots"

    try:
        with Status(
            "[bold cyan]Analyzing evaluation results...", console=console
        ) as status:
            status.update("[bold cyan]Step 1: Parsing evaluation files...")
            allowed_models = None if args.all_models else PAPER_MODEL_ORDER
            expected_scenarios: list[str] | None = None
            expected_runs_per_scenario: int | None = None
            if not args.allow_incomplete_runs:
                expected_scenarios, expected_runs_per_scenario = (
                    _expected_protocol_from_experiment(args.experiment)
                )
                if args.expected_runs_per_scenario is not None:
                    expected_runs_per_scenario = args.expected_runs_per_scenario
            results_df = parse_evaluation_results_detailed(
                eval_results_dir,
                validate_run_counts=not args.allow_incomplete_runs,
                allowed_models=allowed_models,
                expected_scenarios=expected_scenarios,
                expected_runs_per_scenario=expected_runs_per_scenario,
            )
            excluded_counts = dict(results_df.attrs.get("excluded_counts", {}))
            results_csv = write_results_csv(results_df, stats_output_dir)
            results_df = pd.read_csv(results_csv)
            results_df.attrs["excluded_counts"] = excluded_counts

            status.update("[bold cyan]Step 2: Calculating summary statistics...")
            calculate_and_save_stats(
                results_df,
                stats_output_dir,
                write_paper_summary=not args.allow_incomplete_runs,
            )

            status.update("[bold cyan]Step 3: Generating performance plots...")
            figure_files = save_plots_for_paper_theme(
                results_df,
                plots_output_dir,
                show_titles=args.show_titles,
            )
            write_figure_captions(
                plots_output_dir,
                figure_files,
                protocol_complete=not args.allow_incomplete_runs,
                expected_runs_per_scenario=expected_runs_per_scenario,
            )

        print_evaluation_summary(results_df)

    except (ValueError, FileNotFoundError) as e:
        console.print(f"\n[red bold]Error:[/] {e}")
        console.print(f"  Looking in: [dim]{Path(eval_results_dir).resolve()}[/]")
        sys.exit(1)


if __name__ == "__main__":
    main()
