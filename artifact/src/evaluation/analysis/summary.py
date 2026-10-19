import pandas as pd
from rich.box import ROUNDED, SIMPLE
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.evaluation.analysis.constants import (
    MAX_ROUNDS,
    PRIMARY_ROUND_BUDGET,
    ROUND_BUDGETS,
    format_model_name,
    natural_sort_key,
    ordered_model_ids,
)
from src.evaluation.analysis.metrics import (
    compute_metrics_for_group,
    budgeted_success_rate_for_group,
    budgeted_success_stats_for_group,
)

console = Console()


def format_token_count(n: float) -> str:
    return f"{n / 1000:.0f}k" if n >= 1000 else f"{n:.0f}"


def format_duration_ms(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    if value >= 1000:
        return f"{value / 1000:.1f}s"
    return f"{value:.0f}ms"


def create_summary_table(title: str, is_scenario: bool = False) -> Table:
    table = Table(
        box=SIMPLE,
        header_style="bold cyan",
        padding=(0, 1),
        title=f"[bold cyan]{title}[/]",
        title_justify="left",
    )
    table.add_column("Model" if not is_scenario else "Scenario", style="magenta")
    for col in ["SR", "CI95", "N", "R_s", "Msg", "TokIn", "TokOut", "Tok", "C_avg"]:
        table.add_column(col, justify="right")
    if not is_scenario:
        table.add_column("C_tot", justify="right")
    return table


def add_summary_row(
    table: Table,
    name: str,
    df: pd.DataFrame,
    *,
    round_budget: int,
    is_scenario: bool = False,
):
    budget_stats = budgeted_success_stats_for_group(df, round_budget)
    success_mask = df["success"].fillna(False).astype(bool) & (
        pd.to_numeric(df["turns"], errors="coerce") <= round_budget
    )
    succ = df[success_mask]
    rate = float(budget_stats["rate"])
    ci_low = float(budget_stats["ci_low"])
    ci_high = float(budget_stats["ci_high"])
    style = "green" if rate >= 0.8 else "yellow" if rate >= 0.5 else "red"
    row = [
        name,
        f"[{style}]{rate:.0%}[/]",
        f"{ci_low:.0%}-{ci_high:.0%}",
        str(len(df)),
        f"{succ['turns'].mean():.1f}" if not succ.empty else "0.0",
        f"{df['messages'].mean():.0f}",
        format_token_count(float(df["prompt_tokens"].mean())),
        format_token_count(float(df["completion_tokens"].mean())),
        format_token_count(float(df["total_tokens"].mean())),
        f"${df['cost'].mean():.2f}",
    ]
    if not is_scenario:
        row.append(f"${df['cost'].sum():.2f}")
    table.add_row(*row)


def create_timing_summary_table(title: str) -> Table:
    table = Table(
        box=SIMPLE,
        header_style="bold cyan",
        padding=(0, 1),
        title=f"[bold cyan]{title}[/]",
        title_justify="left",
    )
    table.add_column("Model", style="magenta")
    for col in ["Wall", "LLM", "Tool", "Cost"]:
        table.add_column(col, justify="right")
    return table


def create_timing_distribution_table(title: str, name_column: str) -> Table:
    table = Table(
        box=SIMPLE,
        header_style="bold cyan",
        padding=(0, 1),
        title=f"[bold cyan]{title}[/]",
        title_justify="left",
    )
    table.add_column(name_column, style="magenta")
    for col in ["Avg", "P50", "P90", "Min", "Max", "Succ", "Fail"]:
        table.add_column(col, justify="right")
    return table


def add_timing_distribution_row(
    table: Table, name: str, timing_stats: dict[str, float | int | None]
) -> None:
    table.add_row(
        name,
        format_duration_ms(timing_stats.get("avg")),
        format_duration_ms(timing_stats.get("p50")),
        format_duration_ms(timing_stats.get("p90")),
        format_duration_ms(timing_stats.get("min")),
        format_duration_ms(timing_stats.get("max")),
        format_duration_ms(timing_stats.get("avg_success")),
        format_duration_ms(timing_stats.get("avg_failure")),
    )


def create_budgeted_success_table(title: str) -> Table:
    table = Table(
        box=SIMPLE,
        header_style="bold cyan",
        padding=(0, 1),
        title=f"[bold cyan]{title}[/]",
        title_justify="left",
    )
    table.add_column("Model", style="magenta")
    for round_budget in ROUND_BUDGETS:
        table.add_column(f"R≤{round_budget}", justify="right")
    return table


def create_scenario_model_matrix(df: pd.DataFrame, title: str, metric: str) -> Table:
    table = Table(
        box=SIMPLE,
        header_style="bold cyan",
        padding=(0, 1),
        title=f"[bold cyan]{title}[/]",
        title_justify="left",
    )
    sorted_models = ordered_model_ids(df["model"].unique())
    sorted_scenarios = sorted(df["scenario"].unique(), key=natural_sort_key)

    table.add_column("Scenario", style="magenta")
    for model in sorted_models:
        table.add_column(format_model_name(model), justify="right")

    for scenario in sorted_scenarios:
        row = [scenario]
        for model in sorted_models:
            subset = df[(df["scenario"] == scenario) & (df["model"] == model)]
            if subset.empty:
                row.append("-")
                continue
            if metric == "sr":
                value = float(subset["success"].mean())
                style = "green" if value >= 0.8 else "yellow" if value >= 0.5 else "red"
                row.append(f"[{style}]{value:.0%}[/]")
            elif metric == "sr20":
                value = budgeted_success_rate_for_group(subset).get(
                    f"sr_within_rounds_{PRIMARY_ROUND_BUDGET}", 0.0
                )
                style = "green" if value >= 0.8 else "yellow" if value >= 0.5 else "red"
                row.append(f"[{style}]{value:.0%}[/]")
            elif metric == "turns_success":
                succ = subset[subset["success"]]
                row.append(f"{succ['turns'].mean():.1f}" if not succ.empty else "-")
            elif metric == "s60":
                value = budgeted_success_rate_for_group(subset).get(
                    "sr_within_rounds_60", 0.0
                )
                style = "green" if value >= 0.8 else "yellow" if value >= 0.5 else "red"
                row.append(f"[{style}]{value:.0%}[/]")
            else:
                raise ValueError(f"Unknown matrix metric: {metric}")
        table.add_row(*row)

    return table


def print_evaluation_summary(df: pd.DataFrame):
    excluded_counts = dict(df.attrs.get("excluded_counts", {}))
    infra_exhausted = int(excluded_counts.get("infra_exhausted_files", 0))
    console.print(
        Panel(
            Text.from_markup(
                f"[bold cyan]Evaluation Analysis Summary[/]\n"
                f"[grey70]Total runs: {len(df)}  ·  Cost: ${df['cost'].sum():.2f}  ·  Tokens: {df['total_tokens'].sum():,}[/]"
            ),
            box=ROUNDED,
            border_style="cyan",
            expand=False,
        )
    )
    if infra_exhausted:
        console.print(
            f"[yellow]Note:[/] Excluded {infra_exhausted} infra-exhausted runs from benchmark metrics."
        )

    m_table = create_summary_table(
        f"Model Performance (primary endpoint: success within {PRIMARY_ROUND_BUDGET} rounds)"
    )
    for model in ordered_model_ids(df["model"].unique()):
        model_df = df[df["model"] == model]
        add_summary_row(
            m_table,
            format_model_name(model),
            model_df,
            round_budget=PRIMARY_ROUND_BUDGET,
        )
    console.print(m_table)
    console.print(
        "[dim]Cols: SR=Success Rate under the table's stated round budget, CI95=95% Wilson CI, N=Runs, R_s=Avg Rounds on successful runs within that budget, "
        "Msg=Avg Messages, TokIn/TokOut/Tok=Avg Prompt/Completion/Total Tokens, C_avg=Avg Cost, C_tot=Total Cost[/]"
    )

    timing_table = create_timing_summary_table("Trace Timing per Run")
    has_timing = False
    for model in ordered_model_ids(df["model"].unique()):
        model_df = df[df["model"] == model]
        model_metrics = compute_metrics_for_group(model_df)
        row = [format_model_name(model)]
        for metric_key in [
            "avg_rollout_wall_clock_ms",
            "avg_total_llm_ms_raw",
            "avg_total_tool_ms_raw",
            "avg_cost_ms_raw",
        ]:
            metric_value = model_metrics.get(metric_key)
            if metric_value is not None:
                has_timing = True
            row.append(format_duration_ms(metric_value))
        timing_table.add_row(*row)
    if has_timing:
        console.print(timing_table)
        console.print(
            "[dim]Cols: Wall/LLM/Tool/Cost are avg per-run trace timings; '-' means missing in older traces.[/]"
        )

    scenario_timing_table = create_timing_distribution_table(
        "Scenario Wall-Clock Timing", "Scenario"
    )
    has_scenario_timing = False
    for scenario in sorted(df["scenario"].unique(), key=natural_sort_key):
        scenario_df = df[df["scenario"] == scenario]
        scenario_metrics = compute_metrics_for_group(scenario_df)
        wall_clock_stats = scenario_metrics["timing_stats"]["rollout_wall_clock_ms"]
        if int(wall_clock_stats.get("count", 0)) > 0:
            has_scenario_timing = True
        add_timing_distribution_row(scenario_timing_table, scenario, wall_clock_stats)
    if has_scenario_timing:
        console.print(scenario_timing_table)
        console.print(
            "[dim]Cols: Avg/P50/P90/Min/Max are wall-clock timing over timed runs; Succ/Fail are avg wall-clock on successful/failed runs.[/]"
        )

    k_table = create_budgeted_success_table(
        "Budgeted per-run success rate by round budget"
    )
    for model in ordered_model_ids(df["model"].unique()):
        model_df = df[df["model"] == model]
        sak = budgeted_success_rate_for_group(model_df)
        row = [format_model_name(model)]
        for round_budget in ROUND_BUDGETS:
            row.append(f"{sak.get(f'sr_within_rounds_{round_budget}', 0.0):.0%}")
        k_table.add_row(*row)
    console.print(k_table)

    s_table = create_summary_table(
        f"Scenario Breakdown (success within {MAX_ROUNDS} rounds)",
        is_scenario=True,
    )
    for scenario in sorted(df["scenario"].unique(), key=natural_sort_key):
        scenario_df = df[df["scenario"] == scenario]
        add_summary_row(
            s_table,
            scenario,
            scenario_df,
            round_budget=MAX_ROUNDS,
            is_scenario=True,
        )
    console.print(s_table)
    console.print(
        create_scenario_model_matrix(
            df,
            f"Scenario x Model success within {PRIMARY_ROUND_BUDGET} rounds",
            metric="sr20",
        )
    )
    console.print(
        create_scenario_model_matrix(
            df,
            f"Scenario x Model success within {MAX_ROUNDS} rounds",
            metric="s60",
        )
    )
    console.print(
        create_scenario_model_matrix(
            df, "Scenario x Model successful-rounds mean", metric="turns_success"
        )
    )
    console.print()
