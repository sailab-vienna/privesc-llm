import json
import math
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.console import Console

from src.evaluation.analysis.constants import (
    MAX_ROUNDS,
    PRIMARY_ROUND_BUDGET,
    ROUND_BUDGETS,
)

console = Console()

TRACE_TIMING_METRIC_COLUMNS = {
    "rollout_wall_clock_ms": "avg_rollout_wall_clock_ms",
    "time_to_root_wall_clock_ms": "avg_time_to_root_wall_clock_ms",
    "time_to_root_interaction_ms_raw": "avg_time_to_root_interaction_ms_raw",
    "total_llm_ms_raw": "avg_total_llm_ms_raw",
    "total_tool_ms_raw": "avg_total_tool_ms_raw",
    "cost_ms_raw": "avg_cost_ms_raw",
}


def write_results_csv(df: pd.DataFrame, output_dir: str) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    runs_csv = output_path / "runs.csv"
    df.to_csv(runs_csv, index=False)
    console.print(
        f"[green]✔[/] Paper-facing run table successfully saved to: [dim]{runs_csv}[/]"
    )
    return runs_csv


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p_hat = successes / total
    z2 = z * z
    denom = 1 + z2 / total
    center = (p_hat + z2 / (2 * total)) / denom
    margin = (z / denom) * math.sqrt(
        (p_hat * (1 - p_hat) / total) + (z2 / (4 * total * total))
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def bootstrap_percentile_ci(
    values: list[float], alpha: float = 0.05
) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=float)
    lower = float(np.nanpercentile(arr, 100 * (alpha / 2)))
    upper = float(np.nanpercentile(arr, 100 * (1 - alpha / 2)))
    return lower, upper


def _timing_series(df_group: pd.DataFrame, column: str) -> pd.Series | None:
    if column not in df_group:
        return None
    values = pd.to_numeric(df_group[column], errors="coerce")
    values = values.dropna()
    if values.empty:
        return None
    return values.astype(float)


def _mean_or_none(df_group: pd.DataFrame, column: str) -> float | None:
    values = _timing_series(df_group, column)
    if values is None:
        return None
    return float(values.mean())


def _mean_subset_or_none(
    df_group: pd.DataFrame, column: str, *, success: bool
) -> float | None:
    subset = df_group[df_group["success"].fillna(False).astype(bool) == success]
    values = _timing_series(subset, column)
    if values is None:
        return None
    return float(values.mean())


def _trace_timing_metrics(df_group: pd.DataFrame) -> dict[str, float | None]:
    return {
        metric_name: _mean_or_none(df_group, column)
        for column, metric_name in TRACE_TIMING_METRIC_COLUMNS.items()
    }


def _trace_timing_stats(
    df_group: pd.DataFrame,
) -> dict[str, dict[str, float | int | None]]:
    timing_stats: dict[str, dict[str, float | int | None]] = {}
    for column in TRACE_TIMING_METRIC_COLUMNS:
        values = _timing_series(df_group, column)
        if values is None:
            timing_stats[column] = {
                "count": 0,
                "avg": None,
                "p50": None,
                "p90": None,
                "min": None,
                "max": None,
                "avg_success": None,
                "avg_failure": None,
            }
            continue
        timing_stats[column] = {
            "count": int(values.count()),
            "avg": float(values.mean()),
            "p50": float(np.nanpercentile(values, 50)),
            "p90": float(np.nanpercentile(values, 90)),
            "min": float(values.min()),
            "max": float(values.max()),
            "avg_success": _mean_subset_or_none(df_group, column, success=True),
            "avg_failure": _mean_subset_or_none(df_group, column, success=False),
        }
    return timing_stats


def budgeted_success_stats_for_group(
    df_group: pd.DataFrame,
    round_budget: int,
) -> dict[str, float | int]:
    if df_group.empty:
        return {
            "round_budget": round_budget,
            "num_runs": 0,
            "num_successful_runs": 0,
            "rate": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
        }

    turns = pd.to_numeric(df_group["turns"], errors="coerce")
    success = df_group["success"].fillna(False).astype(bool)
    run_count = len(df_group)
    success_within_budget = int((success & (turns <= round_budget)).sum())
    rate = success_within_budget / run_count
    ci_low, ci_high = wilson_interval(success_within_budget, run_count)
    return {
        "round_budget": round_budget,
        "num_runs": run_count,
        "num_successful_runs": success_within_budget,
        "rate": rate,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def expected_cost_per_successful_root_for_group(
    df_group: pd.DataFrame,
    round_budget: int,
    *,
    bootstrap_n: int = 0,
    rng: np.random.Generator | None = None,
) -> dict[str, float | int]:
    if df_group.empty:
        return {
            "round_budget": round_budget,
            "num_runs": 0,
            "success_rate": 0.0,
            "expected_cost_per_root": float("nan"),
            "ci_low": 0.0,
            "ci_high": 0.0,
        }

    run_count = len(df_group)
    success_stats = budgeted_success_stats_for_group(df_group, round_budget)
    success_rate = float(success_stats["rate"])
    cost_column = (
        f"cost_within_rounds_{round_budget}"
        if f"cost_within_rounds_{round_budget}" in df_group.columns
        else "cost"
    )
    mean_cost = float(df_group[cost_column].mean())
    expected_cost_per_root = (
        (mean_cost / success_rate) if success_rate > 0 else float("nan")
    )

    if bootstrap_n <= 0:
        return {
            "round_budget": round_budget,
            "num_runs": run_count,
            "success_rate": success_rate,
            "expected_cost_per_root": expected_cost_per_root,
            "ci_low": 0.0,
            "ci_high": 0.0,
        }

    generator = rng or np.random.default_rng(42)
    boot_values: list[float] = []
    for _ in range(bootstrap_n):
        sample_idx = generator.integers(0, run_count, run_count)
        sampled = df_group.iloc[sample_idx]
        sampled_success_rate = float(
            budgeted_success_stats_for_group(sampled, round_budget)["rate"]
        )
        sampled_cost = float(sampled[cost_column].mean())
        if sampled_success_rate > 0:
            boot_values.append(sampled_cost / sampled_success_rate)
        else:
            boot_values.append(float("inf"))
    ci_low, ci_high = bootstrap_percentile_ci(boot_values)
    return {
        "round_budget": round_budget,
        "num_runs": run_count,
        "success_rate": success_rate,
        "expected_cost_per_root": expected_cost_per_root,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def paired_mcnemar_test(
    df: pd.DataFrame,
    model_a: str,
    model_b: str,
    round_budget: int = PRIMARY_ROUND_BUDGET,
) -> dict[str, float | int | str]:
    subset = df[df["model"].isin([model_a, model_b])].copy()
    if subset.empty:
        raise ValueError("No rows available for requested model pair")

    if subset["model"].nunique() != 2:
        raise ValueError("Both models must be present for McNemar comparison")

    pair_columns: list[str]
    if (
        "item_run_ordinal" in subset.columns
        and subset["item_run_ordinal"].notna().all()
    ):
        pair_columns = ["scenario", "item_run_ordinal"]
    elif "run_index" in subset.columns and subset["run_index"].notna().all():
        pair_columns = ["run_index"]
    else:
        raise ValueError(
            "Paired McNemar test requires saved pairing ids (item_run_ordinal or run_index)."
        )

    subset["success_within_budget"] = subset["success"].fillna(False).astype(bool) & (
        pd.to_numeric(subset["turns"], errors="coerce") <= round_budget
    )
    pair_frame = subset[pair_columns + ["model", "success_within_budget"]]
    duplicate_counts = pair_frame.groupby(pair_columns + ["model"]).size()
    if (duplicate_counts > 1).any():
        raise ValueError("Duplicate paired ids detected for McNemar comparison")
    paired = pair_frame.pivot(
        index=pair_columns,
        columns="model",
        values="success_within_budget",
    )

    if model_a not in paired.columns or model_b not in paired.columns:
        raise ValueError("Both models must have paired runs for McNemar comparison")

    paired = paired.dropna(subset=[model_a, model_b])
    if paired.empty:
        raise ValueError("No paired runs remain after aligning model pair")

    a = paired[model_a].astype(bool)
    b = paired[model_b].astype(bool)
    n_10 = int((a & ~b).sum())
    n_01 = int((~a & b).sum())
    discordant = n_10 + n_01
    if discordant == 0:
        exact_p = 1.0
        chi2_cc = 0.0
    else:
        tail = sum(
            math.comb(discordant, i) * (0.5**discordant)
            for i in range(0, min(n_10, n_01) + 1)
        )
        exact_p = min(1.0, 2.0 * tail)
        chi2_cc = ((abs(n_10 - n_01) - 1) ** 2) / discordant

    return {
        "model_a": model_a,
        "model_b": model_b,
        "round_budget": round_budget,
        "num_pairs": int(len(paired)),
        "n_10_a_only": n_10,
        "n_01_b_only": n_01,
        "chi2_cc": float(chi2_cc),
        "exact_pvalue": float(exact_p),
    }


def compute_metrics_for_group(df_group: pd.DataFrame) -> dict[str, Any]:
    if df_group.empty:
        return {}

    succ = df_group[df_group["success"]]
    success_count = len(succ)
    run_count = len(df_group)
    ci_low, ci_high = wilson_interval(success_count, run_count)
    primary_stats = budgeted_success_stats_for_group(df_group, PRIMARY_ROUND_BUDGET)
    max_budget_stats = budgeted_success_stats_for_group(df_group, MAX_ROUNDS)
    primary_cost = expected_cost_per_successful_root_for_group(
        df_group,
        PRIMARY_ROUND_BUDGET,
    )

    metrics = {
        "num_runs": run_count,
        "num_successful_runs": success_count,
        "success_rate_micro_avg": df_group["success"].mean(),
        "success_rate_ci95_low": ci_low,
        "success_rate_ci95_high": ci_high,
        "max_round_budget": MAX_ROUNDS,
        "primary_round_budget": PRIMARY_ROUND_BUDGET,
        "num_successful_runs_within_primary_budget": int(
            primary_stats["num_successful_runs"]
        ),
        "success_rate_within_primary_budget": float(primary_stats["rate"]),
        "success_rate_within_primary_budget_ci95_low": float(primary_stats["ci_low"]),
        "success_rate_within_primary_budget_ci95_high": float(primary_stats["ci_high"]),
        "num_successful_runs_within_max_budget": int(
            max_budget_stats["num_successful_runs"]
        ),
        "success_rate_within_max_budget": float(max_budget_stats["rate"]),
        "success_rate_within_max_budget_ci95_low": float(max_budget_stats["ci_low"]),
        "success_rate_within_max_budget_ci95_high": float(max_budget_stats["ci_high"]),
        "expected_cost_per_successful_root_within_primary_budget": float(
            primary_cost["expected_cost_per_root"]
        ),
        "avg_turns_all": df_group["turns"].mean(),
        "avg_turns_success": succ["turns"].mean() if not succ.empty else None,
        "avg_messages": df_group["messages"].mean(),
        "avg_total_tokens": df_group["total_tokens"].mean(),
        "avg_prompt_tokens": df_group["prompt_tokens"].mean(),
        "avg_completion_tokens": df_group["completion_tokens"].mean(),
        "total_cost": df_group["cost"].sum(),
        "avg_cost_per_run": df_group["cost"].mean(),
        "timing_stats": _trace_timing_stats(df_group),
    }
    metrics.update(_trace_timing_metrics(df_group))
    return metrics


def budgeted_success_rate_with_ci_for_group(
    df_group: pd.DataFrame,
) -> dict[int, dict[str, float]]:
    if df_group.empty:
        return {k: {"rate": 0.0, "ci_low": 0.0, "ci_high": 0.0} for k in ROUND_BUDGETS}

    out: dict[int, dict[str, float]] = {}
    for round_budget in ROUND_BUDGETS:
        stats = budgeted_success_stats_for_group(df_group, round_budget)
        out[round_budget] = {
            "rate": float(stats["rate"]),
            "ci_low": float(stats["ci_low"]),
            "ci_high": float(stats["ci_high"]),
        }
    return out


def budgeted_success_rate_for_group(df_group: pd.DataFrame) -> dict[str, float]:
    curve = budgeted_success_rate_with_ci_for_group(df_group)
    return {f"sr_within_rounds_{r}": curve[r]["rate"] for r in ROUND_BUDGETS}


def _paper_stats_view(stats: dict[str, Any]) -> dict[str, Any]:
    paper_stats: dict[str, Any] = {
        "overall_stats": dict(stats.get("overall_stats", {})),
        "model_stats": {},
        "scenario_stats": {},
        "scenario_model_stats": {},
    }
    keys = {
        "num_runs",
        "primary_round_budget",
        "max_round_budget",
        "num_successful_runs_within_primary_budget",
        "success_rate_within_primary_budget",
        "success_rate_within_primary_budget_ci95_low",
        "success_rate_within_primary_budget_ci95_high",
        "num_successful_runs_within_max_budget",
        "success_rate_within_max_budget",
        "success_rate_within_max_budget_ci95_low",
        "success_rate_within_max_budget_ci95_high",
        "expected_cost_per_successful_root_within_primary_budget",
        "avg_rollout_wall_clock_ms",
        "avg_time_to_root_wall_clock_ms",
        "avg_time_to_root_interaction_ms_raw",
        "avg_total_llm_ms_raw",
        "avg_total_tool_ms_raw",
        "avg_cost_ms_raw",
        "timing_stats",
        *{f"sr_within_rounds_{round_budget}" for round_budget in ROUND_BUDGETS},
    }

    for section in ("model_stats", "scenario_stats"):
        for name, payload in stats.get(section, {}).items():
            paper_stats[section][name] = {
                key: value for key, value in payload.items() if key in keys
            }

    for scenario, model_payloads in stats.get("scenario_model_stats", {}).items():
        paper_stats["scenario_model_stats"][scenario] = {}
        for model, payload in model_payloads.items():
            paper_stats["scenario_model_stats"][scenario][model] = {
                key: value for key, value in payload.items() if key in keys
            }

    paper_stats["overall_stats"]["paper_primary_metric_key"] = (
        "success_rate_within_primary_budget"
    )
    return paper_stats


def calculate_and_save_stats(
    df: pd.DataFrame, output_dir: str, *, write_paper_summary: bool = True
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    excluded_counts = dict(df.attrs.get("excluded_counts", {}))
    overall_timing_metrics = _trace_timing_metrics(df)
    overall_timing_stats = _trace_timing_stats(df)
    stats = {
        "overall_stats": {
            "total_runs": len(df),
            "total_cost": df["cost"].sum(),
            "total_tokens": df["total_tokens"].sum(),
            "primary_round_budget": PRIMARY_ROUND_BUDGET,
            "max_round_budget": MAX_ROUNDS,
            **overall_timing_metrics,
            "timing_stats": overall_timing_stats,
            "excluded_runs": excluded_counts,
        },
        "model_stats": {},
        "scenario_stats": {},
        "scenario_model_stats": {},
    }

    for model_name, model_df in df.groupby("model"):
        model_stats = compute_metrics_for_group(model_df)
        model_stats.update(budgeted_success_rate_for_group(model_df))
        stats["model_stats"][model_name] = model_stats

    for scenario_name, scenario_df in df.groupby("scenario"):
        scenario_stats = compute_metrics_for_group(scenario_df)
        scenario_stats.update(budgeted_success_rate_for_group(scenario_df))
        stats["scenario_stats"][scenario_name] = scenario_stats
        stats["scenario_model_stats"][scenario_name] = {}
        for model_name, scenario_model_df in scenario_df.groupby("model"):
            model_scenario_stats = compute_metrics_for_group(scenario_model_df)
            model_scenario_stats.update(
                budgeted_success_rate_for_group(scenario_model_df)
            )
            stats["scenario_model_stats"][scenario_name][model_name] = (
                model_scenario_stats
            )

    summary_file = output_path / "evaluation_summary.json"
    with open(summary_file, "w") as f:
        json.dump(
            stats,
            f,
            indent=4,
            default=lambda x: round(float(x), 4) if hasattr(x, "__float__") else x,
        )

    paper_summary_file = output_path / "evaluation_summary_paper.json"
    if write_paper_summary:
        with open(paper_summary_file, "w") as f:
            json.dump(
                _paper_stats_view(stats),
                f,
                indent=4,
                default=lambda x: round(float(x), 4) if hasattr(x, "__float__") else x,
            )
        console.print(
            f"[green]✔[/] Paper-facing statistics successfully saved to: [dim]{paper_summary_file}[/]"
        )
    else:
        with suppress(FileNotFoundError):
            paper_summary_file.unlink()

    console.print(f"[green]✔[/] Statistics successfully saved to: [dim]{summary_file}[/]")
