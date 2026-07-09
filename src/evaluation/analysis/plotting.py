from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import FuncFormatter, PercentFormatter
from matplotlib.transforms import blended_transform_factory
from rich.console import Console

from src.evaluation.analysis.constants import (
    CLAUDE_OPUS_47_MODEL,
    E1_TRACE_MODEL_ORDER,
    DEEPSEEK_V32_MODEL,
    GEMMA4_31B_MODEL,
    MAX_ROUNDS,
    PARETO_ROUND_BUDGET,
    PLOT_EXTENSIONS,
    PRIMARY_ROUND_BUDGET,
    PRIVESC_LLM_4B_MODEL,
    PRIVESC_LLM_NAME,
    QWEN3_4B_MODEL,
    QWEN3_4B_SFT_MODEL,
    ROUND_BUDGETS,
    format_model_name,
    natural_sort_key,
    ordered_model_ids,
    short_scenario_label,
)
from src.evaluation.analysis.metrics import (
    bootstrap_percentile_ci,
    budgeted_success_rate_with_ci_for_group,
    expected_cost_per_successful_root_for_group,
    wilson_interval,
)

console = Console()


@dataclass(frozen=True)
class DeploymentPoint:
    model: str
    deployment: str
    size_b: float
    label_offset: tuple[int, int]
    ha: str
    va: str = "center"


PAPER_TIGHT_STEMS = {
    "2_heatmap_success_rate_by_scenario",
    "2b_heatmap_success_rate_by_scenario_r20",
    "2c_success_vs_round_budget",
    "budgeted_success_by_size",
    "5_average_tool_usage",
    "8_pareto_success_rate_vs_cost",
}

BUDGETED_SUCCESS_DEPLOYMENT_POINTS = [
    DeploymentPoint(
        model=QWEN3_4B_MODEL,
        deployment="Local open-weight",
        size_b=4.0,
        label_offset=(14, 0),
        ha="left",
    ),
    DeploymentPoint(
        model=QWEN3_4B_SFT_MODEL,
        deployment="Local open-weight",
        size_b=4.0,
        label_offset=(14, 0),
        ha="left",
    ),
    DeploymentPoint(
        model=PRIVESC_LLM_4B_MODEL,
        deployment="Local open-weight",
        size_b=4.0,
        label_offset=(14, 0),
        ha="left",
    ),
    DeploymentPoint(
        model=GEMMA4_31B_MODEL,
        deployment="Local open-weight",
        size_b=31.0,
        label_offset=(0, -16),
        ha="center",
        va="top",
    ),
    DeploymentPoint(
        model=DEEPSEEK_V32_MODEL,
        deployment="Hosted open-weight API",
        size_b=671.0,
        label_offset=(-14, 0),
        ha="right",
    ),
    DeploymentPoint(
        model=CLAUDE_OPUS_47_MODEL,
        deployment="Closed API",
        size_b=float("nan"),
        label_offset=(0, -16),
        ha="center",
        va="top",
    ),
]

DEPLOYMENT_MARKERS = {
    "Local open-weight": "^",
    "Hosted open-weight API": "D",
    "Closed API": "o",
}


def _save_plot(fig, output_path: Path, *, stem: str, ext: str) -> None:
    if stem in PAPER_TIGHT_STEMS:
        fig.savefig(
            output_path,
            format=ext,
            transparent=False,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.05,
        )
        return
    fig.savefig(output_path, format=ext, transparent=False, dpi=300)


def _is_privesc_label(label: str) -> bool:
    return label == PRIVESC_LLM_NAME or label.startswith(PRIVESC_LLM_NAME)


def _highlight_axis_model_labels(ax) -> None:
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        if _is_privesc_label(label.get_text()):
            label.set_fontweight("bold")
            label.set_color("#111827")


def _highlight_current_legend(ax) -> None:
    legend = ax.get_legend()
    if legend is None:
        return
    for text in legend.get_texts():
        if _is_privesc_label(text.get_text()):
            text.set_fontweight("bold")
            text.set_color("#111827")


def _style_heatmap_xlabels(ax, *, rotation: float = 25) -> None:
    ax.tick_params(axis="x", pad=6)
    for label in ax.get_xticklabels():
        label.set_rotation(rotation)
        label.set_rotation_mode("anchor")
        label.set_ha("right")
        label.set_va("top")


def _budgeted_success_mask(df: pd.DataFrame, round_budget: int) -> pd.Series:
    return df["success"].fillna(False).astype(bool) & (
        pd.to_numeric(df["turns"], errors="coerce") <= round_budget
    )


def _assert_budgeted_plot_consistency(
    df: pd.DataFrame,
    *,
    model_order: list[str],
    sorted_scenarios: list[str],
) -> None:
    for model in model_order:
        model_df = df[df["model"] == model]
        if model_df.empty:
            continue

        primary_total = int(_budgeted_success_mask(model_df, PRIMARY_ROUND_BUDGET).sum())
        max_total = int(_budgeted_success_mask(model_df, MAX_ROUNDS).sum())
        if max_total < primary_total:
            raise ValueError(
                f"{model}: r<={MAX_ROUNDS} total {max_total} is below "
                f"r<={PRIMARY_ROUND_BUDGET} total {primary_total}"
            )

        heatmap_total = 0
        for scenario in sorted_scenarios:
            scenario_df = model_df[model_df["scenario"] == scenario]
            heatmap_total += int(_budgeted_success_mask(scenario_df, MAX_ROUNDS).sum())
        curve_endpoint = int(_budgeted_success_mask(model_df, MAX_ROUNDS).sum())
        if heatmap_total != curve_endpoint:
            raise ValueError(
                f"{model}: heatmap r<={MAX_ROUNDS} total {heatmap_total} does "
                f"not match budget curve endpoint {curve_endpoint}"
            )


def _model_stroke_width(model_id: str, default: float) -> float:
    return default + 1.0 if model_id == "qwen3-4b-rl" else default


def _model_marker_size(model_id: str, default: float) -> float:
    return default + 1.4 if model_id == "qwen3-4b-rl" else default


def _format_duration_ms(value_ms: float) -> str:
    if value_ms >= 1000:
        return f"{value_ms / 1000:.1f}s"
    return f"{value_ms:.0f}ms"


def _duration_tick_label(value: float, _pos: float) -> str:
    return _format_duration_ms(float(value))


def apply_paper_plot_theme() -> None:
    import seaborn as sns

    sns.set_theme(
        style="whitegrid",
        context="paper",
        rc={
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Nimbus Roman",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "axes.facecolor": "#FFFFFF",
            "figure.facecolor": "#FFFFFF",
            "grid.color": "#E5EAF0",
            "grid.alpha": 0.45,
            "axes.edgecolor": "#D2DAE5",
            "axes.titlesize": 17,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "legend.title_fontsize": 14,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        },
    )


def save_plots_for_paper_theme(
    df: pd.DataFrame,
    output_dir_str: str,
    show_titles: bool = False,
) -> list[str]:
    import seaborn as sns

    output_dir = Path(output_dir_str)
    output_dir.mkdir(exist_ok=True)

    stale_stems = [
        "2c_success_at_k_curves",
        "2b_heatmap_ci_width_by_scenario",
        "3_boxplot_turns_by_scenario",
        "3_boxplot_rounds_by_scenario",
    ]
    stale_names = [f"{stem}.{ext}" for stem in stale_stems for ext in PLOT_EXTENSIONS]
    for stale in stale_names:
        stale_path = output_dir / stale
        if stale_path.exists():
            stale_path.unlink()

    apply_paper_plot_theme()

    sorted_scenarios = sorted(df["scenario"].unique(), key=natural_sort_key)
    scenario_display_map = {s: short_scenario_label(s) for s in sorted_scenarios}
    scenario_display_order = [scenario_display_map[s] for s in sorted_scenarios]
    models_present = sorted(df["model"].unique())
    is_e1_trace_comparison = set(models_present) == set(E1_TRACE_MODEL_ORDER)
    model_order = ordered_model_ids(models_present)
    model_display_map = {m: format_model_name(m) for m in model_order}
    display_order = [model_display_map[m] for m in model_order]
    x_axis_model_order = list(reversed(model_order))
    x_axis_display_order = [model_display_map[m] for m in x_axis_model_order]
    palette_colors = sns.color_palette("colorblind", n_colors=len(model_order))
    model_palette = {
        model: palette_colors[i % len(palette_colors)]
        for i, model in enumerate(model_order)
    }
    display_palette = {
        model_display_map[model]: model_palette[model] for model in model_order
    }
    df_plot = df.copy()
    df_plot["model_display"] = df_plot["model"].map(model_display_map)
    _assert_budgeted_plot_consistency(
        df_plot,
        model_order=model_order,
        sorted_scenarios=sorted_scenarios,
    )

    figure_files: list[str] = []
    FIG_W = 7.2
    FIG_H = 4.8
    FIG_H_TALL = 6.8
    FIG_H_SCENARIO = 6.2

    def _successful_timing_summary(column: str, *, seed: int) -> pd.DataFrame:
        rows: list[dict[str, float | int | str]] = []
        rng = np.random.default_rng(seed)
        bootstrap_n = 2000
        for model in x_axis_model_order:
            model_df = df_plot[df_plot["model"] == model]
            values = pd.to_numeric(model_df[column], errors="coerce").dropna()
            count = len(values)
            if count == 0:
                continue
            mean_value = float(values.mean())
            boot_values: list[float] = []
            for _ in range(bootstrap_n):
                sample_idx = rng.integers(0, count, count)
                sampled = values.iloc[sample_idx]
                boot_values.append(float(sampled.mean()))
            ci_low, ci_high = bootstrap_percentile_ci(boot_values)
            rows.append(
                {
                    "model": model,
                    "model_display": model_display_map[model],
                    "mean_ms": mean_value,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n_success": count,
                }
            )
        return pd.DataFrame(rows)

    def _plot_success_timing_summary(
        summary_df: pd.DataFrame,
        *,
        file_stem: str,
        ylabel: str,
        title: str,
    ) -> None:
        if summary_df.empty:
            return

        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        x_positions = list(range(len(summary_df)))
        values = summary_df["mean_ms"].to_numpy(dtype=float)
        ci_low = summary_df["ci_low"].to_numpy(dtype=float)
        ci_high = summary_df["ci_high"].to_numpy(dtype=float)
        colors = [display_palette[name] for name in summary_df["model_display"]]
        label_offsets: dict[str, tuple[int, int]] = {
            "qwen3_4b_instruct_2507_base_paper_static": (0, 8),
            "qwen/qwen3-4b": (0, 8),
            "qwen/qwen3-4b-sft": (0, 8),
            "qwen/qwen3-4b-sft-rl": (0, 8),
            "prime_rl_t_20260218_164349_step_1000": (0, 8),
            "deepseek/deepseek-v3.2": (0, 8),
            "anthropic/claude-opus-4.6": (0, 8),
            "anthropic/claude-opus-4.7": (0, 8),
            "openai/gpt-5.2": (0, 8),
            "google/gemini-3-flash-preview": (0, 8),
        }
        for idx, color in enumerate(colors):
            ax.errorbar(
                x_positions[idx],
                values[idx],
                yerr=[
                    [max(0.0, values[idx] - ci_low[idx])],
                    [max(0.0, ci_high[idx] - values[idx])],
                ],
                fmt="o",
                color=color,
                ecolor="#243447",
                markersize=7,
                elinewidth=1.4,
                capsize=3,
            )
            model_id = str(summary_df.iloc[idx]["model"])
            dx, dy = label_offsets.get(model_id, (0, 8))
            ax.annotate(
                f"{_format_duration_ms(values[idx])} (n={int(summary_df.iloc[idx]['n_success'])})",
                (x_positions[idx], values[idx]),
                textcoords="offset points",
                xytext=(dx, dy),
                ha="center",
                va="bottom",
                fontsize=8.8,
                color="#243447",
                bbox={
                    "boxstyle": "round,pad=0.1",
                    "facecolor": "#FFFFFF",
                    "alpha": 0.8,
                    "edgecolor": "none",
                },
            )
        if show_titles:
            ax.set_title(title, weight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Model")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(summary_df["model_display"].tolist())
        _highlight_axis_model_labels(ax)
        positive_values = [
            float(v) for v in values if np.isfinite(float(v)) and float(v) > 0
        ]
        max_value = float(np.nanmax(np.maximum(values, ci_high)))
        if (
            len(positive_values) >= 2
            and max(positive_values) / min(positive_values) > 25
        ):
            ax.set_yscale("log")
            ax.set_ylabel(f"{ylabel} (log scale)")
            ax.set_ylim(min(positive_values) / 1.8, max_value * 1.8)
        else:
            ax.set_ylim(0, max_value * 1.15 if max_value > 0 else 1.0)
        ax.yaxis.set_major_formatter(FuncFormatter(_duration_tick_label))
        ax.text(
            0.01,
            0.02,
            "Error bars are 95% bootstrap CI over successful runs only.",
            transform=ax.transAxes,
            fontsize=9.3,
            color="#445",
        )
        fig.tight_layout()
        for ext in PLOT_EXTENSIONS:
            file_name = f"{file_stem}.{ext}"
            _save_plot(fig, output_dir / file_name, stem=file_stem, ext=ext)
            figure_files.append(file_name)
        plt.close(fig)

    model_rows: list[dict[str, float | str]] = []
    for model in x_axis_model_order:
        model_df = df_plot[df_plot["model"] == model]
        successes = int(model_df["success"].sum())
        total = len(model_df)
        rate = successes / total
        ci_low, ci_high = wilson_interval(successes, total)
        model_rows.append(
            {
                "model_display": model_display_map[model],
                "success_rate": rate,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "successes": successes,
                "runs": total,
            }
        )
    model_summary = pd.DataFrame(model_rows)

    fig, ax1 = plt.subplots(figsize=(FIG_W, FIG_H))
    x_positions = list(range(len(model_summary)))
    rates = model_summary["success_rate"].to_numpy()
    ci_low = model_summary["ci_low"].to_numpy()
    ci_high = model_summary["ci_high"].to_numpy()
    colors = [display_palette[n] for n in model_summary["model_display"]]
    for idx, color in enumerate(colors):
        ax1.errorbar(
            x_positions[idx],
            rates[idx],
            yerr=[
                [max(0.0, rates[idx] - ci_low[idx])],
                [max(0.0, ci_high[idx] - rates[idx])],
            ],
            fmt="o",
            color=color,
            ecolor="#243447",
            markersize=7.6,
            elinewidth=1.2,
            capsize=3,
        )
        label = f"{100 * rates[idx]:.1f}%"
        if rates[idx] >= 0.9:
            label_dx = 0
            label_dy = -12
        else:
            label_dx = 0
            label_dy = 6
        ax1.annotate(
            label,
            (x_positions[idx], rates[idx]),
            textcoords="offset points",
            xytext=(label_dx, label_dy),
            ha="center",
            va="top" if label_dy < 0 else "bottom",
            fontsize=8.8,
            color="#243447",
            bbox={
                "boxstyle": "round,pad=0.1",
                "facecolor": "#FFFFFF",
                "alpha": 0.75,
                "edgecolor": "none",
            },
        )
    if show_titles:
        ax1.set_title(
            "Success rate under 60-round budget (95% Wilson CI)", weight="bold"
        )
    ax1.set_ylabel(r"Per-run success rate (root within $\leq 60$ rounds)")
    ax1.set_xlabel("")
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(model_summary["model_display"].tolist())
    _highlight_axis_model_labels(ax1)
    ax1.set_ylim(0, 1.02)
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    reference_baselines = [
        (0.75, "Human baseline (reported)"),
        (0.25, "Traditional tools baseline (reported)"),
    ]
    baseline_text_transform = blended_transform_factory(ax1.transAxes, ax1.transData)
    for y, label in reference_baselines:
        ax1.axhline(y=y, color="#A8B3C2", linestyle="--", linewidth=0.9, alpha=0.7)
        ax1.text(
            0.99,
            y + 0.01,
            label,
            ha="right",
            va="bottom",
            fontsize=8.4,
            color="#6B7280",
            transform=baseline_text_transform,
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "#FFFFFF",
                "alpha": 0.72,
                "edgecolor": "none",
            },
        )
    fig.tight_layout()
    for ext in PLOT_EXTENSIONS:
        file_name = f"1_overall_success_rate.{ext}"
        fig.savefig(output_dir / file_name, format=ext, transparent=False, dpi=300)
        figure_files.append(file_name)
    plt.close(fig)

    model_rows_r20: list[dict[str, float | str]] = []
    for model in x_axis_model_order:
        model_df = df_plot[df_plot["model"] == model]
        success_within_r20 = model_df["success"].fillna(False).astype(bool) & (
            pd.to_numeric(model_df["turns"], errors="coerce") <= PRIMARY_ROUND_BUDGET
        )
        successes_r20 = int(success_within_r20.sum())
        total = len(model_df)
        rate_r20 = successes_r20 / total
        ci_low_r20, ci_high_r20 = wilson_interval(successes_r20, total)
        model_rows_r20.append(
            {
                "model_display": model_display_map[model],
                "success_rate": rate_r20,
                "ci_low": ci_low_r20,
                "ci_high": ci_high_r20,
                "successes": successes_r20,
                "runs": total,
            }
        )
    model_summary_r20 = pd.DataFrame(model_rows_r20)

    fig, ax1b = plt.subplots(figsize=(FIG_W, FIG_H))
    x_positions_r20 = list(range(len(model_summary_r20)))
    rates_r20 = model_summary_r20["success_rate"].to_numpy()
    ci_low_r20 = model_summary_r20["ci_low"].to_numpy()
    ci_high_r20 = model_summary_r20["ci_high"].to_numpy()
    colors_r20 = [display_palette[n] for n in model_summary_r20["model_display"]]
    for idx, color in enumerate(colors_r20):
        ax1b.errorbar(
            x_positions_r20[idx],
            rates_r20[idx],
            yerr=[
                [max(0.0, rates_r20[idx] - ci_low_r20[idx])],
                [max(0.0, ci_high_r20[idx] - rates_r20[idx])],
            ],
            fmt="o",
            color=color,
            ecolor="#243447",
            markersize=7,
            elinewidth=1.4,
            capsize=3,
        )
        label = f"{100 * rates_r20[idx]:.1f}%"
        if rates_r20[idx] >= 0.9:
            label_dx = 0
            label_dy = -12
        else:
            label_dx = 0
            label_dy = 6
        ax1b.annotate(
            label,
            (x_positions_r20[idx], rates_r20[idx]),
            textcoords="offset points",
            xytext=(label_dx, label_dy),
            ha="center",
            va="top" if label_dy < 0 else "bottom",
            fontsize=8.8,
            color="#243447",
            bbox={
                "boxstyle": "round,pad=0.1",
                "facecolor": "#FFFFFF",
                "alpha": 0.75,
                "edgecolor": "none",
            },
        )
    if show_titles:
        ax1b.set_title(
            f"Success rate under {PRIMARY_ROUND_BUDGET}-round budget (95% Wilson CI)",
            weight="bold",
        )
    ax1b.set_ylabel(
        rf"Per-run success rate (root within $\leq {PRIMARY_ROUND_BUDGET}$ rounds)"
    )
    ax1b.set_xlabel("")
    ax1b.set_xticks(x_positions_r20)
    ax1b.set_xticklabels(model_summary_r20["model_display"].tolist())
    _highlight_axis_model_labels(ax1b)
    ax1b.set_ylim(0, 1)
    ax1b.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    baseline_text_transform_r20 = blended_transform_factory(
        ax1b.transAxes, ax1b.transData
    )
    for y, label in reference_baselines:
        ax1b.axhline(y=y, color="#9AA4B2", linestyle="--", linewidth=1.0, alpha=0.75)
        ax1b.text(
            0.99,
            y + 0.01,
            label,
            ha="right",
            va="bottom",
            fontsize=8.8,
            color="#5A6473",
            transform=baseline_text_transform_r20,
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "#FFFFFF",
                "alpha": 0.85,
                "edgecolor": "none",
            },
        )
    fig.tight_layout()
    for ext in PLOT_EXTENSIONS:
        file_name = f"1b_overall_success_rate_r20.{ext}"
        fig.savefig(output_dir / file_name, format=ext, transparent=False, dpi=300)
        figure_files.append(file_name)
    plt.close(fig)

    model_rows_r10: list[dict[str, float | str]] = []
    for model in x_axis_model_order:
        model_df = df_plot[df_plot["model"] == model]
        success_within_r10 = model_df["success"].fillna(False).astype(bool) & (
            pd.to_numeric(model_df["turns"], errors="coerce") <= 10
        )
        successes_r10 = int(success_within_r10.sum())
        total = len(model_df)
        rate_r10 = successes_r10 / total
        ci_low_r10, ci_high_r10 = wilson_interval(successes_r10, total)
        model_rows_r10.append(
            {
                "model_display": model_display_map[model],
                "success_rate": rate_r10,
                "ci_low": ci_low_r10,
                "ci_high": ci_high_r10,
                "successes": successes_r10,
                "runs": total,
            }
        )
    model_summary_r10 = pd.DataFrame(model_rows_r10)

    fig, ax1d = plt.subplots(figsize=(FIG_W, FIG_H))
    x_positions_r10 = list(range(len(model_summary_r10)))
    rates_r10 = model_summary_r10["success_rate"].to_numpy()
    ci_low_r10 = model_summary_r10["ci_low"].to_numpy()
    ci_high_r10 = model_summary_r10["ci_high"].to_numpy()
    colors_r10 = [display_palette[n] for n in model_summary_r10["model_display"]]
    for idx, color in enumerate(colors_r10):
        ax1d.errorbar(
            x_positions_r10[idx],
            rates_r10[idx],
            yerr=[
                [max(0.0, rates_r10[idx] - ci_low_r10[idx])],
                [max(0.0, ci_high_r10[idx] - rates_r10[idx])],
            ],
            fmt="o",
            color=color,
            ecolor="#243447",
            markersize=7,
            elinewidth=1.4,
            capsize=3,
        )
        label = f"{100 * rates_r10[idx]:.1f}%"
        if rates_r10[idx] >= 0.9:
            label_dx = 0
            label_dy = -12
        else:
            label_dx = 0
            label_dy = 6
        ax1d.annotate(
            label,
            (x_positions_r10[idx], rates_r10[idx]),
            textcoords="offset points",
            xytext=(label_dx, label_dy),
            ha="center",
            va="top" if label_dy < 0 else "bottom",
            fontsize=8.8,
            color="#243447",
            bbox={
                "boxstyle": "round,pad=0.1",
                "facecolor": "#FFFFFF",
                "alpha": 0.75,
                "edgecolor": "none",
            },
        )
    if show_titles:
        ax1d.set_title(
            "Success rate under 10-round budget (95% Wilson CI)", weight="bold"
        )
    ax1d.set_ylabel(r"Success rate (root within $\leq 10$ rounds)")
    ax1d.set_xlabel("")
    ax1d.set_xticks(x_positions_r10)
    ax1d.set_xticklabels(model_summary_r10["model_display"].tolist())
    _highlight_axis_model_labels(ax1d)
    ax1d.set_ylim(0, 1)
    ax1d.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    baseline_text_transform_r10 = blended_transform_factory(
        ax1d.transAxes, ax1d.transData
    )
    for y, label in reference_baselines:
        ax1d.axhline(y=y, color="#9AA4B2", linestyle="--", linewidth=1.0, alpha=0.75)
        ax1d.text(
            0.99,
            y + 0.01,
            label,
            ha="right",
            va="bottom",
            fontsize=8.8,
            color="#5A6473",
            transform=baseline_text_transform_r10,
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "#FFFFFF",
                "alpha": 0.85,
                "edgecolor": "none",
            },
        )
    fig.tight_layout()
    for ext in PLOT_EXTENSIONS:
        file_name = f"1d_overall_success_rate_r10.{ext}"
        fig.savefig(output_dir / file_name, format=ext, transparent=False, dpi=300)
        figure_files.append(file_name)
    plt.close(fig)

    rounds_rows: list[dict[str, float | str]] = []
    rounds_bootstrap_n = 2000
    rounds_rng = np.random.default_rng(123)
    for model in x_axis_model_order:
        model_df = df_plot[df_plot["model"] == model]
        succ_turns = pd.to_numeric(
            model_df.loc[model_df["success"].fillna(False).astype(bool), "turns"],
            errors="coerce",
        ).dropna()
        success_count = len(succ_turns)
        if success_count == 0:
            rounds_rows.append(
                {
                    "model": model,
                    "model_display": model_display_map[model],
                    "avg_rounds_until_root": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "n_success": 0,
                }
            )
            continue

        mean_rounds = float(succ_turns.mean())
        boot_values: list[float] = []
        for _ in range(rounds_bootstrap_n):
            sample_idx = rounds_rng.integers(0, success_count, success_count)
            sampled = succ_turns.iloc[sample_idx]
            boot_values.append(float(sampled.mean()))
        ci_low_rounds, ci_high_rounds = bootstrap_percentile_ci(boot_values)
        rounds_rows.append(
            {
                "model": model,
                "model_display": model_display_map[model],
                "avg_rounds_until_root": mean_rounds,
                "ci_low": ci_low_rounds,
                "ci_high": ci_high_rounds,
                "n_success": success_count,
            }
        )

    rounds_df = pd.DataFrame(rounds_rows)
    rounds_df = rounds_df.dropna(subset=["avg_rounds_until_root"]).reset_index(
        drop=True
    )

    fig, ax1c = plt.subplots(figsize=(FIG_W, FIG_H))
    x_positions_rounds = list(range(len(rounds_df)))
    rounds_vals = rounds_df["avg_rounds_until_root"].to_numpy()
    rounds_ci_low = rounds_df["ci_low"].to_numpy()
    rounds_ci_high = rounds_df["ci_high"].to_numpy()
    rounds_colors = [display_palette[n] for n in rounds_df["model_display"]]
    rounds_label_offsets: dict[str, tuple[int, int]] = {
        "qwen3_4b_instruct_2507_base_paper_static": (0, 8),
        "qwen/qwen3-4b": (0, 8),
        "qwen/qwen3-4b-sft": (0, 8),
        "qwen/qwen3-4b-sft-rl": (0, 8),
        "prime_rl_t_20260218_164349_step_1000": (0, 8),
        "deepseek/deepseek-v3.2": (0, 8),
        "anthropic/claude-opus-4.6": (0, 8),
        "anthropic/claude-opus-4.7": (0, 8),
    }
    for idx, color in enumerate(rounds_colors):
        ax1c.errorbar(
            x_positions_rounds[idx],
            rounds_vals[idx],
            yerr=[
                [max(0.0, rounds_vals[idx] - rounds_ci_low[idx])],
                [max(0.0, rounds_ci_high[idx] - rounds_vals[idx])],
            ],
            fmt="o",
            color=color,
            ecolor="#243447",
            markersize=7,
            elinewidth=1.4,
            capsize=3,
        )
        model_id = str(rounds_df.iloc[idx]["model"])
        dx, dy = rounds_label_offsets.get(model_id, (0, 8))
        ax1c.annotate(
            f"{rounds_vals[idx]:.1f} (n={int(rounds_df.iloc[idx]['n_success'])})",
            (x_positions_rounds[idx], rounds_vals[idx]),
            textcoords="offset points",
            xytext=(dx, dy),
            ha="center",
            va="bottom",
            fontsize=8.8,
            color="#243447",
            bbox={
                "boxstyle": "round,pad=0.1",
                "facecolor": "#FFFFFF",
                "alpha": 0.8,
                "edgecolor": "none",
            },
        )
    if show_titles:
        ax1c.set_title("Average rounds until root (successful runs)", weight="bold")
    ax1c.set_ylabel("Average rounds until root")
    ax1c.set_xlabel("Model")
    ax1c.set_xticks(x_positions_rounds)
    ax1c.set_xticklabels(rounds_df["model_display"].tolist())
    _highlight_axis_model_labels(ax1c)
    y_top_rounds = max(20.0, min(MAX_ROUNDS, float(np.nanmax(rounds_ci_high)) + 3.0))
    ax1c.set_ylim(0, y_top_rounds)
    ax1c.text(
        0.01,
        0.02,
        "Error bars are 95% bootstrap CI.",
        transform=ax1c.transAxes,
        fontsize=9.3,
        color="#445",
    )
    fig.tight_layout()
    for ext in PLOT_EXTENSIONS:
        file_name = f"1c_avg_rounds_until_root.{ext}"
        fig.savefig(output_dir / file_name, format=ext, transparent=False, dpi=300)
        figure_files.append(file_name)
    plt.close(fig)

    heatmap_df = df_plot.copy()
    heatmap_df["success_within_max_budget"] = _budgeted_success_mask(
        heatmap_df,
        MAX_ROUNDS,
    )
    success_count_pivot = (
        heatmap_df.groupby(["scenario", "model"])["success_within_max_budget"]
        .sum()
        .unstack(level="model")
    )
    run_count_pivot = (
        df_plot.groupby(["scenario", "model"]).size().unstack(level="model")
    )
    success_rate_pivot = (
        heatmap_df.groupby(["scenario", "model"])["success_within_max_budget"]
        .mean()
        .unstack(level="model")
    )
    success_count_pivot = success_count_pivot.reindex(
        sorted_scenarios, axis="index"
    ).reindex(x_axis_model_order, axis="columns")
    run_count_pivot = run_count_pivot.reindex(sorted_scenarios, axis="index").reindex(
        x_axis_model_order, axis="columns"
    )
    success_rate_pivot = success_rate_pivot.reindex(
        sorted_scenarios, axis="index"
    ).reindex(x_axis_model_order, axis="columns")
    success_rate_pivot.index = [
        scenario_display_map[s] for s in success_rate_pivot.index
    ]
    success_rate_pivot.columns = [
        model_display_map[c] for c in success_rate_pivot.columns
    ]
    annot = success_count_pivot.copy().astype(object)
    for scenario in success_count_pivot.index:
        for model in success_count_pivot.columns:
            successes = success_count_pivot.loc[scenario, model]
            runs = run_count_pivot.loc[scenario, model]
            annot.loc[scenario, model] = f"{int(successes)}/{int(runs)}"
    annot.index = success_rate_pivot.index
    annot.columns = success_rate_pivot.columns
    fig, ax2 = plt.subplots(figsize=(FIG_W, FIG_H_TALL))
    sns.heatmap(
        success_rate_pivot,
        annot=annot,
        fmt="",
        linewidths=0.6,
        linecolor="#EEF2F7",
        cmap=sns.light_palette("#1f77b4", as_cmap=True),
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Success rate", "location": "right", "pad": 0.02},
        ax=ax2,
    )
    cbar = ax2.collections[0].colorbar
    if cbar is not None:
        cbar.set_label("")
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    _style_heatmap_xlabels(ax2, rotation=40 if is_e1_trace_comparison else 25)
    plt.yticks(rotation=0)
    _highlight_axis_model_labels(ax2)
    fig.tight_layout()
    for ext in PLOT_EXTENSIONS:
        stem = "2_heatmap_success_rate_by_scenario"
        file_name = f"{stem}.{ext}"
        _save_plot(fig, output_dir / file_name, stem=stem, ext=ext)
        figure_files.append(file_name)
    plt.close(fig)

    heatmap_r20_df = df_plot.copy()
    heatmap_r20_df["success_within_primary_budget"] = _budgeted_success_mask(
        heatmap_r20_df,
        PRIMARY_ROUND_BUDGET,
    )
    success_count_pivot_r20 = (
        heatmap_r20_df.groupby(["scenario", "model"])["success_within_primary_budget"]
        .sum()
        .unstack(level="model")
    )
    success_rate_pivot_r20 = (
        heatmap_r20_df.groupby(["scenario", "model"])["success_within_primary_budget"]
        .mean()
        .unstack(level="model")
    )
    success_count_pivot_r20 = success_count_pivot_r20.reindex(
        sorted_scenarios, axis="index"
    ).reindex(x_axis_model_order, axis="columns")
    success_rate_pivot_r20 = success_rate_pivot_r20.reindex(
        sorted_scenarios, axis="index"
    ).reindex(x_axis_model_order, axis="columns")
    success_rate_pivot_r20.index = [
        scenario_display_map[s] for s in success_rate_pivot_r20.index
    ]
    success_rate_pivot_r20.columns = [
        model_display_map[c] for c in success_rate_pivot_r20.columns
    ]
    annot_r20 = success_count_pivot_r20.copy().astype(object)
    for scenario in success_count_pivot_r20.index:
        for model in success_count_pivot_r20.columns:
            successes = success_count_pivot_r20.loc[scenario, model]
            runs = run_count_pivot.loc[scenario, model]
            annot_r20.loc[scenario, model] = f"{int(successes)}/{int(runs)}"
    annot_r20.index = success_rate_pivot_r20.index
    annot_r20.columns = success_rate_pivot_r20.columns
    fig, ax2_r20 = plt.subplots(figsize=(FIG_W, FIG_H_TALL))
    sns.heatmap(
        success_rate_pivot_r20,
        annot=annot_r20,
        fmt="",
        linewidths=0.6,
        linecolor="#EEF2F7",
        cmap=sns.light_palette("#1f77b4", as_cmap=True),
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Success rate", "location": "right", "pad": 0.02},
        ax=ax2_r20,
    )
    cbar_r20 = ax2_r20.collections[0].colorbar
    if cbar_r20 is not None:
        cbar_r20.set_label("")
    ax2_r20.set_xlabel("")
    ax2_r20.set_ylabel("")
    _style_heatmap_xlabels(ax2_r20, rotation=40 if is_e1_trace_comparison else 25)
    plt.yticks(rotation=0)
    _highlight_axis_model_labels(ax2_r20)
    fig.tight_layout()
    for ext in PLOT_EXTENSIONS:
        stem = "2b_heatmap_success_rate_by_scenario_r20"
        file_name = f"{stem}.{ext}"
        _save_plot(fig, output_dir / file_name, stem=stem, ext=ext)
        figure_files.append(file_name)
    plt.close(fig)

    ci_rows = []
    for scenario in sorted_scenarios:
        for model in model_order:
            subset = df_plot[
                (df_plot["scenario"] == scenario) & (df_plot["model"] == model)
            ]
            if subset.empty:
                continue
            successes = int(subset["success"].sum())
            ci_low_s, ci_high_s = wilson_interval(successes, len(subset))
            ci_rows.append(
                {
                    "scenario": scenario,
                    "model": model_display_map[model],
                    "ci_width": ci_high_s - ci_low_s,
                }
            )
    ci_width_pivot = (
        pd.DataFrame(ci_rows)
        .pivot(index="scenario", columns="model", values="ci_width")
        .reindex(sorted_scenarios, axis="index")
        .reindex(x_axis_display_order, axis="columns")
    )
    ci_width_pivot.index = [scenario_display_map[s] for s in ci_width_pivot.index]
    fig, ax2b = plt.subplots(figsize=(FIG_W, FIG_H_TALL))
    sns.heatmap(
        ci_width_pivot,
        annot=True,
        fmt=".2f",
        linewidths=0.6,
        linecolor="#EEF2F7",
        cmap="flare",
        cbar_kws={"label": "95% Wilson CI width"},
        ax=ax2b,
    )
    cbar_ci = ax2b.collections[0].colorbar
    if cbar_ci is not None:
        cbar_ci.set_label("")
    if show_titles:
        ax2b.set_title("Appendix: 95% Wilson CI width by scenario", weight="bold")
    ax2b.set_xlabel("Model")
    ax2b.set_ylabel("")
    _style_heatmap_xlabels(ax2b)
    plt.yticks(rotation=0)
    _highlight_axis_model_labels(ax2b)
    fig.tight_layout()
    for ext in PLOT_EXTENSIONS:
        file_name = f"A1_heatmap_wilson_ci_width_by_scenario.{ext}"
        fig.savefig(output_dir / file_name, format=ext, transparent=False, dpi=300)
        figure_files.append(file_name)
    plt.close(fig)

    fig, ax3 = plt.subplots(figsize=(FIG_W, FIG_H))
    round_budgets_plot = [0, *ROUND_BUDGETS]
    marker_cycle = ["o", "s", "D", "^"]
    budget_curve_order = list(reversed(model_order))
    for idx, model in enumerate(budget_curve_order):
        model_df = df_plot[df_plot["model"] == model]
        curve = budgeted_success_rate_with_ci_for_group(model_df)
        turns = pd.to_numeric(model_df["turns"], errors="coerce")
        success = model_df["success"].fillna(False).astype(bool)
        run_count = len(model_df)
        ys: list[float] = []
        ys_low: list[float] = []
        ys_high: list[float] = []
        for r in round_budgets_plot:
            point = curve.get(r)
            if point is None:
                success_within_budget = int(((success) & (turns <= r)).sum())
                rate = success_within_budget / run_count if run_count > 0 else 0.0
                ci_low, ci_high = wilson_interval(success_within_budget, run_count)
                point = {"rate": rate, "ci_low": ci_low, "ci_high": ci_high}
            ys.append(float(point["rate"]))
            ys_low.append(float(point["ci_low"]))
            ys_high.append(float(point["ci_high"]))
        color = model_palette[model]
        ax3.plot(
            round_budgets_plot,
            ys,
            marker=marker_cycle[idx % len(marker_cycle)],
            linewidth=_model_stroke_width(model, 2.2),
            markersize=_model_marker_size(model, 5.8),
            label=model_display_map[model],
            color=color,
            zorder=4 if model == "qwen3-4b-rl" else 2,
        )
        ax3.fill_between(round_budgets_plot, ys_low, ys_high, color=color, alpha=0.075)
    ax3.set_ylim(0, 1.03)
    ax3.set_xlim(min(round_budgets_plot), max(round_budgets_plot))
    ax3.set_xlabel(r"Round budget $r$")
    ax3.set_ylabel(r"Empirical success within $r$ rounds")
    if show_titles:
        ax3.set_title("Budgeted per-run success rate (95% Wilson CI)", weight="bold")
    ax3.set_xticks(round_budgets_plot)
    ax3.axvline(
        x=20,
        color="#9AA4B2",
        linestyle="--",
        linewidth=1.0,
        alpha=0.9,
        zorder=0,
    )
    ax3.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    handles, labels = ax3.get_legend_handles_labels()
    legend_handle_by_label = dict(zip(labels, handles, strict=False))
    ordered_labels = [model_display_map[model] for model in model_order]
    ordered_handles = [legend_handle_by_label[label] for label in ordered_labels]
    if is_e1_trace_comparison:
        ax3.legend(
            ordered_handles,
            ordered_labels,
            title=None,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            borderaxespad=0.0,
            frameon=True,
            framealpha=0.88,
            fontsize=10.5,
        )
    else:
        ax3.legend(
            ordered_handles,
            ordered_labels,
            title=None,
            loc="lower right",
            frameon=True,
            framealpha=0.88,
            fontsize=11.5,
        )
    _highlight_current_legend(ax3)
    fig.tight_layout()
    for ext in PLOT_EXTENSIONS:
        stem = "2c_success_vs_round_budget"
        file_name = f"{stem}.{ext}"
        _save_plot(fig, output_dir / file_name, stem=stem, ext=ext)
        figure_files.append(file_name)
    plt.close(fig)

    required_deployment_models = {
        point.model for point in BUDGETED_SUCCESS_DEPLOYMENT_POINTS
    }
    if required_deployment_models.issubset(set(model_order)):
        from matplotlib.lines import Line2D

        fig = plt.figure(figsize=(FIG_W, FIG_H))
        size_width_ratio = 5.4
        closed_width_ratio = 1.6
        gs = fig.add_gridspec(
            1, 2, width_ratios=[size_width_ratio, closed_width_ratio], wspace=0.06
        )
        ax_size = fig.add_subplot(gs[0, 0])
        ax_closed = fig.add_subplot(gs[0, 1], sharey=ax_size)
        ax_size.set_axisbelow(True)
        ax_closed.set_axisbelow(True)
        ax_size.grid(True, alpha=0.25)
        ax_closed.grid(axis="y", alpha=0.25)
        ax_closed.grid(axis="x", visible=False)

        size_xmin, size_xmax = 2.4, 1100.0
        qwen_ladder_models = (QWEN3_4B_MODEL, QWEN3_4B_SFT_MODEL, PRIVESC_LLM_4B_MODEL)
        qwen_points: dict[str, tuple[float, float]] = {}

        for point in BUDGETED_SUCCESS_DEPLOYMENT_POINTS:
            model = point.model
            model_df = df_plot[df_plot["model"] == model]
            if model_df.empty:
                continue
            turns = pd.to_numeric(model_df["turns"], errors="coerce")
            success = model_df["success"].fillna(False).astype(bool)
            run_count = len(model_df)
            successes_r20 = int(((success) & (turns <= PRIMARY_ROUND_BUDGET)).sum())
            success_r20 = successes_r20 / run_count
            is_privesc = model == PRIVESC_LLM_4B_MODEL
            is_claude = model == CLAUDE_OPUS_47_MODEL
            marker = DEPLOYMENT_MARKERS[point.deployment]
            color = model_palette[model]
            target_ax = ax_closed if is_claude else ax_size
            x_value = 0.5 if is_claude else point.size_b
            target_ax.scatter(
                [x_value],
                [success_r20],
                s=200 if is_privesc else 110,
                marker=marker,
                color=color,
                edgecolors="#243447",
                linewidths=1.2 if is_privesc else 0.6,
                zorder=6 if is_privesc else 5,
            )
            target_ax.annotate(
                model_display_map[model],
                (x_value, success_r20),
                xytext=point.label_offset,
                textcoords="offset points",
                ha=point.ha,
                va=point.va,
                fontsize=10.0,
                fontweight="bold" if is_privesc else "normal",
                color="#243447",
                clip_on=False,
            )
            if model in qwen_ladder_models:
                qwen_points[model] = (x_value, success_r20)

        for src, dst in zip(qwen_ladder_models[:-1], qwen_ladder_models[1:]):
            if src in qwen_points and dst in qwen_points:
                start = qwen_points[src]
                end = qwen_points[dst]
                connector = FancyArrowPatch(
                    start,
                    end,
                    arrowstyle="-|>,head_length=5,head_width=3",
                    mutation_scale=1.0,
                    shrinkA=11,
                    shrinkB=11,
                    linestyle="--",
                    linewidth=1.6,
                    color="#4B5563",
                    alpha=0.85,
                    zorder=2,
                    transform=ax_size.transData,
                )
                ax_size.add_patch(connector)

        ax_size.set_xscale("log")
        ax_size.set_xlim(size_xmin, size_xmax)
        ax_size.set_xticks([4, 10, 30, 100, 300, 1000])
        ax_size.xaxis.set_major_formatter(
            FuncFormatter(lambda x, _: f"{int(x)}B")
        )
        ax_size.set_xlabel("Model size (parameters, log scale)")
        ax_size.set_ylabel(
            rf"Per-run success rate (root within $\leq${PRIMARY_ROUND_BUDGET} rounds)"
        )
        y_min = 0.30
        y_max = 1.05
        ax_size.set_ylim(y_min, y_max)
        ax_size.set_yticks(np.arange(0.3, 1.0001, 0.1))
        ax_size.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))

        ax_closed.set_xlim(0.0, 1.0)
        ax_closed.set_xticks([0.5])
        ax_closed.set_xticklabels(["size\nundisclosed"], fontsize=9.0)
        ax_closed.set_xlabel("")
        ax_closed.tick_params(axis="y", which="both", left=False, labelleft=False)
        ax_size.spines["top"].set_visible(False)
        ax_closed.spines["top"].set_visible(False)
        ax_closed.spines["left"].set_linestyle((0, (3, 3)))
        ax_closed.spines["left"].set_color("#9CA3AF")

        dx = 0.012
        dy = 0.022
        closed_dx = dx * (size_width_ratio / closed_width_ratio)
        ax_size.plot(
            [1 - dx, 1 + dx],
            [-dy, dy],
            transform=ax_size.transAxes,
            color="#6B7280",
            clip_on=False,
            linewidth=1.0,
        )
        ax_size.plot(
            [1 - dx, 1 + dx],
            [1 - dy, 1 + dy],
            transform=ax_size.transAxes,
            color="#6B7280",
            clip_on=False,
            linewidth=1.0,
        )
        ax_closed.plot(
            [-closed_dx, closed_dx],
            [-dy, dy],
            transform=ax_closed.transAxes,
            color="#6B7280",
            clip_on=False,
            linewidth=1.0,
        )
        ax_closed.plot(
            [-closed_dx, closed_dx],
            [1 - dy, 1 + dy],
            transform=ax_closed.transAxes,
            color="#6B7280",
            clip_on=False,
            linewidth=1.0,
        )

        legend_entries = [
            ("Local open-weight", "Local open-weight"),
            ("Hosted open-weight API", "Hosted open-weight"),
            ("Closed API", "Closed API"),
        ]
        marker_legend_handles = [
            Line2D(
                [0],
                [0],
                marker=DEPLOYMENT_MARKERS[key],
                color="w",
                markerfacecolor="#6B7280",
                markeredgecolor="#243447",
                markersize=9.5,
                linewidth=0,
                label=label,
            )
            for key, label in legend_entries
        ]
        if show_titles:
            ax_size.set_title(
                "Budgeted reliability by model size",
                weight="bold",
            )

        ax_size.legend(
            handles=marker_legend_handles,
            loc="upper right",
            bbox_to_anchor=(1.0, 1.0),
            bbox_transform=ax_size.transAxes,
            ncol=1,
            frameon=True,
            framealpha=0.92,
            fontsize=10.0,
            handletextpad=0.6,
            borderpad=0.55,
            borderaxespad=1.1,
            edgecolor="#9CA3AF",
        )
        fig.tight_layout(pad=0.5)
        for ext in PLOT_EXTENSIONS:
            stem = "budgeted_success_by_size"
            file_name = f"{stem}.{ext}"
            _save_plot(fig, output_dir / file_name, stem=stem, ext=ext)
            figure_files.append(file_name)
        plt.close(fig)

    df_successful = df_plot[df_plot["success"]].copy()
    if not df_successful.empty:
        df_successful["scenario_display"] = df_successful["scenario"].map(
            scenario_display_map
        )
        successful_turns = pd.to_numeric(
            df_successful["turns"], errors="coerce"
        ).dropna()
        rounds_axis_top = min(
            float(MAX_ROUNDS),
            max(
                20.0, float(np.ceil((float(successful_turns.max()) + 2.0) / 5.0) * 5.0)
            ),
        )
        show_budget_cap = rounds_axis_top >= float(MAX_ROUNDS)
        model_success_counts = (
            df_successful.groupby("model_display")
            .size()
            .reindex(display_order)
            .fillna(0)
            .astype(int)
        )
        success_count_text = ", ".join(
            f"{model}: n={int(count)}" for model, count in model_success_counts.items()
        )
        fig, ax4 = plt.subplots(figsize=(FIG_W, FIG_H_SCENARIO))
        sns.boxplot(
            data=df_successful,
            x="scenario_display",
            y="turns",
            hue="model_display",
            order=scenario_display_order,
            hue_order=display_order,
            palette=display_palette,
            showfliers=False,
            linewidth=1,
            ax=ax4,
        )
        if show_titles:
            ax4.set_title(
                "Appendix: rounds to completion by scenario (successful runs)",
                weight="bold",
            )
        ax4.set_xlabel("Scenario")
        ax4.set_ylabel("Rounds to completion")
        ax4.set_ylim(0, rounds_axis_top)
        if show_budget_cap:
            ax4.axhline(
                y=MAX_ROUNDS,
                color="#9AA4B2",
                linestyle="--",
                linewidth=1.0,
                alpha=0.9,
            )
            ax4.text(
                0.995,
                MAX_ROUNDS - 0.8,
                "budget cap (R=60)",
                transform=blended_transform_factory(ax4.transAxes, ax4.transData),
                ha="right",
                va="top",
                fontsize=8.6,
                color="#5A6473",
            )
        plt.xticks(rotation=45, ha="right")
        ax4.legend(title="Model")
        _highlight_current_legend(ax4)
        ax4.text(
            0.01,
            0.98,
            success_count_text,
            transform=ax4.transAxes,
            ha="left",
            va="top",
            fontsize=8.6,
            color="#445",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "#FFFFFF",
                "alpha": 0.8,
                "edgecolor": "#D2DAE5",
            },
        )
        fig.tight_layout()
        for ext in PLOT_EXTENSIONS:
            file_name = f"A2_boxplot_rounds_by_scenario.{ext}"
            fig.savefig(output_dir / file_name, format=ext, transparent=False, dpi=300)
            figure_files.append(file_name)
        plt.close(fig)

        fig, ax4b = plt.subplots(figsize=(FIG_W, FIG_H))
        sns.boxplot(
            data=df_successful,
            x="model_display",
            y="turns",
            hue="model_display",
            order=x_axis_display_order,
            palette=display_palette,
            showfliers=False,
            linewidth=1,
            legend=False,
            ax=ax4b,
        )
        if show_titles:
            ax4b.set_title(
                "Efficiency: rounds to completion by model (successful runs)",
                weight="bold",
            )
        ax4b.set_xlabel("Model")
        ax4b.set_ylabel("Rounds to completion")
        ax4b.set_ylim(0, rounds_axis_top)
        if show_budget_cap:
            ax4b.axhline(
                y=MAX_ROUNDS,
                color="#9AA4B2",
                linestyle="--",
                linewidth=1.0,
                alpha=0.9,
            )
            ax4b.text(
                0.995,
                MAX_ROUNDS - 0.8,
                "budget cap (R=60)",
                transform=blended_transform_factory(ax4b.transAxes, ax4b.transData),
                ha="right",
                va="top",
                fontsize=8.6,
                color="#5A6473",
            )
        ax4b.set_xticks(range(len(x_axis_display_order)))
        ax4b.set_xticklabels(
            [
                f"{model}\n(n={int(model_success_counts[model])})"
                for model in x_axis_display_order
            ]
        )
        _highlight_axis_model_labels(ax4b)
        fig.tight_layout()
        for ext in PLOT_EXTENSIONS:
            file_name = f"3b_boxplot_rounds_by_model.{ext}"
            fig.savefig(output_dir / file_name, format=ext, transparent=False, dpi=300)
            figure_files.append(file_name)
        plt.close(fig)

    fig, ax5 = plt.subplots(figsize=(FIG_W, FIG_H))
    message_cap = float(df_plot["messages"].quantile(0.95))
    df_messages = df_plot.copy()
    df_messages["messages_plot"] = df_messages["messages"].clip(upper=message_cap)
    sns.violinplot(
        data=df_messages,
        x="model_display",
        y="messages_plot",
        hue="model_display",
        inner="quartile",
        cut=0,
        order=x_axis_display_order,
        legend=False,
        palette=display_palette,
        ax=ax5,
    )
    if show_titles:
        ax5.set_title("Trace messages per run (all runs)", weight="bold")
    ax5.set_xlabel("Model")
    ax5.set_ylabel("Messages per run (axis clipped at p95)")
    _highlight_axis_model_labels(ax5)
    ax5.text(
        0.01,
        0.02,
        f"Upper axis clipped at p95={message_cap:.0f} messages",
        transform=ax5.transAxes,
        fontsize=9.3,
        color="#445",
    )
    fig.tight_layout()
    for ext in PLOT_EXTENSIONS:
        file_name = f"4_agent_verbosity_distribution.{ext}"
        fig.savefig(output_dir / file_name, format=ext, transparent=False, dpi=300)
        figure_files.append(file_name)
    plt.close(fig)

    fig, ax6 = plt.subplots(figsize=(FIG_W, FIG_H))
    sns.violinplot(
        data=df_plot,
        x="model_display",
        y="total_tokens",
        hue="model_display",
        inner="quartile",
        cut=0,
        order=x_axis_display_order,
        legend=False,
        palette=display_palette,
        ax=ax6,
    )
    if show_titles:
        ax6.set_title("Token usage: total tokens per run (all runs)", weight="bold")
    ax6.set_xlabel("Model")
    ax6.set_ylabel("Total tokens per run")
    _highlight_axis_model_labels(ax6)
    token_min = max(
        1.0, float(df_plot["total_tokens"].replace(0, pd.NA).dropna().min())
    )
    token_max = float(df_plot["total_tokens"].max())
    if token_max / token_min > 200:
        ax6.set_yscale("log")
        ax6.set_ylabel("Total tokens per run (log scale)")
    fig.tight_layout()
    for ext in PLOT_EXTENSIONS:
        file_name = f"4b_token_usage_distribution.{ext}"
        fig.savefig(output_dir / file_name, format=ext, transparent=False, dpi=300)
        figure_files.append(file_name)
    plt.close(fig)

    tool_usage_long = (
        df_plot.melt(
            id_vars="model_display",
            value_vars=["exec_command_count", "test_credentials_count"],
            var_name="tool",
            value_name="calls",
        )
        .assign(
            model_display=lambda d: pd.Categorical(
                d["model_display"], categories=x_axis_display_order, ordered=True
            )
        )
        .sort_values("model_display")
    )
    tool_usage_long = tool_usage_long.replace(
        {
            "tool": {
                "exec_command_count": "Command executions",
                "test_credentials_count": "Credential checks",
            }
        }
    )
    tool_order = ["Command executions", "Credential checks"]
    tool_usage_long["tool"] = pd.Categorical(
        tool_usage_long["tool"], categories=tool_order, ordered=True
    )
    fig, ax7 = plt.subplots(figsize=(FIG_W, FIG_H))
    sns.barplot(
        data=tool_usage_long,
        x="model_display",
        y="calls",
        hue="tool",
        hue_order=tool_order,
        errorbar=("ci", 95),
        n_boot=1000,
        palette=sns.color_palette("colorblind", 2),
        ax=ax7,
    )
    if show_titles:
        ax7.set_title(
            "Agent strategy: tool usage per run (mean ± 95% CI)", weight="bold"
        )
    ax7.set_xlabel("")
    ax7.set_ylabel("Tool calls per run")
    ax7.legend(title="Tool", framealpha=0.88)
    _highlight_axis_model_labels(ax7)
    fig.tight_layout()
    for ext in PLOT_EXTENSIONS:
        stem = "5_average_tool_usage"
        file_name = f"{stem}.{ext}"
        _save_plot(fig, output_dir / file_name, stem=stem, ext=ext)
        figure_files.append(file_name)
    plt.close(fig)

    fig, ax8 = plt.subplots(figsize=(FIG_W, FIG_H))
    df_cost = df_plot.copy()
    df_cost["cost_plot"] = df_cost["cost"].clip(lower=1e-4)
    sns.boxplot(
        data=df_cost,
        x="model_display",
        y="cost_plot",
        hue="model_display",
        order=x_axis_display_order,
        legend=False,
        palette=display_palette,
        linewidth=1,
        showfliers=False,
        width=0.55,
        ax=ax8,
    )
    sns.stripplot(
        data=df_cost,
        x="model_display",
        y="cost_plot",
        order=x_axis_display_order,
        color="#243447",
        alpha=0.35,
        size=2.8,
        jitter=0.22,
        ax=ax8,
    )
    positive_costs = df_cost.loc[df_cost["cost"] > 0, "cost"]
    if not positive_costs.empty:
        cost_span = float(positive_costs.max() / positive_costs.min())
        if cost_span > 25:
            ax8.set_yscale("log")
            ax8.set_ylabel("Cost per run in USD (log scale)")
        else:
            ax8.set_ylabel("Cost per run in USD")
    else:
        ax8.set_ylabel("Cost per run in USD")
    ax8.yaxis.set_major_formatter(
        FuncFormatter(
            lambda y, _: (
                f"${y:.2f}" if y >= 0.1 else (f"${y:.3f}" if y >= 0.01 else f"${y:.4f}")
            )
        )
    )
    if show_titles:
        ax8.set_title("Economic efficiency: cost per run (all runs)", weight="bold")
    ax8.set_xlabel("Model")
    _highlight_axis_model_labels(ax8)
    fig.tight_layout()
    for ext in PLOT_EXTENSIONS:
        file_name = f"6_cost_per_run.{ext}"
        fig.savefig(output_dir / file_name, format=ext, transparent=False, dpi=300)
        figure_files.append(file_name)
    plt.close(fig)

    wall_time_to_root_df = _successful_timing_summary(
        "time_to_root_wall_clock_ms",
        seed=321,
    )
    _plot_success_timing_summary(
        wall_time_to_root_df,
        file_stem="6b_time_to_root_wall_clock",
        ylabel="Wall-clock time to root",
        title="Wall-clock time to root (successful runs)",
    )

    interaction_time_to_root_df = _successful_timing_summary(
        "time_to_root_interaction_ms_raw",
        seed=654,
    )
    _plot_success_timing_summary(
        interaction_time_to_root_df,
        file_stem="6c_time_to_root_interaction",
        ylabel="Interaction time to root",
        title="Interaction time to root (successful runs)",
    )

    econ_rows: list[dict[str, float | str]] = []
    bootstrap_n = 2000
    rng = np.random.default_rng(42)
    for model in x_axis_model_order:
        model_df = df_plot[df_plot["model"] == model].reset_index(drop=True)
        econ_stats = expected_cost_per_successful_root_for_group(
            model_df,
            PRIMARY_ROUND_BUDGET,
            bootstrap_n=bootstrap_n,
            rng=rng,
        )
        econ_rows.append(
            {
                "model": model,
                "model_display": model_display_map[model],
                "expected_cost_per_root": float(econ_stats["expected_cost_per_root"]),
                "ci_low": float(econ_stats["ci_low"]),
                "ci_high": float(econ_stats["ci_high"]),
                "p_success": float(econ_stats["success_rate"]),
            }
        )

    econ_df = pd.DataFrame(econ_rows)
    fig, ax9 = plt.subplots(figsize=(FIG_W, FIG_H))
    x_positions = list(range(len(econ_df)))
    y_vals = econ_df["expected_cost_per_root"].to_numpy()
    positive_vals = [float(v) for v in y_vals if np.isfinite(float(v)) and float(v) > 0]
    min_positive = min(positive_vals) if positive_vals else 1e-4
    max_positive = max(positive_vals) if positive_vals else 1.0
    y_floor = min_positive / 5.0
    econ_label_offsets: dict[str, tuple[int, int]] = {
        "qwen3_4b_instruct_2507_base_paper_static": (0, 8),
        "qwen/qwen3-4b": (0, 8),
        "qwen/qwen3-4b-sft": (0, 8),
        "qwen/qwen3-4b-sft-rl": (0, 8),
        "prime_rl_t_20260218_164349_step_1000": (0, 8),
        "deepseek/deepseek-v3.2": (0, 8),
        "anthropic/claude-opus-4.6": (0, 6),
        "anthropic/claude-opus-4.7": (0, 6),
        "openai/gpt-5.2": (0, 6),
        "google/gemini-3-flash-preview": (0, 6),
    }
    for idx, (_, row) in enumerate(econ_df.iterrows()):
        color = display_palette[str(row["model_display"])]
        model_id = str(row["model"])
        expected_cost = float(row["expected_cost_per_root"])
        y_mid = (
            expected_cost
            if np.isfinite(expected_cost) and expected_cost > 0
            else y_floor
        )
        y_ci_low = float(row["ci_low"])
        y_ci_high = float(row["ci_high"])
        if (
            np.isfinite(y_ci_low)
            and np.isfinite(y_ci_high)
            and y_ci_low > 0
            and y_ci_high > 0
        ):
            ax9.errorbar(
                x_positions[idx],
                y_mid,
                yerr=[
                    [max(0.0, y_mid - max(y_floor, y_ci_low))],
                    [max(0.0, max(y_mid, y_ci_high) - y_mid)],
                ],
                fmt="o",
                color=color,
                ecolor="#243447",
                markersize=7,
                elinewidth=1.3,
                capsize=3,
            )
        else:
            ax9.scatter(
                [x_positions[idx]],
                [y_mid],
                marker="v",
                s=60,
                color=color,
                edgecolors="#243447",
                linewidths=0.6,
                zorder=4,
            )
        dx, dy = econ_label_offsets.get(model_id, (0, 8))
        ax9.annotate(
            f"${expected_cost:.2f}"
            if np.isfinite(expected_cost) and expected_cost > 0
            else "$0.00*",
            (x_positions[idx], y_mid),
            textcoords="offset points",
            xytext=(dx, dy),
            ha="center",
            va="bottom",
            fontsize=8.6,
            color="#243447",
            bbox={
                "boxstyle": "round,pad=0.1",
                "facecolor": "#FFFFFF",
                "alpha": 0.8,
                "edgecolor": "none",
            },
        )
    ax9.set_yscale("log")
    ax9.set_ylim(y_floor / 1.8, max_positive * 2.0)
    ax9.set_ylabel("Expected USD per successful root (log scale)")
    ax9.yaxis.set_major_formatter(
        FuncFormatter(
            lambda y, _: (
                f"${y:.2f}" if y >= 0.1 else (f"${y:.3f}" if y >= 0.01 else f"${y:.4f}")
            )
        )
    )
    ax9.set_xticks(x_positions)
    ax9.set_xticklabels(econ_df["model_display"].tolist())
    _highlight_axis_model_labels(ax9)
    ax9.set_xlabel("Model")
    if show_titles:
        ax9.set_title(
            f"Economic outcome: expected cost per successful root at R≤{PRIMARY_ROUND_BUDGET}",
            weight="bold",
        )
    ax9.text(
        0.01,
        0.02,
        rf"Metric: E[cost]/P(root within ≤{PRIMARY_ROUND_BUDGET} rounds); error bars are 95% bootstrap CI. *Zero-cost points are shown at the plot floor.",
        transform=ax9.transAxes,
        fontsize=9.3,
        color="#445",
    )
    fig.tight_layout()
    for ext in PLOT_EXTENSIONS:
        file_name = f"7_expected_cost_per_successful_root.{ext}"
        fig.savefig(output_dir / file_name, format=ext, transparent=False, dpi=300)
        figure_files.append(file_name)
    plt.close(fig)

    pareto_rows: list[dict[str, float | str]] = []

    def _pareto_category(model: str) -> str:
        if "qwen" in model.lower() or model in {
            "qwen3_4b_instruct_2507_base_paper_static",
            "sft_warm_start_checkpoint500_paper_static_repeat",
            "prime_rl_t_20260218_164349_step_1000",
            PRIVESC_LLM_4B_MODEL,
            GEMMA4_31B_MODEL,
        }:
            return "Local models"
        if model in {DEEPSEEK_V32_MODEL}:
            return "Hosted open-weight"
        return "API models"

    category_markers = {
        "API models": "o",
        "Hosted open-weight": "D",
        "Local models": "^",
    }

    for model in model_order:
        model_df = df_plot[df_plot["model"] == model]
        econ_stats = expected_cost_per_successful_root_for_group(
            model_df,
            PARETO_ROUND_BUDGET,
        )
        p_success = float(econ_stats["success_rate"])
        expected_cost_per_root = float(econ_stats["expected_cost_per_root"])
        pareto_rows.append(
            {
                "model": model,
                "model_display": model_display_map[model],
                "success_rate": p_success,
                "expected_cost_per_root": expected_cost_per_root,
                "category": _pareto_category(model),
            }
        )
    pareto_df = pd.DataFrame(pareto_rows)
    positive_expected_costs = pareto_df.loc[
        np.isfinite(pareto_df["expected_cost_per_root"])
        & (pareto_df["expected_cost_per_root"] > 0),
        "expected_cost_per_root",
    ]
    min_nonzero_cost = (
        float(positive_expected_costs.min())
        if not positive_expected_costs.empty
        else 1e-4
    )
    x_floor = max(1e-4, min_nonzero_cost / 5.0)
    pareto_df["expected_cost_per_root_plot"] = pareto_df[
        "expected_cost_per_root"
    ].apply(
        lambda x: (
            max(x_floor, float(x))
            if np.isfinite(float(x)) and float(x) > 0
            else x_floor
        )
    )

    fig, ax10 = plt.subplots(figsize=(FIG_W, FIG_H))
    label_placements: dict[str, tuple[tuple[int, int], str, str]] = {
        "anthropic/claude-opus-4.6": ((-10, -10), "right", "top"),
        "anthropic/claude-opus-4.7": ((-10, -10), "right", "top"),
        "openai/gpt-5.2": ((10, -10), "left", "top"),
        "deepseek/deepseek-v3.2": ((14, 0), "left", "center"),
        "google/gemini-3-flash-preview": ((10, 8), "left", "bottom"),
        "qwen3-4b": ((14, 0), "left", "bottom"),
        "qwen3-sft": ((14, 0), "left", "center"),
        "qwen3-4b-rl": ((14, 0), "left", "center"),
    }
    default_placement: tuple[tuple[int, int], str, str] = ((10, 8), "left", "bottom")

    for _, row in pareto_df.iterrows():
        model_id = str(row["model"])
        model_display = str(row["model_display"])
        category = str(row["category"])
        ax10.scatter(
            float(row["expected_cost_per_root_plot"]),
            float(row["success_rate"]),
            s=220 if model_id == "qwen3-4b-rl" else 160,
            marker=category_markers[category],
            color=display_palette[model_display],
            edgecolors="#243447",
            linewidths=1.0 if model_id == "qwen3-4b-rl" else 0.6,
            alpha=0.95,
            zorder=6 if model_id == "qwen3-4b-rl" else 5,
            label=model_display,
        )
        offset, ha, va = label_placements.get(model_id, default_placement)
        ax10.annotate(
            model_display,
            (float(row["expected_cost_per_root_plot"]), float(row["success_rate"])),
            textcoords="offset points",
            xytext=offset,
            ha=ha,
            va=va,
            fontsize=10.0,
            color="#243447",
            fontweight="bold" if model_id == "qwen3-4b-rl" else "normal",
            clip_on=False,
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "#FFFFFF",
                "alpha": 0.82,
                "edgecolor": "none",
            },
        )

    qwen_ladder = [QWEN3_4B_MODEL, QWEN3_4B_SFT_MODEL, PRIVESC_LLM_4B_MODEL]
    qwen_points: list[tuple[float, float]] = []
    for model_id in qwen_ladder:
        match = pareto_df[pareto_df["model"] == model_id]
        if not match.empty:
            qwen_points.append(
                (
                    float(match.iloc[0]["expected_cost_per_root_plot"]),
                    float(match.iloc[0]["success_rate"]),
                )
            )
    if len(qwen_points) >= 2:
        for i in range(len(qwen_points) - 1):
            start_x, start_y = qwen_points[i]
            end_x, end_y = qwen_points[i + 1]
            connector = FancyArrowPatch(
                (start_x, start_y),
                (end_x, end_y),
                arrowstyle="-|>,head_length=5,head_width=3",
                mutation_scale=1.0,
                shrinkA=11,
                shrinkB=11,
                linestyle="--",
                linewidth=1.8,
                color="#4B5563",
                alpha=0.85,
                zorder=2,
                transform=ax10.transData,
            )
            ax10.add_patch(connector)

    ax10.set_xscale("log")
    ax10.set_xlabel("Expected cost per successful root (USD, log)")
    ax10.xaxis.set_major_formatter(
        FuncFormatter(
            lambda x, _: (
                f"${x:.2f}" if x >= 0.1 else (f"${x:.3f}" if x >= 0.01 else f"${x:.4f}")
            )
        )
    )
    ax10.set_ylabel(rf"Empirical success within {PARETO_ROUND_BUDGET} rounds")
    has_floored_points = bool(
        (
            ~np.isfinite(pareto_df["expected_cost_per_root"])
            | (pareto_df["expected_cost_per_root"] <= 0)
        ).any()
    )
    x_min = min_nonzero_cost * 0.8
    if has_floored_points:
        x_min = min(x_min, x_floor * 0.8)
    x_max = float(pareto_df["expected_cost_per_root_plot"].max()) * 1.35
    ax10.set_xlim(x_min, x_max)

    tick_candidates: list[float] = []
    decade_min = int(np.floor(np.log10(x_min)))
    decade_max = int(np.ceil(np.log10(x_max)))
    for decade in range(decade_min, decade_max + 1):
        base = 10.0**decade
        for mult in (1.0, 2.0, 5.0):
            tick_val = mult * base
            if x_min <= tick_val <= x_max:
                tick_candidates.append(tick_val)
    if tick_candidates:
        ax10.set_xticks(sorted(set(tick_candidates)))
    y_min = 0.30
    y_max = 1.05
    ax10.set_ylim(y_min, y_max)
    ax10.set_yticks(np.arange(0.3, 1.0001, 0.1))
    ax10.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    if show_titles:
        ax10.set_title(
            f"Pareto view: success@{PARETO_ROUND_BUDGET} vs expected cost per root",
            weight="bold",
            pad=10,
        )
    ax10.grid(True, alpha=0.25)

    from matplotlib.lines import Line2D

    marker_legend_handles = [
        Line2D(
            [0],
            [0],
            marker=category_markers[category],
            color="w",
            markerfacecolor="#6B7280",
            markeredgecolor="#243447",
            markersize=9.5,
            linewidth=0,
            label=category,
        )
        for category in ["API models", "Hosted open-weight", "Local models"]
    ]
    ax10.legend(
        handles=marker_legend_handles,
        title=None,
        loc="lower right",
        frameon=True,
        framealpha=0.88,
        fontsize=11.0,
    )
    fig.tight_layout(pad=0.5)
    for ext in PLOT_EXTENSIONS:
        stem = "8_pareto_success_rate_vs_cost"
        file_name = f"{stem}.{ext}"
        _save_plot(fig, output_dir / file_name, stem=stem, ext=ext)
        figure_files.append(file_name)
    plt.close(fig)

    console.print(f"[green]✔[/] Plots successfully saved to: [dim]{output_dir}[/]")
    plt.style.use("default")
    return figure_files


def save_plots_for_dark_theme(df: pd.DataFrame, output_dir_str: str):
    save_plots_for_paper_theme(df, output_dir_str)
