#!/usr/bin/env python3
"""Generate ACSAC reward-design ablation figures from existing eval artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.analysis.plotting import apply_paper_plot_theme


@dataclass(frozen=True)
class RewardVariant:
    slug: str
    run_id: str
    label: str
    matrix_label: str
    has_round: bool
    has_cost: bool
    selected: bool = False


@dataclass(frozen=True)
class StepStats:
    variant: RewardVariant
    step: int
    model: str
    num_runs: int
    success_r5: int
    success_r20: int
    success_r60: int
    all_solved_scenarios: int
    total_scenarios: int
    path: Path

    @property
    def rate_r5(self) -> float:
        return self.success_r5 / self.num_runs

    @property
    def rate_r20(self) -> float:
        return self.success_r20 / self.num_runs

    @property
    def rate_r60(self) -> float:
        return self.success_r60 / self.num_runs


VARIANTS = [
    RewardVariant(
        slug="outcome",
        run_id="20260515T162238Z",
        label="Outcome",
        matrix_label=r"\rewardvariant{Outcome}",
        has_round=False,
        has_cost=False,
    ),
    RewardVariant(
        slug="outcome_cost",
        run_id="20260518T021707Z",
        label="Outcome+Cost",
        matrix_label=r"\rewardvariant{{+}Cost} $\star$",
        has_round=False,
        has_cost=True,
        selected=True,
    ),
    RewardVariant(
        slug="outcome_round",
        run_id="20260515T162238Z",
        label="Outcome+Round",
        matrix_label=r"\rewardvariant{{+}Round}",
        has_round=True,
        has_cost=False,
    ),
    RewardVariant(
        slug="outcome_round_cost",
        run_id="20260515T162238Z",
        label="Outcome+Round+Cost",
        matrix_label=r"\rewardvariant{{+}Round{+}Cost}",
        has_round=True,
        has_cost=True,
    ),
]


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def one_model_stats(summary_path: Path) -> tuple[str, dict[str, Any], dict[str, Any]]:
    payload = read_json(summary_path)
    model_stats = payload.get("model_stats")
    if not isinstance(model_stats, dict) or len(model_stats) != 1:
        raise ValueError(f"Expected exactly one model in {summary_path}")
    model, stats = next(iter(model_stats.items()))
    if not isinstance(stats, dict):
        raise ValueError(f"Expected model stats object in {summary_path}")
    scenario_stats = payload.get("scenario_stats")
    if not isinstance(scenario_stats, dict) or not scenario_stats:
        raise ValueError(f"Expected non-empty scenario stats in {summary_path}")
    return str(model), stats, scenario_stats


def int_stat(stats: dict[str, Any], key: str, path: Path) -> int:
    value = stats.get(key)
    if not isinstance(value, int):
        raise ValueError(f"Missing integer stat {key!r} in {path}")
    return value


def all_solved_scenario_count(
    scenario_stats: dict[str, Any], summary_path: Path
) -> tuple[int, int]:
    count = 0
    for scenario, stats in scenario_stats.items():
        if not isinstance(stats, dict):
            raise ValueError(
                f"Expected stats object for scenario {scenario!r} in {summary_path}"
            )
        if (
            int_stat(stats, "num_successful_runs_within_max_budget", summary_path)
            == int_stat(stats, "num_runs", summary_path)
        ):
            count += 1
    return count, len(scenario_stats)


def load_variant_steps(source_root: Path, variant: RewardVariant) -> list[StepStats]:
    procedural_root = source_root / variant.slug / variant.run_id / "eval" / "procedural"
    steps: list[StepStats] = []
    for step_root in sorted(
        procedural_root.glob("step_*"), key=lambda path: int(path.name.split("_")[1])
    ):
        summary_path = step_root / "stats" / "evaluation_summary.json"
        if not summary_path.is_file():
            print(f"[WARN] Missing stats: {summary_path}", file=sys.stderr)
            continue
        model, stats, scenario_stats = one_model_stats(summary_path)
        all_solved_scenarios, total_scenarios = all_solved_scenario_count(
            scenario_stats, summary_path
        )
        steps.append(
            StepStats(
                variant=variant,
                step=int(step_root.name.split("_")[1]),
                model=model,
                num_runs=int_stat(stats, "num_runs", summary_path),
                success_r5=int(round(float(stats["sr_within_rounds_5"]) * stats["num_runs"])),
                success_r20=int_stat(
                    stats, "num_successful_runs_within_primary_budget", summary_path
                ),
                success_r60=int_stat(
                    stats, "num_successful_runs_within_max_budget", summary_path
                ),
                all_solved_scenarios=all_solved_scenarios,
                total_scenarios=total_scenarios,
                path=step_root,
            )
        )
    if not steps:
        raise ValueError(f"No step stats found for {variant.slug} under {procedural_root}")
    return steps


def select_reported_checkpoint(steps: list[StepStats]) -> StepStats:
    return max(steps, key=lambda stats: (stats.rate_r20, stats.rate_r60, -stats.step))


def final_step(steps: list[StepStats]) -> StepStats:
    for stats in steps:
        if stats.step == 1000:
            return stats
    raise ValueError(f"Missing step_1000 for {steps[0].variant.slug}")


def success_rate(stats: StepStats, round_budget: int) -> float:
    if round_budget == 20:
        return stats.rate_r20
    if round_budget == 60:
        return stats.rate_r60
    raise ValueError(f"Unsupported stability round budget: {round_budget}")


def percent(value: float) -> str:
    return f"{100 * value:.0f}\\%"


def metric_tabular(stats: StepStats) -> str:
    return (
        rf"\begin{{tabular}}{{@{{}}l@{{\hskip 0.55em}}r@{{}}}}"
        rf"$r{{\le}}5$ & \textbf{{{percent(stats.rate_r5)}}}\\[-0.15ex]"
        rf"$r{{\le}}20$ & \textbf{{{percent(stats.rate_r20)}}}\\[-0.15ex]"
        rf"$r{{\le}}60$ & \textbf{{{percent(stats.rate_r60)}}}\\[-0.15ex]"
        rf"$S_{{\mathrm{{all}}}}$ & "
        rf"\textbf{{{stats.all_solved_scenarios}/{stats.total_scenarios}}}"
        rf"\end{{tabular}}"
    )


CARD_W = 2.85
CARD_H = 2.35
BAND_H = 0.72
GAP = 0.18
LABEL_GUTTER = 0.95
TOP_LABEL_GAP = 0.42


def card_origin(col: int, row: int) -> tuple[float, float]:
    x = LABEL_GUTTER + col * (CARD_W + GAP)
    y_top = -row * (CARD_H + GAP)
    return x, y_top


def card_block(stats: StepStats, col: int, row: int) -> str:
    selected = stats.variant.selected
    body_style = "rcardsel" if selected else "rcard"
    band_style = "rbandsel" if selected else "rband"
    x, y_top = card_origin(col, row)
    x2 = x + CARD_W
    y_bot = y_top - CARD_H
    y_band_bot = y_top - BAND_H
    xc = x + CARD_W / 2
    y_title = y_top - 0.28
    y_step = y_top - 0.54
    y_metrics = y_bot + (CARD_H - BAND_H) / 2 - 0.04
    return "\n".join(
        [
            rf"\filldraw[{body_style}] ({x:.3f},{y_bot:.3f}) rectangle ({x2:.3f},{y_top:.3f});",
            rf"\filldraw[{band_style}] ({x:.3f},{y_band_bot:.3f}) rectangle ({x2:.3f},{y_top:.3f});",
            rf"\node[rtitle] at ({xc:.3f},{y_title:.3f}) {{{stats.variant.matrix_label}}};",
            rf"\node[rstep] at ({xc:.3f},{y_step:.3f}) {{step {stats.step}}};",
            rf"\node[rmetric] at ({xc:.3f},{y_metrics:.3f}) {{{metric_tabular(stats)}}};",
        ]
    )


def axis_labels() -> str:
    lines: list[str] = []
    for col, label in enumerate(("No cost term", "Cost term")):
        x, y_top = card_origin(col, 0)
        xc = x + CARD_W / 2
        lines.append(
            rf"\node[raxis] at ({xc:.3f},{y_top + TOP_LABEL_GAP:.3f}) {{{label}}};"
        )
    for row, label in enumerate(("No round\\\\term", "Round\\\\term")):
        x, y_top = card_origin(0, row)
        yc = y_top - CARD_H / 2
        lines.append(
            rf"\node[raxis,anchor=east] at ({x - 0.18:.3f},{yc:.3f}) {{{label}}};"
        )
    return "\n".join(lines)


def write_matrix_tex(output_path: Path, peaks: dict[str, StepStats]) -> None:
    placements = [
        (peaks["outcome"], 0, 0),
        (peaks["outcome_cost"], 1, 0),
        (peaks["outcome_round"], 0, 1),
        (peaks["outcome_round_cost"], 1, 1),
    ]
    cards = "\n".join(card_block(stats, col, row) for stats, col, row in placements)
    tex = rf"""% Generated by scripts/paper/plot_reward_design.py.
% Caption snippet:
% \caption{{Reward-factor ablation on procedural holdout. Cells show the checkpoint selected by the procedural rule: primary $P(H_{{\mathrm{{root}}}}{{\le}}20)$, with $P(H_{{\mathrm{{root}}}}{{\le}}60)$ recovery and step-1000 stability used only as tie-breakers. Values report success at $r\leq5,20,60$ and $S_{{\mathrm{{all}}}}$, the count of procedural scenario families solved on every rollout within the max budget.}}
\definecolor{{rewardink}}{{RGB}}{{42,47,56}}
\definecolor{{rewardline}}{{RGB}}{{127,133,141}}
\definecolor{{rewardfill}}{{RGB}}{{247,248,250}}
\definecolor{{rewardbar}}{{RGB}}{{233,236,240}}
\definecolor{{rewardselline}}{{RGB}}{{76,120,168}}
\definecolor{{rewardselfill}}{{RGB}}{{244,247,252}}
\definecolor{{rewardselbar}}{{RGB}}{{227,236,247}}

\tikzset{{
  rcard/.style={{rounded corners=5pt, draw=rewardline, fill=rewardfill, line width=0.9pt}},
  rband/.style={{rounded corners=5pt, draw=rewardline, fill=rewardbar, line width=0.8pt}},
  rcardsel/.style={{rounded corners=5pt, draw=rewardselline, fill=rewardselfill, line width=1.1pt}},
  rbandsel/.style={{rounded corners=5pt, draw=rewardselline, fill=rewardselbar, line width=1.0pt}},
  rtitle/.style={{font=\sffamily\bfseries\scriptsize, text=rewardink, align=center, inner sep=0pt}},
  rstep/.style={{font=\sffamily\scriptsize, text=rewardink!72, align=center, inner sep=0pt}},
  raxis/.style={{font=\sffamily\bfseries\scriptsize, text=rewardink, align=center, inner sep=2pt}},
  rmetric/.style={{font=\sffamily\scriptsize, text=rewardink, align=center, inner sep=0pt}},
}}

\begin{{tikzpicture}}[x=1cm,y=1cm,line join=round,line cap=round]
{axis_labels()}
{cards}
\end{{tikzpicture}}
"""
    output_path.write_text(tex, encoding="utf-8")


def plot_dumbbell(
    output_path: Path,
    peaks: list[StepStats],
    finals: list[StepStats],
    *,
    round_budget: int,
) -> None:
    apply_paper_plot_theme()
    fig, ax = plt.subplots(figsize=(7.2, 2.55))
    y_positions = list(range(len(peaks)))
    peak_by_slug = {stats.variant.slug: stats for stats in peaks}
    final_by_slug = {stats.variant.slug: stats for stats in finals}

    for y, variant in enumerate([stats.variant for stats in peaks]):
        peak = peak_by_slug[variant.slug]
        final = final_by_slug[variant.slug]
        peak_rate = success_rate(peak, round_budget)
        final_rate = success_rate(final, round_budget)
        line_color = "#2F343B" if variant.selected else "#8993A3"
        line_width = 2.2 if variant.selected else 1.6
        ax.plot(
            [final_rate, peak_rate],
            [y, y],
            color=line_color,
            linewidth=line_width,
            zorder=1,
        )
        ax.scatter(
            final_rate,
            y,
            marker="o",
            s=38,
            facecolors="white",
            edgecolors=line_color,
            linewidths=1.2,
            label="step 1000" if y == 0 else None,
            zorder=2,
        )
        ax.scatter(
            peak_rate,
            y,
            marker="*" if variant.selected else "o",
            s=105 if variant.selected else 46,
            color="#111111" if variant.selected else "#5B6678",
            label="peak" if y == 0 else None,
            zorder=3,
        )
        ax.text(
            peak_rate + 0.008,
            y,
            f"{100 * peak_rate:.0f}%",
            va="center",
            ha="left",
            fontsize=12,
            fontweight="normal",
            color="#2F343B",
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [
            "Outcome",
            "+Cost *",
            "+Round",
            "+Round+Cost",
        ],
        fontweight="normal",
    )
    ax.invert_yaxis()
    ax.set_ylim(len(peaks) - 0.18, -0.55)
    ax.set_xlabel(
        rf"Procedural $P(H_{{\mathrm{{root}}}}\leq {round_budget})$",
        fontweight="normal",
    )
    ax.set_xlim(0.69, 0.955)
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _pos: f"{100 * value:.0f}%")
    )
    ax.set_ylabel("")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", alpha=0.35)
    for spine in ax.spines.values():
        spine.set_color("#D2DAE5")
        spine.set_linewidth(0.8)
    legend = ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.012, 0.018),
        ncol=2,
        frameon=True,
        framealpha=0.88,
        title=None,
        fontsize=11.0,
        borderaxespad=0.0,
        handlelength=1.0,
        columnspacing=0.55,
        borderpad=0.22,
        labelspacing=0.24,
        handletextpad=0.34,
    )
    legend.get_frame().set_edgecolor("#D2DAE5")
    legend.get_frame().set_linewidth(0.65)
    legend.get_frame().set_facecolor("#FFFFFF")
    fig.subplots_adjust(left=0.21, right=0.99, bottom=0.23, top=0.95)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("outputs/runs/paper/03_reward_ladder/qwen3-4b-instruct-2507"),
    )
    parser.add_argument(
        "--matrix-output",
        type=Path,
        default=Path("paper/ACSAC/figures/reward_ablation_matrix.tex"),
    )
    parser.add_argument(
        "--dumbbell-output-r20",
        type=Path,
        default=Path("paper/ACSAC/figures/reward_stability_dumbbell_r20.pdf"),
    )
    parser.add_argument(
        "--dumbbell-output-r60",
        type=Path,
        default=Path("paper/ACSAC/figures/reward_stability_dumbbell_r60.pdf"),
    )
    args = parser.parse_args()

    all_steps = {
        variant.slug: load_variant_steps(args.source_root, variant)
        for variant in VARIANTS
    }
    peaks = {slug: select_reported_checkpoint(steps) for slug, steps in all_steps.items()}
    finals = {slug: final_step(steps) for slug, steps in all_steps.items()}

    args.matrix_output.parent.mkdir(parents=True, exist_ok=True)
    args.dumbbell_output_r20.parent.mkdir(parents=True, exist_ok=True)
    args.dumbbell_output_r60.parent.mkdir(parents=True, exist_ok=True)
    write_matrix_tex(args.matrix_output, peaks)
    plot_dumbbell(
        args.dumbbell_output_r20,
        [peaks[variant.slug] for variant in VARIANTS],
        [finals[variant.slug] for variant in VARIANTS],
        round_budget=20,
    )
    plot_dumbbell(
        args.dumbbell_output_r60,
        [peaks[variant.slug] for variant in VARIANTS],
        [finals[variant.slug] for variant in VARIANTS],
        round_budget=60,
    )

    print("Extracted reward-design values:")
    for variant in VARIANTS:
        peak = peaks[variant.slug]
        final = finals[variant.slug]
        print(
            f"{variant.label}: peak step {peak.step} "
            f"P(H_root<=5) {peak.success_r5}/{peak.num_runs}, "
            f"P(H_root<=20) {peak.success_r20}/{peak.num_runs}, "
            f"P(H_root<=60) {peak.success_r60}/{peak.num_runs}, "
            f"S_all {peak.all_solved_scenarios}/{peak.total_scenarios}; "
            f"step1000 P(H_root<=60) {final.success_r60}/{final.num_runs}"
        )
    print(f"Wrote {args.matrix_output}")
    print(f"Wrote {args.dumbbell_output_r20}")
    print(f"Wrote {args.dumbbell_output_r60}")


if __name__ == "__main__":
    main()
