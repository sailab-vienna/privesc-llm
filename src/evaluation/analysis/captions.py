from pathlib import Path

from src.evaluation.analysis.constants import (
    MAX_ROUNDS,
    PARETO_ROUND_BUDGET,
    PRIMARY_ROUND_BUDGET,
    RUNS_PER_SCENARIO,
    SCENARIOS_PER_MODEL,
)


def write_figure_captions(
    output_dir_str: str,
    figure_files: list[str],
    protocol_complete: bool,
    expected_runs_per_scenario: int | None = None,
) -> Path:
    output_dir = Path(output_dir_str)
    output_dir.mkdir(exist_ok=True)
    caption_file = output_dir / "figure_captions.md"

    if protocol_complete:
        expected_runs = expected_runs_per_scenario or RUNS_PER_SCENARIO
        runs_per_model = expected_runs * SCENARIOS_PER_MODEL
        setup_line = (
            f"Evaluation setup for all figures: static privilege-escalation benchmark with {SCENARIOS_PER_MODEL} scenarios "
            f"and {expected_runs} runs per scenario (`N={runs_per_model}` runs per model), max interaction budget {MAX_ROUNDS} rounds."
        )
        accounting_line = f"Run accounting is strict: analysis expects exactly `n={expected_runs}` runs for each model-scenario cell and fails if counts are inconsistent."
        runs_per_scenario_label = str(expected_runs)
        runs_per_model_label = str(runs_per_model)
        fig1_counts_label = f"x/{runs_per_model}"
        fig2_parenthetical = f"n={expected_runs} runs per scenario"
    else:
        setup_line = (
            f"Evaluation setup for all figures: static privilege-escalation benchmark with {SCENARIOS_PER_MODEL} scenarios "
            f"and n runs per scenario (`N` runs per model), max interaction budget {MAX_ROUNDS} rounds."
        )
        accounting_line = "Run accounting allows incomplete model-scenario cells; denominators are parameterized as n (per scenario) and N (per model)."
        runs_per_scenario_label = "n"
        runs_per_model_label = "N"
        fig1_counts_label = "x/N"
        fig2_parenthetical = "n varies by scenario/model cell"

    appendix_line = "- Appendix uncertainty heatmap (`A1_heatmap_wilson_ci_width_by_scenario`): Shows 95% Wilson CI width per scenario and model; larger values indicate higher uncertainty due to finite sample size."
    text = f"""# Figure Captions

{setup_line}

{accounting_line}

One round corresponds to one agent turn (one model step producing a command and tool output).

This measures per-run reliability and is not a best-of-k retry metric.

- Figure 1 (`1_overall_success_rate`): Success rate under a 60-round interaction budget. Points show the fraction of runs that achieve root within ≤ 60 rounds ({SCENARIOS_PER_MODEL} scenarios × {runs_per_scenario_label} runs; N={runs_per_model_label} runs per model), with numeric labels shown as percent and counts (`{fig1_counts_label}`). Error bars denote 95% Wilson score confidence intervals. Dashed reference lines mark reused reported baselines from Happe et al.: Human baseline: 75% (reported), Traditional tools baseline: 25% (reported).
- Figure 1b (`1b_overall_success_rate_r20`): Success rate under a stricter {PRIMARY_ROUND_BUDGET}-round interaction budget. Points show the fraction of runs that achieve root within ≤ {PRIMARY_ROUND_BUDGET} rounds ({SCENARIOS_PER_MODEL} scenarios × {runs_per_scenario_label} runs; N={runs_per_model_label} runs per model), with numeric labels shown as percent and counts (`{fig1_counts_label}`). Error bars denote 95% Wilson score confidence intervals.
- Figure 1d (`1d_overall_success_rate_r10`): Success rate under a stricter 10-round interaction budget. Points show the fraction of runs that achieve root within ≤ 10 rounds ({SCENARIOS_PER_MODEL} scenarios × {runs_per_scenario_label} runs; N={runs_per_model_label} runs per model), with numeric labels shown as percent and counts (`{fig1_counts_label}`). Error bars denote 95% Wilson score confidence intervals.
- Figure 1c (`1c_avg_rounds_until_root`): Average rounds until root for successful runs only, by model. Points show mean rounds-to-root, error bars show 95% bootstrap confidence intervals over successful runs, and labels include successful-run sample size (`n`).
- Figure 2 (`2_heatmap_success_rate_by_scenario`): Per-scenario success under a 60-round budget. Each cell reports x/{runs_per_scenario_label} successful runs for that scenario ({fig2_parenthetical}), while color encodes the corresponding success rate. Scenario labels use benchmark names without numeric prefixes.
- Figure 2b (`2b_heatmap_success_rate_by_scenario_r20`): Per-scenario success under a stricter {PRIMARY_ROUND_BUDGET}-round budget. Each cell reports x/{runs_per_scenario_label} successful runs for that scenario ({fig2_parenthetical}), while color encodes the corresponding success rate. Scenario labels use benchmark names without numeric prefixes.
- Figure 2c (`2c_success_vs_round_budget`): Budgeted per-run success rate as a function of interaction budget R. For each R, we report the fraction of runs that achieve root within ≤ R rounds (N={runs_per_model_label} per model). Shaded bands show 95% Wilson score confidence intervals. This measures per-run reliability under a budget and is not a best-of-k retry metric. Budgets evaluated at R ∈ {{5,10,...,60}} rounds.
- Figure 2d (`budgeted_success_by_deployment`): Budgeted static-benchmark reliability by model and deployment class at R≤{PRIMARY_ROUND_BUDGET}. Point colors reuse the Figure 2c model palette, marker shape encodes deployment type, and error bars denote Wilson 95% confidence intervals.
- Figure 3b (`3b_boxplot_rounds_by_model`): Distribution of rounds to completion for successful runs, stratified by model. Per-model annotations report effective sample size (`n`) for successful runs only. Runs that did not achieve root within 60 rounds are excluded; dashed reference lines indicate the budget cap (`R=60`).
- Figure 4 (`4_agent_verbosity_distribution`): Distribution of trace message counts per run by model (all runs). Message counts use trace history length (`history`) including system, user, assistant, and tool messages. Y-axis is clipped at the 95th percentile to preserve central readability.
- Figure 4b (`4b_token_usage_distribution`): Distribution of total tokens per run by model (all runs); y-axis uses log scale when dynamic range is large. Total tokens are computed as prompt + completion tokens from the trace (`total_tokens` fallback to `prompt_tokens + completion_tokens`).
- Figure 5 (`5_average_tool_usage`): Tool usage per run by model with human-readable categories (Command executions, Credential checks). Bars show means; error bars show 95% bootstrap confidence intervals across runs.
- Figure 6 (`6_cost_per_run`): Distribution of per-run API usage cost by model (all runs); y-axis uses log scale when dynamic range is large. Costs use trace `total_cost` and fall back to token-based pricing only when API cost is missing or zero.
- Figure 6b (`6b_time_to_root_wall_clock`): Mean wall-clock time to root for successful runs only, by model. Error bars show 95% bootstrap confidence intervals over successful runs, and labels include successful-run sample size (`n`). This metric uses end-to-end rollout wall clock from session start until the first successful root-yielding tool result.
- Figure 6c (`6c_time_to_root_interaction`): Mean interaction time to root for successful runs only, by model. Error bars show 95% bootstrap confidence intervals over successful runs, and labels include successful-run sample size (`n`). This metric uses cumulative raw LLM-call plus tool-call latency until the first successful root-yielding tool result, excluding non-interaction wait time.
- Figure 7 (`7_expected_cost_per_successful_root`): Expected cost per successful root under the primary {PRIMARY_ROUND_BUDGET}-round budget, computed as `E[cost] / P(root within ≤{PRIMARY_ROUND_BUDGET} rounds)` per model. Error bars show 95% bootstrap confidence intervals.
- Figure 8 (`8_pareto_success_rate_vs_cost`): Pareto-style scatter of per-model success rate (root within ≤{PARETO_ROUND_BUDGET} rounds) versus expected cost per successful root at the same budget, computed as `E[cost] / P(root within ≤{PARETO_ROUND_BUDGET} rounds)`. The x-axis uses log scale and the y-axis spans `50%` to `100%` for readability while preserving comparability across runs. Marker shape encodes model family (API, hosted open-weight, local), and the Qwen post-training progression is connected by directional arrows (Backbone → SFT → SFT-RL) when all points are available.
- Appendix Figure A2 (`A2_boxplot_rounds_by_scenario`): Distribution of rounds to completion for successful runs, stratified by scenario and model, with per-model successful-run sample size (`n`) context.
{appendix_line}

Generated files:
{chr(10).join(f"- `{name}`" for name in sorted(figure_files))}
"""
    caption_file.write_text(text)
    return caption_file
