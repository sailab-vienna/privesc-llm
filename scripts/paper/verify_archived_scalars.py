"""Verify archived scalar outcomes without running models or scenarios."""

import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

if not __debug__:
    raise SystemExit("Run this verifier without Python optimization.")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.paper.replay_trace_token_price_bench import fit_prices
from src.evaluation.analysis.metrics import wilson_interval
from src.evaluation.analysis.parsing import parse_evaluation_results_detailed
from src.utils.pricing import MODEL_PRICING, calculate_cost


def check_stats(rows, stats):
    total = len(rows)
    assert total == stats["num_runs"]
    for budget, suffix in [(20, "primary"), (60, "max")]:
        successes = sum(row["success"] and row["turns"] <= budget for row in rows)
        assert successes == stats[f"num_successful_runs_within_{suffix}_budget"]
        p = successes / total
        assert math.isclose(
            p, stats[f"success_rate_within_{suffix}_budget"], abs_tol=1e-12
        )
        low, high = wilson_interval(successes, total)
        for label, value in [("low", low), ("high", high)]:
            assert math.isclose(
                value,
                stats[f"success_rate_within_{suffix}_budget_ci95_{label}"],
                abs_tol=1e-12,
            )
    for budget in range(5, 61, 5):
        rate = sum(row["success"] and row["turns"] <= budget for row in rows) / total
        assert math.isclose(rate, stats[f"sr_within_rounds_{budget}"], abs_tol=1e-12)


if len(sys.argv) != 2:
    raise SystemExit("Usage: python3 verify_archived_scalars.py EVAL_ARTIFACT_DIR")

artifact = Path(sys.argv[1]).resolve()
base = artifact / "evaluation_artifacts/scalars"
released = artifact / "summaries"
cost_root = artifact / "evaluation_artifacts/provenance/cost"
metadata = json.loads((base / "cohorts.json").read_text())
assert metadata["schema_version"] == 2
summary_refs = metadata["summaries"]
cohorts = {item["id"]: item for item in metadata["cohorts"]}
assert len(metadata["cohorts"]) == len(cohorts) == 64
assert set(cohorts) == set(range(1, 65))
by_cohort = defaultdict(list)
source_files = set()
with (base / "runs.csv").open() as stream:
    reader = csv.DictReader(stream)
    assert reader.fieldnames == [
        "cohort",
        "scenario",
        "success",
        "turns",
        "source_file",
        "source_sha256",
    ]
    for row in reader:
        cohort = int(row["cohort"])
        assert cohort in cohorts
        assert row["success"] in ("0", "1")
        row["success"] = int(row["success"])
        row["turns"] = int(row["turns"])
        assert 0 < row["turns"] <= 60
        assert row["success"] or row["turns"] == 60
        assert Path(row["source_file"]).name == row["source_file"]
        assert re.fullmatch(r"[0-9a-f]{64}", row["source_sha256"])
        identity = (cohorts[cohort]["source_dir"], row["source_file"])
        assert identity not in source_files
        source_files.add(identity)
        by_cohort[cohort].append(row)

totals = Counter()
for key, cohort in cohorts.items():
    rows = by_cohort[key]
    assert len(rows) == cohort["expected_runs"]
    counts = Counter(row["scenario"] for row in rows)
    assert len(counts) == cohort["expected_runs"] // cohort["runs_per_scenario"]
    assert set(counts.values()) == {cohort["runs_per_scenario"]}
    totals[cohort["experiment"]] += len(rows)
assert totals == {
    "rq1": 3000,
    "rq2": 4000,
    "prompt_gemma": 720,
    "prompt_rl": 720,
    "prompt_base": 720,
    "prompt_sft": 720,
    "rebuttal": 720,
}

assert set(summary_refs) == set(totals)
paths = {}
for experiment, reference in summary_refs.items():
    path = released / reference["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == reference["sha256"]
    paths[experiment] = path

for experiment, path in paths.items():
    summary = json.loads(path.read_text())
    expected_models = (
        {item["model"] for item in cohorts.values() if item["experiment"] == experiment}
        if experiment != "rq2"
        else {
            "e4_outcome_step_200",
            "e4_outcome_round_step_400",
            "e4_outcome_cost_step_300",
            "e4_outcome_round_cost_step_200",
        }
    )
    assert len(expected_models) == {"rq1": 6, "rq2": 4, "rebuttal": 6}.get(
        experiment, 3
    )
    assert set(summary["model_stats"]) == expected_models
    assert all(
        set(models) == expected_models
        for models in summary["scenario_model_stats"].values()
    )
    for model, stats in summary["model_stats"].items():
        matches = [
            item
            for item in cohorts.values()
            if item["experiment"] == experiment and item["model"] == model
        ]
        assert len(matches) == 1
        cohort = matches[0]
        rows = by_cohort[cohort["id"]]
        assert set(summary["scenario_model_stats"]) == {row["scenario"] for row in rows}
        check_stats(rows, stats)
        for scenario, models in summary["scenario_model_stats"].items():
            check_stats(
                [row for row in rows if row["scenario"] == scenario], models[model]
            )

headline_path = (
    released
    / "runs/paper/04_static_benchmark/comparison/headline/20260524/eval/static/aggregate/final/stats/evaluation_summary_paper.json"
)
headline = json.loads(headline_path.read_text())
expected_scenarios = sorted(headline["scenario_model_stats"])
headline_frames = {}
trace_roots = [
    path
    for path in (artifact / "headline_static_traces").rglob("traces")
    if any(child.is_dir() for child in path.iterdir())
]
assert len(trace_roots) == 6
for trace_root in trace_roots:
    frame = parse_evaluation_results_detailed(
        str(trace_root),
        expected_scenarios=expected_scenarios,
        expected_runs_per_scenario=10,
    )
    models = frame["model"].unique().tolist()
    assert len(models) == 1 and models[0] not in headline_frames
    headline_frames[models[0]] = frame
assert set(headline_frames) == set(headline["model_stats"])

headline_costs = {}
for model, frame in headline_frames.items():
    stats = headline["model_stats"][model]
    successes = int((frame["success"] & (frame["turns"] <= 20)).sum())
    assert successes == stats["num_successful_runs_within_primary_budget"]
    per_root = (
        math.fsum(float(value) for value in frame["cost_within_rounds_20"]) / successes
    )
    expected = stats["expected_cost_per_successful_root_within_primary_budget"]
    assert math.isclose(per_root, expected, rel_tol=1e-10)
    headline_costs[model] = per_root

measurements = json.loads(
    (cost_root / "rtx5090/verified_5090_msrp_corporate_costs.json").read_text()
)
c_capex = (1999 + 700) / (3 * 365 * 24 * 0.5)
c_energy = 0.675 * (0.1837 * 1.1595)
c_sec = (c_capex + c_energy) / 3600
hardware = measurements["hardware"]
assert math.isclose(c_capex, hardware["capex_hr"], rel_tol=1e-12)
assert math.isclose(c_energy, hardware["energy_hr"], rel_tol=1e-12)
assert math.isclose(c_sec, hardware["c_sec"], rel_tol=1e-12)

pricing_models = {
    "qwen3_4b": "e4_outcome_cost_step_300",
    "gemma4_31b_bnb": "gemma4_31b_it_thinking_gemma4_tools",
}
for slug, pricing_model in pricing_models.items():
    with (cost_root / f"rtx5090/{slug}/vllm_throughput_results.csv").open() as stream:
        bench_row = next(
            row for row in csv.DictReader(stream) if int(row["concurrency"]) == 16
        )
    bench = measurements[slug]["bench"]
    for key, value in bench_row.items():
        assert math.isclose(float(value), float(bench[key]), rel_tol=1e-12)
    mixed_price = (
        c_sec * bench["wall_s"] / (bench["input_tokens"] + bench["output_tokens"]) * 1e6
    )
    assert math.isclose(mixed_price, bench["usd_per_1m_mixed_tokens"], rel_tol=1e-12)

    record = json.loads(
        (cost_root / f"rtx5090/{slug}/trace_replay_price_fit.json").read_text()
    )
    fit = fit_prices(record["bucket_rows"], c_sec)
    for key in (
        "alpha_s_per_input_token",
        "beta_s_per_output_token",
        "input_usd_per_1m_tokens",
        "output_usd_per_1m_tokens",
        "r2",
    ):
        assert math.isclose(fit[key], record["fit"][key], rel_tol=1e-10)
    historical = record["trace_summary"]
    expected_anchor = (
        mixed_price
        * (
            historical["mean_input_tokens_per_run"]
            + historical["mean_output_tokens_per_run"]
        )
        / 1e6
    )
    assert math.isclose(
        record["fit"]["anchored_to_cost_per_run_usd"],
        expected_anchor,
        rel_tol=1e-10,
    )
    raw_cost = (
        math.fsum(
            historical[f"mean_{side}_tokens_per_run"] * fit[f"{side}_usd_per_1m_tokens"]
            for side in ("input", "output")
        )
        / 1e6
    )
    scale = record["fit"]["anchored_to_cost_per_run_usd"] / raw_cost
    assert math.isclose(scale, record["fit"]["anchor_scale"], rel_tol=1e-10)
    prices = tuple(
        fit[f"{side}_usd_per_1m_tokens"] * scale for side in ("input", "output")
    )
    assert all(
        math.isclose(actual, expected, rel_tol=1e-10)
        for actual, expected in zip(prices, MODEL_PRICING[pricing_model], strict=True)
    )

    trace = measurements[slug]["trace"]
    cost_per_run = (
        math.fsum(
            trace[f"mean_{side}"] * price
            for side, price in zip(("input", "output"), prices, strict=True)
        )
        / 1e6
    )
    cost_per_root = cost_per_run / trace["success_rate"]
    assert math.isclose(cost_per_run, measurements[slug]["cost_per_run"], rel_tol=1e-10)
    assert math.isclose(
        cost_per_root, measurements[slug]["cost_per_root"], rel_tol=1e-10
    )
    assert math.isclose(cost_per_root, headline_costs[pricing_model], rel_tol=1e-10)

training_root = cost_root / "training"
sft_seconds = json.loads((training_root / "sft_train_stats.json").read_text())[
    "train_runtime"
]
with (training_root / "rl_timing.csv").open() as stream:
    timing_rows = list(csv.DictReader(stream))
assert [int(row["step"]) for row in timing_rows] == list(range(301))
assert float(timing_rows[-1]["update_seconds"]) == 0
rl_seconds = math.fsum(
    float(row[key])
    for row in timing_rows
    for key in ("update_seconds", "broadcast_seconds", "checkpoint_seconds")
)
assert math.isclose(rl_seconds, 41578.71370762332, rel_tol=1e-12)
training_cost = (sft_seconds + rl_seconds) * 4 / 3600 * 2.29
assert round(training_cost, 2) == 121.08

local_cost = headline_costs[pricing_models["qwen3_4b"]]
gemma_cost = headline_costs[pricing_models["gemma4_31b_bnb"]]
claude_cost = headline_costs["anthropic/claude-opus-4.7"]
break_even = training_cost / (claude_cost - local_cost)
print(
    f"Headline costs: PrivEsc=${local_cost:.8f}, Gemma=${gemma_cost:.8f} "
    f"({gemma_cost / local_cost:.4f}x), Claude=${claude_cost:.8f} "
    f"({claude_cost / local_cost:.4f}x)."
)
print(
    f"Recorded SFT + RL through checkpoint 300: ${training_cost:.2f}; "
    f"Claude break-even={break_even:.1f} successful escalations."
)

deepseek = cohorts[64]
assert deepseek["model"] == "deepseek/deepseek-v4-flash"
with (base / "deepseek_r20_tokens.csv").open() as stream:
    usage = list(csv.DictReader(stream))
assert len(usage) == 120
assert {row["source_file"] for row in usage} == {
    row["source_file"] for row in by_cohort[64]
}
assert all(
    int(row[key]) >= 0
    for row in usage
    for key in ("input_tokens_r20", "output_tokens_r20")
)
assert sum(int(row["input_tokens_r20"]) for row in usage) == 3413857
assert sum(int(row["output_tokens_r20"]) for row in usage) == 284832
price = deepseek["cost_r20"]
assert MODEL_PRICING[deepseek["model"]] == (
    price["input_usd_per_1m_tokens"],
    price["output_usd_per_1m_tokens"],
)
cost = math.fsum(
    calculate_cost(
        deepseek["model"], int(row["input_tokens_r20"]), int(row["output_tokens_r20"])
    )
    for row in usage
)
successes = sum(row["success"] and row["turns"] <= 20 for row in by_cohort[64])
per_root = cost / successes
assert math.isclose(per_root, price["cost_per_successful_root_usd"], rel_tol=1e-10)
print(
    f"DeepSeek V4 Flash API: ${per_root:.8f}/success; {per_root / local_cost:.4f}x the PrivEsc-LLM RTX 5090 estimate."
)

chain_root = released / "runs/paper/04_static_benchmark/chainreactor/base/20260524"
with (chain_root / "runs.csv").open() as stream:
    chain_rows = list(csv.DictReader(stream))
chain = json.loads((chain_root / "stats/evaluation_summary_paper.json").read_text())
assert len(chain_rows) == len({row["scenario"] for row in chain_rows}) == 12
assert set(chain["scenario_stats"]) == {row["scenario"] for row in chain_rows}
for row in chain_rows:
    recorded = chain["scenario_stats"][row["scenario"]]
    for key in ("plan_found", "planner_timed_out", "extract_succeeded"):
        assert row[key] in ("0", "1")
        assert bool(int(row[key])) == recorded[key]
    assert (
        int(row["planner_time_limit_sec"]) == recorded["planner_time_limit_sec"] == 1800
    )
assert (
    sum(int(row["plan_found"]) for row in chain_rows)
    == chain["overall_stats"]["plans_found"]
    == 1
)
assert (
    sum(int(row["planner_timed_out"]) for row in chain_rows)
    == chain["overall_stats"]["planner_timed_out"]
    == 3
)
assert "executed_success" not in chain["overall_stats"]
assert all("executed_success" not in row for row in chain["model_stats"].values())
print("ChainReactor: 1/12 plans, 3 timeouts; execution was not measured.")
print(
    "Scalar checks: 10,600 rollouts, reported rates and CIs, DeepSeek cost, and ChainReactor accounting verified."
)
