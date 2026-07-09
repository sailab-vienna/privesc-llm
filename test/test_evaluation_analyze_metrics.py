import json

import pandas as pd
import pytest

from src.config import _DEFAULT_RL_GENERATORS
from src.evaluation import analyze
from src.evaluation.analysis.summary import (
    create_timing_distribution_table,
    create_timing_summary_table,
)
from src.rl.benchmark import load_benchmark_scenarios
from src.utils.pricing import calculate_cost


def current_trace(
    *,
    model: str,
    scenario: str,
    success: bool,
    turns: int,
    status: str = "completed",
    benchmark_eligible: bool = True,
    max_turns: int = 60,
    error: str | None = None,
    metadata: dict | None = None,
    **extra,
) -> dict:
    payload = {
        "model": model,
        "scenario": scenario,
        "status": status,
        "success": success,
        "turns": turns,
        "history": [],
        "error": error,
        "metadata": {
            "benchmark_eligible": benchmark_eligible,
            "prompt_vars": {"max_turns": max_turns},
        },
    }
    if metadata:
        payload["metadata"].update(metadata)
    payload.update(extra)
    return payload


def test_budgeted_success_rate_uses_rounds_not_tool_calls() -> None:
    df = pd.DataFrame(
        [
            {
                "success": True,
                "turns": 15,
                "first_success_tool_step": 120,
            },
            {
                "success": True,
                "turns": 61,
                "first_success_tool_step": 1,
            },
            {
                "success": False,
                "turns": 5,
                "first_success_tool_step": 2,
            },
        ]
    )

    metrics = analyze.budgeted_success_rate_for_group(df)

    assert metrics["sr_within_rounds_10"] == pytest.approx(0.0)
    assert metrics["sr_within_rounds_20"] == pytest.approx(1 / 3)
    assert metrics["sr_within_rounds_60"] == pytest.approx(1 / 3)


def test_budgeted_success_at_60_equals_micro_success_when_cap_is_60() -> None:
    df = pd.DataFrame(
        [
            {"success": True, "turns": 1},
            {"success": True, "turns": 60},
            {"success": False, "turns": 60},
            {"success": False, "turns": 10},
        ]
    )

    metrics = analyze.budgeted_success_rate_for_group(df)

    assert metrics["sr_within_rounds_60"] == pytest.approx(float(df["success"].mean()))


def test_budgeted_success_rate_is_monotonic() -> None:
    df = pd.DataFrame(
        [
            {"success": True, "turns": 8},
            {"success": True, "turns": 22},
            {"success": True, "turns": 49},
            {"success": False, "turns": 12},
            {"success": False, "turns": 60},
        ]
    )

    metrics = analyze.budgeted_success_rate_for_group(df)
    vals = [metrics[f"sr_within_rounds_{r}"] for r in analyze.ROUND_BUDGETS]
    assert vals == sorted(vals)


def test_budgeted_success_rate_with_ci_returns_valid_bounds() -> None:
    df = pd.DataFrame(
        [
            {"success": True, "turns": 8},
            {"success": True, "turns": 22},
            {"success": False, "turns": 60},
            {"success": False, "turns": 7},
        ]
    )

    curve = analyze.budgeted_success_rate_with_ci_for_group(df)

    for r in analyze.ROUND_BUDGETS:
        assert (
            0.0 <= curve[r]["ci_low"] <= curve[r]["rate"] <= curve[r]["ci_high"] <= 1.0
        )


def test_budgeted_success_stats_returns_counts_and_ci() -> None:
    df = pd.DataFrame(
        [
            {"success": True, "turns": 8},
            {"success": True, "turns": 22},
            {"success": False, "turns": 60},
            {"success": False, "turns": 7},
        ]
    )

    stats = analyze.budgeted_success_stats_for_group(df, 20)

    assert stats["round_budget"] == 20
    assert stats["num_runs"] == 4
    assert stats["num_successful_runs"] == 1
    assert stats["rate"] == pytest.approx(0.25)
    assert 0.0 <= float(stats["ci_low"]) <= 0.25 <= float(stats["ci_high"]) <= 1.0


def test_expected_cost_per_successful_root_uses_requested_budget() -> None:
    df = pd.DataFrame(
        [
            {"success": True, "turns": 10, "cost": 1.0},
            {"success": True, "turns": 30, "cost": 1.0},
            {"success": False, "turns": 60, "cost": 1.0},
            {"success": False, "turns": 60, "cost": 1.0},
        ]
    )

    at_20 = analyze.expected_cost_per_successful_root_for_group(df, 20)
    at_60 = analyze.expected_cost_per_successful_root_for_group(df, 60)

    assert at_20["success_rate"] == pytest.approx(0.25)
    assert at_20["expected_cost_per_root"] == pytest.approx(4.0)
    assert at_60["success_rate"] == pytest.approx(0.5)
    assert at_60["expected_cost_per_root"] == pytest.approx(2.0)


def test_expected_cost_per_successful_root_prefers_budgeted_cost_columns() -> None:
    df = pd.DataFrame(
        [
            {
                "success": True,
                "turns": 10,
                "cost": 100.0,
                "cost_within_rounds_20": 1.0,
                "cost_within_rounds_60": 2.0,
            },
            {
                "success": False,
                "turns": 60,
                "cost": 100.0,
                "cost_within_rounds_20": 3.0,
                "cost_within_rounds_60": 4.0,
            },
        ]
    )

    at_20 = analyze.expected_cost_per_successful_root_for_group(df, 20)
    at_60 = analyze.expected_cost_per_successful_root_for_group(df, 60)

    assert at_20["expected_cost_per_root"] == pytest.approx(4.0)
    assert at_60["expected_cost_per_root"] == pytest.approx(6.0)


def test_compute_metrics_includes_wilson_ci() -> None:
    df = pd.DataFrame(
        [
            {
                "success": True,
                "turns": 4,
                "messages": 10,
                "total_tokens": 100,
                "prompt_tokens": 60,
                "completion_tokens": 40,
                "cost": 0.1,
                "rollout_wall_clock_ms": 1000.0,
                "time_to_root_wall_clock_ms": 800.0,
                "time_to_root_interaction_ms_raw": 300.0,
                "total_llm_ms_raw": 200.0,
                "total_tool_ms_raw": 100.0,
                "cost_ms_raw": 300.0,
            },
            {
                "success": False,
                "turns": 60,
                "messages": 20,
                "total_tokens": 300,
                "prompt_tokens": 240,
                "completion_tokens": 60,
                "cost": 0.2,
                "rollout_wall_clock_ms": None,
                "time_to_root_wall_clock_ms": None,
                "time_to_root_interaction_ms_raw": None,
                "total_llm_ms_raw": 300.0,
                "total_tool_ms_raw": 150.0,
                "cost_ms_raw": 450.0,
            },
        ]
    )

    metrics = analyze.compute_metrics_for_group(df)

    assert metrics["num_runs"] == 2
    assert metrics["num_successful_runs"] == 1
    assert metrics["success_rate_micro_avg"] == pytest.approx(0.5)
    assert 0.0 <= metrics["success_rate_ci95_low"] <= 0.5
    assert 0.5 <= metrics["success_rate_ci95_high"] <= 1.0
    assert metrics["avg_turns_success"] == pytest.approx(4.0)
    assert metrics["primary_round_budget"] == analyze.PRIMARY_BUDGET
    assert metrics["num_successful_runs_within_primary_budget"] == 1
    assert metrics["success_rate_within_primary_budget"] == pytest.approx(0.5)
    assert metrics[
        "expected_cost_per_successful_root_within_primary_budget"
    ] == pytest.approx(0.15 / 0.5)
    assert metrics["avg_rollout_wall_clock_ms"] == pytest.approx(1000.0)
    assert metrics["avg_time_to_root_wall_clock_ms"] == pytest.approx(800.0)
    assert metrics["avg_time_to_root_interaction_ms_raw"] == pytest.approx(300.0)
    assert metrics["avg_total_llm_ms_raw"] == pytest.approx(250.0)
    assert metrics["avg_total_tool_ms_raw"] == pytest.approx(125.0)
    assert metrics["avg_cost_ms_raw"] == pytest.approx(375.0)
    assert metrics["timing_stats"]["rollout_wall_clock_ms"]["count"] == 1
    assert metrics["timing_stats"]["rollout_wall_clock_ms"]["avg_failure"] is None
    assert metrics["timing_stats"]["time_to_root_wall_clock_ms"]["count"] == 1
    assert metrics["timing_stats"]["time_to_root_wall_clock_ms"]["avg_success"] == pytest.approx(
        800.0
    )
    assert metrics["timing_stats"]["total_llm_ms_raw"]["p50"] == pytest.approx(250.0)
    assert metrics["timing_stats"]["total_llm_ms_raw"]["p90"] == pytest.approx(290.0)
    assert metrics["timing_stats"]["total_llm_ms_raw"]["min"] == pytest.approx(200.0)
    assert metrics["timing_stats"]["total_llm_ms_raw"]["max"] == pytest.approx(300.0)
    assert metrics["timing_stats"]["total_llm_ms_raw"]["avg_success"] == pytest.approx(
        200.0
    )
    assert metrics["timing_stats"]["total_llm_ms_raw"]["avg_failure"] == pytest.approx(
        300.0
    )


def test_compute_metrics_no_success_has_none_turns_success() -> None:
    df = pd.DataFrame(
        [
            {
                "success": False,
                "turns": 60,
                "messages": 20,
                "total_tokens": 300,
                "prompt_tokens": 240,
                "completion_tokens": 60,
                "cost": 0.2,
            }
        ]
    )

    metrics = analyze.compute_metrics_for_group(df)

    assert metrics["avg_turns_success"] is None


def test_compute_metrics_avg_turns_success_uses_all_successes() -> None:
    df = pd.DataFrame(
        [
            {
                "success": True,
                "turns": 8,
                "messages": 10,
                "total_tokens": 100,
                "prompt_tokens": 60,
                "completion_tokens": 40,
                "cost": 0.1,
            },
            {
                "success": True,
                "turns": 30,
                "messages": 12,
                "total_tokens": 120,
                "prompt_tokens": 70,
                "completion_tokens": 50,
                "cost": 0.2,
            },
            {
                "success": False,
                "turns": 60,
                "messages": 20,
                "total_tokens": 300,
                "prompt_tokens": 240,
                "completion_tokens": 60,
                "cost": 0.3,
            },
        ]
    )

    metrics = analyze.compute_metrics_for_group(df)

    assert metrics["success_rate_micro_avg"] == pytest.approx(2 / 3)
    assert metrics["success_rate_within_primary_budget"] == pytest.approx(1 / 3)
    assert metrics["avg_turns_success"] == pytest.approx(19.0)


def test_parse_excludes_error_runs_and_runtime_errors(tmp_path) -> None:
    model_dir = tmp_path / "openai-gpt"
    model_dir.mkdir(parents=True)

    valid_trace = current_trace(
        model="openai/gpt-5.2",
        scenario="01_vuln_suid_gtfo",
        success=True,
        turns=3,
        prompt_tokens=1000,
        completion_tokens=1000,
        total_tokens=2000,
        history=[
            {"role": "system", "content": "s"},
            {"role": "assistant", "content": "a"},
            {"role": "tool", "content": '{"got_root": false}'},
            {"role": "assistant", "content": "b"},
            {"role": "tool", "content": '{"got_root": false}'},
            {"role": "assistant", "content": "c"},
            {"role": "tool", "content": '{"got_root": true}'},
        ],
    )
    (model_dir / "run_valid.json").write_text(json.dumps(valid_trace))

    error_run_trace = current_trace(
        model="openai/gpt-5.2",
        scenario="01_vuln_suid_gtfo",
        success=False,
        turns=0,
        status="error",
        benchmark_eligible=False,
        error="Process exited with non-zero exit status 125",
        history=[],
    )
    (model_dir / "error_run_1.json").write_text(json.dumps(error_run_trace))

    runtime_error_trace = current_trace(
        model="openai/gpt-5.2",
        scenario="01_vuln_suid_gtfo",
        success=False,
        turns=0,
        status="error",
        benchmark_eligible=False,
        error="Provider returned error",
        history=[],
    )
    (model_dir / "run_runtime_error.json").write_text(json.dumps(runtime_error_trace))

    df = analyze.parse_evaluation_results_detailed(
        str(tmp_path), validate_run_counts=False
    )

    assert len(df) == 1
    assert bool(df.iloc[0]["success"]) is True
    assert df.iloc[0]["cost"] == pytest.approx(
        calculate_cost(
            valid_trace["model"],
            valid_trace["prompt_tokens"],
            valid_trace["completion_tokens"],
        )
    )


def test_parse_excludes_started_error_runs_with_source_scenario(tmp_path) -> None:
    model_dir = tmp_path / "checkpoint-model"
    model_dir.mkdir(parents=True)

    timed_out_trace = current_trace(
        model="checkpoint-model",
        scenario="error_run_9",
        success=False,
        turns=1,
        status="error",
        benchmark_eligible=False,
        error="Run timed out after 3600s",
        metadata={
            "trace_incomplete": True,
            "source_scenario": "weak_password",
        },
        history=[
            {"role": "user", "content": "Start privilege escalation now."},
            {"role": "assistant", "content": "Investigating."},
        ],
        prompt_tokens=240,
        completion_tokens=60,
        total_tokens=300,
    )
    (model_dir / "error_run_9.json").write_text(json.dumps(timed_out_trace))

    with pytest.raises(ValueError, match="No data was parsed"):
        analyze.parse_evaluation_results_detailed(str(tmp_path), validate_run_counts=False)


def test_parse_uses_history_for_secondary_metrics(tmp_path) -> None:
    model_dir = tmp_path / "local"
    model_dir.mkdir(parents=True)

    trace = current_trace(
        model="qwen3-4b",
        scenario="01_vuln_suid_gtfo",
        success=False,
        turns=2,
        max_turns=2,
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        history=[
            {"role": "system", "content": "s"},
            {
                "role": "assistant",
                "content": "x",
                "tool_calls": [
                    {
                        "function": {
                            "name": "exec_command",
                            "arguments": '{"command":"id"}',
                        }
                    }
                ],
            },
            {"role": "tool", "content": '{"got_root": false}'},
            {
                "role": "assistant",
                "content": "y",
                "tool_calls": [
                    {
                        "function": {
                            "name": "test_credentials",
                            "arguments": '{"user":"root","password":"x"}',
                        }
                    }
                ],
            },
        ],
    )
    (model_dir / "run.json").write_text(json.dumps(trace))

    df = analyze.parse_evaluation_results_detailed(
        str(tmp_path), validate_run_counts=False
    )

    assert len(df) == 1
    assert int(df.iloc[0]["messages"]) == 4
    assert int(df.iloc[0]["exec_command_count"]) == 1
    assert int(df.iloc[0]["test_credentials_count"]) == 1


def test_parse_uses_token_pricing_for_cost_columns(tmp_path) -> None:
    model_dir = tmp_path / "local"
    model_dir.mkdir(parents=True)

    trace = current_trace(
        model="qwen3-4b",
        scenario="01_vuln_suid_gtfo",
        success=True,
        turns=2,
        prompt_tokens=15,
        completion_tokens=6,
        total_tokens=21,
        total_cost=999.0,
        history=[
            {"role": "system", "content": "s"},
            {"role": "assistant", "content": "a"},
            {"role": "tool", "content": '{"got_root": false}'},
            {"role": "assistant", "content": "b"},
            {"role": "tool", "content": '{"got_root": true}'},
        ],
        llm_usage_by_turn=[
            {
                "turn": 1,
                "prompt_tokens": 10,
                "completion_tokens": 1,
                "total_tokens": 11,
                "cost": 0.1,
            },
            {
                "turn": 2,
                "prompt_tokens": 5,
                "completion_tokens": 5,
                "total_tokens": 10,
                "cost": 0.2,
            },
        ],
    )
    (model_dir / "run.json").write_text(json.dumps(trace))

    df = analyze.parse_evaluation_results_detailed(
        str(tmp_path), validate_run_counts=False
    )

    assert len(df) == 1
    assert df.iloc[0]["cost"] == pytest.approx(
        calculate_cost("qwen3-4b", 15, 6)
    )
    assert df.iloc[0]["cost_within_rounds_5"] == pytest.approx(
        calculate_cost("qwen3-4b", 15, 6)
    )
    assert df.iloc[0]["cost_within_rounds_20"] == pytest.approx(
        calculate_cost("qwen3-4b", 15, 6)
    )


def test_parse_zero_total_cost_falls_back_to_token_pricing(tmp_path) -> None:
    model_dir = tmp_path / "anthropic"
    model_dir.mkdir(parents=True)

    trace = current_trace(
        model="anthropic/claude-opus-4.6",
        scenario="01_vuln_suid_gtfo",
        success=True,
        turns=2,
        prompt_tokens=5000,
        completion_tokens=100,
        total_tokens=5100,
        total_cost=0.0,
        history=[
            {"role": "system", "content": "s"},
            {"role": "assistant", "content": "a"},
            {"role": "tool", "content": '{"got_root": false}'},
            {"role": "assistant", "content": "b"},
            {"role": "tool", "content": '{"got_root": true}'},
        ],
    )
    (model_dir / "run.json").write_text(json.dumps(trace))

    df = analyze.parse_evaluation_results_detailed(
        str(tmp_path), validate_run_counts=False
    )

    assert len(df) == 1
    expected = calculate_cost(
        "anthropic/claude-opus-4.6",
        trace["prompt_tokens"],
        trace["completion_tokens"],
    )
    assert df.iloc[0]["cost"] == pytest.approx(expected)


def test_parse_missing_total_cost_falls_back_to_token_pricing(tmp_path) -> None:
    model_dir = tmp_path / "deepseek"
    model_dir.mkdir(parents=True)

    trace = current_trace(
        model="deepseek/deepseek-v3.2",
        scenario="01_vuln_suid_gtfo",
        success=True,
        turns=2,
        prompt_tokens=20_000,
        completion_tokens=2_000,
        total_tokens=22_000,
        history=[
            {"role": "system", "content": "s"},
            {"role": "assistant", "content": "a"},
            {"role": "tool", "content": '{"got_root": false}'},
            {"role": "assistant", "content": "b"},
            {"role": "tool", "content": '{"got_root": true}'},
        ],
    )
    (model_dir / "run.json").write_text(json.dumps(trace))

    df = analyze.parse_evaluation_results_detailed(
        str(tmp_path), validate_run_counts=False
    )

    assert len(df) == 1
    expected = calculate_cost(
        "deepseek/deepseek-v3.2",
        trace["prompt_tokens"],
        trace["completion_tokens"],
    )
    assert df.iloc[0]["cost"] == pytest.approx(expected)


def test_parse_qwen3_4b_uses_fallback_token_pricing(tmp_path) -> None:
    model_dir = tmp_path / "qwen"
    model_dir.mkdir(parents=True)

    trace = current_trace(
        model="qwen/qwen3-4b-sft-rl",
        scenario="01_vuln_suid_gtfo",
        success=True,
        turns=3,
        prompt_tokens=200_000,
        completion_tokens=20_000,
        total_tokens=220_000,
        total_cost=0.0,
        history=[
            {"role": "system", "content": "s"},
            {"role": "assistant", "content": "a"},
            {"role": "tool", "content": '{"got_root": false}'},
            {"role": "assistant", "content": "b"},
            {"role": "tool", "content": '{"got_root": false}'},
            {"role": "assistant", "content": "c"},
            {"role": "tool", "content": '{"got_root": true}'},
        ],
    )
    (model_dir / "run.json").write_text(json.dumps(trace))

    df = analyze.parse_evaluation_results_detailed(
        str(tmp_path), validate_run_counts=False
    )

    assert len(df) == 1
    expected = calculate_cost(
        "qwen/qwen3-4b-sft-rl",
        trace["prompt_tokens"],
        trace["completion_tokens"],
    )
    assert df.iloc[0]["cost"] == pytest.approx(expected)


def test_parse_total_tokens_falls_back_to_prompt_plus_completion(tmp_path) -> None:
    model_dir = tmp_path / "openai"
    model_dir.mkdir(parents=True)

    trace = current_trace(
        model="openai/gpt-5.2",
        scenario="01_vuln_suid_gtfo",
        success=False,
        turns=3,
        max_turns=3,
        prompt_tokens=1200,
        completion_tokens=300,
        total_tokens=0,
        history=[
            {"role": "system", "content": "s"},
            {"role": "assistant", "content": "a"},
            {"role": "tool", "content": '{"got_root": false}'},
            {"role": "assistant", "content": "b"},
            {"role": "tool", "content": '{"got_root": false}'},
            {"role": "assistant", "content": "c"},
            {"role": "tool", "content": '{"got_root": false}'},
        ],
    )
    (model_dir / "run.json").write_text(json.dumps(trace))

    df = analyze.parse_evaluation_results_detailed(
        str(tmp_path), validate_run_counts=False
    )

    assert len(df) == 1
    assert (
        int(df.iloc[0]["total_tokens"])
        == trace["prompt_tokens"] + trace["completion_tokens"]
    )


def test_parse_timing_fields_are_optional_with_cost_fallback(tmp_path) -> None:
    model_dir = tmp_path / "openai"
    model_dir.mkdir(parents=True)

    trace = current_trace(
        model="openai/gpt-5.2",
        scenario="01_vuln_suid_gtfo",
        success=True,
        turns=3,
        timing={
            "rollout_wall_clock_ms": 350.0,
            "time_to_root_wall_clock_ms": 210.0,
            "time_to_root_interaction_ms_raw": 180.0,
            "total_llm_ms_raw": 200.0,
            "total_tool_ms_raw": 121.0,
        },
    )
    (model_dir / "run.json").write_text(json.dumps(trace))

    df = analyze.parse_evaluation_results_detailed(
        str(tmp_path), validate_run_counts=False
    )

    assert len(df) == 1
    assert df.iloc[0]["rollout_wall_clock_ms"] == pytest.approx(350.0)
    assert df.iloc[0]["time_to_root_wall_clock_ms"] == pytest.approx(210.0)
    assert df.iloc[0]["time_to_root_interaction_ms_raw"] == pytest.approx(180.0)
    assert df.iloc[0]["total_llm_ms_raw"] == pytest.approx(200.0)
    assert df.iloc[0]["total_tool_ms_raw"] == pytest.approx(121.0)
    assert df.iloc[0]["cost_ms_raw"] == pytest.approx(321.0)


def test_parse_validate_run_counts_raises_on_missing_runs(tmp_path) -> None:
    model_dir = tmp_path / "openai"
    model_dir.mkdir(parents=True)

    trace = current_trace(
        model="openai/gpt-5.2",
        scenario="01_vuln_suid_gtfo",
        success=True,
        turns=4,
        prompt_tokens=50,
        completion_tokens=20,
        total_tokens=70,
        history=[
            {"role": "system", "content": "s"},
            {"role": "assistant", "content": "a"},
            {"role": "tool", "content": '{"got_root": false}'},
            {"role": "assistant", "content": "b"},
            {"role": "tool", "content": '{"got_root": false}'},
            {"role": "assistant", "content": "c"},
            {"role": "tool", "content": '{"got_root": false}'},
            {"role": "assistant", "content": "d"},
            {"role": "tool", "content": '{"got_root": true}'},
        ],
    )
    (model_dir / "run.json").write_text(json.dumps(trace))

    with pytest.raises(ValueError, match="Unexpected run counts"):
        analyze.parse_evaluation_results_detailed(
            str(tmp_path), validate_run_counts=True
        )


def test_parse_validate_run_counts_raises_on_absent_expected_scenario(tmp_path) -> None:
    model_dir = tmp_path / "openai"
    model_dir.mkdir(parents=True)

    for ordinal in range(10):
        trace = current_trace(
            model="openai/gpt-5.2",
            scenario="01_vuln_suid_gtfo",
            success=True,
            turns=1,
            metadata={"item_run_ordinal": ordinal},
        )
        (model_dir / f"run_{ordinal}.json").write_text(json.dumps(trace))

    with pytest.raises(ValueError, match="02_vuln_password_in_shell_history"):
        analyze.parse_evaluation_results_detailed(
            str(tmp_path),
            validate_run_counts=True,
            expected_scenarios=[
                "01_vuln_suid_gtfo",
                "02_vuln_password_in_shell_history",
            ],
        )


def test_parse_validate_run_counts_raises_on_unexpected_scenario(tmp_path) -> None:
    model_dir = tmp_path / "openai"
    model_dir.mkdir(parents=True)

    for ordinal in range(10):
        expected_trace = current_trace(
            model="openai/gpt-5.2",
            scenario="01_vuln_suid_gtfo",
            success=True,
            turns=1,
            metadata={"item_run_ordinal": ordinal},
        )
        extra_trace = current_trace(
            model="openai/gpt-5.2",
            scenario="debug_extra_scenario",
            success=True,
            turns=1,
            metadata={"item_run_ordinal": ordinal},
        )
        (model_dir / f"expected_{ordinal}.json").write_text(json.dumps(expected_trace))
        (model_dir / f"extra_{ordinal}.json").write_text(json.dumps(extra_trace))

    with pytest.raises(ValueError, match="debug_extra_scenario"):
        analyze.parse_evaluation_results_detailed(
            str(tmp_path),
            validate_run_counts=True,
            expected_scenarios=["01_vuln_suid_gtfo"],
        )


def test_expected_scenarios_from_static_experiment_uses_runner_config() -> None:
    assert analyze._expected_protocol_from_experiment(
        "eval/paper_static_qwen3_4b_base"
    ) == (load_benchmark_scenarios(), 20)


def test_expected_scenarios_from_procedural_experiment_uses_generators() -> None:
    assert analyze._expected_protocol_from_experiment("eval/paper_procedural") == (
        _DEFAULT_RL_GENERATORS,
        20,
    )


def test_parse_raises_on_inconsistent_turn_count(tmp_path) -> None:
    model_dir = tmp_path / "broken"
    model_dir.mkdir(parents=True)

    trace = current_trace(
        model="openai/gpt-5.2",
        scenario="01_vuln_suid_gtfo",
        success=False,
        turns=3,
        max_turns=3,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        history=[
            {"role": "system", "content": "s"},
            {"role": "assistant", "content": "a"},
            {"role": "tool", "content": '{"got_root": false}'},
        ],
    )
    (model_dir / "run.json").write_text(json.dumps(trace))

    with pytest.raises(ValueError, match="turns field disagrees"):
        analyze.parse_evaluation_results_detailed(
            str(tmp_path), validate_run_counts=False
        )


def test_parse_skips_error_runs_without_real_scenario(tmp_path) -> None:
    model_dir = tmp_path / "openai"
    model_dir.mkdir(parents=True)

    valid_trace = current_trace(
        model="openai/gpt-5.2",
        scenario="01_vuln_suid_gtfo",
        success=False,
        turns=1,
        max_turns=1,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        history=[
            {"role": "system", "content": "s"},
            {"role": "assistant", "content": "a"},
            {"role": "tool", "content": '{"got_root": false}'},
        ],
    )
    unknown_error = current_trace(
        model="openai/gpt-5.2",
        scenario="error_run_7",
        success=False,
        turns=0,
        status="error",
        benchmark_eligible=True,
        metadata={"source_scenario": None},
        history=[],
    )
    (model_dir / "run.json").write_text(json.dumps(valid_trace))
    (model_dir / "error_run_7.json").write_text(json.dumps(unknown_error))

    df = analyze.parse_evaluation_results_detailed(
        str(tmp_path), validate_run_counts=False
    )

    assert len(df) == 1
    assert df.iloc[0]["scenario"] == "01_vuln_suid_gtfo"


def test_calculate_and_save_stats_writes_scenario_model_stats(tmp_path) -> None:
    df = pd.DataFrame(
        [
            {
                "model": "m1",
                "scenario": "s1",
                "success": True,
                "turns": 5,
                "cost": 0.1,
                "total_tokens": 100,
                "prompt_tokens": 70,
                "completion_tokens": 30,
                "messages": 10,
                "total_tool_calls": 2,
                "exec_command_count": 2,
                "test_credentials_count": 0,
                "first_success_tool_step": 2,
                "rollout_wall_clock_ms": 1200.0,
                "time_to_root_wall_clock_ms": 900.0,
                "time_to_root_interaction_ms_raw": 600.0,
                "total_llm_ms_raw": 400.0,
                "total_tool_ms_raw": 250.0,
                "cost_ms_raw": 650.0,
            },
            {
                "model": "m2",
                "scenario": "s1",
                "success": False,
                "turns": 60,
                "cost": 0.2,
                "total_tokens": 200,
                "prompt_tokens": 150,
                "completion_tokens": 50,
                "messages": 20,
                "total_tool_calls": 4,
                "exec_command_count": 4,
                "test_credentials_count": 0,
                "first_success_tool_step": None,
                "rollout_wall_clock_ms": None,
                "time_to_root_wall_clock_ms": None,
                "time_to_root_interaction_ms_raw": None,
                "total_llm_ms_raw": 200.0,
                "total_tool_ms_raw": 100.0,
                "cost_ms_raw": 300.0,
            },
        ]
    )

    analyze.calculate_and_save_stats(df, str(tmp_path))
    stats = json.loads((tmp_path / "evaluation_summary.json").read_text())
    paper_stats = json.loads((tmp_path / "evaluation_summary_paper.json").read_text())

    assert "scenario_model_stats" in stats
    assert stats["scenario_model_stats"]["s1"]["m1"][
        "success_rate_micro_avg"
    ] == pytest.approx(1.0)
    assert stats["scenario_model_stats"]["s1"]["m1"][
        "sr_within_rounds_10"
    ] == pytest.approx(1.0)
    assert stats["overall_stats"]["avg_total_llm_ms_raw"] == pytest.approx(300.0)
    assert stats["overall_stats"]["avg_time_to_root_wall_clock_ms"] == pytest.approx(
        900.0
    )
    assert stats["model_stats"]["m1"]["avg_rollout_wall_clock_ms"] == pytest.approx(
        1200.0
    )
    assert stats["model_stats"]["m1"]["avg_time_to_root_interaction_ms_raw"] == pytest.approx(
        600.0
    )
    assert stats["scenario_model_stats"]["s1"]["m2"][
        "avg_cost_ms_raw"
    ] == pytest.approx(300.0)
    assert stats["overall_stats"]["timing_stats"]["total_llm_ms_raw"][
        "p50"
    ] == pytest.approx(300.0)
    assert stats["scenario_stats"]["s1"]["timing_stats"]["total_tool_ms_raw"][
        "min"
    ] == pytest.approx(100.0)
    assert stats["scenario_stats"]["s1"]["timing_stats"]["total_tool_ms_raw"][
        "avg_success"
    ] == pytest.approx(250.0)
    assert stats["scenario_stats"]["s1"]["timing_stats"]["total_tool_ms_raw"][
        "avg_failure"
    ] == pytest.approx(100.0)
    assert "mean_scenario_success_rate" not in stats["model_stats"]["m1"]
    assert (
        paper_stats["overall_stats"]["paper_primary_metric_key"]
        == "success_rate_within_primary_budget"
    )
    assert paper_stats["model_stats"]["m1"]["avg_time_to_root_wall_clock_ms"] == pytest.approx(
        900.0
    )
    assert paper_stats["model_stats"]["m1"]["timing_stats"]["time_to_root_wall_clock_ms"][
        "avg"
    ] == pytest.approx(900.0)
    assert "success_rate_micro_avg" not in paper_stats["model_stats"]["m1"]


def test_calculate_and_save_stats_can_suppress_paper_summary(tmp_path) -> None:
    df = pd.DataFrame(
        [
            {
                "model": "m1",
                "scenario": "s1",
                "success": True,
                "turns": 1,
                "cost": 0.0,
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "messages": 1,
                "total_tool_calls": 0,
                "exec_command_count": 0,
                "test_credentials_count": 0,
                "first_success_tool_step": None,
            }
        ]
    )
    stale_paper_summary = tmp_path / "evaluation_summary_paper.json"
    stale_paper_summary.write_text("{}")

    analyze.calculate_and_save_stats(df, str(tmp_path), write_paper_summary=False)

    assert (tmp_path / "evaluation_summary.json").exists()
    assert not stale_paper_summary.exists()


def test_paired_mcnemar_uses_saved_pairing_ids() -> None:
    df = pd.DataFrame(
        [
            {
                "model": "a/m1",
                "scenario": "01_s",
                "item_run_ordinal": 0,
                "run_index": 0,
                "success": True,
                "turns": 5,
            },
            {
                "model": "a/m2",
                "scenario": "01_s",
                "item_run_ordinal": 0,
                "run_index": 0,
                "success": False,
                "turns": 60,
            },
            {
                "model": "a/m1",
                "scenario": "02_s",
                "item_run_ordinal": 0,
                "run_index": 1,
                "success": False,
                "turns": 60,
            },
            {
                "model": "a/m2",
                "scenario": "02_s",
                "item_run_ordinal": 0,
                "run_index": 1,
                "success": True,
                "turns": 10,
            },
        ]
    )

    stats = analyze.paired_mcnemar_test(df, "a/m1", "a/m2", round_budget=20)

    assert stats["num_pairs"] == 2
    assert stats["n_10_a_only"] == 1
    assert stats["n_01_b_only"] == 1
    assert stats["exact_pvalue"] == pytest.approx(1.0)


def test_summary_table_headers_use_consistent_abbreviations() -> None:
    table = analyze.create_summary_table("x")
    headers = [col.header for col in table.columns]
    assert headers == [
        "Model",
        "SR",
        "CI95",
        "N",
        "R_s",
        "Msg",
        "TokIn",
        "TokOut",
        "Tok",
        "C_avg",
        "C_tot",
    ]


def test_timing_summary_table_headers_are_compact() -> None:
    table = create_timing_summary_table("x")
    headers = [col.header for col in table.columns]
    assert headers == ["Model", "Wall", "LLM", "Tool", "Cost"]


def test_timing_distribution_table_headers_are_compact() -> None:
    table = create_timing_distribution_table("x", "Scenario")
    headers = [col.header for col in table.columns]
    assert headers == ["Scenario", "Avg", "P50", "P90", "Min", "Max", "Succ", "Fail"]


def test_scenario_model_matrix_uses_scenario_and_model_axes() -> None:
    df = pd.DataFrame(
        [
            {"model": "a/m1", "scenario": "01_s", "success": True, "turns": 5},
            {"model": "a/m2", "scenario": "01_s", "success": False, "turns": 60},
            {"model": "a/m1", "scenario": "02_s", "success": True, "turns": 10},
            {"model": "a/m2", "scenario": "02_s", "success": True, "turns": 20},
        ]
    )

    table = analyze.create_scenario_model_matrix(df, "x", metric="sr")
    headers = [col.header for col in table.columns]

    assert headers == ["Scenario", "m1", "m2"]
    assert len(table.rows) == 2
