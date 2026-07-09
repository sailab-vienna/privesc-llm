import json
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console

from src.evaluation.analysis.constants import ROUND_BUDGETS, RUNS_PER_SCENARIO
from src.evaluation.analysis.trace_payload import (
    trace_item_run_ordinal,
    trace_metadata,
    trace_scenario_name,
)
from src.runner.result_policy import is_benchmark_eligible_payload
from src.utils.pricing import calculate_cost

console = Console()


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _tool_payload(message: dict[str, Any]) -> dict[str, Any] | None:
    content = message.get("content")
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None
    return None


def first_success_tool_step(messages: list[dict[str, Any]]) -> int | None:
    tool_step = 0
    for msg in messages:
        if msg.get("role") != "tool":
            continue
        tool_step += 1
        payload = _tool_payload(msg)
        if isinstance(payload, dict) and payload.get("got_root") is True:
            return tool_step
    return None


def _got_root_from_messages(messages: list[dict[str, Any]]) -> bool | None:
    saw_got_root_signal = False
    for message in messages:
        if message.get("role") != "tool":
            continue
        payload = _tool_payload(message)
        if not isinstance(payload, dict) or "got_root" not in payload:
            continue
        saw_got_root_signal = True
        if payload.get("got_root") is True:
            return True
    if saw_got_root_signal:
        return False
    return None


def _assistant_turn_count(messages: list[dict[str, Any]]) -> int:
    return sum(1 for message in messages if message.get("role") == "assistant")


def _per_turn_llm_usage(content: dict[str, Any]) -> list[dict[str, Any]]:
    raw_usage = content.get("llm_usage_by_turn")
    if isinstance(raw_usage, list):
        return [entry for entry in raw_usage if isinstance(entry, dict)]
    usage = content.get("usage")
    if isinstance(usage, dict):
        nested = usage.get("llm_usage_by_turn")
        if isinstance(nested, list):
            return [entry for entry in nested if isinstance(entry, dict)]
    return []


def _budget_costs(
    content: dict[str, Any],
    model_name: str,
    full_run_cost: float,
) -> dict[int, float]:
    turns = _per_turn_llm_usage(content)
    if not turns:
        return {round_budget: full_run_cost for round_budget in ROUND_BUDGETS}
    if any(not bool(turn.get("available", True)) for turn in turns):
        return {round_budget: full_run_cost for round_budget in ROUND_BUDGETS}

    cumulative_costs: list[float] = []
    running_cost = 0.0
    for turn in turns:
        prompt_tokens = int(turn.get("prompt_tokens", 0) or 0)
        completion_tokens = int(turn.get("completion_tokens", 0) or 0)
        running_cost += calculate_cost(model_name, prompt_tokens, completion_tokens)
        cumulative_costs.append(running_cost)

    if not cumulative_costs:
        return {round_budget: full_run_cost for round_budget in ROUND_BUDGETS}

    out: dict[int, float] = {}
    last_cost = cumulative_costs[-1]
    for round_budget in ROUND_BUDGETS:
        idx = min(round_budget, len(cumulative_costs)) - 1
        out[round_budget] = last_cost if idx < 0 else cumulative_costs[idx]
    return out


def parse_evaluation_results_detailed(
    root_dir: str,
    validate_run_counts: bool = True,
    allowed_models: list[str] | None = None,
    expected_scenarios: list[str] | None = None,
    expected_runs_per_scenario: int | None = None,
) -> pd.DataFrame:
    data = []
    root_path = Path(root_dir)

    skipped_error_runs = 0
    skipped_runtime_errors = 0
    skipped_infra_runs = 0

    for file_path in sorted(root_path.glob("*/*.json")):
        if file_path.stem.startswith("error_run_"):
            skipped_error_runs += 1
            continue

        try:
            with open(file_path, "r") as f:
                content = json.load(f)

            metadata = trace_metadata(content)
            if not is_benchmark_eligible_payload(content):
                skipped_infra_runs += 1
                continue

            if content.get("status") == "error" and not metadata:
                skipped_runtime_errors += 1
                continue

            model_name = content.get("model", file_path.parent.name)
            scenario_name = trace_scenario_name(content)
            if not scenario_name:
                console.print(
                    f"[yellow]Warning:[/] Skipping file without 'scenario' field: {file_path.name}"
                )
                continue
            if content.get("status") == "error" and str(scenario_name).startswith(
                "error_run_"
            ):
                skipped_error_runs += 1
                continue
            hist = content.get("history", [])
            if not isinstance(hist, list):
                raise TypeError("history payload must be a list of messages")

            tools = [tc for m in hist if m.get("tool_calls") for tc in m["tool_calls"]]
            execs = sum(1 for tc in tools if tc["function"]["name"] == "exec_command")
            creds = sum(
                1 for tc in tools if tc["function"]["name"] == "test_credentials"
            )
            first_success_step = first_success_tool_step(hist)

            success = bool(content.get("success"))
            tool_success = _got_root_from_messages(hist)
            if tool_success is not None and success != tool_success:
                raise ValueError(
                    f"Trace success flag disagrees with tool outcomes in {file_path.name}"
                )

            turns = int(content.get("turns", 0))
            assistant_turns = _assistant_turn_count(hist)
            if hist and turns != assistant_turns:
                raise ValueError(
                    f"Trace turns field disagrees with assistant-turn count in {file_path.name}: {turns} != {assistant_turns}"
                )

            p_tok = int(content.get("prompt_tokens") or 0)
            c_tok = int(content.get("completion_tokens") or 0)
            t_tok_raw = content.get("total_tokens")
            t_tok = (
                int(t_tok_raw) if t_tok_raw not in (None, 0, "0") else (p_tok + c_tok)
            )
            tok_sum = p_tok + c_tok

            cost = calculate_cost(model_name, p_tok, c_tok) if tok_sum > 0 else 0.0

            usage_turns = _per_turn_llm_usage(content)
            if usage_turns and len(usage_turns) != turns:
                raise ValueError(
                    f"Per-turn usage length disagrees with turns field in {file_path.name}: {len(usage_turns)} != {turns}"
                )
            budget_costs = _budget_costs(content, model_name, cost)

            timing = content.get("timing")
            if not isinstance(timing, dict):
                timing = {}
            total_llm_ms_raw = _optional_float(timing.get("total_llm_ms_raw"))
            total_tool_ms_raw = _optional_float(timing.get("total_tool_ms_raw"))
            cost_ms_raw = _optional_float(timing.get("cost_ms_raw"))
            if (
                cost_ms_raw is None
                and total_llm_ms_raw is not None
                and total_tool_ms_raw is not None
            ):
                cost_ms_raw = total_llm_ms_raw + total_tool_ms_raw

            row = {
                "model": model_name,
                "scenario": scenario_name,
                "success": success,
                "turns": turns,
                "cost": cost,
                "total_tokens": t_tok,
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "messages": len(hist),
                "total_tool_calls": len(tools),
                "exec_command_count": execs,
                "test_credentials_count": creds,
                "first_success_tool_step": first_success_step,
                "run_index": content.get("run_index"),
                "item_run_ordinal": trace_item_run_ordinal(content),
                "rollout_wall_clock_ms": _optional_float(
                    timing.get("rollout_wall_clock_ms")
                ),
                "time_to_root_wall_clock_ms": _optional_float(
                    timing.get("time_to_root_wall_clock_ms")
                ),
                "time_to_root_interaction_ms_raw": _optional_float(
                    timing.get("time_to_root_interaction_ms_raw")
                ),
                "total_llm_ms_raw": total_llm_ms_raw,
                "total_tool_ms_raw": total_tool_ms_raw,
                "cost_ms_raw": cost_ms_raw,
            }
            for round_budget, budget_cost in budget_costs.items():
                row[f"cost_within_rounds_{round_budget}"] = budget_cost
            data.append(row)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            console.print(
                f"[red]Warning:[/] Could not process file {file_path}. Error: {e}"
            )

    if not data:
        raise ValueError(
            "No data was parsed. Check the directory path and file structure."
        )

    if skipped_error_runs or skipped_runtime_errors or skipped_infra_runs:
        console.print(
            "[yellow]Note:[/] Excluded invalid runs from analysis: "
            f"error_run files={skipped_error_runs}, runtime error files={skipped_runtime_errors}, infra-exhausted files={skipped_infra_runs}"
        )

    df = pd.DataFrame(data)
    df.attrs["excluded_counts"] = {
        "error_run_files": skipped_error_runs,
        "runtime_error_files": skipped_runtime_errors,
        "infra_exhausted_files": skipped_infra_runs,
    }

    if allowed_models is not None:
        before_count = len(df)
        allowed = set(allowed_models)
        df = df[df["model"].isin(allowed)].copy()
        dropped = before_count - len(df)
        if dropped > 0:
            console.print(
                f"[yellow]Note:[/] Excluded {dropped} runs from models not in PAPER_MODEL_ORDER."
            )
        if df.empty:
            raise ValueError(
                "No runs remain after model filtering. Check PAPER_MODEL_ORDER and model aliases."
            )

    if validate_run_counts:
        expected_runs = expected_runs_per_scenario or RUNS_PER_SCENARIO
        counts = df.groupby(["model", "scenario"]).size()
        if expected_scenarios is not None:
            expected_scenario_set = set(expected_scenarios)
            unexpected_scenarios = sorted(
                set(df["scenario"].unique()) - expected_scenario_set
            )
            if unexpected_scenarios:
                raise ValueError(
                    "Unexpected scenarios in paper evaluation results: "
                    + ", ".join(unexpected_scenarios)
                )
            models = sorted(df["model"].unique())
            expected_index = pd.MultiIndex.from_product(
                [models, expected_scenarios], names=["model", "scenario"]
            )
            counts = counts.reindex(expected_index, fill_value=0)
        mismatched = counts[counts != expected_runs]
        if not mismatched.empty:
            raise ValueError(
                "Unexpected run counts (paper protocol requires exactly "
                f"{expected_runs} runs per model-scenario):\n{mismatched}"
            )

    return df
