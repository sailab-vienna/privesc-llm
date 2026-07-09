import json
import logging
import os
from typing import Any

import matplotlib.pyplot as plt
from rich import get_console
from rich.table import Table

from src.config import AppConfig, plain_config_dict
from src.dataset.privesc.collection_stats import selected_metadata_histograms
from src.dataset.privesc.sft_preprocessing import SFTPromptNormalizer

console = get_console()
log = logging.getLogger("privesc_sft")


def save_outputs(
    cfg: AppConfig,
    preprocessor: SFTPromptNormalizer,
    examples: list[dict[str, Any]],
    *,
    output_dir: str,
    filtered_stats: dict[str, int],
) -> None:
    save_tools(examples, output_dir)
    save_scenario_datasets(preprocessor, examples, output_dir)
    print_filter_stats(filtered_stats)
    save_stats(cfg, examples, output_dir)
    save_config_snapshot(cfg, output_dir)


def save_tools(examples: list[dict[str, Any]], output_dir: str) -> None:
    tool_values = [example.get("tools", {}) for example in examples]
    tools = tool_values[0] if tool_values else {}
    base_tools_json = json.dumps(tools, sort_keys=True)
    mismatched_tool_defs = sum(
        1
        for value in tool_values
        if json.dumps(value if value is not None else {}, sort_keys=True)
        != base_tools_json
    )
    if mismatched_tool_defs > 0:
        log.warning(
            "Detected %d traces with differing tool definitions; writing first definition",
            mismatched_tool_defs,
        )

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "tools.json"), "w") as f:
        json.dump(tools, f, indent=2)


def save_scenario_datasets(
    preprocessor: SFTPromptNormalizer,
    examples: list[dict[str, Any]],
    output_dir: str,
) -> None:
    for scenario, group in group_by_scenario(examples).items():
        scenario_dir = os.path.join(output_dir, scenario)
        save_dataset(preprocessor, group, os.path.join(scenario_dir, "traces.jsonl"))
        write_json(os.path.join(scenario_dir, "stats.json"), calculate_stats(group))


def save_dataset(
    preprocessor: SFTPromptNormalizer,
    examples: list[dict[str, Any]],
    output_path: str,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for example in examples:
            out_obj = example.copy()
            out_obj["messages"] = normalized_messages(preprocessor, example)
            f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
    log.info("Saved %d examples to %s", len(examples), output_path)


def normalized_messages(
    preprocessor: SFTPromptNormalizer, example: dict[str, Any]
) -> list[dict[str, Any]]:
    messages = [msg.copy() for msg in example["messages"]]
    meta = json.loads(example.get("metadata", "{}"))
    preprocessor.normalize_message_preamble(messages, meta)
    for msg in messages:
        if msg.get("role") == "tool" and isinstance(msg.get("content"), dict):
            msg["content"] = json.dumps(msg["content"])
    return messages


def save_stats(
    cfg: AppConfig,
    examples: list[dict[str, Any]],
    output_dir: str,
) -> None:
    stats = calculate_stats(examples)
    write_json(os.path.join(output_dir, "stats.json"), stats)
    print_stats_tables(stats)
    if not cfg.datasets.sft.no_plots:
        generate_plots(stats, output_dir)


def save_config_snapshot(cfg: AppConfig, output_dir: str) -> None:
    payload = plain_config_dict(cfg, resolve=False)
    write_json(os.path.join(output_dir, "config_snapshot.json"), payload)


def calculate_stats(examples: list[dict[str, Any]]) -> dict[str, Any]:
    if not examples:
        return {}
    return (
        count_messages_and_tools(examples)
        | token_and_turn_stats(examples)
        | {"metadata_histograms": selected_metadata_histograms(_example_metadata(examples))}
    )


def _example_metadata(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metadata_values: list[dict[str, Any]] = []
    for example in examples:
        try:
            metadata = json.loads(example.get("metadata", "{}"))
        except json.JSONDecodeError:
            metadata = {}
        if isinstance(metadata, dict):
            metadata_values.append(metadata)
    return metadata_values


def count_messages_and_tools(examples: list[dict[str, Any]]) -> dict[str, Any]:
    total_messages = 0
    total_toolcalls = 0
    total_tokens = 0
    scenarios: dict[str, int] = {}
    toolcall_types: dict[str, int] = {}

    for example in examples:
        messages = example["messages"]
        total_messages += len(messages)
        total_tokens += example.get("num_tokens", 0)

        scenario = example.get("scenario", "unknown")
        scenarios[scenario] = scenarios.get(scenario, 0) + 1

        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            for tool_call in msg.get("tool_calls", []):
                total_toolcalls += 1
                name = tool_call.get("function", {}).get("name", "unknown")
                toolcall_types[name] = toolcall_types.get(name, 0) + 1

    num_runs = len(examples)
    return {
        "num_runs": num_runs,
        "total_messages": total_messages,
        "avg_messages_per_run": total_messages / num_runs if num_runs else 0,
        "total_toolcalls": total_toolcalls,
        "avg_toolcalls_per_run": total_toolcalls / num_runs if num_runs else 0,
        "total_tokens": total_tokens,
        "avg_tokens_per_run": total_tokens / num_runs if num_runs else 0,
        "scenarios": scenarios,
        "toolcall_types": toolcall_types,
    }


def token_and_turn_stats(examples: list[dict[str, Any]]) -> dict[str, Any]:
    token_thresholds = {"16k": 16 * 1024, "32k": 32 * 1024, "64k": 64 * 1024}
    all_turns = [ex.get("turns", 0) for ex in examples]
    all_tokens = [ex.get("num_tokens", 0) for ex in examples]
    scenario_long_ctx: dict[str, dict[str, int]] = {}
    scenario_turns: dict[str, list[int]] = {}

    for example in examples:
        scenario = example.get("scenario", "unknown")
        tokens = example.get("num_tokens", 0)
        turns = example.get("turns", 0)
        scenario_turns.setdefault(scenario, []).append(turns)
        scenario_long_ctx.setdefault(scenario, {"16k": 0, "32k": 0, "64k": 0})
        for label, threshold in token_thresholds.items():
            if tokens > threshold:
                scenario_long_ctx[scenario][label] += 1

    return {
        "traces_gt_16k": sum(1 for t in all_tokens if t > token_thresholds["16k"]),
        "traces_gt_32k": sum(1 for t in all_tokens if t > token_thresholds["32k"]),
        "traces_gt_64k": sum(1 for t in all_tokens if t > token_thresholds["64k"]),
        "scenario_long_ctx": scenario_long_ctx,
        "scenario_avg_turns": {
            key: sum(values) / len(values) if values else 0
            for key, values in scenario_turns.items()
        },
        "min_turns": min(all_turns) if all_turns else 0,
        "max_turns": max(all_turns) if all_turns else 0,
        "avg_turns": sum(all_turns) / len(all_turns) if all_turns else 0,
        "min_tokens": min(all_tokens) if all_tokens else 0,
        "max_tokens": max(all_tokens) if all_tokens else 0,
    }


def print_filter_stats(filtered_stats: dict[str, int]) -> None:
    if not filtered_stats:
        return

    aggregated: dict[str, int] = {}
    for reason, count in filtered_stats.items():
        category = reason.split(" (")[0]
        aggregated[category] = aggregated.get(category, 0) + count

    from rich import box

    table = Table(
        title="◆ Filter Stats",
        title_style="bold bright_white",
        box=box.SIMPLE_HEAVY,
        border_style="bright_red",
        show_header=True,
        header_style="bold bright_white",
    )
    table.add_column("Reason", style="bright_white")
    table.add_column("Count", justify="right", style="bright_white")
    for reason, count in sorted(aggregated.items(), key=lambda item: item[1], reverse=True):
        table.add_row(reason, str(count))
    console.print(table)
    console.print()


def print_stats_tables(stats: dict[str, Any]) -> None:
    if not stats:
        return
    console.print(summary_table(stats))
    console.print(scenario_table(stats))


def summary_table(stats: dict[str, Any]) -> Table:
    from rich import box

    table = Table(
        title="◆ Global Stats",
        title_style="bold bright_white",
        box=box.SIMPLE_HEAVY,
        border_style="bright_cyan",
        show_header=True,
        header_style="bold bright_white",
    )
    table.add_column("Metric", style="bright_white")
    table.add_column("Value", style="bright_white")
    table.add_row("Total Runs", str(stats["num_runs"]))
    table.add_row("Total Tokens", f"{stats['total_tokens']:,}")
    table.add_row("Avg Tokens/Run", f"{stats['avg_tokens_per_run']:.0f}")
    table.add_row("Avg Turns/Run", f"{stats['avg_turns']:.1f}")
    table.add_row("Max Tokens", f"{stats['max_tokens']}")
    table.add_row(
        "Traces > 16k",
        f"[bright_yellow]{stats['traces_gt_16k']}[/] / {stats['num_runs']} ({stats['traces_gt_16k'] / stats['num_runs']:.0%})",
    )
    if stats["traces_gt_32k"] > 0:
        table.add_row(
            "Traces > 32k",
            f"[bright_red]{stats['traces_gt_32k']}[/] / {stats['num_runs']}",
        )
    return table


def scenario_table(stats: dict[str, Any]) -> Table:
    from rich import box

    table = Table(
        title="◆ Scenarios",
        title_style="bold bright_white",
        box=box.SIMPLE_HEAVY,
        border_style="bright_magenta",
        show_header=True,
        header_style="bold bright_white",
    )
    table.add_column("Scenario", style="bright_white")
    table.add_column("Runs", justify="right", style="bright_white")
    table.add_column(">16k", justify="right", style="bright_yellow")
    table.add_column("Turns", justify="right", style="bright_cyan")

    scenarios = stats.get("scenarios", {})
    long_ctx = stats.get("scenario_long_ctx", {})
    avg_turns = stats.get("scenario_avg_turns", {})
    for name, count in sorted(scenarios.items(), key=lambda item: item[1], reverse=True):
        gt16 = long_ctx.get(name, {}).get("16k", 0)
        gt16_str = str(gt16) if gt16 > 0 else "-"
        if gt16 > count * 0.5:
            gt16_str = f"[bright_yellow]{gt16_str}[/]"
        table.add_row(name, str(count), gt16_str, f"{avg_turns.get(name, 0):.1f}")
    return table


def generate_plots(stats: dict[str, Any], output_dir: str) -> None:
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    scenarios = sorted(stats.get("scenarios", {}).items())
    if scenarios:
        names, counts = zip(*scenarios)
        plt.figure(figsize=(12, 6))
        plt.bar(names, counts)
        plt.xticks(rotation=45, ha="right")
        plt.title("Number of Runs per Scenario")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "runs_per_scenario.png"))
        plt.close()
    console.print(f"[dim]Plots saved to {plots_dir}[/]", style="dim")


def group_by_scenario(examples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for example in examples:
        scenario = example.get("scenario", "unknown")
        groups.setdefault(scenario, []).append(example)
    return groups


def write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
