"""Tests for runner schedule building logic."""

import json
import os
from pathlib import Path
import tempfile
from typing import cast

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.config import (
    AppConfig,
    RunnerConfig,
    SourceConfig,
    ScenarioConfig,
    AgentConfig,
    PromptsConfig,
    SSHConfig,
    PrivEscRewardConfig,
)
from src.runner.schedule import build_schedule, count_existing, update_cfg_with_instance
from src.scenarios.procedural import ProceduralScenarioSource
from src.scenarios.static import StaticScenarioSource
from src.scenarios.types import ScenarioInstance


@pytest.fixture
def tmpdir():
    """Provide a temp directory with traces subdir structure."""
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "traces", "model"))
        yield d


def cfg(
    tmpdir,
    source_type="static",
    mode="trace_collection",
    scenarios=None,
    generators=None,
    runs_per_item=5,
    max_runs=None,
):
    """Create minimal test config."""
    return OmegaConf.structured(
        AppConfig(
            runner=RunnerConfig(
                mode=mode,
                runs_per_item=runs_per_item,
                max_runs=max_runs,
                output_dir=tmpdir,
                source=SourceConfig(
                    type=source_type,
                    scenarios=scenarios or [],
                    generators=generators or [],
                ),
            ),
            scenario=ScenarioConfig(name="test_scenario", image="test:latest"),
            agent=AgentConfig(model="test/model", max_turns=12),
            prompts=PromptsConfig(
                system_template="privilege_escalation.jinja",
                start_instruction="Start privilege escalation now.",
                no_tool_calls_nudge="No tool calls received.",
                template_vars={
                    "user": "user",
                    "password": "password",
                    "max_turns": 12,
                    "term_cols": 120,
                    "term_rows": 40,
                },
            ),
            ssh=SSHConfig(),
        )
    )


def write_trace(tmpdir, filename):
    """Write a current-format successful trace file."""
    path = os.path.join(tmpdir, "traces", "model", filename)
    payload = {
        "status": "completed",
        "success": True,
        "turns": 1,
        "error": None,
        "metadata": {
            "benchmark_eligible": True,
            "prompt_vars": {"max_turns": 60},
        },
    }
    with open(path, "w") as f:
        json.dump(payload, f)


def _valid_trace_history() -> list[dict]:
    return [
        {
            "role": "system",
            "content": "SECRET SOLUTION DATA\n{}",
        },
        {
            "role": "user",
            "content": "Start privilege escalation now.",
        },
        {
            "role": "assistant",
            "content": (
                "I have enough observed context to follow the discovered "
                "privilege escalation path and verify root access cleanly."
            ),
            "tool_calls": [],
        },
    ]


def write_trace_with_metadata(
    tmpdir,
    filename: str,
    *,
    generator_name: str,
    seed: int,
    item_run_ordinal: int = 0,
    mode: str = "trace_collection",
    success: bool = True,
    turns: int = 2,
    benchmark_eligible: bool = True,
    pre_repair_rejected: bool = False,
    metadata_overrides: dict | None = None,
) -> None:
    path = os.path.join(tmpdir, "traces", "model", filename)
    metadata = {
        "benchmark_eligible": benchmark_eligible,
        "prompt_vars": {"max_turns": 60},
        "generator_name": generator_name,
        "seed": seed,
        "item_run_ordinal": item_run_ordinal,
    }
    if metadata_overrides:
        metadata.update(metadata_overrides)
    if pre_repair_rejected:
        metadata["pre_repair_rejected"] = True
    payload: dict[str, object] = {
        "mode": mode,
        "status": "completed",
        "success": success,
        "turns": turns,
        "error": None,
        "metadata": metadata,
    }
    if success:
        payload["history"] = _valid_trace_history()
    with open(path, "w") as f:
        json.dump(payload, f)


def make_static_source(scenarios):
    return StaticScenarioSource(scenarios=scenarios)


def make_procedural_source(generators):
    return ProceduralScenarioSource(generators=generators)


def test_counts_matching_files(tmpdir):
    write_trace(tmpdir, "my_scenario_123_20250101_120000.json")
    write_trace(tmpdir, "my_scenario_456_20250101_120001.json")
    write_trace(tmpdir, "other_scenario_789_20250101_120002.json")

    output_dir = os.path.join(tmpdir, "traces", "model")
    assert count_existing(output_dir, "my_scenario_") == 2
    assert count_existing(output_dir, "other_scenario_") == 1


def test_missing_dir_returns_zero():
    assert count_existing("/nonexistent/path", "my_scenario_") == 0


def test_count_existing_skips_error_traces(tmpdir):
    path = os.path.join(tmpdir, "traces", "model", "a_1_20250101_120000.json")
    payload = {
        "scenario": "a",
        "status": "error",
        "success": False,
        "turns": 0,
        "error": "Process exited with non-zero exit status 125",
        "metadata": {
            "benchmark_eligible": False,
            "prompt_vars": {"max_turns": 60},
        },
    }
    with open(path, "w") as f:
        json.dump(payload, f)

    output_dir = os.path.join(tmpdir, "traces", "model")
    assert count_existing(output_dir, "a_") == 0


def test_count_existing_skips_completed_failures_before_max_turns(tmpdir):
    path = os.path.join(tmpdir, "traces", "model", "a_1_20250101_120000.json")
    payload = {
        "scenario": "a",
        "status": "completed",
        "success": False,
        "turns": 12,
        "error": None,
        "metadata": {"benchmark_eligible": True, "prompt_vars": {"max_turns": 60}},
    }
    with open(path, "w") as f:
        json.dump(payload, f)

    output_dir = os.path.join(tmpdir, "traces", "model")
    assert count_existing(output_dir, "a_") == 0


def test_single_scenario_equal_distribution(tmpdir):
    """Single scenario gets all runs."""
    c = cfg(tmpdir, scenarios=["scen_a"], runs_per_item=5)
    schedule, desc = build_schedule(c, make_static_source(["scen_a"]))

    # 5 runs / 1 scenario = 5 runs each, indices: 0, 1, 2, 3, 4
    assert schedule == [0, 1, 2, 3, 4]
    assert "Static" in desc
    assert "1 scenario" in desc
    assert "5 runs" in desc


def test_multi_scenario_equal_distribution(tmpdir):
    """Multiple scenarios split runs equally."""
    c = cfg(tmpdir, scenarios=["a", "b", "c"], runs_per_item=3)
    schedule, desc = build_schedule(c, make_static_source(["a", "b", "c"]))

    # 3 runs each -> 9 total
    # Indices: a gets 0,3,6; b gets 1,4,7; c gets 2,5,8
    assert len(schedule) == 9
    assert "3 scenarios" in desc


def test_skips_scenario_with_existing_traces(tmpdir):
    """Scenarios with existing traces are skipped up to runs_per_item."""
    write_trace(tmpdir, "a_42_20250101_120000.json")
    write_trace(tmpdir, "a_43_20250101_120001.json")

    c = cfg(tmpdir, scenarios=["a", "b"], runs_per_item=3)
    schedule, desc = build_schedule(c, make_static_source(["a", "b"]))

    # 3 runs each
    # 'a' has 2 existing, needs 1 more
    # 'b' has 0 existing, needs 3
    # Total: 4 runs
    assert len(schedule) == 4
    assert "4 runs" in desc


def test_round_robin_order_with_skips(tmpdir):
    """Round robin should interleave items while honoring skips."""
    write_trace(tmpdir, "a_0_20250101_120000.json")
    write_trace(tmpdir, "a_1_20250101_120001.json")

    c = cfg(tmpdir, scenarios=["a", "b"], runs_per_item=3)
    schedule, _ = build_schedule(c, make_static_source(["a", "b"]))

    # a has 2 existing -> needs 1 (next run index 4)
    # b has 0 existing -> needs 3 (indices 1,3,5)
    assert schedule == [4, 1, 3, 5]


def test_empty_when_all_complete_static(tmpdir):
    """Returns empty schedule when all scenarios have enough traces."""
    for i in range(3):
        write_trace(tmpdir, f"a_{i}_20250101_12000{i}.json")
        write_trace(tmpdir, f"b_{i}_20250101_12000{i}.json")

    c = cfg(tmpdir, scenarios=["a", "b"], runs_per_item=3)
    schedule, desc = build_schedule(c, make_static_source(["a", "b"]))

    # 3 each, both have 3 existing
    assert schedule == []
    assert "3 runs" in desc


def test_caps_total_runs(tmpdir):
    """Caps total runs when max_runs is set."""
    c = cfg(tmpdir, scenarios=["a", "b", "c"], runs_per_item=3, max_runs=5)
    schedule, desc = build_schedule(c, make_static_source(["a", "b", "c"]))

    assert len(schedule) == 5
    assert "cap 5" in desc


def test_equal_distribution_procedural(tmpdir):
    """Generators split runs equally."""
    generators = ["sudo_gtfobins", "suid_gtfobins"]
    c = cfg(tmpdir, source_type="procedural", generators=generators, runs_per_item=5)
    schedule, desc = build_schedule(c, make_procedural_source(generators))

    # 5 runs each
    assert len(schedule) == 10
    assert "Procedural" in desc
    assert "2 generators" in desc


def test_skips_generator_with_existing_traces(tmpdir):
    """Generators with existing traces are skipped."""
    write_trace_with_metadata(
        tmpdir,
        "sudo_gtfobins_42_20250101_120000.json",
        generator_name="sudo_gtfobins",
        seed=42,
    )
    write_trace_with_metadata(
        tmpdir,
        "sudo_gtfobins_43_20250101_120001.json",
        generator_name="sudo_gtfobins",
        seed=43,
        item_run_ordinal=1,
    )

    generators = ["sudo_gtfobins", "suid_gtfobins"]
    c = cfg(tmpdir, source_type="procedural", generators=generators, runs_per_item=3)
    schedule, _ = build_schedule(c, make_procedural_source(generators))

    # 3 each
    # sudo has 2 existing, needs 1
    # suid has 0 existing, needs 3
    # Total: 4 runs
    assert len(schedule) == 4


def test_procedural_resume_skips_duplicate_existing_seed_keys(tmpdir):
    generators = ["sudo_gtfobins", "suid_gtfobins"]

    write_trace_with_metadata(
        tmpdir,
        "sudo_gtfobins_a.json",
        generator_name="sudo_gtfobins",
        seed=42,
    )
    write_trace_with_metadata(
        tmpdir,
        "sudo_gtfobins_b.json",
        generator_name="sudo_gtfobins",
        seed=42,
    )
    write_trace_with_metadata(
        tmpdir,
        "suid_gtfobins_a.json",
        generator_name="suid_gtfobins",
        seed=43,
    )

    c = cfg(tmpdir, source_type="procedural", generators=generators, runs_per_item=2)
    schedule, _ = build_schedule(c, make_procedural_source(generators))

    assert schedule == [2, 3]


def test_trace_collection_resume_skips_but_does_not_count_rejected_seed(tmpdir):
    generators = ["sudo_gtfobins", "suid_gtfobins"]

    write_trace_with_metadata(
        tmpdir,
        "sudo_gtfobins_rejected.json",
        generator_name="sudo_gtfobins",
        seed=42,
        success=False,
        turns=1,
        benchmark_eligible=False,
        pre_repair_rejected=True,
    )

    c = cfg(tmpdir, source_type="procedural", generators=generators, runs_per_item=1)
    schedule, _ = build_schedule(c, make_procedural_source(generators))

    assert schedule == [0, 1]


def test_trace_collection_resume_skips_but_does_not_count_failed_seed(tmpdir):
    generators = ["sudo_gtfobins", "suid_gtfobins"]

    write_trace_with_metadata(
        tmpdir,
        "sudo_gtfobins_failed.json",
        generator_name="sudo_gtfobins",
        seed=42,
        success=False,
        turns=60,
    )

    c = cfg(tmpdir, source_type="procedural", generators=generators, runs_per_item=1)
    schedule, _ = build_schedule(c, make_procedural_source(generators))

    assert schedule == [0, 1]


def test_trace_collection_resume_skips_but_does_not_count_audit_rejected_seed(tmpdir):
    generators = ["weak_password", "suid_gtfobins"]

    write_trace_with_metadata(
        tmpdir,
        "weak_password_root_root.json",
        generator_name="weak_password",
        seed=42,
        success=True,
        metadata_overrides={"root_password": "root"},
    )

    c = cfg(tmpdir, source_type="procedural", generators=generators, runs_per_item=1)
    schedule, _ = build_schedule(c, make_procedural_source(generators))

    assert schedule == [0, 1]


def test_trace_collection_resume_does_not_count_missing_history(tmpdir):
    generators = ["sudo_gtfobins", "suid_gtfobins"]

    write_trace_with_metadata(
        tmpdir,
        "sudo_gtfobins_missing_history.json",
        generator_name="sudo_gtfobins",
        seed=42,
        success=False,
        turns=12,
    )
    path = Path(tmpdir) / "traces" / "model" / "sudo_gtfobins_missing_history.json"
    payload = json.loads(path.read_text())
    payload["success"] = True
    payload.pop("history", None)
    path.write_text(json.dumps(payload))

    c = cfg(tmpdir, source_type="procedural", generators=generators, runs_per_item=1)
    schedule, _ = build_schedule(c, make_procedural_source(generators))

    assert schedule == [0, 1]


def test_procedural_seed_dedupe_disabled_outside_trace_collection(tmpdir):
    generators = ["sudo_gtfobins", "suid_gtfobins"]

    write_trace_with_metadata(
        tmpdir,
        "sudo_gtfobins_a.json",
        generator_name="sudo_gtfobins",
        seed=42,
    )
    write_trace_with_metadata(
        tmpdir,
        "sudo_gtfobins_b.json",
        generator_name="sudo_gtfobins",
        seed=42,
    )
    write_trace_with_metadata(
        tmpdir,
        "suid_gtfobins_a.json",
        generator_name="suid_gtfobins",
        seed=43,
    )

    c = cfg(
        tmpdir,
        source_type="procedural",
        mode="evaluation",
        generators=generators,
        runs_per_item=2,
    )
    schedule, _ = build_schedule(c, make_procedural_source(generators))

    assert schedule == [3]


def test_empty_when_all_complete_procedural(tmpdir):
    """Returns empty schedule when all generators have enough traces."""
    for gen in ["sudo_gtfobins", "suid_gtfobins"]:
        for i in range(3):
            write_trace_with_metadata(
                tmpdir,
                f"{gen}_{i}_20250101_12000{i}.json",
                generator_name=gen,
                seed=i,
                item_run_ordinal=i,
            )

    generators = ["sudo_gtfobins", "suid_gtfobins"]
    c = cfg(tmpdir, source_type="procedural", generators=generators, runs_per_item=3)
    schedule, desc = build_schedule(c, make_procedural_source(generators))

    assert schedule == []
    assert "3 runs" in desc


def test_caps_when_max_runs_less_than_items(tmpdir):
    """Caps schedule when max_runs < number of generators."""
    generators = [
        "capabilities_gtfobins",
        "suid_gtfobins",
        "sudo_gtfobins",
        "cron_wildcard",
        "cron_writable_script",
        "password_file",
        "password_history",
        "password_reuse",
        "ssh_key_reuse",
        "weak_password",
    ]
    c = cfg(
        tmpdir,
        source_type="procedural",
        generators=generators,
        runs_per_item=1,
        max_runs=3,
    )
    schedule, desc = build_schedule(c, make_procedural_source(generators))

    # cap 3 -> first 3 scheduled
    assert len(schedule) == 3
    assert "10 generators" in desc
    assert "3 runs" in desc


def test_indices_interleave_for_round_robin(tmpdir):
    """Schedule indices should work with run_index % num_items."""
    c = cfg(tmpdir, scenarios=["a", "b"], runs_per_item=3)
    source = StaticScenarioSource(scenarios=["a", "b"])
    schedule, _ = build_schedule(c, source)

    # With 2 scenarios and 3 runs each:
    # 'a' (idx=0) should get indices where idx % 2 == 0: 0, 2, 4
    # 'b' (idx=1) should get indices where idx % 2 == 1: 1, 3, 5
    a_indices = [i for i in schedule if i % 2 == 0]
    b_indices = [i for i in schedule if i % 2 == 1]

    assert len(a_indices) == 3
    assert len(b_indices) == 3


def test_resume_generates_correct_indices(tmpdir):
    """Resume should generate indices that continue the sequence."""
    write_trace(tmpdir, "a_0_20250101_120000.json")  # a has 1 existing

    c = cfg(tmpdir, scenarios=["a", "b"], runs_per_item=3)
    source = StaticScenarioSource(scenarios=["a", "b"])
    schedule, _ = build_schedule(c, source)

    # a needs 2 more (has 1 of 3), b needs 3
    # a's next indices: runs 1,2 -> 1*2+0=2, 2*2+0=4
    # b's indices: runs 0,1,2 -> 0*2+1=1, 1*2+1=3, 2*2+1=5
    a_indices = sorted([i for i in schedule if i % 2 == 0])
    b_indices = sorted([i for i in schedule if i % 2 == 1])

    assert a_indices == [2, 4]  # a continues from run 1
    assert b_indices == [1, 3, 5]  # b starts from run 0


def test_static_resume_refills_missing_ordinal_when_duplicate_exists(tmpdir):
    for ordinal in [0, 1, 3]:
        write_trace_with_metadata(
            tmpdir,
            f"a_{ordinal}.json",
            generator_name="a",
            seed=ordinal,
            item_run_ordinal=ordinal,
            mode="evaluation",
        )
    write_trace_with_metadata(
        tmpdir,
        "a_dup.json",
        generator_name="a",
        seed=3,
        item_run_ordinal=3,
        mode="evaluation",
    )

    c = cfg(tmpdir, scenarios=["a"], runs_per_item=4, mode="evaluation")
    schedule, _ = build_schedule(c, make_static_source(["a"]))

    assert schedule == [2]


def test_static_resume_preserves_legacy_counts_for_items_without_ordinals(tmpdir):
    write_trace(tmpdir, "a_legacy.json")
    write_trace_with_metadata(
        tmpdir,
        "b_0.json",
        generator_name="b",
        seed=0,
        item_run_ordinal=0,
        mode="evaluation",
    )

    c = cfg(tmpdir, scenarios=["a", "b"], runs_per_item=2, mode="evaluation")
    schedule, _ = build_schedule(c, make_static_source(["a", "b"]))

    assert schedule == [2, 3]


def test_static_resume_ignores_out_of_range_ordinals(tmpdir):
    write_trace_with_metadata(
        tmpdir,
        "a_bad_ordinal.json",
        generator_name="a",
        seed=99,
        item_run_ordinal=9,
        mode="evaluation",
    )

    c = cfg(tmpdir, scenarios=["a"], runs_per_item=2, mode="evaluation")
    schedule, _ = build_schedule(c, make_static_source(["a"]))

    assert schedule == [0, 1]


def test_static_resume_keeps_valid_ordinals_despite_invalid_extra_trace(tmpdir):
    write_trace_with_metadata(
        tmpdir,
        "a_0.json",
        generator_name="a",
        seed=0,
        item_run_ordinal=0,
        mode="evaluation",
    )
    write_trace_with_metadata(
        tmpdir,
        "a_bad_ordinal.json",
        generator_name="a",
        seed=99,
        item_run_ordinal=9,
        mode="evaluation",
    )

    c = cfg(tmpdir, scenarios=["a"], runs_per_item=2, mode="evaluation")
    schedule, _ = build_schedule(c, make_static_source(["a"]))

    assert schedule == [1]


def test_default_cfg_backend_does_not_clobber_instance_backend(tmpdir, monkeypatch):
    monkeypatch.delenv("PRIVESC_SCENARIO_BACKEND", raising=False)
    c = cfg(tmpdir, scenarios=["a"], runs_per_item=1)
    instance = ScenarioInstance(
        id="a",
        config=ScenarioConfig(name="a", backend="local_docker"),
    )

    run_cfg = update_cfg_with_instance(c, instance)

    assert run_cfg.scenario.backend == "local_docker"


def test_hydra_scenario_backend_override_applies_to_procedural_instance(
    monkeypatch,
) -> None:
    monkeypatch.delenv("PRIVESC_SCENARIO_BACKEND", raising=False)
    conf_dir = str((Path.cwd() / "conf").resolve())
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                "+experiment=trace/standard_guided_deepseek_training",
                "scenario.backend=local_docker",
            ],
        )

    source = ProceduralScenarioSource(
        generators=list(cfg.runner.source.generators),
        base_seed=int(cfg.runner.source.seed),
        random_seed=bool(cfg.runner.source.random_seed),
        randomize_env=bool(
            OmegaConf.select(cfg, "runner.source.randomize_env", default=True)
        ),
    )
    instance = source.build(0)

    run_cfg = update_cfg_with_instance(cast(AppConfig, cfg), instance)

    assert instance.config.backend == "remote_ssh"
    assert run_cfg.scenario.backend == "local_docker"


def test_env_backend_override_applies_to_run_config(tmpdir, monkeypatch):
    c = cfg(tmpdir, scenarios=["a"], runs_per_item=1)
    instance = ScenarioInstance(
        id="a",
        config=ScenarioConfig(name="a", backend="local_docker"),
    )

    monkeypatch.setenv("PRIVESC_SCENARIO_BACKEND", "remote_ssh")
    run_cfg = update_cfg_with_instance(c, instance)

    assert run_cfg.scenario.backend == "remote_ssh"


def test_update_cfg_with_instance_preserves_frozen_reward_config(tmpdir):
    c = cfg(tmpdir, scenarios=["a"], runs_per_item=1, mode="evaluation")
    c.reward = PrivEscRewardConfig(mode="outcome_cost")
    instance = ScenarioInstance(
        id="a",
        config=ScenarioConfig(name="a", backend="local_docker"),
    )

    run_cfg = update_cfg_with_instance(c, instance)

    assert run_cfg.reward.mode == "outcome_cost"
    assert run_cfg.scenario.name == "a"


def test_update_cfg_with_instance_accepts_full_procedural_generator_schema() -> None:
    conf_dir = str((Path.cwd() / "conf").resolve())
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(
            config_name="config",
            overrides=["runner=trace_collection/standard_training"],
        )

    source = ProceduralScenarioSource(
        generators=list(cfg.runner.source.generators),
        base_seed=int(cfg.runner.source.seed),
        random_seed=bool(cfg.runner.source.random_seed),
        randomize_env=bool(
            OmegaConf.select(cfg, "runner.source.randomize_env", default=True)
        ),
    )
    instance = source.build(0)

    run_cfg = update_cfg_with_instance(cast(AppConfig, cfg), instance)

    assert run_cfg.scenario.name == instance.id
    assert run_cfg.scenario.solution == instance.solution


def test_update_cfg_with_instance_resolves_interpolated_reward_config() -> None:
    conf_dir = str((Path.cwd() / "conf").resolve())
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(config_name="config", overrides=["+experiment=eval/paper_procedural"])

    source = ProceduralScenarioSource(
        generators=list(cfg.runner.source.generators),
        base_seed=int(cfg.runner.source.seed),
        random_seed=bool(cfg.runner.source.random_seed),
        randomize_env=bool(
            OmegaConf.select(cfg, "runner.source.randomize_env", default=True)
        ),
    )
    instance = source.build(0)

    run_cfg = update_cfg_with_instance(cast(AppConfig, cfg), instance)

    assert run_cfg.reward.h_max == cfg.rl.prime_rl.max_turns
