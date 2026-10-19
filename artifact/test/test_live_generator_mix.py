from collections import Counter
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import yaml

from src.config import SFTQualityFilterConfig, _DEFAULT_RL_GENERATORS
from src.generators import GENERATOR_REGISTRY
from src.generators.profile_audit import generator_profile_diff_rows
from src.gym.prompts import render_system_prompt_from_template
from src.scenarios.procedural import ProceduralScenarioSource


EXPECTED_STANDARD_GENERATORS = [
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

EXPECTED_TRAINING_PROFILE_DEFAULTS = [
    "training/lowpriv_usernames",
    "training/capabilities_gtfobins",
    "training/suid_gtfobins",
    "training/sudo_gtfobins",
    "training/cron_wildcard",
    "training/cron_writable_script",
    "training/password_file",
    "training/password_history",
    "training/password_reuse",
    "training/ssh_key_reuse",
    "training/weak_password",
    "_self_",
]

EXPECTED_HOLDOUT_PROFILE_DEFAULTS = [
    "holdout/lowpriv_usernames",
    "holdout/capabilities_gtfobins",
    "holdout/suid_gtfobins",
    "holdout/sudo_gtfobins",
    "holdout/cron_wildcard",
    "holdout/cron_writable_script",
    "holdout/password_file",
    "holdout/password_history",
    "holdout/password_reuse",
    "holdout/ssh_key_reuse",
    "holdout/weak_password",
    "_self_",
]

_CONFIG_GENERATOR_PATHS = {
    "conf/rl/prime_rl/default.yaml": ("source", "generators"),
    "conf/experiment/eval/paper_procedural.yaml": (
        "runner",
        "source",
        "generators",
    ),
    "conf/experiment/train/paper_prime_rl.yaml": (
        "rl",
        "prime_rl",
        "source",
        "generators",
    ),
    "conf/runner/eval_procedural.yaml": ("source", "generators"),
    "conf/runner/paper_procedural.yaml": ("source", "generators"),
    "conf/runner/trace_collection/standard_training.yaml": ("source", "generators"),
    "conf/runner/trace_collection/standard_validation.yaml": ("source", "generators"),
}


def _load_yaml(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _select(data: dict, key_path: tuple[str, ...]):
    value = data
    for key in key_path:
        value = value[key]
    return value


def _compose(overrides: list[str]):
    with initialize_config_dir(
        config_dir=str(Path("conf").resolve()), version_base=None
    ):
        return compose(config_name="config", overrides=overrides)


def test_default_rl_generator_list_matches_standard_mix():
    assert _DEFAULT_RL_GENERATORS == EXPECTED_STANDARD_GENERATORS


def test_generator_registry_matches_active_standard_mix():
    assert list(GENERATOR_REGISTRY) == EXPECTED_STANDARD_GENERATORS
    assert "credential_artifact" not in GENERATOR_REGISTRY
    assert "sudo_wildcard" not in GENERATOR_REGISTRY


def _profile_config(profile: str) -> dict:
    cfg = _compose([f"generators={profile}"])
    data = OmegaConf.to_container(cfg.generators, resolve=True)
    assert isinstance(data, dict)
    return data


def _assert_generator_profile(cfg, profile: str) -> None:
    data = OmegaConf.to_container(cfg.generators, resolve=True)
    assert isinstance(data, dict)
    expected = _profile_config(profile)

    for key in (
        "capabilities_allowlist",
        "suid_allowlist",
        "sudo_allowlist",
        "cron_backup_dirs",
        "password_file",
        "password_history",
        "ssh_key_configs",
        "weak_password",
    ):
        assert data[key] == expected[key], key


def test_default_generator_profile_is_training():
    defaults = _load_yaml("conf/generators/default.yaml")["defaults"]
    training_defaults = _load_yaml("conf/generators/training.yaml")["defaults"]
    holdout_defaults = _load_yaml("conf/generators/holdout.yaml")["defaults"]

    assert defaults == ["training", "_self_"]
    assert training_defaults == EXPECTED_TRAINING_PROFILE_DEFAULTS
    assert holdout_defaults == EXPECTED_HOLDOUT_PROFILE_DEFAULTS


def test_holdout_generator_profile_values_are_disjoint_from_training():
    training = _profile_config("training")
    holdout = _profile_config("holdout")
    rows = generator_profile_diff_rows(training, holdout)

    for row in rows:
        assert row["training_count"], row["category"]
        assert row["validation_count"], row["category"]
        assert row["overlap_count"] == 0, row["category"]


def test_password_file_profiles_cover_each_template_kind_and_placement():
    for profile in ("training", "holdout"):
        config = _profile_config(profile)["password_file"]
        template_kinds = {template["kind"] for template in config["file_templates"]}
        placement_kinds = set(config["placement_locations"])

        assert template_kinds == {"mail", "config", "note"}, profile
        assert placement_kinds == {"mail", "config", "note"}, profile
        for kind, locations in config["placement_locations"].items():
            assert locations, f"{profile} {kind}"


def test_sudoers_file_names_are_varied_in_each_profile():
    for profile in ("training", "holdout"):
        names = _profile_config(profile)["sudo_layout"]["sudoers_file_names"]
        assert len(names) > 1, profile


def test_sudoers_file_names_compose_into_structured_generator_config():
    for profile in ("training", "holdout"):
        cfg = _compose([f"generators={profile}"])
        assert len(cfg.generators.sudo_layout.sudoers_file_names) > 1


def test_cron_job_names_compose_into_structured_generator_config():
    for profile in ("training", "holdout"):
        cfg = _compose([f"generators={profile}"])
        assert cfg.generators.cron_wildcard.job_directory == "/etc/cron.d"
        assert cfg.generators.cron_wildcard.job_names
        assert cfg.generators.cron_writable_script.job_directory == "/etc/cron.d"
        assert cfg.generators.cron_writable_script.job_names


def test_live_generator_lists_match_across_active_configs():
    for path, key_path in _CONFIG_GENERATOR_PATHS.items():
        assert _select(_load_yaml(path), key_path) == EXPECTED_STANDARD_GENERATORS, path


def test_default_trace_collection_runners_alias_standard():
    training_defaults = _load_yaml("conf/runner/trace_collection_training.yaml")[
        "defaults"
    ]
    validation_defaults = _load_yaml("conf/runner/trace_collection_validation.yaml")[
        "defaults"
    ]

    assert training_defaults == ["trace_collection/standard_training", "_self_"]
    assert validation_defaults == ["trace_collection/standard_validation", "_self_"]


def test_standard_trace_budgets_and_output_hierarchy():
    training = _load_yaml("conf/runner/trace_collection/standard_training.yaml")
    validation = _load_yaml("conf/runner/trace_collection/standard_validation.yaml")
    composed_validation = _compose(["runner=trace_collection/standard_validation"])

    assert _select(training, ("source", "generators")) == EXPECTED_STANDARD_GENERATORS
    assert (
        list(composed_validation.runner.source.generators)
        == EXPECTED_STANDARD_GENERATORS
    )
    assert (
        training["output_dir"]
        == "outputs/traces/trace_collection/${datasets.sft.profile}/${datasets.sft.regime}/${datasets.sft.teacher}/${datasets.sft.reasoning_variant}/training"
    )
    assert (
        validation["output_dir"]
        == "outputs/traces/trace_collection/${datasets.sft.profile}/${datasets.sft.regime}/${datasets.sft.teacher}/${datasets.sft.reasoning_variant}/validation"
    )
    assert training["runs_per_item"] * len(EXPECTED_STANDARD_GENERATORS) == 2000
    assert (
        composed_validation.runner.runs_per_item * len(EXPECTED_STANDARD_GENERATORS)
        == 200
    )


def test_standard_sft_collection_configs_use_axis_hierarchy():
    training = _compose(["datasets/sft=collection/standard/training"])
    validation = _compose(["datasets/sft=collection/standard/validation"])

    assert training.datasets.sft.profile == "standard"
    assert training.datasets.sft.regime == "unguided"
    assert training.datasets.sft.teacher == "deepseek"
    assert training.datasets.sft.reasoning_variant == "long_reasoning"
    assert (
        training.datasets.sft.trace_root
        == "outputs/traces/trace_collection/standard/unguided/deepseek/long_reasoning"
    )
    assert validation.datasets.sft.source_dir == "validation"
    assert validation.datasets.sft.exclude_source_dir == "training"
    assert (
        str(training.datasets.sft.output_dir)
        == "outputs/data/privesc_sft/standard/unguided/deepseek/long_reasoning/training"
    )
    assert (
        str(training.datasets.sft.validation_output_dir)
        == "outputs/data/privesc_sft/standard/unguided/deepseek/long_reasoning/validation"
    )
    assert (
        str(validation.datasets.sft.output_dir)
        == "outputs/data/privesc_sft/standard/unguided/deepseek/long_reasoning/validation"
    )


def test_planned_experiment_axes_compose_independently():
    cfg = _compose(
        [
            "datasets/sft/regime=unguided",
            "datasets/sft/teacher=deepseek",
            "generators/decoys=on",
        ]
    )

    assert cfg.datasets.sft.regime == "unguided"
    assert cfg.datasets.sft.teacher == "deepseek"
    assert cfg.datasets.sft.teacher_model == "deepseek/deepseek-v4-flash"
    assert cfg.generators.decoys_enabled is True


def test_sft_quality_configs_are_regime_specific():
    quality_files = {
        path.relative_to("conf/datasets/sft/quality").as_posix()
        for path in Path("conf/datasets/sft/quality").rglob("*.yaml")
    }
    guided = _compose(["datasets/sft/quality=guided"])
    unguided = _compose(["datasets/sft/quality=unguided"])

    assert quality_files == {"shared.yaml", "guided.yaml", "unguided.yaml"}
    assert guided.datasets.sft.quality.max_turns == 20
    assert guided.datasets.sft.quality.live_turn_max_attempts == 3
    assert guided.datasets.sft.quality.reject_on_secret_solution_leakage is True
    assert guided.datasets.sft.quality.verify_solution_in_prompt is True
    assert (
        "secret solution"
        in guided.datasets.sft.quality.secret_solution_leakage_keywords
    )
    assert "hint in the secret data" in (
        guided.datasets.sft.quality.secret_solution_leakage_keywords
    )
    assert guided.datasets.sft.quality.secret_solution_leakage_keywords == (
        SFTQualityFilterConfig().secret_solution_leakage_keywords
    )
    assert unguided.datasets.sft.quality.max_turns == 20
    assert unguided.datasets.sft.quality.live_turn_max_attempts == 3
    assert unguided.datasets.sft.quality.reject_on_secret_solution_leakage is False
    assert unguided.datasets.sft.quality.verify_solution_in_prompt is False
    assert unguided.datasets.sft.quality.secret_solution_leakage_keywords == []


def test_reasoning_variant_derivation_experiment_configs_are_reproducible():
    short_cfg = _compose(
        ["+experiment=derive/standard_unguided_deepseek_short_reasoning_validation"]
    )
    assert short_cfg.reasoning_variant_derivation.variant == "short_reasoning"
    assert short_cfg.reasoning_variant_derivation.source_variant == "long_reasoning"
    assert short_cfg.datasets.sft.reasoning_variant == "short_reasoning"
    assert short_cfg.datasets.sft.regime == "unguided"
    assert short_cfg.datasets.sft.teacher == "deepseek"
    assert short_cfg.datasets.sft.teacher_model == "deepseek/deepseek-v4-flash"
    assert short_cfg.datasets.sft.source_dir == "validation"
    assert short_cfg.datasets.sft.exclude_source_dir == "training"
    assert (
        short_cfg.datasets.sft.trace_root
        == "outputs/traces/trace_collection/standard/unguided/deepseek/short_reasoning"
    )
    assert short_cfg.agent.model == short_cfg.datasets.sft.teacher_model

    no_reasoning_cfg = _compose(
        ["+experiment=derive/standard_guided_deepseek_no_reasoning_training"]
    )
    assert no_reasoning_cfg.reasoning_variant_derivation.variant == "no_reasoning"
    assert no_reasoning_cfg.datasets.sft.reasoning_variant == "no_reasoning"
    assert no_reasoning_cfg.datasets.sft.regime == "guided"
    assert no_reasoning_cfg.datasets.sft.source_dir == "training"
    assert no_reasoning_cfg.datasets.sft.exclude_source_dir is None


def test_trace_collection_prompt_configs_are_guided_and_unguided():
    prompt_files = {
        path.name for path in Path("conf/prompts").glob("trace_collection*.yaml")
    }

    assert prompt_files == {
        "trace_collection_guided.yaml",
        "trace_collection_unguided.yaml",
    }


def test_decoy_switch_has_realistic_experiment_defaults():
    cfg = _compose(["generators/decoys=on"])

    assert cfg.generators.decoys_enabled is True
    assert cfg.generators.suid_layout.decoy_count_min == 10
    assert cfg.generators.suid_layout.decoy_count_max >= (
        cfg.generators.suid_layout.decoy_count_min
    )
    assert cfg.generators.suid_layout.decoy_count_max <= 12
    assert set(cfg.generators.suid_layout.decoy_path_templates) == {
        "/opt/{service}/libexec/{binary}",
        "/var/lib/{service}/bin/{binary}",
    }
    assert cfg.generators.sudo_layout.decoy_count_min == 1
    assert cfg.generators.sudo_layout.decoy_count_max == 2
    assert set(cfg.generators.sudo_layout.decoy_path_templates) == {
        "/usr/local/bin/{name}",
        "/usr/local/sbin/{name}",
    }
    assert "/opt" not in cfg.generators.password_file.placement_locations.config
    assert "/srv" not in cfg.generators.password_file.placement_locations.config
    assert "/etc/app" in cfg.generators.password_file.placement_locations.config
    assert cfg.generators.password_file.placement_locations.mail == [
        "/var/spool/mail",
        "{user_home}/Mail",
        "{user_home}/mail",
        "{user_home}/.local/share/mail",
    ]


def test_trace_experiments_encode_regime_teacher_split_and_generator_profile():
    expected = {
        "standard_guided_deepseek_training": (
            "guided",
            "deepseek",
            "deepseek/deepseek-v4-flash",
            "training",
            "training",
        ),
        "standard_guided_deepseek_validation": (
            "guided",
            "deepseek",
            "deepseek/deepseek-v4-flash",
            "validation",
            "holdout",
        ),
        "standard_unguided_deepseek_training": (
            "unguided",
            "deepseek",
            "deepseek/deepseek-v4-flash",
            "training",
            "training",
        ),
        "standard_unguided_deepseek_validation": (
            "unguided",
            "deepseek",
            "deepseek/deepseek-v4-flash",
            "validation",
            "holdout",
        ),
    }

    for experiment, (
        regime,
        teacher,
        teacher_model,
        split,
        generators_profile,
    ) in expected.items():
        cfg = _compose([f"+experiment=trace/{experiment}"])

        assert cfg.datasets.sft.profile == "standard"
        assert cfg.datasets.sft.regime == regime
        assert cfg.datasets.sft.teacher == teacher
        assert cfg.datasets.sft.teacher_model == teacher_model
        if teacher == "deepseek":
            assert cfg.agent.api_base == "https://openrouter.ai/api/v1"
            assert cfg.agent.model == "deepseek/deepseek-v4-flash"
        assert cfg.datasets.sft.quality.verify_solution_in_prompt is (
            regime == "guided"
        )
        assert cfg.datasets.sft.quality.reject_on_secret_solution_leakage is (
            regime == "guided"
        )
        assert cfg.runner.source.generators == EXPECTED_STANDARD_GENERATORS
        _assert_generator_profile(cfg, generators_profile)
        assert cfg.generators.decoys_enabled is False
        assert cfg.datasets.sft.reasoning_variant == "long_reasoning"
        assert cfg.runner.output_dir.endswith(
            f"/standard/{regime}/{teacher}/long_reasoning/{split}"
        )
        assert cfg.runner.runs_per_item == (200 if split == "training" else 20)
        assert cfg.datasets.sft.max_per_generator == (
            200 if split == "training" else 20
        )


def test_paper_procedural_eval_uses_holdout_generator_profile():
    cfg = _compose(["+experiment=eval/paper_procedural"])

    assert cfg.runner.source.generators == EXPECTED_STANDARD_GENERATORS
    _assert_generator_profile(cfg, "holdout")


def test_deepseek_reasoning_params_are_regime_specific():
    guided = _compose(["+experiment=trace/standard_guided_deepseek_training"])
    unguided = _compose(["+experiment=trace/standard_unguided_deepseek_training"])

    guided_params = OmegaConf.to_container(guided.agent.params, resolve=True)
    unguided_params = OmegaConf.to_container(unguided.agent.params, resolve=True)

    provider = {"only": ["deepseek"], "allow_fallbacks": False}
    assert guided_params == {
        "extra_body": {
            "thinking": {"type": "disabled"},
            "provider": provider,
        }
    }
    assert unguided_params == {
        "reasoning_effort": "high",
        "extra_body": {
            "thinking": {"type": "enabled"},
            "provider": provider,
        },
    }


def test_trace_experiments_can_enable_decoys_as_a_switch():
    cfg = _compose(
        [
            "+experiment=trace/standard_guided_deepseek_training",
            "generators/decoys=on",
        ]
    )

    assert cfg.datasets.sft.profile == "standard"
    assert cfg.datasets.sft.regime == "guided"
    assert cfg.datasets.sft.teacher == "deepseek"
    assert cfg.generators.decoys_enabled is True


def test_unguided_trace_prompt_matches_evaluation_prompt():
    evaluation = _compose(["prompts=evaluation"])
    unguided = _compose(["+experiment=trace/standard_unguided_deepseek_training"])

    assert unguided.prompts.system_template == "trace_collection.jinja"
    assert unguided.prompts.template_vars["include_solution_guidance"] is False
    assert unguided.prompts.start_instruction == evaluation.prompts.start_instruction
    assert (
        unguided.prompts.no_tool_calls_nudge == evaluation.prompts.no_tool_calls_nudge
    )
    template_vars = {
        "user": "lowpriv",
        "password": "trustno1",
        "max_turns": 15,
        "term_cols": 80,
        "term_rows": 24,
        "include_solution_guidance": False,
        "solution": {"description": "hidden"},
    }
    trace_prompt = render_system_prompt_from_template(
        "trace_collection.jinja",
        template_vars,
    )
    eval_prompt = render_system_prompt_from_template(
        "privilege_escalation.jinja",
        template_vars,
    )
    assert trace_prompt == eval_prompt
    assert "SECRET SOLUTION DATA" not in trace_prompt


def test_guided_trace_prompt_uses_hidden_solution_template():
    guided = _compose(["+experiment=trace/standard_guided_deepseek_training"])

    assert guided.prompts.system_template == "trace_collection.jinja"
    assert guided.prompts.template_vars["include_solution_guidance"] is True
    prompt = render_system_prompt_from_template(
        "trace_collection.jinja",
        {
            "user": "lowpriv",
            "password": "trustno1",
            "max_turns": 15,
            "term_cols": 80,
            "term_rows": 24,
            "include_solution_guidance": True,
            "solution": {"description": "hidden"},
        },
    )
    assert "SECRET SOLUTION DATA" in prompt


def test_live_mix_generates_one_round_without_generation_failures():
    source = ProceduralScenarioSource(
        generators=EXPECTED_STANDARD_GENERATORS,
        base_seed=42,
        generator_configs={"decoys_enabled": False},
    )
    counts = Counter()

    for run_index in range(len(EXPECTED_STANDARD_GENERATORS)):
        instance = source.build(run_index)
        solution = instance.solution
        counts[instance.id] += 1

        assert instance.id in EXPECTED_STANDARD_GENERATORS
        assert instance.config.setup_script
        assert isinstance(solution, dict)
        assert set(solution) == {
            "description",
            "vulnerability",
            "exploit_tool_calls",
            "hint",
            "scenario",
        }
        assert solution["exploit_tool_calls"]
        assert instance.metadata["generator_name"] == instance.id

    assert counts == Counter({name: 1 for name in EXPECTED_STANDARD_GENERATORS}), counts
