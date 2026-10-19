"""Tests for Prime-RL environment argument helpers."""

from typing import cast

from src.config import (
    AgentConfig,
    AppConfig,
    PrimeRLRLConfig,
    PrivEscRewardConfig,
    PromptsConfig,
    RLConfig,
    RewardConfigLike,
    ScenarioConfig,
    SSHConfig,
    SourceConfig,
)
from src.rl.prime_rl.env_args import build_train_env_args


_DEFAULT_SSH = SSHConfig(user="user", key_path="/tmp/key", servers="localhost:22")
_DEFAULT_AGENT = AgentConfig(
    api_key="k", api_base="https://example.com", model="m", max_turns=10
)
_DEFAULT_PROMPTS = PromptsConfig(
    system_template="{task}",
    start_instruction="Start",
    no_tool_calls_nudge="Use tools.",
    template_vars={"task": "x"},
)


def _make_cfg(reward: RewardConfigLike = None) -> AppConfig:
    return AppConfig(
        ssh=_DEFAULT_SSH,
        scenario=ScenarioConfig(),
        agent=_DEFAULT_AGENT,
        prompts=_DEFAULT_PROMPTS,
        reward=cast(PrivEscRewardConfig, reward or PrivEscRewardConfig()),
        rl=RLConfig(
            prime_rl=PrimeRLRLConfig(
                source=SourceConfig(type="procedural", generators=["password_file"]),
                output_dir="x",
            )
        ),
    )


def _expected_reward(mode: str = "outcome_round_cost", *, h_max: int = 20) -> dict:
    data = PrivEscRewardConfig(mode=mode).to_dict()
    data["h_max"] = h_max
    return data


def test_build_train_env_args_accepts_reward_mapping_from_hydra():
    cfg = _make_cfg(reward={"mode": "outcome"})

    env_args = build_train_env_args(cfg, num_examples=4)

    assert env_args["reward"] == _expected_reward("outcome")


def test_build_train_env_args_preserves_outcome_round_cost_parameters():
    cfg = _make_cfg(
        reward={
            "mode": "outcome_round_cost",
            "h_max": 99,
            "lambda_cost": 0.2,
            "c_ref_ms": 900.0,
            "llm_ms_clip_ms": 250.0,
            "tool_ms_clip_ms": 125.0,
            "iface_penalty": 0.01,
        }
    )
    cfg.rl.prime_rl.max_turns = 12

    env_args = build_train_env_args(cfg, num_examples=4)

    assert env_args["reward"] == {
        "mode": "outcome_round_cost",
        "h_max": 12,
        "lambda_cost": 0.2,
        "c_ref_ms": 900.0,
        "llm_ms_clip_ms": 250.0,
        "tool_ms_clip_ms": 125.0,
        "iface_penalty": 0.01,
    }


def test_build_train_env_args_normalizes_deprecated_reward_aliases():
    cfg = _make_cfg(
        reward={
            "mode": "outcome_speed_cost",
        }
    )

    env_args = build_train_env_args(cfg, num_examples=4)

    assert env_args["reward"] == {
        **_expected_reward("outcome_round_cost"),
    }


def test_build_train_env_args_uses_scenario_parallel_tool_limit():
    cfg = _make_cfg()
    cfg.scenario.max_parallel_tool_calls = 3

    env_args = build_train_env_args(cfg, num_examples=4)

    assert env_args["max_parallel_tool_calls"] == 3


def test_build_train_env_args_passes_resolved_generator_configs():
    cfg = _make_cfg()
    cfg.generators.decoys_enabled = True
    cfg.generators.password_reuse.service_users = ["deploy"]

    env_args = build_train_env_args(cfg, num_examples=4)

    assert env_args["generator_configs"]["decoys_enabled"] is True
    assert env_args["generator_configs"]["password_reuse"]["service_users"] == [
        "deploy"
    ]
