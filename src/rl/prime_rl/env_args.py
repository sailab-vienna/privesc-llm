"""Prime-RL environment argument helpers."""

from __future__ import annotations

from math import ceil
from typing import Any

from src.config import AppConfig, plain_config_dict, resolve_reward_config
from src.gym.prompts import base_template_vars_for_rl


def _build_shared_env_args(cfg: AppConfig, *, num_examples: int) -> dict[str, Any]:
    reward_cfg = resolve_reward_config(cfg.reward)
    reward_args = reward_cfg.to_dict()
    reward_args["h_max"] = cfg.rl.prime_rl.max_turns
    return {
        "max_turns": cfg.rl.prime_rl.max_turns,
        "system_template": cfg.prompts.system_template,
        "system_template_vars": base_template_vars_for_rl(
            dict(cfg.prompts.template_vars or {})
        ),
        "start_instruction": cfg.prompts.start_instruction,
        "no_tool_calls_nudge": cfg.prompts.no_tool_calls_nudge,
        "enable_auto_tool_choice": cfg.rl.prime_rl.inference_enable_auto_tool_choice,
        "num_examples": num_examples,
        "base_command_timeout": cfg.scenario.base_command_timeout,
        "slow_command_timeout": cfg.scenario.slow_command_timeout,
        "max_command_timeout": cfg.scenario.max_command_timeout,
        "max_parallel_tool_calls": cfg.scenario.max_parallel_tool_calls,
        "reward": reward_args,
    }


def build_train_env_args(cfg: AppConfig, *, num_examples: int) -> dict[str, Any]:
    source = cfg.rl.prime_rl.source
    return {
        **_build_shared_env_args(cfg, num_examples=num_examples),
        "source_type": "procedural",
        "scenario_backend": cfg.rl.prime_rl.scenario_backend,
        "generators": list(source.generators),
        "generator_configs": plain_config_dict(
            cfg.generators,
            error_label="cfg.generators",
        ),
        "seed": source.seed,
        "random_seed": bool(source.random_seed),
    }


def build_eval_env_args(
    cfg: AppConfig, *, benchmark_scenarios: list[str], num_examples: int
) -> dict[str, Any]:
    return {
        **_build_shared_env_args(cfg, num_examples=num_examples),
        "source_type": "static",
        "scenario_backend": cfg.rl.prime_rl.scenario_backend,
        "scenarios": benchmark_scenarios,
    }


def derive_train_num_examples(cfg: AppConfig) -> int:
    prime_cfg = cfg.rl.prime_rl
    generators = list(prime_cfg.source.generators)
    if not generators:
        raise ValueError(
            "Procedural RL training requires rl.prime_rl.source.generators"
        )
    if prime_cfg.rollouts_per_example < 1:
        raise ValueError("rl.prime_rl.rollouts_per_example must be >= 1")
    if prime_cfg.batch_size < 1:
        raise ValueError("rl.prime_rl.batch_size must be >= 1")

    problems_per_step = ceil(prime_cfg.batch_size / prime_cfg.rollouts_per_example)
    return max(len(generators), problems_per_step)


def build_train_envs(cfg: AppConfig, env_id: str) -> list[dict[str, Any]]:
    generators = list(cfg.rl.prime_rl.source.generators)
    if len(set(generators)) != len(generators):
        raise ValueError("rl.prime_rl.source.generators must be unique")

    total_examples = derive_train_num_examples(cfg)
    base_args = build_train_env_args(cfg, num_examples=total_examples)
    base_count, remainder = divmod(total_examples, len(generators))

    envs: list[dict[str, Any]] = []
    for idx, generator in enumerate(generators):
        per_env_examples = base_count + (1 if idx < remainder else 0)
        envs.append(
            {
                "id": env_id,
                "name": f"privesc/{generator}",
                "args": {
                    **base_args,
                    "generators": [generator],
                    "num_examples": per_env_examples,
                },
            }
        )
    return envs
