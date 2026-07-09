"""Live local_docker checks for sampled procedural solution guidance."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.generators.base import load_gtfobins_allowlist
from src.generators.capabilities_gtfobins import CapabilitiesGtfobinsGenerator
from src.generators.credential_artifact import (
    PasswordFileGenerator,
    PasswordHistoryGenerator,
)
from src.generators.cron_wildcard import CronWildcardGenerator
from src.generators.cron_writable_script import CronWritableScriptGenerator
from src.generators.password_reuse import PasswordReuseGenerator
from src.generators.ssh_key_reuse import SshKeyReuseGenerator
from src.generators.sudo_gtfobins import SudoGtfobinsGenerator, load_sudo_allowlist
from src.generators.suid_gtfobins import SuidGtfobinsGenerator
from src.generators.weak_password import WeakPasswordGenerator
from src.gym.scenario import PrivEscScenario

from test.solution_utils import run_exploit

from .test_generators_base import local_docker_scenario_config, local_docker_ssh_config

LIVE_GUIDANCE_GENERATORS = (
    "capabilities_gtfobins",
    "suid_gtfobins",
    "sudo_gtfobins",
    "password_file",
    "password_history",
    "password_reuse",
    "weak_password",
    "ssh_key_reuse",
    "cron_wildcard",
    "cron_writable_script",
)


def _preferred_binary(allowlist: list[str], preferred: tuple[str, ...]) -> str:
    normalized = [str(item) for item in allowlist]
    for candidate in preferred:
        if candidate in normalized:
            return candidate
    if normalized:
        return sorted(normalized)[0]
    raise AssertionError("Need at least one allowlisted binary")


_SUID_BINARY = _preferred_binary(
    load_gtfobins_allowlist("suid_allowlist"),
    ("chroot", "xargs", "rsync", "php", "node"),
)
_SUDO_BINARY = _preferred_binary(
    load_sudo_allowlist(),
    ("zip", "mawk", "taskset", "timeout"),
)
_CAPABILITIES_BINARY = _preferred_binary(
    load_gtfobins_allowlist("capabilities_allowlist"),
    ("python", "perl", "php", "ruby", "node"),
)


def _password_reuse_sample() -> object:
    return PasswordReuseGenerator(
        config={
            "patterns": [
                {"name": "root_account_reuse", "add_service_distractor": True}
            ],
            "service_users": ["backup"],
            "decoys_enabled": True,
        }
    ).generate(seed=0)


def _weak_password_sample() -> object:
    generator = WeakPasswordGenerator()
    return WeakPasswordGenerator(
        config={
            "patterns": [
                {
                    "name": "uid0_username_password",
                    "target_role": "uid0",
                    "password_source": "target_username",
                }
            ],
            "common_passwords": list(generator.common_passwords),
            "uid0_users": ["monitoring"],
            "decoy_users": ["runner"],
            "decoy_count_min": generator.decoy_count_min,
            "decoy_count_max": generator.decoy_count_max,
        }
    ).generate(seed=0)


def _sample_scenario(generator_name: str):
    if generator_name == "capabilities_gtfobins":
        return CapabilitiesGtfobinsGenerator(
            config_dir=Path("conf/gtfobins/catalog/capabilities")
        ).generate(seed=0, binary_name=_CAPABILITIES_BINARY)
    if generator_name == "suid_gtfobins":
        return SuidGtfobinsGenerator(
            config_dir=Path("conf/gtfobins/catalog/suid")
        ).generate(seed=0, binary_name=_SUID_BINARY)
    if generator_name == "sudo_gtfobins":
        return SudoGtfobinsGenerator(
            config_dir=Path("conf/gtfobins/catalog/sudo")
        ).generate(seed=0, binary_name=_SUDO_BINARY)
    if generator_name == "password_file":
        return PasswordFileGenerator().generate(seed=0)
    if generator_name == "password_history":
        return PasswordHistoryGenerator().generate(seed=0)
    if generator_name == "password_reuse":
        return _password_reuse_sample()
    if generator_name == "weak_password":
        return _weak_password_sample()
    if generator_name == "ssh_key_reuse":
        return SshKeyReuseGenerator().generate(seed=0)
    if generator_name == "cron_wildcard":
        return CronWildcardGenerator().generate(seed=0)
    if generator_name == "cron_writable_script":
        return CronWritableScriptGenerator().generate(seed=0)
    raise AssertionError(f"Unexpected generator: {generator_name}")


def _step_commands(scenario) -> list[str]:
    return [
        step["arguments"]["command"]
        for step in scenario.solution["exploit_tool_calls"]
        if step["function"] == "exec_command"
    ]


@pytest.mark.parametrize("generator_name", LIVE_GUIDANCE_GENERATORS)
def test_sampled_solution_guidance_has_required_recon_steps(
    generator_name: str,
) -> None:
    scenario = _sample_scenario(generator_name)
    steps = scenario.solution["exploit_tool_calls"]
    commands = _step_commands(scenario)
    meta = scenario.metadata

    if generator_name in {
        "capabilities_gtfobins",
        "suid_gtfobins",
        "sudo_gtfobins",
    }:
        assert commands == [meta["preferred_exploit_cmd"]]
    elif generator_name == "password_file":
        assert commands == [f"cat {meta['artifact_path']}"]
    elif generator_name == "password_history":
        assert commands[0] == meta["artifact_search_cmd"]
        assert meta["artifact_path"] not in commands[0]
        assert commands[1] == f"cat {meta['artifact_path']}"
    elif generator_name == "password_reuse":
        assert commands == []
        assert steps[0]["arguments"] == {
            "user": "root",
            "password": scenario.container_password,
        }
    elif generator_name == "weak_password":
        assert steps[0]["arguments"]["command"] == meta["uid0_discovery_cmd"]
        assert steps[1]["arguments"] == {
            "user": meta["intended_target_user"],
            "password": meta["credential_candidates"][0]["password"],
        }
    elif generator_name == "ssh_key_reuse":
        assert commands == [f"ls -la {meta['ssh_dir']}", meta["exploit_cmd"]]
    elif generator_name == "cron_wildcard":
        assert commands == [meta["exploit_cmd"], meta["wait_cmd"]]
    elif generator_name == "cron_writable_script":
        assert commands == [meta["exploit_cmd"], meta["wait_cmd"]]
    else:
        raise AssertionError(generator_name)

    root_steps = [step for step in steps if step["expected_result"]["got_root"]]
    assert len(root_steps) == 1
    assert steps[-1] == root_steps[0]


@pytest.mark.parametrize("generator_name", LIVE_GUIDANCE_GENERATORS)
def test_sampled_solution_guidance_has_single_primary_root_path(
    generator_name: str,
) -> None:
    scenario = _sample_scenario(generator_name)
    steps = scenario.solution["exploit_tool_calls"]

    assert not scenario.solution.get("alternative_exploits")
    root_steps = [step for step in steps if step["expected_result"]["got_root"]]
    assert len(root_steps) == 1
    assert steps[-1] == root_steps[0]
    assert steps[-1]["function"] in {"exec_command", "test_credentials"}


@pytest.mark.parametrize("generator_name", LIVE_GUIDANCE_GENERATORS)
def test_sampled_solution_text_avoids_exact_password_values(
    generator_name: str,
) -> None:
    scenario = _sample_scenario(generator_name)
    meta = scenario.metadata
    text = " ".join(
        [
            scenario.solution["description"],
            scenario.solution["vulnerability"],
            scenario.solution["hint"],
        ]
    )

    if generator_name in {"password_file", "password_history"}:
        sensitive_values = [meta["root_password"]]
    elif generator_name in {"password_reuse", "weak_password"}:
        sensitive_values = [meta["root_password"], meta["user_password"]]
    else:
        sensitive_values = []

    for sensitive_value in sensitive_values:
        assert sensitive_value not in text


@pytest.mark.asyncio
@pytest.mark.slow
@pytest.mark.parametrize("generator_name", LIVE_GUIDANCE_GENERATORS)
async def test_sampled_solutions_reach_root_in_local_docker(
    generator_name: str,
) -> None:
    scenario = _sample_scenario(generator_name)
    scenario_config = local_docker_scenario_config(scenario)

    async with PrivEscScenario(local_docker_ssh_config(), scenario_config) as sc:
        final_result = None
        for exploit in scenario.solution["exploit_tool_calls"]:
            final_result = await run_exploit(
                sc, exploit, scenario.category, is_alternative=False
            )

        assert final_result is not None
        assert final_result.got_root is True
