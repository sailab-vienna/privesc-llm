import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.config import ScenarioConfig, SSHConfig
from src.gym.scenario import PrivEscScenario
from .solution_utils import run_exploit

SCENARIOS_DIR = Path(__file__).parent.parent / "conf" / "scenarios"

ALL_SCENARIOS = [
    "01_vuln_suid_gtfo",
    "02_vuln_password_in_shell_history",
    "03_vuln_sudo_no_password",
    "05_vuln_sudo_gtfo",
    "06_vuln_docker",
    "07_root_password_reuse_mysql",
    "08_root_password_reuse",
    "09_root_password_root",
    "10_root_allows_lowpriv_to_ssh",
    "13_file_with_root_password",
]

# Cron scenarios are slow (require waiting for cron execution) - run separately
SLOW_SCENARIOS = [
    "11_cron_calling_user_wildcard",
    "12_cron_calling_user_file",
]

TIMEOUTS = {
    "06_vuln_docker": 10,
}


def load_scenario_yaml(scenario_name: str) -> dict[str, Any]:
    """Load scenario YAML including defaults and solution."""
    default_file = SCENARIOS_DIR / "default.yaml"
    scenario_file = SCENARIOS_DIR / f"{scenario_name}.yaml"

    defaults = {}
    if default_file.exists():
        with open(default_file) as f:
            defaults = yaml.safe_load(f) or {}

    if not scenario_file.exists():
        raise FileNotFoundError(f"Scenario config not found: {scenario_file}")

    with open(scenario_file) as f:
        data = yaml.safe_load(f) or {}

    data.pop("defaults", None)
    return {**defaults, **data}


def load_scenario_config(scenario_name: str) -> ScenarioConfig:
    """Load ScenarioConfig from YAML."""
    data = load_scenario_yaml(scenario_name)
    data.pop("solution", None)
    data["name"] = scenario_name

    # Resolve env var interpolations
    for key in ["container_user", "container_password"]:
        val = data.get(key, "")
        if isinstance(val, str) and val.startswith("${"):
            data[key] = os.getenv(
                "SCENARIO_USER" if "user" in key else "SCENARIO_PASS",
                "lowpriv" if "user" in key else "trustno1",
            )

    backend_override = os.getenv("PRIVESC_SCENARIO_BACKEND")
    if backend_override:
        data["backend"] = backend_override

    return ScenarioConfig(**data)


def get_test_configs(scenario_name: str) -> tuple[SSHConfig, ScenarioConfig]:
    """Create SSH and scenario configs from environment."""
    scenario_config = load_scenario_config(scenario_name)

    ssh_config = SSHConfig(
        user=os.getenv("PRIVESC_USER", "root"),
        key_path=os.getenv("PRIVESC_KEY", ""),
        servers=os.getenv("PRIVESC_SSH_SERVERS", ""),
    )
    if scenario_config.backend == "remote_ssh" and not ssh_config.key_path:
        raise ValueError("Set PRIVESC_KEY environment variable.")

    if scenario_name in TIMEOUTS:
        scenario_config.base_command_timeout = TIMEOUTS[scenario_name]
    else:
        scenario_config.base_command_timeout = 1

    return ssh_config, scenario_config


class TestPrivEscSolutions:
    """Test that scenarios are exploitable using their embedded solutions."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("scenario_name", ALL_SCENARIOS)
    async def test_scenario_exploit(self, scenario_name: str):
        await self._run_exploit_test(scenario_name)

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.parametrize("scenario_name", SLOW_SCENARIOS)
    async def test_slow_scenario_exploit(self, scenario_name: str):
        await self._run_exploit_test(scenario_name)

    async def _run_exploit_test(self, scenario_name: str):
        # Load solution from YAML
        data = load_scenario_yaml(scenario_name)
        solution = data.get("solution") or {}
        if not solution:
            pytest.skip(f"No solution in {scenario_name}.yaml")

        try:
            ssh_config, scenario_config = get_test_configs(scenario_name)
        except ValueError as e:
            pytest.skip(f"Environment not configured: {e}")

        async with PrivEscScenario(ssh_config, scenario_config) as scenario:
            for exploit in solution.get("exploit_tool_calls", []):
                await run_exploit(
                    scenario, exploit, scenario_name, is_alternative=False
                )

            for exploit in solution.get("alternative_exploits", []):
                await run_exploit(scenario, exploit, scenario_name, is_alternative=True)
