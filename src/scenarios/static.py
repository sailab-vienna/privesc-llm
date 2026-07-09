"""Static scenario source - loads scenarios from YAML config files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from src.config import ScenarioConfig
from .types import ScenarioInstance

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENARIOS_DIR = PROJECT_ROOT / "conf" / "scenarios"


def load_yaml_with_defaults(scenario_name: str) -> dict[str, Any]:
    """Load scenario YAML and merge with defaults."""
    scenario_path = SCENARIOS_DIR / f"{scenario_name}.yaml"
    default_path = SCENARIOS_DIR / "default.yaml"

    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario config not found: {scenario_path}")

    defaults = _load_yaml(default_path) if default_path.exists() else {}
    scenario_data = _load_yaml(scenario_path)

    if "defaults" in scenario_data:
        for default_name in scenario_data["defaults"]:
            if isinstance(default_name, str):
                default_file = SCENARIOS_DIR / f"{default_name}.yaml"
                if default_file.exists():
                    defaults.update(_load_yaml(default_file))

    merged = {**defaults, **scenario_data}
    merged.pop("defaults", None)
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def build_config_from_yaml(scenario_name: str, data: dict[str, Any]) -> ScenarioConfig:
    """Build ScenarioConfig from parsed YAML data."""
    container_user = _resolve_env_value(
        data.get("container_user", "lowpriv"),
        env_key="SCENARIO_USER",
        default="lowpriv",
    )
    container_password = _resolve_env_value(
        data.get("container_password", "trustno1"),
        env_key="SCENARIO_PASS",
        default="trustno1",
    )

    return ScenarioConfig(
        name=scenario_name,
        backend=data.get("backend", "remote_ssh"),
        image=data.get("image") or f"privesc_{scenario_name}",
        base_command_timeout=data.get("base_command_timeout", 1),
        slow_command_timeout=data.get("slow_command_timeout", 10),
        max_command_timeout=data.get("max_command_timeout", 90),
        term_cols=data.get("term_cols", 80),
        term_rows=data.get("term_rows", 24),
        container_user=container_user,
        container_password=container_password,
        extra_args=data.get("extra_args", []),
        pre_command=data.get("pre_command"),
        command=data.get("command"),
        ready_command=data.get("ready_command"),
        ready_command_timeout=data.get("ready_command_timeout"),
        log_message=data.get("log_message"),
        use_registry_mirror=data.get("use_registry_mirror", False),
        manual_sshd_start=data.get("manual_sshd_start", False),
        setup_script=data.get("setup_script"),
    )


def _resolve_env_value(value: Any, env_key: str, default: str) -> str:
    if isinstance(value, str) and value.startswith("${"):
        return os.getenv(env_key, default)
    return value if isinstance(value, str) else default


class StaticScenarioSource:
    """Loads scenarios from conf/scenarios/*.yaml files.

    Uses round-robin selection across scenarios with the same pattern as procedural:
      - scenario = scenarios[run_index % len(scenarios)]
    """

    def __init__(self, scenarios: str | list[str]) -> None:
        self._scenarios = [scenarios] if isinstance(scenarios, str) else list(scenarios)
        if not self._scenarios:
            raise ValueError("At least one scenario must be specified")

    @property
    def scenarios(self) -> list[str]:
        """List of scenario names."""
        return self._scenarios

    def build(self, run_index: int) -> ScenarioInstance:
        """Build a scenario instance from a run index.

        Scenario is selected round-robin: scenarios[run_index % len(scenarios)]
        """
        scenario_name = self._scenarios[run_index % len(self._scenarios)]
        yaml_data = load_yaml_with_defaults(scenario_name)

        solution = yaml_data.pop("solution", None)
        config = build_config_from_yaml(scenario_name, yaml_data)

        return ScenarioInstance(
            id=scenario_name,
            config=config,
            solution=solution,
            metadata={"source_type": "static", "scenario_name": scenario_name},
        )
