"""Solution loading from scenario YAML files."""

import logging
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_solution_data(scenario_name: str, relative_to: str = "") -> dict[str, Any]:
    """Load solution data from conf/scenarios/<scenario>.yaml."""
    scenario_path = PROJECT_ROOT / "conf" / "scenarios" / f"{scenario_name}.yaml"

    if not scenario_path.exists():
        log.warning("No solution found for scenario %s", scenario_name)
        return {}

    try:
        with open(scenario_path) as f:
            data = yaml.safe_load(f) or {}
        return data.get("solution", {})
    except (yaml.YAMLError, OSError) as e:
        log.warning("Failed to load solution for %s: %s", scenario_name, e)
        return {}
