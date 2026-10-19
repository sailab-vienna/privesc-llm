"""Scenario source abstraction for static and procedural scenarios."""

from typing import Any

from src.config import SourceConfig, plain_config_dict
from src.generators.base import load_generator_profile_config
from .types import ScenarioInstance, ScenarioSource
from .static import StaticScenarioSource
from .procedural import ProceduralScenarioSource


def build_scenario_source(
    source_cfg: SourceConfig,
    generators_cfg: Any = None,
) -> StaticScenarioSource | ProceduralScenarioSource:
    """Build a scenario source from configuration."""
    resolved_generators_cfg = plain_config_dict(generators_cfg)
    if source_cfg.generator_profile:
        resolved_generators_cfg = load_generator_profile_config(
            source_cfg.generator_profile
        )
        if not resolved_generators_cfg:
            raise ValueError(
                f"Unknown generator profile: {source_cfg.generator_profile}"
            )

    if source_cfg.type == "static":
        scenarios = source_cfg.scenarios
        if not scenarios:
            raise ValueError("Static source requires at least one scenario")
        return StaticScenarioSource(scenarios=scenarios)

    elif source_cfg.type == "procedural":
        if not source_cfg.generators:
            raise ValueError("Procedural source requires 'generators'")

        return ProceduralScenarioSource(
            generators=source_cfg.generators,
            base_seed=source_cfg.seed,
            random_seed=source_cfg.random_seed,
            randomize_env=source_cfg.randomize_env,
            generator_configs=resolved_generators_cfg,
        )

    raise ValueError(f"Unknown source type: {source_cfg.type}")


__all__ = [
    "ScenarioInstance",
    "ScenarioSource",
    "StaticScenarioSource",
    "ProceduralScenarioSource",
    "build_scenario_source",
]
