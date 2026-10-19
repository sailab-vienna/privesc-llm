"""Procedural scenario source - generates scenarios using registered generators."""

from __future__ import annotations

import random
from typing import Any

from src.generators import GENERATOR_REGISTRY
from src.generators.base import GeneratedScenario
from src.generators.factory import build_generator
from .types import ScenarioInstance


class ProceduralScenarioSource:
    """Generates scenarios using procedural generators with deterministic seeding.

    Uses round-robin selection across generators with deterministic seeding.
    The run_index determines both which generator is used and the seed:
      - generator = generators[run_index % len(generators)]
      - seed = base_seed + run_index

    When randomize_env=False (ablation mode), all runs for a given generator share
    the same environment layout. The trace seed (stored in metadata) still varies per
    run so deduplication and split isolation remain correct.
    """

    def __init__(
        self,
        generators: list[str],
        base_seed: int = 42,
        random_seed: bool = False,
        randomize_env: bool = True,
        generator_configs: dict[str, Any] | None = None,
    ) -> None:
        self._generators = generators
        self._base_seed = base_seed
        self._random_seed = random_seed
        self._randomize_env = randomize_env
        self._generator_configs = generator_configs or {}

        for gen_name in generators:
            if gen_name not in GENERATOR_REGISTRY:
                raise ValueError(
                    f"Unknown generator '{gen_name}'. "
                    f"Available: {list(GENERATOR_REGISTRY.keys())}"
                )

    @property
    def generators(self) -> list[str]:
        """List of generator names."""
        return self._generators

    def _build_generator(self, generator_name: str):
        return build_generator(generator_name, self._generator_configs)

    def build(self, run_index: int) -> ScenarioInstance:
        """Build a scenario instance from a run index.

        Generator is selected round-robin: generators[run_index % len(generators)]
        Seed is deterministic: base_seed + run_index

        When randomize_env=False, the generator receives a fixed per-generator seed
        so all runs for that generator produce the same environment. The trace seed
        (used for deduplication) still increments normally.
        """
        if self._random_seed:
            # Deterministic pseudo-random seed so trace collection is reproducible.
            # Generator profiles define split leakage boundaries; base_seed keeps
            # trace IDs reproducible and collision-free within each profile.
            rng = random.Random(self._base_seed + run_index)
            seed = rng.randrange(2**31)
        else:
            seed = self._base_seed + run_index

        generator_idx = run_index % len(self._generators)
        generator_name = self._generators[generator_idx]
        generator = self._build_generator(generator_name)

        layout_seed = (
            self._base_seed + generator_idx if not self._randomize_env else seed
        )
        generated: GeneratedScenario = generator.generate(layout_seed)

        config = generated.to_scenario_config()

        metadata: dict = {
            "source_type": "procedural",
            "generator_name": generator_name,
            "seed": seed,
            "category": generated.category,
            "hint": generated.hint,
            **generated.metadata,
        }
        if not self._randomize_env:
            metadata["layout_seed"] = layout_seed

        return ScenarioInstance(
            id=generator_name,
            config=config,
            solution=generated.solution,
            metadata=metadata,
        )
