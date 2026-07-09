"""Core types for scenario source abstraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from src.config import ScenarioConfig


@dataclass(frozen=True)
class ScenarioInstance:
    """A fully resolved scenario ready for execution."""

    id: str
    config: ScenarioConfig
    solution: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ScenarioSource(Protocol):
    """Protocol for scenario sources.

    Both static and procedural sources use round-robin selection:
      - item = items[run_index % len(items)]
      - seed = base_seed + run_index (procedural only)

    Where "items" is either scenarios (static) or generators (procedural).

    The scheduler (runner.build_schedule) produces indices based on:
      - Runs per item (runs_per_item)
      - Optional max_runs cap
      - Which items already have enough traces (skip logic)

    Sources consume indices and produce ScenarioInstance objects.
    """

    def build(self, run_index: int) -> ScenarioInstance: ...
