"""Shared helpers for static benchmark evaluation."""

from __future__ import annotations

from pathlib import Path

from omegaconf import ListConfig, OmegaConf


def load_benchmark_scenarios() -> list[str]:
    """Load static benchmark scenario names from conf/runner/eval_benchmark.yaml."""
    cfg_path = (
        Path(__file__).resolve().parents[2] / "conf" / "runner" / "eval_benchmark.yaml"
    )
    data = OmegaConf.load(cfg_path)
    scenarios = OmegaConf.select(data, "source.scenarios")
    if not isinstance(scenarios, (list, ListConfig)) or not scenarios:
        raise ValueError(f"{cfg_path}: source.scenarios must be a non-empty list")
    return [str(s) for s in scenarios]
